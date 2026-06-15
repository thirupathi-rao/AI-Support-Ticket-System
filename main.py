import pandas as pd
import numpy as np
import re
import os
import streamlit as st
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("groq_api_key")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)
# llm = OllamaLLM(
#     model="qwen3.5:4b", 
#     temperature=0,
#     base_url="http://127.0.0.1:11434" 
# )

# --- 1. DATA INGESTION ---
@st.cache_data
def load_data():
    """Loads dataset with robust fallback."""
    try:
        df = pd.read_csv("support_tickets.csv")
    except FileNotFoundError:
        try:
            df = pd.read_excel("support_tickets (2).xlsx")
        except FileNotFoundError:
            st.error("Dataset not found! Please place support_tickets.csv in the directory.")
            st.stop()

    df['resolution_time_hrs'] = df['resolution_time_hrs'].fillna(-1)
    df['customer_rating'] = df['customer_rating'].fillna(-1)
    return df

df = load_data()

# --- 2. ANOMALY DETECTION ---
def detect_anomalies(data):
    """Flags anomalies using statistical and rule-based logic."""
    # Rule 1: SLA Breach (Critical priority, Open status, response time > 24 hrs)
    sla_breach = data[(data['status'] == 'Open') & 
                      (data['priority'] == 'Critical') & 
                      (data['response_time_hrs'] > 24)]
    
    # Rule 2: Statistical Anomaly (Resolution time > 2 std deviations from mean)
    valid_res = data[data['resolution_time_hrs'] > 0]
    if not valid_res.empty:
        mean_res = np.mean(valid_res['resolution_time_hrs'])
        std_res = np.std(valid_res['resolution_time_hrs'])
        long_res = valid_res[valid_res['resolution_time_hrs'] > (mean_res + 2 * std_res)]
    else:
        long_res = pd.DataFrame()
    
    return sla_breach, long_res

# --- 3. LANGGRAPH PANDAS AGENT ---
class GraphState(TypedDict):
    question: str
    code: str
    result: str
    error: str
    iterations: int

@st.cache_resource
def build_agent_graph():
    """Builds and compiles the LangGraph StateGraph (cached for performance)."""
    
    # Ensure you have your Groq API key set in your environment
    # os.environ["GROQ_API_KEY"] = "your_actual_key_here" # Uncomment and paste if not set in terminal
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

def generate_code_node(state: GraphState):
        question = state["question"]
        error = state.get("error", "")
        iterations = state.get("iterations", 0)
        
        priorities = df['priority'].unique().tolist()
        statuses = df['status'].unique().tolist()
        
        # PROMPT UPDATED: Clarified exactly when to filter the -1 values
        prompt_str = f"""
        You are an expert Python data analyst. You have a pandas DataFrame named `df`.
        
        Data Context:
        - Unique Priorities: {priorities}
        - Unique Statuses: {statuses}
        
        First 3 rows:
        {df.head(3).to_string()}
        
        CRITICAL RULE: `customer_rating` and `resolution_time_hrs` use -1 for nulls. 
        ONLY filter these out (e.g., df['customer_rating'] >= 0) if you are calculating an average, min, or max on those specific columns. 
        Do NOT filter them out if you are just counting rows, filtering by status, or looking for unresolved tickets!
        
        User Question: {question}
        
        Task: Write Python pandas code to answer the question.
        - Assume `df` is loaded.
        - Assign your final calculated answer to a variable named `final_answer`.
        - Return ONLY the exact Python code. Do NOT wrap the code in markdown.
        """
        
        if error:
            prompt_str += f"\n\nYOUR PREVIOUS ATTEMPT FAILED.\nError:\n{error}\nRewrite the code to fix this."
        
        raw_response = llm.invoke(prompt_str)
        
        # Robust cleaning
        clean_code = re.sub(r'^```[a-zA-Z]*\n|\n```$', '', raw_response.content.strip())
    
        return {"code": clean_code, "iterations": iterations + 1, "error": ""}

def execute_code_node(state: GraphState):
    """Node: Safely executes the generated code."""
    code = state["code"]
    local_vars = {"df": df, "pd": pd, "np": np}
    
    try:
        exec(code, {}, local_vars)
        if 'final_answer' in local_vars:
            # FIX: We add '"code": code' to the return dict so the UI can display it
            return {"result": local_vars['final_answer'], "error": "", "code": code}
        else:
            # Fallback if it forgot the variable
            return {"result": eval(code, {"df": df, "pd": pd, "np": np}), "error": "", "code": code}
    except Exception as e:
        # Pass code here too, so we can see what caused the error!
        return {"error": str(e), "result": "", "code": code}

# --- 5. DEFINE CONDITIONAL ROUTING ---

def decide_next_step(state: GraphState):
    """Edge: Decides whether to finish or self-correct."""
    error = state.get("error", "")
    iterations = state.get("iterations", 0)
    
    if error and iterations < 3:
        # If there's an error and we haven't hit the retry limit, route back to the coder
        print(f"--- ERROR CAUGHT: ROUTING TO SELF-CORRECTION (Attempt {iterations}) ---")
        return "generate_code"
    else:
        # If successful, or we ran out of retries, end the graph
        return END

# --- 6. COMPILE THE GRAPH ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("generate_code", generate_code_node)
workflow.add_node("execute_code", execute_code_node)

# Add edges
workflow.set_entry_point("generate_code")
workflow.add_edge("generate_code", "execute_code")
workflow.add_conditional_edges("execute_code", decide_next_step)

# Compile
app_graph = workflow.compile()

# --- 4. STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI Support Agent", layout="wide", page_icon="🎫")
st.title("🎫 End-to-End AI Support Ticket System")

# --- SIDEBAR: Metrics & Anomalies ---
with st.sidebar:
    st.header("System Health")
    st.metric("Total Tickets Ingested", len(df))
    st.metric("Currently Open Tickets", len(df[df['status'] == 'Open']))
    
    st.divider()
    st.header("🚨 Detected Anomalies")
    sla_breach, long_res = detect_anomalies(df)
    
    if not sla_breach.empty:
        st.error(f"⚠️ {len(sla_breach)} Critical SLA Breaches (>24h response)")
        st.dataframe(sla_breach[['ticket_id', 'response_time_hrs', 'agent_id']], hide_index=True)
    else:
        st.success("No Critical SLA Breaches.")
        
    if not long_res.empty:
        st.warning(f"⏳ {len(long_res)} Statistically Abnormal Resolution Times")
        st.dataframe(long_res[['ticket_id', 'resolution_time_hrs', 'agent_id']], hide_index=True)
    else:
        st.success("Resolution times within normal variance.")

# --- MAIN AREA: Natural Language Querying ---
st.subheader("Natural Language Query")
st.markdown("Ask questions about the data, e.g., *'Which agent resolved the most tickets this month?'*")

# Chat Interface
prompt_input = st.chat_input("Enter your query here...")

if prompt_input:
    # Display user question
    st.chat_message("user").write(prompt_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data via LangGraph & Groq..."):
            
            # Setup initial state for the graph
            initial_state = {"question": prompt_input, "iterations": 0}
            final_node_output = {}
            
            # Run the graph
            for output in app_graph.stream(initial_state):
                for key, value in output.items():
                    final_node_output = value
            
            # Display Results
            if final_node_output.get("error"):
                st.error(f"Graph failed after max retries. Last Error: {final_node_output['error']}")
            else:
                st.write("**Answer:**")
                st.write(final_node_output.get("result", "No result returned"))
                
            # Expose the final executed code for the evaluator to see
            with st.expander("View LangGraph Execution Logic"):
                st.code(final_node_output.get("code", "Code not available"), language="python")