🎫 End-to-End AI Support Ticket System

📌 Overview

This project is an end-to-End AI system designed to ingest customer support ticket data, detect anomalies, and allow users to query the dataset using Natural Language.

Built for the DOTMappers AI Engineer Assessment, it features a Streamlit frontend and a custom LangGraph execution pipeline powered by Groq (Llama 3.3 70B) to translate natural language into secure, executable Pandas code.

🚀 Quick Start (Single Command Setup)

1. Prerequisites

Ensure you have Python 3.10+ installed.

# Clone the repository
git clone <your-repo-link>
cd <repo-folder>

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt


2. Environment Variables

This application uses Groq's high-speed inference API to bypass local hardware limitations and ensure lightning-fast responses.

# Mac/Linux
export GROQ_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY="your_api_key_here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"


3. Run the Application

Start the system with a single command:

streamlit run app.py


🧠 Architecture & Design Choices

To ensure deterministic, accurate code generation, this system avoids fragile standard LLM "agents" (which often struggle with JSON tool-calling) in favor of a Custom LangGraph State Machine.

How the LangGraph Pipeline Works:

Context Injection: The user's natural language query is injected into a prompt alongside the exact schema, data types, and unique categorical values of the dataset.

Code Generation Node (generate_code): The LLM acts as a pure reasoning engine, generating raw pandas Python code to solve the query.

Execution Node (execute_code): The system uses Python's exec() to safely run the generated code against the active Pandas DataFrame in memory.

The Self-Correcting Loop (Reflection): If the generated code throws a Python error (e.g., KeyError or SyntaxError), the graph catches the traceback, appends it to the prompt, and routes back to the LLM to debug its own code. It attempts this up to 3 times before failing gracefully.

Why Llama-3.3-70b-versatile via Groq?

Due to local compute constraints, running a 70B parameter model locally was not feasible for rapid development. Groq provides LPU-accelerated inference via a free tier, satisfying the "zero-cost" assessment constraint while providing the immense reasoning power required for flawless Pandas code generation.

📊 Features & Assessment Criteria Fulfillment

Data Ingestion: Loads support_tickets.csv into a Pandas DataFrame, caching it in Streamlit to prevent redundant disk reads. Includes a fallback to .xlsx for robust local development.

Natural Language Querying: Users can ask plain-english questions. The LLM translates this to Pandas code and executes it in real-time.

Anomaly Detection: Flags tickets based on two robust rules:

SLA Breach: Critical tickets that are 'Open' and have a response time > 24 hours.

Statistical Anomalies: Uses numpy to flag resolution times that exceed 2 standard deviations from the dataset mean.

UI Integration: Exposes all features through a clean, responsive Streamlit dashboard.

🧪 Example Queries

You can copy and paste these into the Streamlit chat interface:

"How many tickets are currently open?"

"Which agent resolved the most tickets this month?"

"What is the average customer rating for Technical category tickets?"

"How many critical tickets are unresolved?"

"Which agent has the lowest average customer rating?"

⚠️ Known Limitations & Future Scaling

Security (exec usage): Currently, the system uses exec() to evaluate code in the local environment. While fine for a controlled MVP, a true production system would route the generated code to an isolated, sandboxed environment (like an E2B container or a dedicated Docker pod) to prevent arbitrary code execution vulnerabilities.

Context Window Limits: The prompt currently injects df.head(3) and unique categorical values. If scaled to a massive SQL database with 100+ tables, this architecture would pivot to a Vector Database (RAG) to fetch relevant schema snippets dynamically before generation.