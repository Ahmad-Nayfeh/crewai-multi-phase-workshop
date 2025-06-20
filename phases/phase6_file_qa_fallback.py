"""
PHASE 6: General File QA Agents (PDF, JSON, CSV) — Fallback Tools (Improved)

Goal:
- Enable CrewAI agents to answer questions from PDF, JSON, or CSV files using robust custom Python tools.
- Reduce token usage and tool retries with clear type-checking and prompt guidance.

Requirements:
- outputs/sample_phase6.pdf, outputs/sample_phase6.json, outputs/sample_phase6.csv
- pip install PyPDF2

Outputs:
- Answers printed to console.
"""

from dotenv import load_dotenv
load_dotenv()

import os, json, csv
import PyPDF2

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# --- Check/print model for confidence ---
import os
print(f"Model set to: {os.getenv('OPENAI_MODEL_NAME')}")

PDF_PATH = "outputs/sample_phase6.pdf"
JSON_PATH = "outputs/sample_phase6.json"
CSV_PATH = "outputs/sample_phase6.csv"

# --- TOOLS ---

@tool("Simple PDF Text Extractor")
def extract_pdf_text(query: str) -> str:
    """
    Query must be a plain string (e.g. 'Amazon' or 'Python').
    Reads all PDF text and returns lines that mention the query string (case-insensitive).
    """
    if not isinstance(query, str):
        return "ERROR: Query must be a string like 'Amazon'."
    if not os.path.isfile(PDF_PATH):
        return "PDF not found."
    try:
        with open(PDF_PATH, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            all_text = ""
            for page in reader.pages:
                all_text += page.extract_text() or ""
            lines = [line.strip() for line in all_text.splitlines() if query.lower() in line.lower()]
            if lines:
                return "\n".join(lines)
            else:
                return f"No PDF lines mention '{query}'."
    except Exception as e:
        return f"Failed to read PDF: {e}"

@tool("Simple JSON Reader")
def read_json(query: str) -> str:
    """
    Query must be a plain string (e.g. 'Berlin').
    Returns JSON rows from the file containing the query string (case-insensitive).
    """
    if not isinstance(query, str):
        return "ERROR: Query must be a string like 'Berlin'."
    if not os.path.isfile(JSON_PATH):
        return "JSON file not found."
    try:
        with open(JSON_PATH) as f:
            data = json.load(f)
        results = [row for row in data if any(query.lower() in str(v).lower() for v in row.values())]
        if not results:
            return f"No matches for '{query}'."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Simple CSV Reader")
def read_csv(query: str) -> str:
    """
    Query must be a plain string (e.g. 'Manager').
    Returns CSV rows from the file containing the query string (case-insensitive).
    """
    if not isinstance(query, str):
        return "ERROR: Query must be a string like 'Manager'."
    if not os.path.isfile(CSV_PATH):
        return "CSV file not found."
    try:
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            results = [row for row in reader if any(query.lower() in str(v).lower() for v in row.values())]
        if not results:
            return f"No matches for '{query}'."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {e}"

# --- AGENTS ---
pdf_agent = Agent(
    role="PDF File Analyst",
    goal="Find lines in the PDF matching the user's query.",
    backstory="Knows how to scan PDF text for answers. Always use the query string, not full questions.",
    tools=[extract_pdf_text],
    verbose=True
)
json_agent = Agent(
    role="JSON Data Analyst",
    goal="Find answers in the JSON data file.",
    backstory="Knows how to filter structured data. Always use the query string, not full questions.",
    tools=[read_json],
    verbose=True
)
csv_agent = Agent(
    role="CSV Data Analyst",
    goal="Find answers in the CSV data file.",
    backstory="Can analyze spreadsheet data. Always use the query string, not full questions.",
    tools=[read_csv],
    verbose=True
)

# --- TASKS (Customize Questions) ---
pdf_question = "Amazon"     # Use a company or keyword exactly as written in your CV PDF
json_question = "Berlin"
csv_question = "Manager"

pdf_task = Task(
    description=f"Use the Simple PDF Text Extractor to find info about '{pdf_question}' in the file. Only pass the string to the tool, not a full question.",
    expected_output="PDF lines mentioning the query.",
    agent=pdf_agent,
    tools=[extract_pdf_text]
)
json_task = Task(
    description=f"Use the Simple JSON Reader to find info about '{json_question}' in the file. Only pass the string to the tool.",
    expected_output="A summary of any matching rows.",
    agent=json_agent,
    tools=[read_json]
)
csv_task = Task(
    description=f"Use the Simple CSV Reader to find info about '{csv_question}' in the file. Only pass the string to the tool.",
    expected_output="A summary of any matching rows.",
    agent=csv_agent,
    tools=[read_csv]
)

pdf_crew = Crew(
    agents=[pdf_agent],
    tasks=[pdf_task],
    process=Process.sequential,
    verbose=True
)
json_crew = Crew(
    agents=[json_agent],
    tasks=[json_task],
    process=Process.sequential,
    verbose=True
)
csv_crew = Crew(
    agents=[csv_agent],
    tasks=[csv_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("\n==== PDF FILE QA ====")
    if os.path.isfile(PDF_PATH):
        pdf_result = pdf_crew.kickoff()
        print("PDF answer:", pdf_result)
    else:
        print(f"(No PDF found at {PDF_PATH})")

    print("\n==== JSON FILE QA ====")
    if os.path.isfile(JSON_PATH):
        json_result = json_crew.kickoff()
        print("JSON answer:", json_result)
    else:
        print(f"(No JSON found at {JSON_PATH})")

    print("\n==== CSV FILE QA ====")
    if os.path.isfile(CSV_PATH):
        csv_result = csv_crew.kickoff()
        print("CSV answer:", csv_result)
    else:
        print(f"(No CSV found at {CSV_PATH})")

    print("\n==================")
