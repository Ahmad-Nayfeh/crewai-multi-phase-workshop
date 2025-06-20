"""
PHASE 6: Retrieval-Augmented Generation (RAG) with PDF

Goal:
- Let an agent search inside a PDF and answer a user question using CrewAI's PDFSearchTool.
- Learn the core RAG pattern: semantic search, agent context, knowledge-augmented answers.

Skills:
- RAG tools, semantic search, context feeding, multi-modal knowledge workflows.

Requirements:
- Place a PDF in outputs/sample_phase6.pdf or update the path below.

Outputs:
- Answer printed to console.
"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
import os

# --- PDF FILE ---
pdf_path = "outputs/sample_phase6.pdf"

if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"Sample PDF not found at {pdf_path}. Please download a PDF and place it there.")

# --- AGENT ---
rag_agent = Agent(
    role="Knowledge Assistant",
    goal="Find and answer questions using info from a provided PDF document.",
    backstory="Expert at extracting and summarizing knowledge from files.",
    tools=[PDFSearchTool()],
    verbose=True
)

# --- TASK ---
question = "What is this PDF about?"  # Try changing to a specific fact from your document

rag_task = Task(
    description=f"Search the PDF at '{pdf_path}' and answer this question based only on the PDF: {question}",
    expected_output="A concise answer based on the PDF content.",
    agent=rag_agent,
    tools=[PDFSearchTool()]
)

# --- CREW ---
crew = Crew(
    agents=[rag_agent],
    tasks=[rag_task],
    process=Process.sequential,
    verbose=True
)

# --- RUN ---
if __name__ == "__main__":
    result = crew.kickoff(inputs={"pdf": pdf_path})  # CrewAI expects PDF path as input for the tool
    print("\n====== RAG (PDF) PIPELINE COMPLETE ======")
    print("Answer:", result)
    print("===========================")
