"""
PHASE 4: File Tools & Agent File Collaboration

Goal:
- Learn how agents can write to and read from files using CrewAI tools.
- Demonstrate a pipeline where the Developer agent writes code to a file,
  and the QA agent reads and reviews that file.

Skills:
- FileWriterTool, FileReadTool, artifact pipeline, agent-to-agent file collaboration.

Outputs:
- Python code written to outputs/generated_code_phase4.py
- QA review printed to console.

Agents:
- Developer: Writes function to .py file
- QA: Reads .py file, reviews for correctness

"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import FileWriterTool, FileReadTool

import os

# --- AGENTS ---
developer = Agent(
    role="Developer",
    goal="Write a Python function to a file.",
    backstory="Expert Python developer, always documents code.",
    tools=[FileWriterTool()],
    verbose=True
)

qa = Agent(
    role="QA Reviewer",
    goal="Read and review the generated code file.",
    backstory="Meticulous reviewer, finds bugs and gives feedback.",
    tools=[FileReadTool()],
    verbose=True
)

# --- TASKS ---
function_name = "is_even"
code_filename = "outputs/generated_code_phase4.py"

write_task = Task(
    description=f"Write a Python function named '{function_name}' that checks if a number is even. Save the code to '{code_filename}'. Add a docstring and comments.",
    expected_output=f"A Python function in '{code_filename}' that checks if a number is even.",
    agent=developer,
    tools=[FileWriterTool()],
    output_file=code_filename
)

review_task = Task(
    description=f"Read the file '{code_filename}', review the function for correctness, docstring, and suggest at least one improvement.",
    expected_output="A brief QA review with at least one suggestion.",
    agent=qa,
    tools=[FileReadTool()],
    context=[write_task]
)

# --- CREW ---
crew = Crew(
    agents=[developer, qa],
    tasks=[write_task, review_task],
    process=Process.sequential,
    verbose=True
)

# --- RUN ---
if __name__ == "__main__":
    # Ensure output folder exists
    os.makedirs("outputs", exist_ok=True)
    result = crew.kickoff()
    print("\n====== FILE TOOLS PIPELINE COMPLETE ======")
    print(f"Function written to: {code_filename}")
    print("QA review:\n")
    print(result)
    print("===========================")
