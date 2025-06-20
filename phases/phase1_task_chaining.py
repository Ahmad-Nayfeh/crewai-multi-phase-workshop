"""
PHASE 1: Task Chaining with Three Agents

Goal:
- Build a 3-agent pipeline: Researcher → Analyst → Writer.
- Learn: multi-step task context, deeper chaining, sequential process.

Skills:
- Three agents, multi-task chaining, output context, bullet list and paragraph outputs.

Agents:
- Researcher: Finds key facts.
- Analyst: Analyzes and highlights insights.
- Writer: Turns analysis into a simple report.

Outputs:
- Console output and a Markdown report in outputs/report_phase1.md.
"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool

# --- AGENTS ---
researcher = Agent(
    role="Researcher",
    goal="Find key facts about CrewAI workflows",
    backstory="Expert at online research and info gathering.",
    tools=[SerperDevTool()],
    verbose=True
)

analyst = Agent(
    role="Analyst",
    goal="Analyze research and extract actionable insights.",
    backstory="Skilled in seeing the big picture and hidden value.",
    verbose=True
)

writer = Agent(
    role="Report Writer",
    goal="Write a clear, simple report from analysis.",
    backstory="Experienced in technical writing and user docs.",
    tools=[FileWriterTool()],
    verbose=True
)

# --- TASKS ---
topic = "CrewAI workflow best practices"

research_task = Task(
    description=f"Research: '{topic}'. List 3 important best practices for using CrewAI.",
    expected_output="A bullet list of 3 best practices for CrewAI workflows.",
    agent=researcher,
    tools=[SerperDevTool()]
)

analysis_task = Task(
    description="Analyze the research findings. Summarize main themes and highlight one surprising insight.",
    expected_output="A 2-3 sentence paragraph summarizing main insights and a bullet of the most surprising one.",
    agent=analyst,
    context=[research_task]
)

report_task = Task(
    description="Write a user-friendly Markdown report with a short intro, analysis, and closing.",
    expected_output="A Markdown report. Use headings and at least one bullet list.",
    agent=writer,
    context=[analysis_task],
    markdown=True,
    output_file="outputs/report_phase1.md"
)

# --- CREW ---
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,
    verbose=True
)

# --- RUN ---
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n====== REPORT CREATED ======")
    print("The Markdown report is saved in outputs/report_phase1.md")
    print("Report preview:\n")
    print(result)
    print("===========================")
