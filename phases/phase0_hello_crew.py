"""
PHASE 0: Hello CrewAI

Goal:
- Run your first two-agent CrewAI pipeline.
- Learn: basic agent/task structure, context passing, and console output.

Skills:
- CrewAI installation, API keys, agent definition, task chaining, process running.
"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# --- AGENTS ---
researcher = Agent(
    role="Researcher",
    goal="Find key facts about CrewAI",
    backstory="An expert at online research and fact-finding.",
    tools=[SerperDevTool()],
    verbose=True
)

writer = Agent(
    role="Summary Writer",
    goal="Write a concise summary from research findings.",
    backstory="Skilled at making complex info easy for beginners.",
    verbose=True
)

# --- TASKS ---
topic = "CrewAI basics and use cases"

research_task = Task(
    description=f"Research: '{topic}'. List 3 key facts or use cases.",
    expected_output="A bullet list of 3 key facts about CrewAI.",
    agent=researcher,
    tools=[SerperDevTool()]
)

summary_task = Task(
    description="Summarize the research findings in plain English, max 100 words.",
    expected_output="A readable, 2-3 sentence summary for beginners.",
    agent=writer,
    context=[research_task]
)

# --- CREW ---
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, summary_task],
    process=Process.sequential,
    verbose=True
)

# --- RUN ---
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n====== FINAL SUMMARY ======")
    print(result)
    print("===========================")
