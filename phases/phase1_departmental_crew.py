# phase1_departmental_crew.py

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# ----- 1. Define the AGENTS -----

researcher = Agent(
    role="Technology Research Analyst",
    goal="Research and collect up-to-date, relevant information about a given topic using online resources.",
    backstory="Expert in finding and validating online information for technical audiences.",
    tools=[SerperDevTool()],
    verbose=True
)

analyst = Agent(
    role="Insights Analyst",
    goal="Analyze research findings and extract actionable insights for a business audience.",
    backstory="Experienced in reviewing technical research and converting it into business-friendly insights.",
    verbose=True
)

report_writer = Agent(
    role="Report Writer",
    goal="Turn analysis into a polished, readable report for non-technical readers.",
    backstory="Skilled in clear, engaging writing and technical report formatting.",
    verbose=True
)

# ----- 2. Define the TASKS -----

topic = "Modern CrewAI workflow automation"

research_task = Task(
    description=f"Research the topic: '{topic}'. List the 5 most important facts, examples, or trends about CrewAI workflow automation.",
    expected_output="A bullet-point list of 5 essential facts/examples about CrewAI workflow automation.",
    agent=researcher,
    tools=[SerperDevTool()]
)

analysis_task = Task(
    description="Analyze the research findings. Summarize the main themes, highlight one surprising insight, and suggest a practical application.",
    expected_output="A paragraph summarizing the main themes, a highlight of one surprising insight, and a suggested practical use case.",
    agent=analyst,
    context=[research_task]
)

report_writing_task = Task(
    description="Write a clear, well-formatted markdown report that explains the findings and analysis to a non-technical audience. Use headings and bullet points where appropriate.",
    expected_output="A markdown report with a brief intro, sections for findings and analysis, and a closing statement.",
    agent=report_writer,
    context=[analysis_task],
    markdown=True,
    output_file="report.md"
)

# ----- 3. Create the CREW -----

crew = Crew(
    agents=[researcher, analyst, report_writer],
    tasks=[research_task, analysis_task, report_writing_task],
    process=Process.sequential,
    verbose=True
)

# ----- 4. Run the Crew and Print the Final Result -----

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n========== REPORT FILE CREATED ==========")
    print("The final report has been written to 'report.md' in your workspace!")
    print("\nSummary of report:\n")
    print(result)
    print("\n=========================================")
