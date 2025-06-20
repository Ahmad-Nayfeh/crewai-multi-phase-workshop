# phase0_hello_crew.py

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# ----- 1. Define the AGENTS -----

researcher = Agent(
    role="Web Research Specialist",
    goal="Find relevant and up-to-date information about a given topic",
    backstory="You are skilled at using search engines and filtering high-quality sources.",
    tools=[SerperDevTool()],
    verbose=True
)

summary_writer = Agent(
    role="Summary Writer",
    goal="Write concise and informative summaries of research findings",
    backstory="You are an expert at distilling complex info into readable summaries for beginners.",
    verbose=True
)

# ----- 2. Define the TASKS -----

topic = "CrewAI framework use cases"

research_task = Task(
    description=f"Research the topic: '{topic}'. List the 5 most important/useful things a developer should know. Use your web search tool.",
    expected_output="A bullet-point list of the 5 most important/useful facts for developers about CrewAI use cases.",
    agent=researcher,
    tools=[SerperDevTool()]
)

summary_task = Task(
    description="Take the research findings and write a short, easy-to-understand summary (max 150 words).",
    expected_output="A concise summary for beginners, covering the main findings from the research task.",
    agent=summary_writer,
    context=[research_task]
)

# ----- 3. Create the CREW -----

crew = Crew(
    agents=[researcher, summary_writer],
    tasks=[research_task, summary_task],
    process=Process.sequential,  # Task order matters!
    verbose=True
)

# ----- 4. Run the Crew and Print the Summary -----

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n========== FINAL SUMMARY ==========\n")
    print(result)
    print("\n===================================")
