"""
PHASE 2: Flows & Custom Python Tools

Goal:
- Learn CrewAI Flows for multi-step orchestration.
- Use a custom Python tool (@tool) in your workflow.

Skills:
- Flow class, state, @start and @listen event methods, passing context, using and calling custom tools.

Outputs:
- Prints report and "simulated Telegram post" to console.

Agents:
- Researcher: Web research
- Analyst: Analyze findings
- Writer: Markdown report

Custom Tool:
- post_to_telegram: Simulates posting to Telegram (prints to console)
"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.flow.flow import Flow, start, listen
from crewai.tools import tool
from pydantic import BaseModel

# --- AGENTS ---
researcher = Agent(
    role="Researcher",
    goal="Find the most recent news about CrewAI.",
    backstory="Always up-to-date on AI frameworks.",
    tools=[SerperDevTool()],
    verbose=True
)

analyst = Agent(
    role="Analyst",
    goal="Summarize and highlight the significance of CrewAI news.",
    backstory="Great at making sense of new trends.",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write a Markdown summary of CrewAI developments.",
    backstory="Communicates complex news simply.",
    verbose=True
)

# --- CREW DEFINITION FUNCTION ---
def make_crew(topic):
    research_task = Task(
        description=f"Research the latest about '{topic}'. List the top 2-3 new things.",
        expected_output="A bullet list of the most significant recent CrewAI news.",
        agent=researcher,
        tools=[SerperDevTool()]
    )
    analysis_task = Task(
        description="Analyze the research findings and explain why they're important.",
        expected_output="A 2-3 sentence explanation of the most significant insight.",
        agent=analyst,
        context=[research_task]
    )
    report_task = Task(
        description="Write a Markdown summary report of the findings and analysis.",
        expected_output="A Markdown report, with headings and bullet points.",
        agent=writer,
        context=[analysis_task],
        markdown=True
    )
    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True
    )

# --- CUSTOM TOOL ---
@tool("Post to Telegram")
def post_to_telegram(report_text: str) -> str:
    """
    Simulates posting a report to Telegram by printing it to the console.
    """
    print("\n==== [Simulated Telegram Post] ====")
    print(report_text)
    print("==== [End Telegram Post] ====\n")
    return "Posted to Telegram (simulated)!"

# --- FLOW DEFINITION ---
class FlowState(BaseModel):
    topic: str = ""
    report: str = ""

class CrewFlow(Flow[FlowState]):
    @start()
    def get_topic(self):
        self.state.topic = "CrewAI framework"
        print(f"\n[Flow] Topic: {self.state.topic}")
        return self.state.topic

    @listen(get_topic)
    def run_crew(self, topic):
        print("\n[Flow] Running crew...")
        crew = make_crew(topic)
        result = crew.kickoff()
        self.state.report = str(result)
        return self.state.report

    @listen(run_crew)
    def post_report(self, report):
        print("[Flow] Posting the report to Telegram...")
        return post_to_telegram.run(report)

# --- RUN ---
if __name__ == "__main__":
    flow = CrewFlow()
    result = flow.kickoff()
    print("\n====== FLOW COMPLETE ======")
    print("Final step result:", result)
    print("===========================")
