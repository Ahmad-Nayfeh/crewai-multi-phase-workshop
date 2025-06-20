# phase2_flow_and_custom_tool.py

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.flow.flow import Flow, start, listen
from crewai.tools import tool

# ----- 1. Define AGENTS -----

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

# ----- 2. Define the CREW from Phase 1 -----

def make_crew(topic):
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
        markdown=True
    )

    return Crew(
        agents=[researcher, analyst, report_writer],
        tasks=[research_task, analysis_task, report_writing_task],
        process=Process.sequential,
        verbose=True
    )

# ----- 3. Custom Tool: post_to_telegram -----

@tool("Post to Telegram")
def post_to_telegram(report_text: str) -> str:
    """
    Simulates posting a report to Telegram by printing it to the console.
    """
    print("\n\n==== [Simulated Telegram Post] ====")
    print(report_text)
    print("==== [End Telegram Post] ====\n\n")
    return "Posted to Telegram (simulated)!"

# ----- 4. Define the Flow -----

from pydantic import BaseModel

class FlowState(BaseModel):
    topic: str = ""
    report: str = ""

class ReportOrchestrationFlow(Flow[FlowState]):
    @start()
    def get_topic(self):
        # You could make this dynamic; we'll use a fixed value for this demo
        self.state.topic = "CrewAI workflow automation in 2025"
        print(f"\n[Flow] Topic set to: {self.state.topic}")
        return self.state.topic

    @listen(get_topic)
    def run_crew(self, topic):
        print("\n[Flow] Running the research-analysis-report crew...")
        crew = make_crew(topic)
        result = crew.kickoff()
        self.state.report = str(result)
        return self.state.report

    @listen(run_crew)
    def post_report_to_telegram(self, report):
        print("[Flow] Posting the report to Telegram...")
        return post_to_telegram.run(report)

# ----- 5. Run the Flow -----

if __name__ == "__main__":
    flow = ReportOrchestrationFlow()
    result = flow.kickoff()
    print("\n========== FLOW COMPLETE ==========\n")
    print("Final step result:", result)
    print("\nCheck the console above for the simulated Telegram post!\n")
