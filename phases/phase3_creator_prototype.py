"""
PHASE 3: Multi-Crew Planning and Code Execution Pipeline

Goal:
- Chain two specialized crews: planning crew → execution crew, orchestrated by a Flow.
- Pass structured info (JSON plan) from one crew to the next.
- Let an agent write code (with CodeInterpreterTool/Docker), then QA agent reviews it.

Skills:
- Multi-crew flows, JSON output as contract, code execution tool, QA automation, error handling.

Agents:
- Intake: Clarify user intent.
- Planner: Output JSON function spec.
- Developer: Write Python function from JSON.
- QA: Review, suggest improvements.

Outputs:
- Console logs. The QA agent's review appears at the end.
"""

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel
import json

# --- PLANNING CREW ---
intake_agent = Agent(
    role="Intake Specialist",
    goal="Clarify and rephrase user requests for Python functions.",
    backstory="Acts as the interface between users and the planning team.",
    verbose=True
)

planner_agent = Agent(
    role="Function Planner",
    goal="Create a clear JSON plan for a Python function.",
    backstory="Expert at transforming user requests into technical specs.",
    verbose=True
)

def make_planning_crew(user_request):
    intake_task = Task(
        description=f"Clarify and rephrase this request: '{user_request}'.",
        expected_output="A one-sentence, clarified description of the function.",
        agent=intake_agent
    )
    plan_task = Task(
        description="Based on the clarification, create a JSON plan: {\"function_name\": str, \"description\": str}.",
        expected_output="A valid JSON object with 'function_name' and 'description'.",
        agent=planner_agent,
        context=[intake_task]
    )
    return Crew(
        agents=[intake_agent, planner_agent],
        tasks=[intake_task, plan_task],
        process=Process.sequential,
        verbose=True
    )

# --- EXECUTION CREW ---
developer_agent = Agent(
    role="Developer",
    goal="Implement the planned function in Python.",
    backstory="Writes robust Python code based on specs.",
    tools=[CodeInterpreterTool()],
    allow_code_execution=True,
    code_execution_mode="safe",
    verbose=True
)

qa_agent = Agent(
    role="QA Reviewer",
    goal="Review and critique the function code.",
    backstory="Looks for correctness, clarity, and suggests improvements.",
    verbose=True
)

def make_execution_crew(plan_json):
    dev_task = Task(
        description=f"Write a complete Python function based on this JSON: {json.dumps(plan_json)}",
        expected_output="A valid, well-commented Python function.",
        agent=developer_agent,
        tools=[CodeInterpreterTool()]
    )
    qa_task = Task(
        description="Review the function for correctness and suggest at least one improvement.",
        expected_output="QA feedback paragraph.",
        agent=qa_agent,
        context=[dev_task]
    )
    return Crew(
        agents=[developer_agent, qa_agent],
        tasks=[dev_task, qa_task],
        process=Process.sequential,
        verbose=True
    )

# --- FLOW ---
class CreatorState(BaseModel):
    user_request: str = ""
    plan_json: dict = {}
    qa_feedback: str = ""

class CreatorFlow(Flow[CreatorState]):
    @start()
    def get_user_request(self):
        self.state.user_request = "Write a Python function to check if a number is prime."
        print(f"\n[Flow] User Request: {self.state.user_request}")
        return self.state.user_request

    @listen(get_user_request)
    def run_planning_crew(self, user_request):
        print("[Flow] Running planning crew...")
        planning_crew = make_planning_crew(user_request)
        result = planning_crew.kickoff()
        # Extract JSON from result
        raw = result.raw.strip()
        if raw.startswith("```"):
            import re
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        plan_json = json.loads(raw)
        self.state.plan_json = plan_json
        print(f"[Flow] Plan JSON: {plan_json}")
        return plan_json

    @listen(run_planning_crew)
    def run_execution_crew(self, plan_json):
        print("[Flow] Running execution crew...")
        execution_crew = make_execution_crew(plan_json)
        result = execution_crew.kickoff()
        self.state.qa_feedback = str(result)
        print("[Flow] QA Feedback:\n", self.state.qa_feedback)
        return self.state.qa_feedback

# --- RUN ---
if __name__ == "__main__":
    flow = CreatorFlow()
    result = flow.kickoff()
    print("\n====== CREATOR FLOW COMPLETE ======")
    print("Final QA feedback:\n")
    print(result)
    print("===========================")
