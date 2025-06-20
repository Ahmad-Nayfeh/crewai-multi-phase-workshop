# phase3_creator_prototype.py

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel
import json

# ----- 1. PLANNING CREW -----

intake_agent = Agent(
    role="Intake Specialist",
    goal="Clearly understand the user's request for a Python function.",
    backstory="You are the front desk—make sure user needs are captured simply and accurately.",
    verbose=True
)

planner_agent = Agent(
    role="Python Function Planner",
    goal="Design a plan for the requested Python function in structured JSON.",
    backstory="Expert at turning requirements into clear function specifications.",
    verbose=True
)

def make_planning_crew(user_request):
    intake_task = Task(
        description=f"Restate and clarify this user request: '{user_request}'.",
        expected_output="A one-sentence restatement of the user's need for a Python function.",
        agent=intake_agent
    )
    plan_task = Task(
        description="Based on the intake clarification, create a JSON plan for a Python function. Use this schema: {\"function_name\": str, \"description\": str}.",
        expected_output="A JSON object with 'function_name' and 'description'.",
        agent=planner_agent,
        context=[intake_task]
    )
    return Crew(
        agents=[intake_agent, planner_agent],
        tasks=[intake_task, plan_task],
        process=Process.sequential,
        verbose=True
    )

# ----- 2. EXECUTION CREW -----

developer_agent = Agent(
    role="Python Developer",
    goal="Write Python code for the planned function.",
    backstory="Expert coder who follows clear requirements and best practices.",
    tools=[CodeInterpreterTool()],
    allow_code_execution=True,
    code_execution_mode="safe",  # Always prefer Docker for safety
    verbose=True
)

qa_agent = Agent(
    role="QA Reviewer",
    goal="Review Python code and provide helpful, concise feedback.",
    backstory="Code reviewer known for catching bugs and suggesting improvements.",
    verbose=True
)

def make_execution_crew(json_plan):
    dev_task = Task(
        description=f"Write a complete, well-commented Python function based on this JSON plan: {json.dumps(json_plan, indent=2)}",
        expected_output="A Python code block with the function implementation.",
        agent=developer_agent,
        tools=[CodeInterpreterTool()]
    )
    qa_task = Task(
        description="Review the function code for correctness, clarity, and safety. Give at least one suggestion for improvement if possible.",
        expected_output="A short QA feedback paragraph.",
        agent=qa_agent,
        context=[dev_task]
    )
    return Crew(
        agents=[developer_agent, qa_agent],
        tasks=[dev_task, qa_task],
        process=Process.sequential,
        verbose=True
    )

# ----- 3. FLOW ORCHESTRATION -----

class CreatorFlowState(BaseModel):
    user_request: str = ""
    plan_json: dict = {}
    qa_feedback: str = ""

class CreatorPrototypeFlow(Flow[CreatorFlowState]):
    @start()
    def get_user_request(self):
        # You could use input(), but for demo, let's use a variable
        self.state.user_request = "Create a Python function that calculates the factorial of a number."
        print(f"\n[Flow] User Request: {self.state.user_request}")
        return self.state.user_request

    @listen(get_user_request)
    def run_planning_crew(self, user_request):
        print("[Flow] Running planning crew...")
        planning_crew = make_planning_crew(user_request)
        result = planning_crew.kickoff()
        # Try to extract JSON from output (robust to some LLM output formatting)
        try:
            plan_json = None
            if isinstance(result, dict):
                plan_json = result
            else:
                # Sometimes the output is a string containing JSON
                plan_json = json.loads(result.raw)
            self.state.plan_json = plan_json
            print(f"[Flow] Plan JSON: {plan_json}")
            return plan_json
        except Exception as e:
            print("[Flow][ERROR] Could not parse plan JSON:", e)
            raise

    @listen(run_planning_crew)
    def run_execution_crew(self, plan_json):
        print("[Flow] Running execution crew...")
        execution_crew = make_execution_crew(plan_json)
        result = execution_crew.kickoff()
        # result is QA agent's feedback
        self.state.qa_feedback = str(result)
        print("[Flow] QA Feedback:", self.state.qa_feedback)
        return self.state.qa_feedback

# ----- 4. RUN THE FLOW -----

if __name__ == "__main__":
    flow = CreatorPrototypeFlow()
    result = flow.kickoff()
    print("\n========== CREATOR FLOW COMPLETE ==========")
    print("Final QA feedback:\n")
    print(result)
    print("\n===========================================")
