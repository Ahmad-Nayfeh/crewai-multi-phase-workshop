"""
PHASE 5: API Integration — Telegram Bot

Goal:
- Learn to integrate any external API (Telegram) with CrewAI using a custom tool.
- Agent sends a message to your Telegram via your bot.
- You can reuse this pattern for *any* HTTP API (Google Sheets, Slack, etc).

Skills:
- Secure API key usage, custom @tool, POST requests, agent-to-user automation.

Requirements:
- TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
- requests library

Outputs:
- Message sent to your Telegram from your CrewAI pipeline!
"""

from dotenv import load_dotenv
load_dotenv()

import os
import requests

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# --- CUSTOM TOOL ---
@tool("Send Telegram Message")
def send_telegram_message(text: str) -> str:
    """
    Sends a text message to a Telegram chat using your bot.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in environment."
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": chat_id,
        "text": text
    })
    if resp.status_code == 200:
        return "✅ Message sent to Telegram!"
    else:
        return f"❌ Failed to send: {resp.text}"

# --- AGENT ---
notifier = Agent(
    role="Notifier",
    goal="Send a summary to the user's Telegram using the bot API.",
    backstory="Expert in messaging and notifications.",
    tools=[send_telegram_message],
    verbose=True
)

# --- TASK ---
notify_task = Task(
    description="Send the message '👋 Hello from CrewAI Phase 5! Your pipeline works.' to the user's Telegram using your tool.",
    expected_output="Confirmation that the message was sent.",
    agent=notifier,
    tools=[send_telegram_message]
)

# --- CREW ---
crew = Crew(
    agents=[notifier],
    tasks=[notify_task],
    process=Process.sequential,
    verbose=True
)

# --- RUN ---
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n====== TELEGRAM NOTIFY PIPELINE COMPLETE ======")
    print("Result:", result)
    print("===========================")
