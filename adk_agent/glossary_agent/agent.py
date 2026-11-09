"""発展(a): 同じ指示・同じ読み取りツールを Google ADK で組み立てる。"""
from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
