"""Google ADK の設定を確認する。モデル/APIは呼ばない。"""
import importlib.metadata
from adk_agent.glossary_agent.agent import root_agent
from app.agent import INSTRUCTIONS
from app import tools

assert root_agent.instruction == INSTRUCTIONS
assert root_agent.tools == [tools.search_glossary, tools.get_lecture_summary, tools.make_quiz]
assert root_agent.name == "glossary_agent"
print("google-adk", importlib.metadata.version("google-adk"))
print("instruction matches: True")
print("tools:", [tool.__name__ for tool in root_agent.tools])
print("model:", root_agent.model)
print("API calls: 0 (configuration check only)")
