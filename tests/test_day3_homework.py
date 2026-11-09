"""Day3: 実モデルを呼ばず、ツールと承認境界を検証する。"""
import json

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from app import tools
from app.agent import agent, confirm_deferred_requests, run_with_approval


def test_quiz_uses_requested_lecture_definition(monkeypatch):
    monkeypatch.setattr(tools.random, "choice", lambda candidates: candidates[0])
    expected = next(e for e in tools.load_glossary() if e["lecture"] == 2)
    quiz = json.loads(tools.make_quiz(2))
    assert quiz == {"question": f"「{expected['term']}」の意味は？", "answer": expected["definition"], "lecture": 2}


@pytest.mark.parametrize("lecture", [0, -1, 999])
def test_quiz_handles_missing_lecture(lecture):
    assert "error" in json.loads(tools.make_quiz(lecture))


def test_quiz_is_exposed_and_called():
    result = agent.run_sync("確認問題", model=TestModel(call_tools=["make_quiz"]))
    parts = [p for m in result.all_messages() for p in m.parts]
    assert any(isinstance(p, ToolCallPart) and p.tool_name == "make_quiz" for p in parts)
    assert any(isinstance(p, ToolReturnPart) and p.tool_name == "make_quiz" for p in parts)


def test_save_requires_approval_before_writing(tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    pending = agent.run_sync("学習を記録", model=TestModel(call_tools=["save_learning_log"]))
    assert isinstance(pending.output, DeferredToolRequests)
    assert len(pending.output.approvals) == 1
    assert pending.output.approvals[0].tool_name == "save_learning_log"
    assert not log_dir.exists()


@pytest.mark.parametrize("approved", [True, False])
def test_approval_and_denial_have_correct_side_effects(tmp_path, monkeypatch, approved):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    model = TestModel(call_tools=["save_learning_log"])
    pending = agent.run_sync("学習を記録", model=model)
    call = pending.output.approvals[0]
    result = agent.run_sync(
        model=model,
        message_history=pending.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: approved}),
    )
    assert isinstance(result.output, str)
    paths = list(log_dir.glob("*.json"))
    assert len(paths) == (1 if approved else 0)
    if approved:
        record = json.loads(paths[0].read_text(encoding="utf-8"))
        assert record["topic"] == call.args_as_dict()["topic"]
        assert record["content"] == call.args_as_dict()["content"]


@pytest.mark.parametrize("reply, approved", [("", False), ("no", False), ("n", False), ("y", True), ("YES", True), ("maybe", False)])
def test_cli_confirmation_defaults_to_denial(monkeypatch, capsys, reply, approved):
    pending = agent.run_sync("学習を記録", model=TestModel(call_tools=["save_learning_log"]))
    monkeypatch.setattr("builtins.input", lambda _: reply)
    decisions = confirm_deferred_requests(pending.output)
    assert list(decisions.approvals.values()) == [approved]
    display = capsys.readouterr().out
    assert "save_learning_log" in display and "topic" in display and "content" in display


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_interrupted_confirmation_denies(monkeypatch, error):
    pending = agent.run_sync("学習を記録", model=TestModel(call_tools=["save_learning_log"]))
    def stop(_):
        raise error
    monkeypatch.setattr("builtins.input", stop)
    assert list(confirm_deferred_requests(pending.output).approvals.values()) == [False]


@pytest.mark.parametrize("reply", ["y", "n"])
def test_complete_cli_approval_loop(tmp_path, monkeypatch, reply):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    monkeypatch.setattr("builtins.input", lambda _: reply)
    result = run_with_approval(agent, "学習を記録", TestModel(call_tools=["save_learning_log"]))
    assert isinstance(result.output, str)
    assert len(list(log_dir.glob("*.json"))) == (1 if reply == "y" else 0)
    assert result.usage.requests == 2


def test_log_path_is_generated_and_existing_logs_are_not_overwritten(tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    first = json.loads(tools.save_learning_log("../../escape", "日本語の記録"))
    second = json.loads(tools.save_learning_log("../../escape", "日本語の記録"))
    assert first["saved"] and second["saved"]
    assert first["path"] != second["path"]
    paths = list(log_dir.glob("*.json"))
    assert len(paths) == 2
    assert all(p.parent == log_dir for p in paths)
    assert all(json.loads(p.read_text(encoding="utf-8"))["content"] == "日本語の記録" for p in paths)


@pytest.mark.parametrize("topic, content", [("", "memo"), ("theme", "  "), ("a" * 201, "memo"), ("theme", "a" * 20001)])
def test_invalid_log_does_not_create_directory(tmp_path, monkeypatch, topic, content):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    assert "error" in json.loads(tools.save_learning_log(topic, content))
    assert not log_dir.exists()
