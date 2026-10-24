import builtins
import sys
from types import SimpleNamespace

import pytest

import app.ask as ask_module


def test_ask_calls_generate_content_once_and_prints_metrics(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
            thoughts_token_count=3,
            total_token_count=22,
        ),
    )

    models = SimpleNamespace(generate_content=lambda **kwargs: response)
    client = SimpleNamespace(models=models)
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.time, "perf_counter", iter([10.0, 10.5]).__next__)

    assert ask_module.ask("質問") == "回答"

    output = capsys.readouterr().out
    assert "[応答時間] 0.50秒" in output
    assert "入力: 12, 出力: 7, 思考: 3, 合計: 22" in output


def test_chat_reuses_one_chat_and_exit(monkeypatch, capsys):
    responses = iter([
        SimpleNamespace(text="一つ目", usage_metadata=None),
        SimpleNamespace(
            text="二つ目",
            usage_metadata=SimpleNamespace(
                prompt_token_count=0,
                candidates_token_count=0,
                thoughts_token_count=None,
                total_token_count=0,
            ),
        ),
    ])
    sent_messages = []
    create_calls = []

    class FakeChat:
        def send_message(self, message):
            sent_messages.append(message)
            return next(responses)

    fake_chat = FakeChat()

    def create_chat(**kwargs):
        create_calls.append(kwargs)
        return fake_chat

    chats = SimpleNamespace(create=create_chat)
    client = SimpleNamespace(chats=chats)
    inputs = iter(["", "   ", "最初の質問", "次の質問", "exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr(ask_module.time, "perf_counter", iter([1.0, 1.25, 2.0, 2.1]).__next__)

    ask_module.run_chat(client)

    output = capsys.readouterr().out
    assert len(create_calls) == 1
    assert create_calls[0] == {"model": ask_module.MODEL}
    assert len(sent_messages) == 2
    assert sent_messages == ["最初の質問", "次の質問"]
    assert "入力: 不明, 出力: 不明, 思考: 不明, 合計: 不明" in output
    assert "入力: 0, 出力: 0, 思考: 不明, 合計: 0" in output
    assert "会話を終了します。" in output


def test_single_mode_uses_default_question_when_arg_is_omitted(monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(sys, "argv", ["app.ask"])
    monkeypatch.setattr(ask_module, "ask", lambda question: asked.append(question) or "回答")

    ask_module.main()

    output = capsys.readouterr().out
    assert asked == [ask_module.DEFAULT_QUESTION]
    assert ask_module.DEFAULT_QUESTION in output


def test_chat_rejects_position_argument(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.ask", "--chat", "質問"])

    with pytest.raises(SystemExit):
        ask_module.main()


def test_response_text_is_explicit_when_text_is_none():
    response = SimpleNamespace(text=None)
    assert ask_module.response_text(response) == "本文を取得できませんでした。"


def test_print_response_metrics_keeps_zero_and_unknown_separate(capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=None,
            thoughts_token_count=0,
            total_token_count=None,
        )
    )

    ask_module.print_response_metrics(response, 0.125)

    output = capsys.readouterr().out
    assert "[応答時間] 0.12秒" in output
    assert "入力: 0, 出力: 不明, 思考: 0, 合計: 不明" in output
