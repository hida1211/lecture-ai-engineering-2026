"""第1回: 素のLLMに1回だけ質問するスクリプトと会話モード。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保持する対話モードを開始する

単発モードはツールも会話履歴も持ちません。--chatモードでは同じ会話へ質問を送り、
履歴を保持しながら対話します。どちらのモードでも講座に固有のことを聞くと、
実際の講座資料と一致しない可能性があります。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def format_token_count(value: int | None) -> str:
    return "不明" if value is None else str(value)


def print_response_metrics(response: object, elapsed_seconds: float) -> None:
    print(f"[応答時間] {elapsed_seconds:.2f}秒")

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        input_tokens = output_tokens = thoughts_tokens = total_tokens = None
    else:
        input_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        thoughts_tokens = getattr(usage, "thoughts_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)

    print(
        "[トークン数] "
        f"入力: {format_token_count(input_tokens)}, "
        f"出力: {format_token_count(output_tokens)}, "
        f"思考: {format_token_count(thoughts_tokens)}, "
        f"合計: {format_token_count(total_tokens)}"
    )


def response_text(response: object) -> str:
    text = getattr(response, "text", None)
    return "本文を取得できませんでした。" if text is None else text


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    start = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed_seconds = time.perf_counter() - start
    print_response_metrics(response, elapsed_seconds)

    return response_text(response)


def run_chat(client: object) -> None:
    chat = client.chats.create(model=MODEL)
    print("会話モードです。終了するには exit と入力してください。")

    while True:
        try:
            question = input("[質問] ")
        except (EOFError, KeyboardInterrupt):
            print("\n会話を終了します。")
            break

        if not question.strip():
            continue

        if question.strip().lower() == "exit":
            print("会話を終了します。")
            break

        start = time.perf_counter()
        response = chat.send_message(question.strip())
        elapsed_seconds = time.perf_counter() - start
        print(f"\n[回答]\n{response_text(response)}")
        print_response_metrics(response, elapsed_seconds)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持する対話モードで開始する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            if args.question is not None:
                parser.error("--chatでは質問を位置引数で指定できません")
            client = genai.Client(api_key=get_api_key())
            run_chat(client)
        else:
            question = args.question if args.question is not None else DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
