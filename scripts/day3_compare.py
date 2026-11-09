"""基礎の比較実行。既定では計画のみ表示し、APIは呼ばない。"""
import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from app import tools
from app.agent import INSTRUCTIONS, build_model, print_trace
from app.config import MODEL

QUESTION = "LoRAとQLoRAの違いは？"
REMOVED_LINE = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
VARIANT_INSTRUCTIONS = INSTRUCTIONS.replace(REMOVED_LINE, "")
READ_TOOLS = [tools.search_glossary, tools.get_lecture_summary, tools.make_quiz]


def build_variant(instructions):
    return Agent(instructions=instructions, tools=READ_TOOLS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Gemini APIを実際に呼ぶ。無料枠・課金・送信内容を確認してから実行")
    parser.add_argument("--output-dir", type=Path, default=Path("lectures/03/evidence/live"))
    args = parser.parse_args()
    assert INSTRUCTIONS.count(REMOVED_LINE) == 1
    print(f"モデル: {MODEL} / 同じ質問: {QUESTION}")
    print("変更はINSTRUCTIONSの検索必須の1行だけ。ツール・docstring・生成設定は同じ。")
    print("検索docstringには引き続き『答える前に使う』とあるため、呼び出しが減るとは限らない。")
    if not args.run_live:
        print("計画のみ表示。API呼び出しなし。実行には --run-live を指定する。")
        return
    # 送る内容は公開講座の指示・用語集・質問だけ。私的な学習ログはツールに含めない。
    model = build_model()
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, instructions in [("before", INSTRUCTIONS), ("after", VARIANT_INSTRUCTIONS)]:
        result = build_variant(instructions).run_sync(
            QUESTION, model=model,
            model_settings={"temperature": 0, "max_tokens": 2048},
            usage_limits=UsageLimits(request_limit=6, tool_calls_limit=12),
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print(f"variant={name}; model={MODEL}; question={QUESTION}")
            print(result.output)
            print_trace(result)
        (run_dir / f"{name}.txt").write_text(stream.getvalue(), encoding="utf-8")
        (run_dir / f"{name}_messages.json").write_bytes(result.all_messages_json())
        print(stream.getvalue())
    print(f"実行記録: {run_dir}")


if __name__ == "__main__":
    main()
