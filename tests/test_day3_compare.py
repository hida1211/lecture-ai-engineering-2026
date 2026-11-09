from scripts.day3_compare import INSTRUCTIONS, VARIANT_INSTRUCTIONS, REMOVED_LINE, READ_TOOLS, main


def test_basic_variant_changes_exactly_one_line():
    assert INSTRUCTIONS.count(REMOVED_LINE) == 1
    assert VARIANT_INSTRUCTIONS == INSTRUCTIONS.replace(REMOVED_LINE, "")
    assert [tool.__name__ for tool in READ_TOOLS] == ["search_glossary", "get_lecture_summary", "make_quiz"]


def test_default_is_dry_run(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["day3_compare"])
    def fail_if_called():
        raise AssertionError("No model call allowed in the default mode")
    monkeypatch.setattr("scripts.day3_compare.build_model", fail_if_called)
    main()
    assert "API呼び出しなし" in capsys.readouterr().out
