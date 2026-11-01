"""`onpar doctor`: every install check at once, and a next step that is a command."""

from __future__ import annotations

import json
from pathlib import Path

from onpar import cli, doctor


def test_doctor_runs_every_check_and_names_the_next_command(tmp_path, monkeypatch):
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    r = doctor.run()
    assert [c.name for c in r.checks][:3] == ["hardware", "capture gateway", "compiled extension"]
    assert r.next_step.startswith("onpar ")
    assert all(c.fix for c in r.checks if c.ok is not True), "every gap names its fix"


def test_a_log_without_its_key_is_blocking_and_exit_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "captures.log").write_bytes(b"\x00" * 64)
    assert cli.main(["doctor"]) == 1
    assert "capture key" in capsys.readouterr().out


def test_doctor_json_is_machine_readable_and_a_clean_machine_exits_zero(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["doctor", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc["blocking"] == []
    assert {c["name"] for c in doc["checks"]} >= {"hardware", "eval set", "serving engines"}


def test_an_eval_set_in_the_cwd_points_at_prove(tmp_path, monkeypatch):
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    Path("evalset.json").write_text("{}")
    assert doctor.run().next_step.startswith("onpar prove evalset.json")


def test_the_commands_an_agent_calls_all_speak_json(capsys):
    """The first-run audit found --json on 15 of 27 verbs. These four were the
    ones an agent reaches for between stages."""
    for argv in (["catalog-sources", "--json"], ["cache", "--json"]):
        assert cli.main(argv) == 0
        json.loads(capsys.readouterr().out)


def test_a_bad_context_value_is_a_usage_error_not_a_world_error():
    """exit 2 means fix the invocation; exit 1 means retry. `banana` is the former."""
    import pytest

    with pytest.raises(SystemExit) as e:
        cli.main(["fit", "--context", "banana"])
    assert e.value.code == 2


def test_catalog_with_an_id_shows_the_entry_instead_of_a_usage_error(capsys):
    assert cli.main(["catalog", "qwen3-30b-a3b"]) == 0
    out = capsys.readouterr().out
    assert "repo" in out and "onpar fit --explain qwen3-30b-a3b" in out
    assert cli.main(["catalog", "qwen3-30b-a3b", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "qwen3-30b-a3b"
    assert cli.main(["catalog", "nope"]) == 1
