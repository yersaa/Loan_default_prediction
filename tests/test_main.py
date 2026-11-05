"""Консольное меню: повторный ввод, обработка ошибок, демо-режим."""
import builtins

import pytest

import main as app_main
from src import config as C
from src.data_io import load_json


def _feed(monkeypatch, answers):
    iterator = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _="": next(iterator))


@pytest.mark.unit
def test_ask_number_retries_until_valid(monkeypatch, capsys):
    _feed(monkeypatch, ["abc", "1000", "12,5"])
    assert app_main.ask_number("x", 5, 0, 100) == 12.5
    out = capsys.readouterr().out
    assert "Нужно число" in out and "Допустимо" in out


@pytest.mark.unit
def test_ask_number_default_on_enter(monkeypatch):
    _feed(monkeypatch, [""])
    assert app_main.ask_number("x", 5, 0, 100) == 5


@pytest.mark.unit
def test_ask_choice_retries_and_ignores_case(monkeypatch, capsys):
    _feed(monkeypatch, ["z", "rent"])
    assert app_main.ask_choice("x", ["RENT", "OWN"], "OWN") == "RENT"
    assert "Выберите" in capsys.readouterr().out


@pytest.mark.unit
def test_run_step_reports_missing_file_and_does_not_mark_done(tmp_project, monkeypatch, capsys):
    def boom():
        raise FileNotFoundError("нет данных")
    monkeypatch.setitem(app_main.STEPS, "2", boom)
    assert app_main.run_step("2") is False
    assert "нет данных" in capsys.readouterr().out
    assert load_json(C.STATE_FILE) is None


@pytest.mark.unit
def test_run_step_handles_interrupt(tmp_project, monkeypatch, capsys):
    def stop():
        raise KeyboardInterrupt
    monkeypatch.setitem(app_main.STEPS, "2", stop)
    assert app_main.run_step("2") is False
    assert "Прервано" in capsys.readouterr().out


@pytest.mark.unit
def test_run_step_marks_success(tmp_project, monkeypatch):
    monkeypatch.setitem(app_main.STEPS, "2", lambda: None)
    assert app_main.run_step("2") is True
    assert app_main.MENU["2"] in load_json(C.STATE_FILE)


@pytest.mark.unit
def test_menu_handles_unknown_choice_and_exit(tmp_project, monkeypatch, capsys):
    _feed(monkeypatch, ["42", "0"])
    app_main.menu_loop()
    assert "Нет такого пункта" in capsys.readouterr().out


@pytest.mark.unit
def test_menu_runs_selected_step(tmp_project, monkeypatch):
    calls = []
    monkeypatch.setitem(app_main.STEPS, "4", lambda: calls.append("4"))
    _feed(monkeypatch, ["4", "0"])
    app_main.menu_loop()
    assert calls == ["4"]


@pytest.mark.unit
def test_pipeline_continues_after_failure(tmp_project, monkeypatch):
    calls = []
    for key in app_main.PIPELINE:
        monkeypatch.setitem(app_main.STEPS, key, lambda k=key: calls.append(k))

    def fail():
        raise FileNotFoundError("x")
    monkeypatch.setitem(app_main.STEPS, "3", fail)
    assert app_main.run_pipeline() == ["3"]
    assert "8" in calls


@pytest.mark.unit
def test_prepare_is_skipped_in_sample_mode(tmp_project, monkeypatch, capsys):
    monkeypatch.setattr(C, "SAMPLE_MODE", True)
    app_main.step_prepare()
    assert "Демо-режим" in capsys.readouterr().out


@pytest.mark.unit
def test_activate_sample_redirects_outputs(monkeypatch):
    for name in ("PARQUET", "DB_PATH", "REPORTS", "FIGURES", "MODELS", "STATE_FILE", "SAMPLE_MODE"):
        monkeypatch.setattr(C, name, getattr(C, name))
    published = C.ROOT / "reports"
    C.activate_sample()
    assert C.SAMPLE_MODE is True
    assert C.PARQUET == C.SAMPLE_PARQUET
    assert C.REPORTS != published and C.SAMPLE_RUN in C.REPORTS.parents
    assert C.SAMPLE_RUN in C.MODELS.parents
    assert C.SHIPPED_MODELS == C.ROOT / "models"


@pytest.mark.unit
def test_parse_args():
    args = app_main.parse_args(["--sample", "--run-all"])
    assert args.sample and args.run_all and not args.make_sample


@pytest.mark.integration
def test_full_step_chain_on_synthetic_parquet(tmp_project, loans, fast_best, monkeypatch, capsys):
    """Шаги меню 2–5 на синтетическом Parquet, включая сообщения пользователю."""
    C.PARQUET.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(C.PARQUET, index=False)
    for key in ("2", "3", "4", "5"):
        assert app_main.run_step(key) is True
    assert "Графики" in capsys.readouterr().out
