"""Сквозные проверки того, что увидит проверяющий сразу после клонирования: дашборд и демо-пайплайн."""
import os
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src.data_io import load_json

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not C.SAMPLE_PARQUET.exists(), reason="демо-выборка ещё не собрана")
def test_dashboard_runs_and_scores_an_application():
    app = AppTest.from_file(str(C.ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Аналитика", "Скоринг заявки", "Модели"]
    app.button[0].click().run()
    assert not app.exception
    assert any("PD =" in m.value for m in app.metric)


@pytest.mark.skipif(not C.SAMPLE_PARQUET.exists(), reason="демо-выборка ещё не собрана")
def test_sample_pipeline_runs_without_raw_data_and_keeps_published_reports():
    """main.py --sample --run-all: всё, кроме бенчмарков на исходном CSV, отрабатывает на выборке."""
    published = C.ROOT / "reports" / "model_metrics.csv"
    before = published.read_bytes() if published.exists() else None
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([sys.executable, "main.py", "--sample", "--run-all"], cwd=C.ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert "Traceback" not in done.stderr, done.stderr[-2000:]
    run = C.SAMPLE_RUN
    assert (run / "reports" / "model_metrics.csv").exists()
    assert (run / "models" / "registry.json").exists()
    assert load_json(run / "models" / "metadata.json")["sample_mode"] is True
    if before is not None:
        assert published.read_bytes() == before
