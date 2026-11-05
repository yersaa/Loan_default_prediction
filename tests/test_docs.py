"""Числа в README и ER-диаграмма не должны расходиться с результатами и схемой."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import config as C

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
METRICS = pd.read_csv(ROOT / "reports" / "model_metrics.csv").set_index("model")
README_ROWS = {
    "Только подгрейд LC (LogReg)": "Grade only (LogReg)",
    "**Logistic Regression, выбрана**": "Logistic Regression",
    "| LightGBM |": "LightGBM",
    "Logistic Regression (поля заёмщика)": "Logistic Regression (borrower only)",
    "LightGBM (поля заёмщика)": "LightGBM (borrower only)",
}


@pytest.mark.parametrize("label, model", README_ROWS.items())
def test_readme_metrics_match_reports(label, model):
    row = next(line for line in README.splitlines() if label in line).replace("*", "")
    for column, digits in (("valid_auc", 3), ("roc_auc", 3), ("gini", 3), ("ks", 3), ("pr_auc", 3), ("brier", 4)):
        assert f"{METRICS.loc[model, column]:.{digits}f}" in row, f"{model}: {column}"


def test_readme_names_the_model_chosen_by_the_rule():
    selection = json.loads((ROOT / "reports" / "model_selection.json").read_text(encoding="utf-8"))
    assert selection["selected"] == "Logistic Regression"
    gap = METRICS.loc["LightGBM", "valid_auc"] - METRICS.loc["Logistic Regression", "valid_auc"]
    assert gap < C.SIMPLICITY_MARGIN_AUC


def test_readme_cohort_sizes_match_metadata():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    assert f"{meta['train_rows']:,}".replace(",", " ") in README
    assert f"{meta['test_rows']:,}".replace(",", " ") in README


def test_er_diagram_parser_reads_every_table():
    sys.path.insert(0, str(ROOT / "docs"))
    from build_er import LAYOUT, RELATIONS, parse_schema

    tables = parse_schema((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))
    assert set(tables) == set(LAYOUT)
    assert ("is_matured", "INTEGER", "") in tables["outcomes"]
    assert ("loan_id", "INTEGER", "PK, FK") in tables["borrowers"]
    assert {parent for parent, *_ in RELATIONS} <= set(tables)
