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
    "| Logistic Regression |": "Logistic Regression",
    "**LightGBM, выбран**": "LightGBM",
    "Logistic Regression (поля заёмщика)": "Logistic Regression (borrower only)",
    "LightGBM (поля заёмщика)": "LightGBM (borrower only)",
}


@pytest.mark.parametrize("label, model", README_ROWS.items())
def test_readme_metrics_match_reports(label, model):
    row = next(line for line in README.splitlines() if label in line).replace("*", "")
    for column, digits in (("valid_auc", 3), ("roc_auc", 3), ("pr_auc", 3), ("ece", 4)):
        assert f"{METRICS.loc[model, column]:.{digits}f}" in row, f"{model}: {column}"


def test_readme_names_the_model_chosen_by_the_rule():
    selection = json.loads((ROOT / "reports" / "model_selection.json").read_text(encoding="utf-8"))
    assert selection["selected"] == "LightGBM"
    assert "**LightGBM, выбран**" in README
    gap = METRICS.loc["LightGBM", "valid_auc"] - METRICS.loc["Logistic Regression", "valid_auc"]
    assert gap >= C.SIMPLICITY_MARGIN_AUC  # отрыв больше запаса, поэтому правило оставляет LightGBM


def test_readme_cohort_sizes_match_metadata():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    assert f"{meta['train_rows']:,}".replace(",", " ") in README
    assert f"{meta['test_rows']:,}".replace(",", " ") in README


def test_notebook_is_executed_without_errors():
    """Ноутбук в репозитории сохранён с результатами: каждая ячейка кода выполнена и ни одна не упала."""
    import nbformat

    notebook = nbformat.read(ROOT / "project_walkthrough.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 25
    for cell in code_cells:
        assert cell.execution_count is not None, cell.source[:60]
        assert all(output.output_type != "error" for output in cell.outputs), cell.source[:60]


def test_every_notebook_cell_is_explained_in_the_guide():
    """У каждой ячейки кода есть объяснение в пояснительной записке, у каждого графика — подпись."""
    import nbformat

    sys.path.insert(0, str(ROOT / "docs"))
    from guide_text import CAPTIONS, EXPLAIN

    notebook = nbformat.read(ROOT / "project_walkthrough.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    keys = [cell.metadata.get("guide") for cell in code_cells]
    assert len(keys) == len(set(keys)), "ключи ячеек должны быть уникальны"
    assert set(keys) <= set(EXPLAIN)
    with_images = {cell.metadata["guide"] for cell in code_cells
                   if any("image/png" in output.get("data", {}) for output in cell.outputs)}
    assert len(with_images) >= 20
    assert with_images <= set(CAPTIONS)


def test_er_diagram_parser_reads_every_table():
    sys.path.insert(0, str(ROOT / "docs"))
    from build_er import LAYOUT, RELATIONS, parse_schema

    tables = parse_schema((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))
    assert set(tables) == set(LAYOUT)
    assert ("is_matured", "INTEGER", "") in tables["outcomes"]
    assert ("loan_id", "INTEGER", "PK, FK") in tables["borrowers"]
    assert {parent for parent, *_ in RELATIONS} <= set(tables)
