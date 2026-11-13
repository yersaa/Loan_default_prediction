"""Единый источник чисел для отчёта и презентации: всё читается из reports/, ничего не вписано руками."""
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

AUTHOR = "Kilybayev Yersaiyn"
GROUP = "AAI-2602M"
COURSE = "Programming for Data Analysis and Database"
TITLE = "Прогноз дефолта по займам LendingClub"

# Единственные числа, которые вписываются вручную: обновите после `pytest --cov=src --cov=main --cov=app`
TEST_COUNT, COVERAGE_PCT = 114, 94

BRAND = {"pine": "14342B", "green": "2E7D5B", "mint": "DCEAE3", "tint": "F1F6F3", "amber": "D9901A",
         "rust": "B9472E", "ink": "1C2421", "muted": "5B6B65", "line": "C9D6D0", "white": "FFFFFF"}


def ru(value: float, digits: int = 3) -> str:
    """Десятичная запятая, как принято в русском тексте."""
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def _json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@dataclass
class Results:
    metrics: pd.DataFrame
    selection: dict
    censoring: pd.DataFrame
    approval: pd.DataFrame
    recovery: pd.DataFrame
    errors: pd.DataFrame
    shap_full: pd.DataFrame
    shap_borrower: pd.DataFrame
    benchmarks: dict
    audit: dict
    ingest: dict
    tuning: dict
    default_by_grade: pd.DataFrame
    vintage: pd.DataFrame
    purpose: pd.DataFrame
    dti: pd.DataFrame
    fico_home: pd.DataFrame
    income: pd.DataFrame
    loss: pd.DataFrame
    state: pd.DataFrame
    metadata: dict
    feature_count: int

    def m(self, model: str, column: str) -> float:
        return float(self.metrics.set_index("model").loc[model, column])

    def fit_mb(self, model: str) -> float:
        """Память обучения модели, замеренная в отдельном процессе (пункт меню 8)."""
        return float(self.benchmarks["model_fit_memory"][model]["fit_peak_mb"])

    def censored(self, cohort_prefix: str, model: str, column: str) -> float:
        rows = self.censoring[self.censoring["cohort"].str.startswith(cohort_prefix)]
        return float(rows.set_index("model").loc[model, column])

    def approve(self, model: str, rate: float, column: str) -> float:
        rows = self.approval[(self.approval["model"] == model) & (self.approval["approval_rate"] == rate)]
        return float(rows.iloc[0][column])


def load() -> Results:
    return Results(
        metrics=_csv("model_metrics.csv"), selection=_json("model_selection.json"),
        censoring=_csv("censoring_sensitivity.csv"), approval=_csv("approval_table.csv"),
        recovery=_csv("grade_recovery.csv"), errors=_csv("error_by_grade.csv"),
        shap_full=_csv("shap_importance.csv"), shap_borrower=_csv("shap_importance_borrower.csv"),
        benchmarks=_json("benchmarks.json"), audit=_json("audit.json"), ingest=_json("ingest.json"),
        tuning=_json("tuning.json"), default_by_grade=_csv("sql/default_by_grade.csv"),
        vintage=_csv("sql/vintage_by_year_term.csv"), purpose=_csv("sql/purpose_having.csv"),
        dti=_csv("sql/dti_deciles.csv"), fico_home=_csv("sql/fico_home_matrix.csv"),
        income=_csv("sql/income_vs_grade_avg.csv"), loss=_csv("sql/loss_by_grade.csv"),
        state=_csv("sql/state_rank.csv"),
        metadata=json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8")),
        feature_count=len(json.loads((ROOT / "models" / "feature_defaults.json").read_text(encoding="utf-8"))),
    )
