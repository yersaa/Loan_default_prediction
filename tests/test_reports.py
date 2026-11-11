"""Аудит, графики, таблица одобрения и бенчмарки на синтетических данных."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.audit import invalid_values, iqr_outliers, plot_missing, print_audit, run_audit
from src.benchmark import (
    TRAIN_SNAPSHOT,
    fit_memory,
    pandas_aggregate,
    results_match,
    run_benchmarks,
    save_train_snapshot,
    sql_aggregate,
)
from src.database import build_database, run_all_queries
from src.eda import run_eda
from src.explain import approval_table, error_analysis, grade_recovery, missed_defaults_profile, run_explain
from src.features import modeling_frame
from src.models import train_and_compare


@pytest.mark.unit
def test_iqr_outliers_counts_extreme_value():
    s = pd.Series(list(range(100)) + [10_000])
    info = iqr_outliers(s)
    assert info["count"] == 1
    assert info["high"] < 10_000


@pytest.mark.unit
def test_invalid_values_detects_each_rule():
    df = pd.DataFrame({
        "annual_inc": [0.0, 10.0], "dti": [999.0, 1.0], "revol_util": [200.0, 1.0],
        "fico_range_low": [800.0, 1.0], "fico_range_high": [700.0, 2.0],
        "earliest_cr_line": pd.to_datetime(["2020-01-01", "2000-01-01"]),
        "issue_d": pd.to_datetime(["2010-01-01", "2010-01-01"]), "int_rate": [50.0, 10.0],
    })
    assert all(n == 1 for n in invalid_values(df).values())


@pytest.mark.unit
def test_approval_table_by_hand():
    """Четыре займа: одобряя 50% с наименьшей PD, берём первые два."""
    test = pd.DataFrame({C.TARGET: [0.0, 1.0, 1.0, 1.0], "total_pymnt": [110.0, 50.0, 40.0, 30.0],
                         "loan_amnt": [100.0, 100.0, 100.0, 100.0]})
    table = approval_table(test, {"m": np.array([0.1, 0.2, 0.3, 0.4])}, rates=(0.5, 1.0))
    half = table[table["approval_rate"] == 0.5].iloc[0]
    assert half["default_rate_pct"] == pytest.approx(50.0)
    assert half["net_return_pct"] == pytest.approx(-20.0)
    assert half["max_pd"] == pytest.approx(0.2)
    everything = table[table["approval_rate"] == 1.0].iloc[0]
    assert everything["default_rate_pct"] == pytest.approx(75.0)
    assert everything["net_return_pct"] == pytest.approx(-42.5)


@pytest.mark.unit
def test_error_analysis_and_profile():
    test = pd.DataFrame({"grade": ["A", "A", "A", "B", "B", "B"], C.TARGET: [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                         "int_rate": 10.0, "fico": 700.0, "dti": 10.0, "annual_inc": 5e4, "loan_amnt": 1e4,
                         "revol_util": 50.0})
    p = np.array([0.1, 0.2, 0.05, 0.8, 0.1, 0.7])
    errors = error_analysis(test, p, 0.5).set_index("grade")
    assert errors.loc["A", "missed_defaults_pct"] == 100
    assert errors.loc["B", "missed_defaults_pct"] == 50
    assert errors.loc["B", "false_alarms_pct"] == 100
    assert missed_defaults_profile(test, p, 0.5).shape == (6, 3)


@pytest.mark.unit
def test_results_match_detects_difference():
    a = pd.DataFrame({"issue_year": [2015], "grade": ["A"], "loans": [10], "default_rate": [0.1]})
    b = a.assign(default_rate=[0.2])
    assert results_match(a, a.copy())
    assert not results_match(a, b)
    assert not results_match(a, a.assign(grade=["B"]))


@pytest.mark.integration
def test_audit_and_console_output(tmp_project, loans, capsys):
    report = run_audit(loans.copy())
    assert report["duplicate_ids"] == 0
    assert report["rows_after_removing_summary_lines"] == len(loans)
    assert report["raw_columns"] is None  # исходного CSV в тесте нет
    assert (C.FIGURES / "01_missing.png").exists()
    print_audit({**report, "missing_pct": {"x": 1.0}, "summary_lines_removed": 0})
    assert "Дубликатов id: 0" in capsys.readouterr().out


@pytest.mark.integration
def test_plot_missing_without_gaps(tmp_project):
    plot_missing(pd.Series({"a": 1.5, "b": 0.0}))
    assert (C.FIGURES / "01_missing.png").exists()


@pytest.mark.integration
def test_eda_and_explain_create_all_artifacts(tmp_project, loans, fast_best):
    build_database(loans.copy())
    run_all_queries()
    frame = modeling_frame(loans.copy())
    names = run_eda(loans.copy(), frame)
    for expected in ("02_default_by_grade.png", "03_vintage.png", "05_corr_heatmap.png", "07_default_by_state.html",
                     "08_monthly_rolling.html"):
        assert expected in names

    metrics, fitted, preds, preds_raw, test = train_and_compare(frame, fast_best)
    result = run_explain(fitted, preds, preds_raw, test, metrics)
    for figure in ("09_roc_pr.png", "10_confusion_matrix.png", "11_calibration.png", "12_shap_summary.png",
                   "12b_shap_borrower.png", "13_approval.png"):
        assert (C.FIGURES / figure).exists()
    assert set(result["shap_borrower"]["feature"]) == set(C.BORROWER_NUM + C.BORROWER_CAT)
    approval = result["approval"]
    # Самая дисциплинированная политика: при меньшей доле одобрения дефолтность не выше
    gbm = approval[approval["model"] == "LightGBM"].sort_values("approval_rate")
    assert gbm["default_rate_pct"].iloc[0] <= gbm["default_rate_pct"].iloc[-1]
    assert not grade_recovery(test, preds).empty


@pytest.mark.integration
def test_fit_memory_measures_one_model(tmp_project, loans):
    C.PARQUET.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(C.PARQUET, index=False)
    snapshot = save_train_snapshot()
    assert snapshot == C.REPORTS / TRAIN_SNAPSHOT
    assert C.TARGET in pd.read_parquet(snapshot).columns
    assert fit_memory("logistic") >= 0
    with pytest.raises(StopIteration):
        fit_memory("no_such_model")


@pytest.mark.integration
def test_benchmarks_need_tuned_models_before_measuring(tmp_project, loans):
    build_database(loans.copy())
    with pytest.raises(FileNotFoundError, match="пункта меню 6"):
        run_benchmarks()


@pytest.mark.integration
def test_pandas_and_sql_aggregates_agree(tmp_project, loans):
    C.PARQUET.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(C.PARQUET, index=False)
    build_database(loans.copy())
    assert results_match(pandas_aggregate(), sql_aggregate())
