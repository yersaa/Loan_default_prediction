"""Таргет, созревание займов, очистка и защита от утечек."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import (
    add_target,
    build_features,
    clean_values,
    emp_to_years,
    is_matured,
    modeling_frame,
    months_between,
    temporal_split,
    term_to_months,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value, expected", [("10+ years", 10.0), ("< 1 year", 0.0), ("3 years", 3.0), ("1 year", 1.0)])
def test_emp_to_years(value, expected):
    assert emp_to_years(value) == expected


def test_emp_to_years_missing():
    assert np.isnan(emp_to_years(np.nan))


def test_term_to_months():
    assert term_to_months(pd.Series(["36 months", "60 months"])).tolist() == [36.0, 60.0]


@pytest.mark.parametrize("issue, term, expected", [
    ("2015-12-01", 36, True),    # ровно на границе выгрузки
    ("2016-01-01", 36, False),
    ("2013-12-01", 60, True),
    ("2014-01-01", 60, False),
    ("2010-06-01", 60, True),
])
def test_is_matured_boundaries(issue, term, expected):
    result = is_matured(pd.Series([pd.Timestamp(issue)]), pd.Series([term]))
    assert bool(result.iloc[0]) is expected


def test_months_between_counts_calendar_months():
    """Дни не учитываются; конец задаётся колонкой дат или одной датой."""
    start = pd.Series(pd.to_datetime(["2010-01-31", "2014-11-01"]))
    end = pd.Series(pd.to_datetime(["2010-02-01", "2015-01-15"]))
    assert months_between(start, end).tolist() == [1, 2]
    assert months_between(start, pd.Timestamp(2018, 12, 1)).tolist() == [107, 49]


def test_add_target_maps_statuses():
    df = pd.DataFrame({"loan_status": ["Fully Paid", "Charged Off", "Default", "Current", "In Grace Period",
                                       "Does not meet the credit policy. Status:Charged Off"]})
    target = add_target(df)[C.TARGET]
    assert target.tolist()[:3] == [0.0, 1.0, 1.0]
    assert target.iloc[3:5].isna().all()
    assert target.iloc[5] == 1.0


def test_clean_values_removes_invalid_and_duplicates():
    df = pd.DataFrame({"id": [1, 2, 3, 3], "annual_inc": [-5.0, 100.0, 100.0, 100.0],
                       "dti": [10.0, 999.0, -1.0, 5.0], "revol_util": [10.0, 151.0, 20.0, 20.0]})
    cleaned = clean_values(df)
    assert len(cleaned) == 3
    assert np.isnan(cleaned["annual_inc"].iloc[0])
    assert cleaned["dti"].iloc[1:].isna().all()
    assert np.isnan(cleaned["revol_util"].iloc[1])


def test_modeling_frame_keeps_only_matured_resolved(loans):
    frame = modeling_frame(loans.copy())
    assert frame[C.TARGET].notna().all()
    assert is_matured(frame["issue_d"], frame["term_months"]).all()
    assert frame["issue_d"].dt.year.between(*C.MODEL_YEARS).all()
    assert frame["issue_d"].is_monotonic_increasing


def test_censored_cohort_is_larger_and_contains_unmatured(loans):
    censored = modeling_frame(loans.copy(), matured_only=False)
    matured = modeling_frame(loans.copy())
    assert len(censored) > len(matured)
    assert (~is_matured(censored["issue_d"], censored["term_months"])).any()


def test_temporal_split_never_overlaps(loans):
    w = temporal_split(modeling_frame(loans.copy()))
    assert w["fit"]["issue_d"].max() < w["valid"]["issue_d"].min()
    assert w["train"]["issue_d"].max() < w["test"]["issue_d"].min()
    assert set(w["test"]["id"]).isdisjoint(w["train"]["id"])
    assert w["test"]["issue_d"].dt.year.eq(C.TEST_YEAR).all()


def test_test_window_is_36_month_only(loans):
    """Следствие правила созревания: 60-месячные займы 2015 г. ещё не истекли."""
    w = temporal_split(modeling_frame(loans.copy()))
    assert w["test"]["term_months"].eq(36).all()


def test_no_leakage_in_features():
    assert not set(C.LEAKAGE_COLS) & set(C.FEATURES)
    assert C.TARGET not in C.FEATURES


def test_borrower_features_exclude_lendingclub_decisions():
    borrower = set(C.BORROWER_NUM + C.BORROWER_CAT)
    assert not borrower & C.LC_ASSIGNED
    assert {"fico", "dti", "annual_inc", "term_months", "purpose"} <= borrower
    assert not {"grade", "sub_grade", "int_rate", "installment", "installment_to_income"} & borrower


def test_build_features_formulas(loans):
    row = build_features(add_target(loans.head(1).copy())).iloc[0]
    assert row["fico"] == pytest.approx((row["fico_range_low"] + row["fico_range_high"]) / 2)
    assert row["loan_to_income"] == pytest.approx(row["loan_amnt"] / row["annual_inc"], rel=1e-5)
    assert row["installment_to_income"] == pytest.approx(row["installment"] * 12 / row["annual_inc"], rel=1e-5)
