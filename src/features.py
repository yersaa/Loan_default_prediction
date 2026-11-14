"""Целевая переменная, отбор созревших займов, очистка, новые признаки и временное разбиение."""
import numpy as np
import pandas as pd

from src import config as C


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """is_default: 1 — списан/дефолт, 0 — погашен, NaN — кредит ещё не завершён."""
    df[C.TARGET] = df["loan_status"].astype("object").map(C.STATUS_TO_TARGET).astype("float32")
    return df


def emp_to_years(value: object) -> float:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3."""
    if pd.isna(value):
        return np.nan
    if value.startswith("<"):
        return 0.0
    return float(value.split()[0].rstrip("+"))


def term_to_months(term: pd.Series) -> pd.Series:
    """'36 months' -> 36.0."""
    return term.astype("object").str.extract(r"(\d+)")[0].astype("float32")


def months_between(start: pd.Series, end: pd.Series | pd.Timestamp) -> pd.Series:
    """Сколько месяцев прошло от start до end: считаются только год и месяц, дни не важны."""
    end_year, end_month = (end.dt.year, end.dt.month) if isinstance(end, pd.Series) else (end.year, end.month)
    return (end_year - start.dt.year) * 12 + (end_month - start.dt.month)


def is_matured(issue_d: pd.Series, term_months: pd.Series) -> pd.Series:
    """Срок займа истёк к дате выгрузки.

    Незрелые займы отбрасывать только по статусу нельзя: среди ещё не истёкших завершены
    либо досрочно погашенные, либо уже списанные, поэтому дефолтность по ним смещена.
    """
    snapshot = pd.Timestamp(C.SNAPSHOT_YEAR, C.SNAPSHOT_MONTH, 1)
    return months_between(issue_d, snapshot) >= term_months


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Заменяет на NaN заведомо некорректные значения (правила обоснованы в аудите)."""
    df.loc[df["annual_inc"] <= 0, "annual_inc"] = np.nan
    df.loc[(df["dti"] < 0) | (df["dti"] >= 999), "dti"] = np.nan
    df.loc[df["revol_util"] > 150, "revol_util"] = np.nan
    return df.drop_duplicates(subset="id")


def add_income_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Общий код для обучения и скоринга: одна формула отношений к доходу."""
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"]).astype("float32")
    df["installment_to_income"] = (df["installment"] * 12 / df["annual_inc"]).astype("float32")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки, которые известны до выдачи займа."""
    df = clean_values(df)
    df["fico"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).astype("float32")
    df["credit_hist_months"] = months_between(df["earliest_cr_line"], df["issue_d"]).astype("float32")
    df["term_months"] = term_to_months(df["term"])
    df["emp_years"] = df["emp_length"].astype("object").map(emp_to_years).astype("float32")
    return add_income_ratios(df)


def modeling_frame(df: pd.DataFrame, matured_only: bool = True) -> pd.DataFrame:
    """Завершённые займы в окне MODEL_YEARS; по умолчанию только созревшие."""
    df = build_features(add_target(df))
    year = df["issue_d"].dt.year
    mask = df[C.TARGET].notna() & year.between(*C.MODEL_YEARS)
    if matured_only:
        mask &= is_matured(df["issue_d"], df["term_months"])
    return df.loc[mask].sort_values("issue_d", kind="stable").reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> dict:
    """Окна по году выдачи, без перемешивания.

    fit (2007–2013) и valid (2014) нужны для порога и калибровки,
    train (2007–2014) — для финальной модели, test (2015) — только для оценки.
    """
    year = df["issue_d"].dt.year
    return {"fit": df[year < C.VALID_YEAR], "valid": df[year == C.VALID_YEAR],
            "train": df[year < C.TEST_YEAR], "test": df[year == C.TEST_YEAR]}
