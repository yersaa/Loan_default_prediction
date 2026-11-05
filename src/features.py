"""Целевая переменная, очистка, новые признаки и временное разбиение без утечки."""
import numpy as np
import pandas as pd

from src import config as C


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """is_default: 1 — списан/дефолт, 0 — погашен, NaN — кредит ещё не завершён."""
    df[C.TARGET] = df["loan_status"].astype("object").map(C.STATUS_TO_TARGET).astype("float32")
    return df


def emp_to_years(value) -> float:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3."""
    if pd.isna(value):
        return np.nan
    if value.startswith("<"):
        return 0.0
    return float(value.split()[0].rstrip("+"))


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Заменяет на NaN заведомо некорректные значения (правила обоснованы в аудите)."""
    df.loc[df["annual_inc"] <= 0, "annual_inc"] = np.nan
    df.loc[(df["dti"] < 0) | (df["dti"] >= 999), "dti"] = np.nan
    df.loc[df["revol_util"] > 150, "revol_util"] = np.nan
    return df.drop_duplicates(subset="id")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки, которые известны на момент подачи заявки."""
    df = clean_values(df)
    df["fico"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).astype("float32")
    months = (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12 \
        + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
    df["credit_hist_months"] = months.astype("float32")
    df["term_months"] = df["term"].astype("object").str.extract(r"(\d+)")[0].astype("float32")
    df["emp_years"] = df["emp_length"].astype("object").map(emp_to_years).astype("float32")
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"]).astype("float32")
    df["installment_to_income"] = (df["installment"] * 12 / df["annual_inc"]).astype("float32")
    return df


def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Только завершённые займы в окне MODEL_YEARS."""
    df = build_features(add_target(df))
    year = df["issue_d"].dt.year
    mask = df[C.TARGET].notna() & year.between(*C.MODEL_YEARS)
    return df.loc[mask].sort_values("issue_d").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_year=C.TEST_YEAR):
    """Train — займы до test_year, test — займы test_year. Без перемешивания."""
    year = df["issue_d"].dt.year
    train, test = df[year < test_year], df[year == test_year]
    return train[C.FEATURES], train[C.TARGET].astype(int), test[C.FEATURES], test[C.TARGET].astype(int)
