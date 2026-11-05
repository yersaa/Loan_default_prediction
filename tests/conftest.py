"""Общие фикстуры: синтетические займы в формате Parquet и изолированные пути."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_io import optimize_dtypes

SUB_GRADES = [f"{g}{i}" for g in "ABCDE" for i in range(1, 6)]


def synthetic_loans(n: int = 12_000, seed: int = 7) -> pd.DataFrame:
    """Займы 2009–2017, где дефолт зависит от грейда, FICO и DTI. Формат совпадает с load_dataset."""
    rng = np.random.default_rng(seed)
    grade_idx = rng.integers(0, len(SUB_GRADES), n)
    sub_grade = np.array(SUB_GRADES)[grade_idx]
    issue = pd.to_datetime(rng.choice(pd.date_range("2009-01-01", "2017-12-01", freq="MS").to_numpy(), n))
    term = rng.choice(["36 months", "60 months"], n, p=[0.75, 0.25])
    fico_low = rng.integers(660, 800, n).astype(float)
    dti = rng.uniform(0, 40, n)
    risk = -2.6 + 0.12 * grade_idx - 0.01 * (fico_low - 700) + 0.02 * dti
    defaulted = rng.random(n) < 1 / (1 + np.exp(-risk))
    term_months = np.where(term == "36 months", 36, 60)
    elapsed = (2018 - issue.year) * 12 + (12 - issue.month)
    matured = elapsed >= term_months
    status = np.where(defaulted, "Charged Off", "Fully Paid")
    status = np.where(~matured & (rng.random(n) < 0.5), "Current", status)
    loan_amnt = rng.integers(1, 35, n) * 1000.0
    rate = 6 + 1.0 * grade_idx / 2 + rng.random(n)
    frame = pd.DataFrame({
        "id": np.arange(1, n + 1), "issue_d": issue.astype("datetime64[ns]"), "loan_amnt": loan_amnt, "term": term,
        "int_rate": rate, "installment": loan_amnt * (rate / 1200) / (1 - (1 + rate / 1200) ** -term_months),
        "grade": [g[0] for g in sub_grade], "sub_grade": sub_grade,
        "emp_length": rng.choice(["< 1 year", "3 years", "10+ years"], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "annual_inc": rng.uniform(20_000, 150_000, n),
        "verification_status": rng.choice(["Not Verified", "Source Verified", "Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], n),
        "addr_state": rng.choice(["CA", "NY", "TX", "IL"], n), "dti": dti,
        "delinq_2yrs": rng.integers(0, 3, n).astype(float),
        "earliest_cr_line": pd.Timestamp("2000-01-01"),
        "fico_range_low": fico_low, "fico_range_high": fico_low + 4,
        "inq_last_6mths": rng.integers(0, 4, n).astype(float), "open_acc": rng.integers(2, 20, n).astype(float),
        "pub_rec": 0.0, "revol_bal": rng.uniform(0, 30_000, n), "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.integers(5, 40, n).astype(float), "initial_list_status": rng.choice(["w", "f"], n),
        "application_type": "Individual", "mort_acc": rng.integers(0, 3, n).astype(float),
        "pub_rec_bankruptcies": 0.0, "loan_status": status,
        "total_pymnt": loan_amnt * np.where(defaulted, 0.6, 1.2), "recoveries": np.where(defaulted, 100.0, 0.0),
        "last_pymnt_d": pd.Timestamp("2018-01-01"),
    })
    frame["earliest_cr_line"] = frame["earliest_cr_line"].astype("datetime64[ns]")
    frame["last_pymnt_d"] = frame["last_pymnt_d"].astype("datetime64[ns]")
    return optimize_dtypes(frame)


@pytest.fixture(scope="session")
def loans() -> pd.DataFrame:
    return synthetic_loans()


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Все пути проекта указывают во временную папку: тесты не трогают настоящие отчёты и модели."""
    monkeypatch.setattr(C, "PARQUET", tmp_path / "data" / "loans.parquet")
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "db" / "lending.db")
    monkeypatch.setattr(C, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(C, "FIGURES", tmp_path / "reports" / "figures")
    monkeypatch.setattr(C, "MODELS", tmp_path / "models")
    monkeypatch.setattr(C, "SHIPPED_MODELS", tmp_path / "models")
    monkeypatch.setattr(C, "STATE_FILE", tmp_path / "reports" / "state.json")
    monkeypatch.setattr(C, "RAW_CSV", tmp_path / "datasets" / "raw.csv.gz")
    monkeypatch.setattr(C, "SAMPLE_PARQUET", tmp_path / "data" / "sample" / "loans_sample.parquet")
    for folder in (C.REPORTS, C.FIGURES, C.MODELS):
        folder.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture(scope="session")
def fast_best() -> dict:
    """Лёгкие гиперпараметры: тесты проверяют логику, а не качество."""
    lgbm = {"params": {"model__num_leaves": 7, "model__n_estimators": 30}}
    logreg = {"params": {"model__C": 1.0}}
    return {"Logistic Regression": logreg, "Logistic Regression (borrower only)": logreg,
            "LightGBM": lgbm, "LightGBM (borrower only)": lgbm}
