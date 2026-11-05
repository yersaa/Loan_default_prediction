"""Схема SQLite, загрузка и аналитические запросы: результаты сверяются с Pandas."""
import sqlite3

import pandas as pd
import pytest

from src import config as C
from src.database import build_database, connect, load_queries, run_all_queries, run_query, split_tables
from src.features import is_matured, term_to_months

pytestmark = pytest.mark.integration


@pytest.fixture
def db(tmp_project, loans):
    build_database(loans.copy())
    return C.DB_PATH


def _matured_resolved(loans: pd.DataFrame) -> pd.DataFrame:
    df = loans.copy()
    df["y"] = df["loan_status"].astype("object").map(C.STATUS_TO_TARGET)
    return df[df["y"].notna() & is_matured(df["issue_d"], term_to_months(df["term"]))]


def test_all_tables_loaded_and_row_counts_match(db, loans):
    with connect() as conn:
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608 — имена из кода
                  for t in ("loans", "borrowers", "outcomes", "dim_grade", "dim_purpose", "dim_state", "dim_home")}
    assert counts["loans"] == counts["borrowers"] == counts["outcomes"] == len(loans)
    assert counts["dim_grade"] == loans["sub_grade"].nunique()


def test_foreign_keys_hold(db):
    with connect() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_foreign_key_violation_is_rejected(db):
    with connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO outcomes (loan_id, loan_status, is_matured) VALUES (-1, 'x', 0)")


def test_connection_closes_after_block(db):
    with connect() as conn:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_outcomes_table_keeps_post_issue_fields_out_of_loans(loans):
    tables = split_tables(loans.copy())
    assert not {"total_pymnt", "recoveries", "is_default", "loan_status"} & set(tables["loans"].columns)
    assert not {"total_pymnt", "recoveries", "is_default"} & set(tables["borrowers"].columns)


def test_requirement_at_least_eight_queries_of_each_kind():
    queries = load_queries()
    text = " ".join(queries.values()).upper()
    assert len(queries) >= 8
    for clause in ("JOIN", "GROUP BY", "HAVING", "WITH ", "OVER (", "LAG(", "RANK(", "NTILE("):
        assert clause in text


def test_every_query_runs_and_returns_rows(db):
    results = run_all_queries()
    assert all(not df.empty or name == "purpose_having" for name, df in results.items())
    for name in results:
        assert (C.REPORTS / "sql" / f"{name}.csv").exists()


def test_default_by_grade_matches_pandas(db, loans):
    sql = run_query("default_by_grade").set_index("grade")
    expected = _matured_resolved(loans).groupby("grade", observed=True)["y"].agg(["size", "mean"])
    assert sql["loans"].to_dict() == expected["size"].to_dict()
    for grade in expected.index:
        assert sql.loc[grade, "default_rate_pct"] == pytest.approx(100 * expected.loc[grade, "mean"], abs=0.01)


def test_vintage_query_reports_censoring(db, loans):
    vintage = run_query("vintage_by_year_term")
    row = vintage[(vintage.issue_year == 2017) & (vintage.term_months == 60)].iloc[0]
    assert row["resolved_pct"] < 100
    old = vintage[(vintage.issue_year == 2010) & (vintage.term_months == 36)].iloc[0]
    assert old["resolved_pct"] == 100


def test_rolling_window_uses_three_months(db):
    rolling = run_query("monthly_rolling")
    expected = (rolling["default_rate_pct"].iloc[0:3].mean())
    assert rolling["rolling_3m_pct"].iloc[2] == pytest.approx(expected, abs=0.02)


def test_run_query_without_database(tmp_project):
    with pytest.raises(FileNotFoundError, match="пункт меню 3"):
        run_query("default_by_grade")
