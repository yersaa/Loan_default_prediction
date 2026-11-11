"""SQLite: создание нормализованной схемы, загрузка данных и выполнение аналитических запросов."""
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src import config as C
from src.features import add_target, build_features, is_matured

REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, states in REGIONS.items() for s in states.split()}


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Соединение с включёнными внешними ключами: коммит при успехе, всегда закрывается."""
    path = path or C.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def dimension(values: pd.Series, id_col: str, name_col: str) -> pd.DataFrame:
    """Справочник: уникальные значения + суррогатный ключ."""
    uniq = sorted(values.dropna().astype(str).unique())
    return pd.DataFrame({id_col: range(1, len(uniq) + 1), name_col: uniq})


def split_tables(df: pd.DataFrame) -> dict:
    """Раскладывает плоскую таблицу на справочники и три таблицы фактов."""
    df = build_features(add_target(df))
    purpose = dimension(df["purpose"], "purpose_id", "purpose")
    home = dimension(df["home_ownership"], "home_id", "home_ownership")
    grade = df[["sub_grade", "grade"]].astype(str).drop_duplicates().sort_values("sub_grade")
    states = pd.DataFrame({"state_code": sorted(df["addr_state"].astype(str).unique())})
    states["region"] = states["state_code"].map(STATE_REGION).fillna("Other")

    loans = pd.DataFrame({
        "loan_id": df["id"],
        "issue_d": df["issue_d"].dt.strftime("%Y-%m-%d"),
        "loan_amnt": df["loan_amnt"], "term_months": df["term_months"].astype(int),
        "int_rate": df["int_rate"], "installment": df["installment"],
        "sub_grade": df["sub_grade"].astype(str),
        "purpose_id": df["purpose"].astype(str).map(dict(zip(purpose.purpose, purpose.purpose_id, strict=True))),
        "state_code": df["addr_state"].astype(str),
        "verification_status": df["verification_status"].astype(str),
        "application_type": df["application_type"].astype(str),
        "initial_list_status": df["initial_list_status"].astype(str),
    })
    borrower_cols = ["emp_years", "annual_inc", "dti", "fico", "credit_hist_months", "delinq_2yrs",
                     "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
                     "mort_acc", "pub_rec_bankruptcies"]
    borrowers = df[["id"] + borrower_cols].rename(columns={"id": "loan_id"})
    home_ids = dict(zip(home.home_ownership, home.home_id, strict=True))
    borrowers.insert(1, "home_id", df["home_ownership"].astype(str).map(home_ids))
    outcomes = pd.DataFrame({
        "loan_id": df["id"], "loan_status": df["loan_status"].astype(str),
        "is_default": df[C.TARGET].astype("Int8"),
        "is_matured": is_matured(df["issue_d"], df["term_months"]).astype(int),
        "total_pymnt": df["total_pymnt"], "recoveries": df["recoveries"],
        "last_pymnt_d": df["last_pymnt_d"].dt.strftime("%Y-%m-%d"),
    })
    return {"dim_grade": grade, "dim_purpose": purpose, "dim_state": states, "dim_home": home,
            "loans": loans, "borrowers": borrowers, "outcomes": outcomes}


def require_database() -> None:
    """connect() создаёт файл при первом обращении, поэтому наличие базы проверяется до соединения."""
    if not C.DB_PATH.exists() or C.DB_PATH.stat().st_size == 0:
        raise FileNotFoundError("База не создана. Сначала выполните пункт меню 3.")


def build_database(df: pd.DataFrame) -> dict:
    """Создаёт схему из sql/schema.sql и загружает таблицы чанками.

    База собирается во временном файле: прерванная загрузка не оставляет частичные таблицы.
    """
    start = time.time()
    tables = split_tables(df)
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.DB_PATH.with_name(C.DB_PATH.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with connect(tmp) as conn:
            conn.executescript((C.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
            conn.execute("PRAGMA synchronous = OFF")
            for name, table in tables.items():
                table.to_sql(name, conn, if_exists="append", index=False, chunksize=100_000)
                print(f"  {name:<12}{len(table):>12,} строк")
        tmp.replace(C.DB_PATH)
    finally:
        tmp.unlink(missing_ok=True)
    stats = {name: len(t) for name, t in tables.items()}
    stats["seconds"] = round(time.time() - start, 1)
    stats["db_mb"] = round(C.DB_PATH.stat().st_size / 2**20, 1)
    return stats


def load_queries(path: Path | None = None) -> dict:
    """Разбирает файл с запросами вида '-- name: <имя>' в словарь {имя: SQL}."""
    text = (path or C.SQL_DIR / "queries.sql").read_text(encoding="utf-8")
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)[1:]
    return {parts[i]: parts[i + 1].strip() for i in range(0, len(parts), 2)}


def run_query(name: str, queries: dict | None = None) -> pd.DataFrame:
    """Выполняет запрос по имени и возвращает результат в Pandas."""
    require_database()
    queries = queries or load_queries()
    with connect() as conn:
        return pd.read_sql_query(queries[name], conn)


def run_all_queries(save: bool = True) -> dict:
    """Выполняет все запросы, сохраняет результаты в reports/sql/*.csv."""
    queries, results = load_queries(), {}
    out = C.REPORTS / "sql"
    out.mkdir(parents=True, exist_ok=True)
    for name in queries:
        start = time.time()
        results[name] = run_query(name, queries)
        print(f"  {name:<24}{len(results[name]):>5} строк  {time.time() - start:5.2f} c")
        if save:
            results[name].to_csv(out / f"{name}.csv", index=False)
    return results
