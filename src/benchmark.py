"""Сравнение способов обработки данных: CSV vs Parquet, Pandas vs SQL (время и память).

Каждый замер идёт в отдельном новом процессе и повторяется несколько раз; в отчёт попадает медиана.
Так задачи не влияют друг на друга через кэши и уже занятую процессом память.
"""
import multiprocessing as mp
import statistics
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import config as C
from src.data_io import load_dataset, mb, memory_report, save_json
from src.database import connect
from src.profiling import measure

CSV_COLS = ["issue_d", "grade", "loan_amnt", "int_rate", "annual_inc", "loan_status"]
REPEATS = 3

AGG_SQL = """
SELECT CAST(substr(l.issue_d, 1, 4) AS INTEGER) AS issue_year, g.grade,
       COUNT(*) AS loans, AVG(o.is_default) AS default_rate
FROM loans l
JOIN outcomes o  ON o.loan_id = l.loan_id
JOIN dim_grade g ON g.sub_grade = l.sub_grade
WHERE o.is_default IS NOT NULL
GROUP BY issue_year, g.grade
ORDER BY issue_year, g.grade
"""


def read_csv_gz() -> pd.DataFrame:
    return pd.read_csv(C.RAW_CSV, usecols=CSV_COLS)


def read_parquet() -> pd.DataFrame:
    return load_dataset(CSV_COLS)


def pandas_aggregate() -> pd.DataFrame:
    df = load_dataset(["issue_d", "grade", "loan_status"])
    df["is_default"] = df["loan_status"].astype("object").map(C.STATUS_TO_TARGET)
    df = df[df["is_default"].notna()]
    grouped = df.groupby([df["issue_d"].dt.year.rename("issue_year"), "grade"], observed=True)["is_default"]
    return grouped.agg(loans="size", default_rate="mean").reset_index().astype({"grade": str})


def sql_aggregate() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(AGG_SQL, conn)


TASKS = {"read_csv_gz": read_csv_gz, "read_parquet": read_parquet,
         "pandas_aggregate": pandas_aggregate, "sql_aggregate": sql_aggregate}


def _isolated_task(name: str, sample_mode: bool) -> tuple:
    """Выполняется в новом процессе: импорты уже оплачены, в замер входит только сама задача."""
    if sample_mode:
        C.activate_sample()
    with measure() as m:
        result = TASKS[name]()
    return m.seconds, m.peak_mb, mb(result)


def run_task(name: str, repeats: int = REPEATS) -> dict:
    runs = []
    for _ in range(repeats):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            runs.append(pool.submit(_isolated_task, name, C.SAMPLE_MODE).result())
    seconds, peaks, sizes = zip(*runs, strict=True)
    return {"seconds": round(statistics.median(seconds), 2), "peak_mb": round(statistics.median(peaks), 1),
            "df_mb": sizes[0], "runs_seconds": [round(s, 2) for s in seconds]}


def csv_vs_parquet() -> dict:
    if not C.RAW_CSV.exists():
        raise FileNotFoundError(f"Нет файла {C.RAW_CSV}: сравнение CSV и Parquet требует исходный CSV.")
    csv_gz = {**run_task("read_csv_gz"), "file_mb": round(C.RAW_CSV.stat().st_size / 2**20, 1)}
    parquet = {**run_task("read_parquet"), "file_mb": round(C.PARQUET.stat().st_size / 2**20, 1)}
    return {"columns": CSV_COLS, "csv_gz": csv_gz, "parquet": parquet}


def results_match(pandas_result: pd.DataFrame, sql_result: pd.DataFrame) -> bool:
    keys = ["issue_year", "grade"]
    merged = pandas_result.merge(sql_result, on=keys, how="outer", suffixes=("_pd", "_sql"), indicator=True)
    return bool((merged["_merge"] == "both").all()
                and (merged["loans_pd"] == merged["loans_sql"]).all()
                and np.allclose(merged["default_rate_pd"], merged["default_rate_sql"]))


def pandas_vs_sql() -> dict:
    return {"task": "дефолтность по году и грейду",
            "pandas_parquet": run_task("pandas_aggregate"), "sqlite": run_task("sql_aggregate"),
            "results_match": results_match(pandas_aggregate(), sql_aggregate())}


def run_benchmarks() -> dict:
    report = {"method": f"отдельный процесс на замер, медиана {REPEATS} запусков, прирост RSS",
              "memory": memory_report(), "csv_vs_parquet": csv_vs_parquet(), "pandas_vs_sql": pandas_vs_sql()}
    save_json(report, C.REPORTS / "benchmarks.json")
    return report
