"""Сравнение способов обработки данных: CSV vs Parquet, Pandas vs SQL (время и память)."""
import time
import tracemalloc

import pandas as pd

from src import config as C
from src.data_io import load_dataset, mb, memory_report, save_json
from src.database import connect

AGG_SQL = """
SELECT substr(l.issue_d, 1, 4) AS issue_year, g.grade,
       COUNT(*) AS loans, AVG(o.is_default) AS default_rate
FROM loans l
JOIN outcomes o  ON o.loan_id = l.loan_id
JOIN dim_grade g ON g.sub_grade = l.sub_grade
WHERE o.is_default IS NOT NULL
GROUP BY issue_year, g.grade
"""


def timed(func, *args, **kwargs):
    """Возвращает (результат, секунды, пиковая память Python в МБ)."""
    tracemalloc.start()
    start = time.time()
    result = func(*args, **kwargs)
    seconds = time.time() - start
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return result, round(seconds, 2), round(peak, 1)


def csv_vs_parquet() -> dict:
    cols = ["issue_d", "grade", "loan_amnt", "int_rate", "annual_inc", "loan_status"]
    csv_df, csv_s, csv_peak = timed(pd.read_csv, C.RAW_CSV, usecols=cols)
    pq_df, pq_s, pq_peak = timed(load_dataset, cols)
    return {"columns": cols,
            "csv_gz": {"seconds": csv_s, "peak_mb": csv_peak, "df_mb": mb(csv_df), "file_mb": round(C.RAW_CSV.stat().st_size / 2**20, 1)},
            "parquet": {"seconds": pq_s, "peak_mb": pq_peak, "df_mb": mb(pq_df), "file_mb": round(C.PARQUET.stat().st_size / 2**20, 1)}}


def pandas_aggregate() -> pd.DataFrame:
    df = load_dataset(["issue_d", "grade", "loan_status"])
    df["is_default"] = df["loan_status"].astype("object").map(C.STATUS_TO_TARGET)
    df = df[df["is_default"].notna()]
    return (df.groupby([df["issue_d"].dt.year.rename("issue_year"), "grade"], observed=True)["is_default"]
            .agg(loans="size", default_rate="mean").reset_index())


def sql_aggregate() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(AGG_SQL, conn)


def pandas_vs_sql() -> dict:
    pdf, pd_s, pd_peak = timed(pandas_aggregate)
    sdf, sql_s, sql_peak = timed(sql_aggregate)
    same = abs(pdf["default_rate"].sum() - sdf["default_rate"].sum()) < 1e-6 and len(pdf) == len(sdf)
    return {"task": "дефолтность по году и грейду",
            "pandas_parquet": {"seconds": pd_s, "peak_mb": pd_peak},
            "sqlite": {"seconds": sql_s, "peak_mb": sql_peak},
            "results_match": bool(same)}


def run_benchmarks() -> dict:
    report = {"memory": memory_report(), "csv_vs_parquet": csv_vs_parquet(), "pandas_vs_sql": pandas_vs_sql()}
    save_json(report, C.REPORTS / "benchmarks.json")
    return report
