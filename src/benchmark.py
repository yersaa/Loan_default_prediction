"""Сравнение способов обработки данных: CSV vs Parquet, Pandas vs SQL, а также память обучения моделей.

Каждый замер идёт в отдельном новом процессе и повторяется несколько раз; в отчёт попадает медиана.
Так задачи не влияют друг на друга через кэши и уже занятую процессом память.
"""
import multiprocessing as mp
import statistics
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data_io import load_dataset, load_json, mb, memory_report, save_json
from src.database import connect, require_database
from src.profiling import measure

CSV_COLS = ["issue_d", "grade", "loan_amnt", "int_rate", "annual_inc", "loan_status"]
REPEATS = 3
TRAIN_SNAPSHOT = "train_snapshot.parquet"  # временный файл для замеров памяти моделей

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
    """Задача для замера: прочитать 6 колонок из исходного CSV.gz."""
    return pd.read_csv(C.RAW_CSV, usecols=CSV_COLS)


def read_parquet() -> pd.DataFrame:
    """Задача для замера: прочитать те же 6 колонок из Parquet."""
    return load_dataset(CSV_COLS)


def pandas_aggregate() -> pd.DataFrame:
    """Задача для замера: дефолтность по году и грейду средствами Pandas (данные из Parquet)."""
    df = load_dataset(["issue_d", "grade", "loan_status"])
    df["is_default"] = df["loan_status"].astype("object").map(C.STATUS_TO_TARGET)
    df = df[df["is_default"].notna()]
    grouped = df.groupby([df["issue_d"].dt.year.rename("issue_year"), "grade"], observed=True)["is_default"]
    return grouped.agg(loans="size", default_rate="mean").reset_index().astype({"grade": str})


def sql_aggregate() -> pd.DataFrame:
    """Задача для замера: та же агрегация одним SQL-запросом к SQLite."""
    require_database()
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


def run_isolated(func: Callable, arg: str, repeats: int = REPEATS) -> list:
    """Запускает func(arg, sample_mode) в новом процессе repeats раз и возвращает результаты всех запусков."""
    runs = []
    for _ in range(repeats):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            runs.append(pool.submit(func, arg, C.SAMPLE_MODE).result())
    return runs


def run_task(name: str, repeats: int = REPEATS) -> dict:
    """Медиана времени и памяти одной задачи из TASKS."""
    seconds, peaks, sizes = zip(*run_isolated(_isolated_task, name, repeats), strict=True)
    return {"seconds": round(statistics.median(seconds), 2), "peak_mb": round(statistics.median(peaks), 1),
            "df_mb": sizes[0], "runs_seconds": [round(s, 2) for s in seconds]}


def save_train_snapshot() -> Path:
    """Сохраняет обучающее окно (признаки и таргет) в отдельный Parquet для замеров памяти.

    Если готовить данные в том же процессе, где идёт замер, память, освобождённая после подготовки,
    переиспользуется при обучении, и прирост занижается (у LightGBM получался 0 МБ).
    Поэтому процесс замера читает только готовый обучающий набор.
    """
    from src.features import modeling_frame, temporal_split  # тяжёлые импорты нужны только этой задаче

    train = temporal_split(modeling_frame(load_dataset()))["train"]
    path = C.REPORTS / TRAIN_SNAPSHOT
    train[C.FEATURES + [C.TARGET]].to_parquet(path, index=False)
    return path


def fit_memory(slug: str) -> float:
    """Прирост пиковой памяти процесса (МБ) за время обучения одной модели на train.

    Обучающий набор загружается до замера: измеряется только model.fit.
    """
    from src.models import build_model, model_specs

    spec = next(s for s in model_specs() if s.slug == slug)
    best = load_json(C.REPORTS / "tuning.json", default={})
    train = pd.read_parquet(C.REPORTS / TRAIN_SNAPSHOT)
    model = build_model(spec, best)
    with measure() as m:
        model.fit(train[C.FEATURES], train[C.TARGET].astype(int))
    return m.peak_mb


def _isolated_fit(slug: str, sample_mode: bool) -> float:
    """Выполняется в новом процессе, чтобы модели не влияли на замеры друг друга."""
    if sample_mode:
        C.activate_sample()
    return fit_memory(slug)


def model_fit_memory(repeats: int = REPEATS) -> dict:
    """Память обучения каждой модели: {название модели: медиана и все запуски, МБ}."""
    from src.models import model_specs

    snapshot = save_train_snapshot()
    report = {}
    try:
        for spec in model_specs():
            peaks = run_isolated(_isolated_fit, spec.slug, repeats)
            report[spec.name] = {"fit_peak_mb": round(statistics.median(peaks), 1),
                                 "runs_mb": [round(p, 1) for p in peaks]}
            print(f"  {spec.name:<36} память обучения {report[spec.name]['fit_peak_mb']:>7.1f} МБ")
    finally:
        snapshot.unlink(missing_ok=True)  # временный файл, в отчёты не входит
    return report


def csv_vs_parquet() -> dict:
    """Время, память и размер файла при чтении одних и тех же колонок из CSV.gz и из Parquet."""
    if not C.RAW_CSV.exists():
        raise FileNotFoundError(f"Нет файла {C.RAW_CSV}: сравнение CSV и Parquet требует исходный CSV.")
    csv_gz = {**run_task("read_csv_gz"), "file_mb": round(C.RAW_CSV.stat().st_size / 2**20, 1)}
    parquet = {**run_task("read_parquet"), "file_mb": round(C.PARQUET.stat().st_size / 2**20, 1)}
    return {"columns": CSV_COLS, "csv_gz": csv_gz, "parquet": parquet}


def results_match(pandas_result: pd.DataFrame, sql_result: pd.DataFrame) -> bool:
    """True, если Pandas и SQL дали одинаковые группы, число займов и дефолтность."""
    keys = ["issue_year", "grade"]
    merged = pandas_result.merge(sql_result, on=keys, how="outer", suffixes=("_pd", "_sql"), indicator=True)
    return bool((merged["_merge"] == "both").all()
                and (merged["loans_pd"] == merged["loans_sql"]).all()
                and np.allclose(merged["default_rate_pd"], merged["default_rate_sql"]))


def pandas_vs_sql() -> dict:
    """Время и память одной агрегации в Pandas и в SQLite, плюс проверка, что результаты совпали."""
    return {"task": "дефолтность по году и грейду",
            "pandas_parquet": run_task("pandas_aggregate"), "sqlite": run_task("sql_aggregate"),
            "results_match": results_match(pandas_aggregate(), sql_aggregate())}


def run_benchmarks() -> dict:
    """Все сравнения пункта меню 8. Результат сохраняется в reports/benchmarks.json."""
    require_database()  # обе проверки — до долгих замеров CSV и памяти, а не после них
    if not (C.REPORTS / "tuning.json").exists():
        raise FileNotFoundError("Нет reports/tuning.json: память моделей замеряется после пункта меню 6.")
    report = {"method": f"отдельный процесс на замер, медиана {REPEATS} запусков, прирост RSS",
              "memory": memory_report(), "csv_vs_parquet": csv_vs_parquet(), "pandas_vs_sql": pandas_vs_sql(),
              "model_fit_memory": model_fit_memory()}
    save_json(report, C.REPORTS / "benchmarks.json")
    return report
