"""Чтение сырого CSV чанками, конвертация в Parquet, оптимизация типов и замеры памяти."""
import gzip
import json
import time
import zlib
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config as C
from src.features import is_matured


def mb(df: pd.DataFrame) -> float:
    """Объём DataFrame в памяти, МБ."""
    return round(df.memory_usage(deep=True).sum() / 2**20, 1)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет служебные строки-итоги, приводит id, term и даты к нормальному виду."""
    df = df[df["loan_amnt"].notna()].copy()  # строки "Total amount funded..." без данных
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int64")
    df["term"] = df["term"].str.strip()
    for col in C.DATE_COLS:
        df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce").astype("datetime64[ns]")
    for col in C.FLOAT32:
        df[col] = df[col].astype("float32")
    for col in C.CATEGORICAL:
        df[col] = df[col].astype("object")
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """category для строк с малым числом значений, float32 для чисел."""
    for col in df.columns:
        if col in C.CATEGORICAL:
            df[col] = df[col].astype("category")
        elif col in C.FLOAT32:
            df[col] = df[col].astype("float32")
    return df


def csv_to_parquet(src: Path | None = None, dst: Path | None = None, chunksize: int | None = None) -> dict:
    """Читает CSV.gz чанками только с нужными колонками и пишет в один Parquet-файл.

    Пишет во временный файл и заменяет им результат только после успеха: прерванная
    конвертация не оставляет усечённый, но читаемый Parquet вместо прежнего.
    """
    src, dst = src or C.RAW_CSV, dst or C.PARQUET
    if not src.exists():
        raise FileNotFoundError(f"Нет файла {src}. Скачайте его с Kaggle (см. README).")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    start, rows_raw, rows, schema = time.time(), 0, 0, None
    writer = None
    try:
        try:
            reader = pd.read_csv(src, usecols=C.USECOLS, dtype={"id": str}, chunksize=chunksize or C.CHUNKSIZE,
                                 low_memory=False)
            for i, chunk in enumerate(reader, 1):
                rows_raw += len(chunk)
                chunk = clean_chunk(chunk)[C.USECOLS]
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp, schema, compression="snappy")
                writer.write_table(table.cast(schema))
                rows += len(chunk)
                print(f"  чанк {i}: всего {rows:,} строк")
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            raise ValueError(f"Файл {src} повреждён или скачан не полностью: {e}") from e
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"В файле {src} нет строк с данными.")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    stats = {"rows_raw": rows_raw, "rows": rows, "seconds": round(time.time() - start, 1),
             "csv_gz_mb": round(src.stat().st_size / 2**20, 1),
             "parquet_mb": round(dst.stat().st_size / 2**20, 1)}
    save_json(stats, C.REPORTS / "ingest.json")
    print(f"Готово: {stats}")
    return stats


def load_dataset(columns: list | None = None, path: Path | None = None) -> pd.DataFrame:
    """Загружает Parquet и сразу оптимизирует типы."""
    path = path or C.PARQUET
    if not path.exists():
        raise FileNotFoundError("Нет Parquet-файла. Сначала выполните пункт меню 1.")
    return optimize_dtypes(pd.read_parquet(path, columns=columns))


def make_sample(src: Path | None = None, dst: Path | None = None) -> dict:
    """Демо-выборка для запуска без Kaggle. Созревшие займы взяты с запасом, чтобы хватило на обучение,
    поэтому доли и дефолтность по выборке НЕ отражают полный набор."""
    df = load_dataset(path=src or C.PARQUET)
    resolved = df["loan_status"].astype("object").isin(C.STATUS_TO_TARGET)
    in_window = df["issue_d"].dt.year.between(*C.MODEL_YEARS)
    term = df["term"].astype("object").str.extract(r"(\d+)")[0].astype("float32")
    mask = resolved & in_window & is_matured(df["issue_d"], term)
    n_matured = min(C.SAMPLE_MATURED_ROWS, int(mask.sum()))
    n_other = min(C.SAMPLE_OTHER_ROWS, int((~mask).sum()))
    sample = pd.concat([df[mask].sample(n_matured, random_state=C.RANDOM_STATE),
                        df[~mask].sample(n_other, random_state=C.RANDOM_STATE)])
    sample = sample.sort_values("issue_d", kind="stable")
    dst = dst or C.SAMPLE_PARQUET
    dst.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(dst, compression="snappy", index=False)
    return {"rows": len(sample), "matured": n_matured, "mb": round(dst.stat().st_size / 2**20, 2)}


def memory_report(n_rows: int | None = None) -> dict:
    """Сравнивает память: все колонки vs выбранные колонки vs выбранные + оптимизация типов."""
    n_rows = n_rows or C.MEMORY_SAMPLE_ROWS
    if not C.RAW_CSV.exists():
        raise FileNotFoundError(f"Нет файла {C.RAW_CSV}: для замера памяти нужен исходный CSV.")
    full = pd.read_csv(C.RAW_CSV, nrows=n_rows, low_memory=False)
    selected = full[C.USECOLS].copy()
    optimized = optimize_dtypes(clean_chunk(selected))
    n_loaded = len(optimized)
    report = {
        "sample_rows": n_loaded,
        "n_columns_raw": full.shape[1],
        "all_cols_mb": mb(full),
        "selected_cols_mb": mb(selected),
        "selected_optimized_mb": mb(optimized),
    }
    if C.PARQUET.exists():
        loaded = load_dataset()
        scale = len(loaded) / n_loaded
        report["full_parquet_loaded_optimized_mb"] = mb(loaded)
        report["full_dataset_estimate_mb"] = {k.removesuffix("_mb"): round(report[k] * scale)
                                              for k in ("all_cols_mb", "selected_cols_mb", "selected_optimized_mb")}
    save_json(report, C.REPORTS / "memory_report.json")
    return report
