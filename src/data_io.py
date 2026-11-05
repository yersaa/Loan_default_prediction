"""Чтение сырого CSV чанками, конвертация в Parquet, оптимизация типов и замеры памяти."""
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config as C


def mb(df: pd.DataFrame) -> float:
    """Объём DataFrame в памяти, МБ."""
    return round(df.memory_usage(deep=True).sum() / 2**20, 1)


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


def csv_to_parquet(src=C.RAW_CSV, dst=C.PARQUET, chunksize=C.CHUNKSIZE) -> dict:
    """Читает CSV.gz чанками только с нужными колонками и пишет в один Parquet-файл."""
    if not src.exists():
        raise FileNotFoundError(f"Нет файла {src}. Скачайте его с Kaggle (см. README).")
    dst.parent.mkdir(parents=True, exist_ok=True)
    start, rows, writer, schema = time.time(), 0, None, None
    reader = pd.read_csv(src, usecols=C.USECOLS, dtype={"id": str}, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        chunk = clean_chunk(chunk)[C.USECOLS]
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(dst, schema, compression="snappy")
        writer.write_table(table.cast(schema))
        rows += len(chunk)
        print(f"  чанк {i}: всего {rows:,} строк")
    writer.close()
    stats = {"rows": rows, "seconds": round(time.time() - start, 1),
             "csv_gz_mb": round(src.stat().st_size / 2**20, 1),
             "parquet_mb": round(dst.stat().st_size / 2**20, 1)}
    print(f"Готово: {stats}")
    return stats


def load_dataset(columns=None, path=C.PARQUET) -> pd.DataFrame:
    """Загружает Parquet и сразу оптимизирует типы."""
    if not path.exists():
        raise FileNotFoundError("Нет Parquet-файла. Сначала выполните пункт меню 1.")
    return optimize_dtypes(pd.read_parquet(path, columns=columns))


def memory_report(n_rows=200_000) -> dict:
    """Сравнивает память: все 151 колонка vs выбранные колонки vs выбранные + оптимизация типов."""
    full = pd.read_csv(C.RAW_CSV, nrows=n_rows, low_memory=False)
    selected = full[C.USECOLS].copy()
    optimized = optimize_dtypes(clean_chunk(selected))
    scale = 2_260_668 / n_rows
    report = {
        "sample_rows": n_rows,
        "all_151_cols_mb": mb(full),
        "selected_cols_mb": mb(selected),
        "selected_optimized_mb": mb(optimized),
    }
    report["full_dataset_estimate_mb"] = {k.replace("_mb", ""): round(v * scale)
                                          for k, v in report.items() if k.endswith("_mb")}
    if C.PARQUET.exists():
        report["full_parquet_loaded_optimized_mb"] = mb(load_dataset())
    save_json(report, C.REPORTS / "memory_report.json")
    return report


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default
