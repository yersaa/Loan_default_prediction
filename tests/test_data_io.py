"""Чтение чанками, Parquet, выборка и служебные функции."""
import gzip

import pandas as pd
import pytest

from src import config as C
from src.data_io import (
    clean_chunk,
    csv_to_parquet,
    load_dataset,
    load_json,
    make_sample,
    mb,
    memory_report,
    optimize_dtypes,
    save_json,
)
from src.profiling import measure


def _write_raw_csv(path, loans, summary_rows=2):
    raw = loans.copy()
    for col in ("issue_d", "earliest_cr_line", "last_pymnt_d"):
        raw[col] = raw[col].dt.strftime("%b-%Y")
    raw = raw[C.USECOLS].astype(object)
    raw["unused_column"] = 1
    summary = pd.DataFrame([{"id": "Total amount funded in policy code 1: 123"} for _ in range(summary_rows)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        pd.concat([raw, summary]).to_csv(handle, index=False)


@pytest.mark.unit
def test_clean_chunk_drops_summary_lines_and_parses():
    chunk = pd.DataFrame({c: [None, None] for c in C.USECOLS})
    chunk["id"] = ["10", "Total amount funded"]
    chunk["loan_amnt"] = [1000.0, None]
    chunk["term"] = [" 36 months", None]
    chunk["issue_d"] = ["Dec-2015", None]
    cleaned = clean_chunk(chunk)
    assert len(cleaned) == 1
    assert cleaned["id"].iloc[0] == 10
    assert cleaned["term"].iloc[0] == "36 months"
    assert cleaned["issue_d"].iloc[0] == pd.Timestamp("2015-12-01")


@pytest.mark.unit
def test_optimize_dtypes_reduces_memory():
    df = pd.DataFrame({"grade": ["A", "B"] * 1000, "loan_amnt": [1.0, 2.0] * 1000})
    before = df.memory_usage(deep=True).sum()
    optimized = optimize_dtypes(df.copy())
    assert optimized["grade"].dtype == "category"
    assert optimized["loan_amnt"].dtype == "float32"
    assert optimized.memory_usage(deep=True).sum() < before


@pytest.mark.unit
def test_mb_positive():
    assert mb(pd.DataFrame({"a": range(100_000)})) > 0


@pytest.mark.unit
def test_json_roundtrip_and_defaults(tmp_path):
    path = tmp_path / "nested" / "x.json"
    save_json({"а": 1}, path)
    assert load_json(path) == {"а": 1}
    assert load_json(tmp_path / "missing.json", default={"d": 1}) == {"d": 1}
    path.write_text("{broken", encoding="utf-8")
    assert load_json(path, default=[]) == []


@pytest.mark.unit
def test_load_dataset_missing_file(tmp_project):
    with pytest.raises(FileNotFoundError, match="пункт меню 1"):
        load_dataset()


@pytest.mark.unit
def test_csv_to_parquet_missing_source(tmp_project):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        csv_to_parquet()


@pytest.mark.integration
def test_csv_to_parquet_roundtrip(tmp_project, loans):
    subset = loans.head(1500)
    _write_raw_csv(C.RAW_CSV, subset)
    stats = csv_to_parquet(chunksize=400)
    assert stats["rows"] == 1500
    assert stats["rows_raw"] == 1502
    assert load_json(C.REPORTS / "ingest.json")["rows_raw"] == 1502
    loaded = load_dataset()
    assert len(loaded) == 1500
    assert loaded["id"].is_unique
    assert set(C.USECOLS) == set(loaded.columns)
    assert loaded["issue_d"].dtype.kind == "M"
    assert load_dataset(["id", "grade"]).shape[1] == 2


def _write_valid_parquet(loans):
    C.PARQUET.parent.mkdir(parents=True, exist_ok=True)
    loans.head(100).to_parquet(C.PARQUET, index=False)
    return C.PARQUET.read_bytes()


@pytest.mark.integration
def test_truncated_download_keeps_previous_parquet(tmp_project, loans):
    """Обрезанный .gz: раньше на месте хорошего файла оставался усечённый, но читаемый Parquet."""
    before = _write_valid_parquet(loans)
    _write_raw_csv(C.RAW_CSV, loans.head(1500))
    raw = C.RAW_CSV.read_bytes()
    C.RAW_CSV.write_bytes(raw[:len(raw) // 2])
    with pytest.raises(ValueError, match="скачан не полностью"):
        csv_to_parquet(chunksize=200)
    assert C.PARQUET.read_bytes() == before
    assert not list(C.PARQUET.parent.glob("*.tmp"))


@pytest.mark.integration
def test_interrupted_conversion_keeps_previous_parquet(tmp_project, loans, monkeypatch):
    before = _write_valid_parquet(loans)
    _write_raw_csv(C.RAW_CSV, loans.head(1500))
    calls = {"n": 0}

    def interrupt_on_second_chunk(chunk):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return clean_chunk(chunk)
    monkeypatch.setattr("src.data_io.clean_chunk", interrupt_on_second_chunk)
    with pytest.raises(KeyboardInterrupt):
        csv_to_parquet(chunksize=400)
    assert C.PARQUET.read_bytes() == before
    assert not list(C.PARQUET.parent.glob("*.tmp"))


@pytest.mark.integration
def test_make_sample_prefers_matured(tmp_project, loans, monkeypatch):
    monkeypatch.setattr(C, "SAMPLE_MATURED_ROWS", 300)
    monkeypatch.setattr(C, "SAMPLE_OTHER_ROWS", 100)
    C.PARQUET.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(C.PARQUET, index=False)
    info = make_sample()
    assert info == {"rows": 400, "matured": 300, "mb": info["mb"]}
    assert len(pd.read_parquet(C.SAMPLE_PARQUET)) == 400


@pytest.mark.integration
def test_memory_report(tmp_project, loans):
    _write_raw_csv(C.RAW_CSV, loans.head(500), summary_rows=0)
    with pytest.raises(FileNotFoundError):
        csv_to_parquet(src=C.RAW_CSV.with_name("nope.csv.gz"))
    csv_to_parquet()
    report = memory_report(n_rows=300)
    assert report["selected_optimized_mb"] < report["selected_cols_mb"]
    assert report["n_columns_raw"] == len(C.USECOLS) + 1
    assert "full_dataset_estimate_mb" in report
    assert (C.REPORTS / "memory_report.json").exists()


@pytest.mark.unit
def test_memory_report_missing_raw(tmp_project):
    with pytest.raises(FileNotFoundError):
        memory_report()


@pytest.mark.unit
def test_measure_records_native_memory_growth():
    with measure() as m:
        block = bytearray(50 * 2**20)
        block[0] = 1
    assert m.seconds >= 0
    assert m.peak_mb >= 40
