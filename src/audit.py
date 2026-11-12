"""Аудит качества данных: пропуски, дубликаты, некорректные значения, выбросы (IQR), покрытие колонок,
проверка, что поля кредитного отчёта относятся к моменту заявки."""
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.data_io import load_json, save_json
from src.style import BLUE, save

HIGH_MISSING_PCT = 50
OUTLIER_COLS = ["loan_amnt", "annual_inc", "dti", "revol_bal", "int_rate"]
BUREAU_MAX_EXCESS_MONTHS = 12  # допуск: кредитный отчёт запрашивают за несколько месяцев до выдачи


def iqr_outliers(s: pd.Series) -> dict:
    """Число выбросов по правилу 1.5 * IQR."""
    q1, q3 = s.quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n = int(((s < low) | (s > high)).sum())
    return {"low": round(float(low), 2), "high": round(float(high), 2), "count": n,
            "share_pct": round(100 * n / s.notna().sum(), 2)}


def invalid_values(df: pd.DataFrame) -> dict:
    """Проверки бизнес-логики."""
    checks = {
        "annual_inc <= 0": df["annual_inc"] <= 0,
        "dti < 0 или >= 999": (df["dti"] < 0) | (df["dti"] >= 999),
        "revol_util > 150%": df["revol_util"] > 150,
        "fico_range_low > fico_range_high": df["fico_range_low"] > df["fico_range_high"],
        "earliest_cr_line позже issue_d": df["earliest_cr_line"] > df["issue_d"],
        "int_rate вне 5–31%": ~df["int_rate"].between(5, 31),
    }
    return {name: int(mask.sum()) for name, mask in checks.items()}


def bureau_timing(df: pd.DataFrame) -> dict:
    """Проверяет, что поля кредитного отчёта не обновлялись после выдачи займа.

    mo_sin_old_rev_tl_op — возраст самого старого возобновляемого счёта в месяцах. На момент заявки он не может
    заметно превышать длину кредитной истории (issue_d - earliest_cr_line). Если бы LendingClub обновлял поле
    к дате выгрузки (2018 г.), у старых займов превышение составляло бы годы.
    """
    issued, first_credit = df["issue_d"].dt, df["earliest_cr_line"].dt
    history = (issued.year - first_credit.year) * 12 + (issued.month - first_credit.month)
    excess = (df["mo_sin_old_rev_tl_op"] - history).dropna()
    return {"rows_checked": len(excess),
            "max_excess_months": float(excess.max()) if len(excess) else None,
            "rows_over_12_months": int((excess > BUREAU_MAX_EXCESS_MONTHS).sum())}


def raw_column_audit(n_rows: int | None = None) -> dict | None:
    """Покрытие всех колонок исходного файла (на первых строках): обоснование отбора нужных колонок."""
    if not C.RAW_CSV.exists():
        return None
    raw = pd.read_csv(C.RAW_CSV, nrows=n_rows or C.MEMORY_SAMPLE_ROWS, low_memory=False)
    missing = raw.isna().mean() * 100
    return {"rows_checked": len(raw), "n_columns": raw.shape[1],
            "columns_over_50pct_missing": int((missing > HIGH_MISSING_PCT).sum()),
            "kept_columns": len(C.USECOLS),
            "known_leakage_columns_excluded": sorted(set(C.LEAKAGE_COLS) & set(raw.columns))}


def plot_missing(missing: pd.Series) -> None:
    """Горизонтальная диаграмма: доля пропусков по колонкам, где они есть."""
    top = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(top.index, top.values, color=BLUE, height=0.7)
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=2, fontsize=9)
    ax.set(title="Доля пропусков в выбранных колонках", xlabel="Пропуски, %")
    ax.grid(axis="y", visible=False)
    save(fig, "01_missing.png")


def run_audit(df: pd.DataFrame) -> dict:
    """Собирает все проверки качества в один словарь и сохраняет его в reports/audit.json."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    missing = (df.isna().mean() * 100).round(2)
    plot_missing(missing)
    ingest = load_json(C.REPORTS / "ingest.json", default={})
    report = {
        "rows_raw": ingest.get("rows_raw"),
        "rows_after_removing_summary_lines": len(df),
        "summary_lines_removed": ingest["rows_raw"] - len(df) if "rows_raw" in ingest else None,
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "missing_pct": {k: v for k, v in missing.sort_values(ascending=False).items() if v > 0},
        "invalid_values": invalid_values(df),
        "bureau_timing": bureau_timing(df),
        "iqr_outliers": {c: iqr_outliers(df[c]) for c in OUTLIER_COLS},
        "loan_status": df["loan_status"].value_counts().to_dict(),
        "issue_year_range": [int(df["issue_d"].dt.year.min()), int(df["issue_d"].dt.year.max())],
        "raw_columns": raw_column_audit(),
    }
    save_json(report, C.REPORTS / "audit.json")
    return report


def print_audit(report: dict) -> None:
    """Печатает краткую сводку аудита в консоль."""
    print(f"Строк: {report['rows_after_removing_summary_lines']:,} "
          f"(удалено служебных строк: {report['summary_lines_removed']})")
    print(f"Дубликатов id: {report['duplicate_ids']}")
    print("Пропуски, % (топ-8):")
    for col, pct in list(report["missing_pct"].items())[:8]:
        print(f"  {col:<22}{pct:>6}")
    print("Некорректные значения:")
    for name, n in report["invalid_values"].items():
        print(f"  {name:<34}{n:>8,}")
    timing = report["bureau_timing"]
    print(f"Поля кредитного отчёта: проверено {timing['rows_checked']:,} займов, обновлённых после выдачи: "
          f"{timing['rows_over_12_months']}")
    print("Выбросы IQR, %:")
    for col, info in report["iqr_outliers"].items():
        print(f"  {col:<12}{info['share_pct']:>6}  (граница сверху {info['high']:,})")
