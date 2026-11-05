"""Аудит качества данных: пропуски, дубликаты, некорректные значения, выбросы (IQR)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.data_io import save_json

RAW_ROWS = 2_260_701  # строк в исходном CSV, включая служебные итоги


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


def plot_missing(missing: pd.Series) -> None:
    top = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot.barh(ax=ax, color="#4C72B0")
    ax.set(title="Доля пропусков по колонкам, %", xlabel="%")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "01_missing.png", dpi=120)
    plt.close(fig)


def run_audit(df: pd.DataFrame) -> dict:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    missing = (df.isna().mean() * 100).round(2)
    plot_missing(missing)
    report = {
        "rows_raw": RAW_ROWS,
        "rows_after_removing_summary_lines": len(df),
        "summary_lines_removed": RAW_ROWS - len(df),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "missing_pct": {k: v for k, v in missing.sort_values(ascending=False).items() if v > 0},
        "invalid_values": invalid_values(df),
        "iqr_outliers": {c: iqr_outliers(df[c]) for c in ["loan_amnt", "annual_inc", "dti", "revol_bal", "int_rate"]},
        "loan_status": df["loan_status"].value_counts().to_dict(),
        "issue_year_range": [int(df["issue_d"].dt.year.min()), int(df["issue_d"].dt.year.max())],
    }
    save_json(report, C.REPORTS / "audit.json")
    return report


def print_audit(report: dict) -> None:
    print(f"Строк: {report['rows_after_removing_summary_lines']:,} "
          f"(удалено служебных строк: {report['summary_lines_removed']})")
    print(f"Дубликатов id: {report['duplicate_ids']}")
    print("Пропуски, % (топ-8):")
    for col, pct in list(report["missing_pct"].items())[:8]:
        print(f"  {col:<22}{pct:>6}")
    print("Некорректные значения:")
    for name, n in report["invalid_values"].items():
        print(f"  {name:<34}{n:>8,}")
    print("Выбросы IQR, %:")
    for col, info in report["iqr_outliers"].items():
        print(f"  {col:<12}{info['share_pct']:>6}  (граница сверху {info['high']:,})")
