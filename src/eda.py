"""EDA: статические графики (matplotlib/seaborn) и интерактивные (Plotly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src import config as C
from src.database import run_query

sns.set_theme(style="whitegrid")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(C.FIGURES / name, dpi=120)
    plt.close(fig)


def plot_default_by_grade() -> None:
    """Дефолтность и ставка по грейдам (данные из SQL)."""
    df = run_query("default_by_grade")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["grade"], df["default_rate_pct"], color="#C44E52")
    ax.set(title="Дефолтность по грейду LendingClub", xlabel="Грейд", ylabel="Дефолт, %")
    ax2 = ax.twinx()
    ax2.plot(df["grade"], df["avg_int_rate"], "o-", color="#4C72B0")
    ax2.set_ylabel("Средняя ставка, %", color="#4C72B0")
    ax2.grid(False)
    save(fig, "02_default_by_grade.png")


def plot_vintage() -> None:
    """Винтажи: дефолтность по году выдачи и сроку."""
    df = run_query("vintage_by_year_term")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for term, part in df.groupby("term_months"):
        ax.plot(part["issue_year"], part["default_rate_pct"], "o-", label=f"{term} мес.")
    ax.axvspan(2015.5, 2018.5, color="grey", alpha=0.15, label="много незавершённых займов")
    ax.set(title="Дефолтность по году выдачи (завершённые займы)", xlabel="Год выдачи", ylabel="Дефолт, %")
    ax.legend()
    save(fig, "03_vintage.png")


def plot_int_rate_box(df: pd.DataFrame) -> None:
    """Распределение ставки по грейдам: boxplot показывает выбросы по IQR."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sample = df.sample(200_000, random_state=C.RANDOM_STATE)
    sns.boxplot(data=sample, x="grade", y="int_rate", order=list("ABCDEFG"), ax=ax,
                fliersize=1, color="#8172B3")
    ax.set(title="Ставка по грейдам (boxplot, IQR)", xlabel="Грейд", ylabel="Ставка, %")
    save(fig, "04_int_rate_by_grade.png")


def plot_correlations(model_df: pd.DataFrame) -> None:
    """Корреляции числовых признаков и целевой переменной (Спирмен — устойчив к выбросам)."""
    cols = ["int_rate", "fico", "dti", "annual_inc", "loan_amnt", "term_months", "revol_util",
            "inq_last_6mths", "credit_hist_months", "loan_to_income", C.TARGET]
    corr = model_df[cols].sample(200_000, random_state=C.RANDOM_STATE).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Корреляции Спирмена (выборка 200 тыс.)")
    save(fig, "05_corr_heatmap.png")


def plot_dti_deciles() -> None:
    df = run_query("dti_deciles")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["dti_decile"].astype(str), df["default_rate_pct"], color="#55A868")
    ax.set(title="Дефолтность по децилям DTI (NTILE в SQL)", xlabel="Дециль DTI", ylabel="Дефолт, %")
    save(fig, "06_dti_deciles.png")


def plot_state_map() -> None:
    """Интерактивная карта дефолтности по штатам."""
    df = run_query("state_rank")
    fig = px.choropleth(df, locations="state_code", locationmode="USA-states", scope="usa",
                        color="default_rate_pct", color_continuous_scale="Reds",
                        hover_data=["loans", "region", "rank_overall"],
                        title="Дефолтность по штатам, % (завершённые займы)")
    fig.write_html(C.FIGURES / "07_default_by_state.html", include_plotlyjs="cdn")


def plot_monthly_rolling() -> None:
    """Интерактивный график: дефолтность по месяцу выдачи + скользящее среднее (окно в SQL)."""
    df = run_query("monthly_rolling")
    df = df[df["issue_month"] >= "2010-01"]
    fig = px.line(df, x="issue_month", y=["default_rate_pct", "rolling_3m_pct"],
                  title="Дефолтность по месяцу выдачи и скользящее среднее (3 мес.)",
                  labels={"value": "Дефолт, %", "issue_month": "Месяц выдачи", "variable": ""})
    fig.write_html(C.FIGURES / "08_monthly_rolling.html", include_plotlyjs="cdn")


def run_eda(df: pd.DataFrame, model_df: pd.DataFrame) -> list:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    steps = [plot_default_by_grade, plot_vintage, plot_dti_deciles, plot_state_map, plot_monthly_rolling]
    for step in steps:
        step()
    plot_int_rate_box(df)
    plot_correlations(model_df)
    return sorted(p.name for p in C.FIGURES.iterdir())
