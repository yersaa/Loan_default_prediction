"""EDA: статические графики (matplotlib/seaborn) и интерактивные (Plotly)."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src import config as C
from src.database import run_query
from src.style import AQUA, BLUE, BLUES_SEQUENTIAL, DIVERGING, MUTED, ORANGE, VIOLET, save


def plot_default_by_grade() -> None:
    """Дефолтность и ставка по грейдам (данные из SQL). Две величины — две панели, а не две оси."""
    df = run_query("default_by_grade")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    bars = ax1.bar(df["grade"], df["default_rate_pct"], color=BLUE, width=0.7)
    ax1.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax1.set(title="Дефолтность по грейду", xlabel="Грейд", ylabel="Дефолт, %")
    ax1.grid(axis="x", visible=False)
    bars = ax2.bar(df["grade"], df["avg_int_rate"], color=ORANGE, width=0.7)
    ax2.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax2.set(title="Средняя ставка по грейду", xlabel="Грейд", ylabel="Ставка, %")
    ax2.grid(axis="x", visible=False)
    save(fig, "02_default_by_grade.png")


def plot_vintage() -> None:
    """Цензурирование: чем моложе займы, тем меньше завершено и тем выше дефолтность среди завершённых."""
    df = run_query("vintage_by_year_term")
    df = df[df["issue_year"] >= 2010]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for (term, part), color in zip(df.groupby("term_months"), [BLUE, ORANGE], strict=False):
        ax1.plot(part["issue_year"], part["default_rate_pct"], "o-", color=color, label=f"{term} мес.")
        ax2.plot(part["issue_year"], part["resolved_pct"], "o-", color=color, label=f"{term} мес.")
    ax1.set(title="Дефолтность среди завершённых займов", xlabel="Год выдачи", ylabel="Дефолт, %")
    ax2.set(title="Доля завершённых займов", xlabel="Год выдачи", ylabel="Завершено, %", ylim=(0, 105))
    ax1.legend()
    ax2.legend()
    save(fig, "03_vintage.png")


def plot_int_rate_box(df: pd.DataFrame) -> None:
    """Распределение ставки по грейдам: boxplot показывает выбросы по IQR."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sample = df.sample(min(C.EDA_SAMPLE_ROWS, len(df)), random_state=C.RANDOM_STATE)
    sns.boxplot(data=sample, x="grade", y="int_rate", order=list("ABCDEFG"), ax=ax, fliersize=1,
                color=VIOLET, boxprops={"alpha": 0.75}, flierprops={"markeredgecolor": MUTED})
    ax.set(title="Ставка по грейдам (boxplot, IQR)", xlabel="Грейд", ylabel="Ставка, %")
    save(fig, "04_int_rate_by_grade.png")


def plot_correlations(model_df: pd.DataFrame) -> None:
    """Корреляции числовых признаков и целевой переменной (Спирмен — устойчив к выбросам)."""
    cols = ["int_rate", "fico", "dti", "annual_inc", "loan_amnt", "term_months", "revol_util",
            "inq_last_6mths", "credit_hist_months", "loan_to_income", C.TARGET]
    sample = model_df[cols].sample(min(C.EDA_SAMPLE_ROWS, len(model_df)), random_state=C.RANDOM_STATE)
    corr = sample.corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8},
                linewidths=1, linecolor="#fcfcfb")
    ax.set_title(f"Корреляции Спирмена (созревшие займы, выборка {len(sample):,})")
    ax.grid(False)
    save(fig, "05_corr_heatmap.png")


def plot_dti_deciles() -> None:
    """Дефолтность по десяти равным группам долговой нагрузки (данные из SQL-запроса dti_deciles)."""
    df = run_query("dti_deciles")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{a:.0f}–{b:.0f}" for a, b in zip(df["dti_from"], df["dti_to"], strict=True)]
    bars = ax.bar(labels, df["default_rate_pct"], color=AQUA, width=0.7)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set(title="Дефолтность по децилям DTI (NTILE в SQL)", xlabel="Диапазон DTI в дециле, %", ylabel="Дефолт, %")
    ax.grid(axis="x", visible=False)
    save(fig, "06_dti_deciles.png")


def plot_state_map() -> None:
    """Интерактивная карта дефолтности по штатам."""
    df = run_query("state_rank")
    fig = px.choropleth(df, locations="state_code", locationmode="USA-states", scope="usa",
                        color="default_rate_pct", color_continuous_scale=BLUES_SEQUENTIAL,
                        hover_data=["loans", "region", "rank_overall"],
                        labels={"default_rate_pct": "Дефолт, %"},
                        title="Дефолтность по штатам, % (созревшие займы)")
    fig.write_html(C.FIGURES / "07_default_by_state.html", include_plotlyjs="cdn")


def plot_monthly_rolling() -> None:
    """Интерактивный график: дефолтность 36-месячных займов по месяцу выдачи + скользящее среднее (окно в SQL)."""
    df = run_query("monthly_rolling")
    df = df[df["issue_month"] >= "2010-01"]
    fig = px.line(df, x="issue_month", y=["default_rate_pct", "rolling_3m_pct"],
                  color_discrete_sequence=[BLUE, ORANGE],
                  title="Дефолтность займов на 36 мес. по месяцу выдачи и скользящее среднее за 3 мес.",
                  labels={"value": "Дефолт, %", "issue_month": "Месяц выдачи", "variable": ""})
    fig.write_html(C.FIGURES / "08_monthly_rolling.html", include_plotlyjs="cdn")


def run_eda(df: pd.DataFrame, model_df: pd.DataFrame) -> list:
    """Строит все графики EDA и возвращает список файлов в папке с графиками."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for step in [plot_default_by_grade, plot_vintage, plot_dti_deciles, plot_state_map, plot_monthly_rolling]:
        step()
    plot_int_rate_box(df)
    plot_correlations(model_df)
    return sorted(p.name for p in C.FIGURES.iterdir())
