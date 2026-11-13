"""Мини-дашборд Streamlit: итоги проекта, аналитика по займам с фильтрами, оценка заявки и сравнение моделей.
Запуск: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C
from src.data_io import load_dataset, load_json
from src.features import add_target, is_matured, term_to_months
from src.style import BLUE, BLUES_SEQUENTIAL, ORANGE

st.set_page_config(page_title="LendingClub PD", layout="wide")

FULL_DATA_HINT = "Демо-выборка: доли и дефолтность не отражают полный набор. Полные данные: пункт меню 1."
MODEL_NAMES = {
    "Dummy (prior)": "Константа (доля дефолтов)", "Grade only (LogReg)": "Только грейд LendingClub",
    "Logistic Regression": "Логистическая регрессия", "LightGBM": "LightGBM", "XGBoost": "XGBoost",
    "Neural network (MLP)": "Нейросеть (MLP)", "Ensemble (LightGBM + XGBoost + MLP)": "Ансамбль",
    "Logistic Regression (borrower only)": "Логистическая регрессия, поля заёмщика",
    "LightGBM (borrower only)": "LightGBM, поля заёмщика",
}
METRICS_HELP = """
**ROC-AUC** — умеет ли модель отличать рискованных заёмщиков от надёжных: 0,5 — угадывание, 1 — идеал.
По ней выбирается модель. **PR-AUC** — как хорошо модель находит именно дефолты (у угадывания около 0,15).
**ECE** — можно ли верить цифре PD: средний разрыв между прогнозом и фактом.
**Recall** — какую долю дефолтов модель поймала при пороге; **precision** — какая доля отказов пришлась на настоящие
дефолты. **Accuracy** здесь обманчива: модель «все вернут» получает 0,85, не найдя ни одного дефолта.
"""


def dataset_path() -> Path:
    """Полный Parquet, если он есть; иначе демо-выборка из репозитория."""
    return C.PARQUET if C.PARQUET.exists() else C.SAMPLE_PARQUET


@st.cache_data
def get_loans() -> pd.DataFrame:
    """Созревшие завершённые займы для вкладки «Аналитика». Результат кэшируется: файл читается один раз."""
    cols = ["issue_d", "grade", "purpose", "addr_state", "term", "loan_amnt", "int_rate", "loan_status"]
    df = add_target(load_dataset(cols, path=dataset_path()))
    df = df[df[C.TARGET].notna() & is_matured(df["issue_d"], term_to_months(df["term"]))].copy()
    df["year"] = df["issue_d"].dt.year
    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Рисует фильтры в боковой панели и возвращает займы, которые под них подходят."""
    st.sidebar.header("Фильтры")
    years = st.sidebar.slider("Год выдачи", int(df.year.min()), int(df.year.max()), (2012, 2015))
    grades = st.sidebar.multiselect("Грейд", list("ABCDEFG"), default=list("ABCDEFG"))
    term_options = sorted(df.term.cat.categories)
    terms = st.sidebar.multiselect("Срок", term_options, default=term_options)
    purposes = st.sidebar.multiselect("Цель кредита", sorted(df.purpose.cat.categories))
    mask = df.year.between(*years) & df.grade.isin(grades) & df.term.isin(terms)
    if purposes:
        mask &= df.purpose.isin(purposes)
    return df[mask]


def about_tab() -> None:
    """Вкладка «О проекте»: задача и главные результаты полного запуска в четырёх цифрах."""
    st.markdown(
        "По анкете заёмщика модель оценивает **вероятность дефолта (PD)** — шанс, что он не вернёт займ. "
        "Обучена на займах 2007–2014 годов, проверена на займах 2015 года. Используются только данные, "
        "известные в момент заявки.")
    metrics_path = C.ROOT / "reports" / "model_metrics.csv"
    selection = load_json(C.ROOT / "reports" / "model_selection.json")
    if not metrics_path.exists() or not selection:
        st.info("Результаты полного запуска не найдены (пункт меню 6).")
        return
    metrics = pd.read_csv(metrics_path).set_index("model")
    chosen = selection["selected"]
    row = metrics.loc[chosen]
    approval = pd.read_csv(C.ROOT / "reports" / "approval_table.csv")
    approve = approval[approval["model"] == chosen].set_index("approval_rate")["default_rate_pct"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Займов в данных", "2,26 млн")
    k2.metric(f"ROC-AUC ({MODEL_NAMES[chosen]})", f"{row['roc_auc']:.3f}",
              f"{row['roc_auc'] - metrics.loc['Grade only (LogReg)', 'roc_auc']:+.3f} к грейду LendingClub")
    k3.metric("Ловит дефолтов", f"{row['recall']:.0%}", f"при пороге PD {row['threshold']:.2f}", delta_color="off")
    k4.metric("Дефолт при одобрении 80%", f"{approve[0.8]:.1f}%", f"{approve[0.8] - approve[1.0]:+.1f} п. п.",
              delta_color="inverse")
    st.markdown(
        "- **Вкладка «Аналитика»** — дефолтность по грейдам, годам и штатам с фильтрами.\n"
        "- **Вкладка «Скоринг заявки»** — введите анкету и получите PD и решение.\n"
        "- **Вкладка «Модели»** — сравнение девяти моделей и что означает каждая метрика.\n\n"
        "Весь проект по шагам — в ноутбуке `project_walkthrough.ipynb`.")


def analytics_tab(df: pd.DataFrame) -> None:
    """Вкладка «Аналитика»: четыре показателя и графики дефолтности по грейду, году и штату."""
    st.caption("Только созревшие займы: срок истёк к декабрю 2018 г., поэтому дефолтность не смещена.")
    if df.empty:
        st.warning("Нет данных под выбранные фильтры.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Созревших займов", f"{len(df):,}")
    k2.metric("Дефолтность", f"{df[C.TARGET].mean():.1%}")
    k3.metric("Средняя ставка", f"{df.int_rate.mean():.1f}%")
    k4.metric("Средняя сумма", f"${df.loan_amnt.mean():,.0f}")

    left, right = st.columns(2)
    by_grade = df.groupby("grade", observed=True)[C.TARGET].mean().mul(100).reset_index()
    left.plotly_chart(px.bar(by_grade, x="grade", y=C.TARGET, title="Дефолт по грейду, %",
                             color_discrete_sequence=[BLUE],
                             labels={C.TARGET: "Дефолт, %", "grade": "Грейд"}))
    by_year = df.groupby(["year", "term"], observed=True)[C.TARGET].mean().mul(100).reset_index()
    right.plotly_chart(px.line(by_year, x="year", y=C.TARGET, color="term", markers=True,
                               color_discrete_sequence=[BLUE, ORANGE], title="Дефолт по году выдачи, %",
                               labels={C.TARGET: "Дефолт, %", "year": "Год", "term": "Срок"}))
    by_state = df.groupby("addr_state", observed=True)[C.TARGET].agg(["mean", "size"]).reset_index()
    by_state["default_pct"] = by_state["mean"] * 100
    st.plotly_chart(px.choropleth(by_state, locations="addr_state", locationmode="USA-states", scope="usa",
                                  color="default_pct", hover_data=["size"], color_continuous_scale=BLUES_SEQUENTIAL,
                                  labels={"default_pct": "Дефолт, %", "size": "Займов"},
                                  title="Дефолт по штатам, %"))


def scoring_tab() -> None:
    """Вкладка «Скоринг заявки»: форма анкеты и вероятность дефолта от каждой модели."""
    from src.models import APPLICATION_LIMITS as L
    from src.models import HOME_OPTIONS, SUB_GRADES, predict_pd
    with st.form("application"):
        c1, c2, c3 = st.columns(3)
        raw = {
            "loan_amnt": c1.number_input("Сумма, $", *L["loan_amnt"], 15000, step=500),
            "term_months": c1.selectbox("Срок, мес.", [36, 60]),
            "int_rate": c1.number_input("Ставка, %", float(L["int_rate"][0]), float(L["int_rate"][1]), 13.0),
            "sub_grade": c1.selectbox("Подгрейд", SUB_GRADES, index=10),
            "annual_inc": c2.number_input("Годовой доход, $", *L["annual_inc"], 65000, step=1000),
            "dti": c2.number_input("DTI, %", float(L["dti"][0]), float(L["dti"][1]), 18.0),
            "fico": c2.number_input("FICO", *L["fico"], 700),
            "emp_years": c2.slider("Стаж, лет", *L["emp_years"], 5),
            "revol_util": c3.number_input("Использование кредитных линий, %",
                                          float(L["revol_util"][0]), float(L["revol_util"][1]), 50.0),
            "inq_last_6mths": c3.number_input("Запросов в бюро за 6 мес.", *L["inq_last_6mths"], 1),
            "credit_hist_months": c3.number_input("Кредитная история, мес.", *L["credit_hist_months"], 180),
            "home_ownership": c3.selectbox("Жильё", HOME_OPTIONS, index=1),
            "purpose": c3.selectbox("Цель", ["debt_consolidation", "credit_card", "home_improvement",
                                             "small_business", "major_purchase", "medical", "car", "other"]),
            "verification_status": "Source Verified",
            "initial_list_status": "w", "application_type": "Individual",
        }
        submitted = st.form_submit_button("Оценить")
    if submitted:
        try:
            result = predict_pd(raw)
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            return
        cols = st.columns(len(result))
        for col, (name, res) in zip(cols, result.items(), strict=True):
            col.metric(MODEL_NAMES.get(name, name), f"PD = {res['pd']:.1%}")
            show = col.error if res["decision"] == "отказ" else col.success
            show(f"Решение: {res['decision']} (порог {res['threshold']:.2f})")
        st.markdown(
            "**Как читать.** PD — вероятность, что заёмщик не вернёт займ. Если она не ниже порога, модель "
            "советует отказать. Порог подобран на займах 2014 года; в реальном банке его выбирает кредитор "
            "под свой риск-аппетит. Главная модель проекта — LightGBM; «поля заёмщика» — вариант без грейда "
            "и ставки, которые назначает сам LendingClub.")
        st.caption("Калибровка выполнена на займах на 36 мес., поэтому PD для срока 60 мес. менее надёжна.")


def models_tab() -> None:
    """Вкладка «Модели»: таблицы и графики, сохранённые при обучении (пункт меню 6)."""
    path = C.REPORTS / "model_metrics.csv"
    if not path.exists():
        st.info("Модели ещё не обучены (пункт меню 6).")
        return
    st.subheader("Сравнение моделей на созревших займах 2015 г.")
    metrics = pd.read_csv(path)
    selection = load_json(C.REPORTS / "model_selection.json")
    chosen = selection["selected"] if selection else None
    table = pd.DataFrame({
        "Модель": metrics["model"].map(lambda m: MODEL_NAMES.get(m, m) + (" ✔ выбрана" if m == chosen else "")),
        "ROC-AUC": metrics["roc_auc"], "PR-AUC": metrics["pr_auc"], "ECE": metrics["ece"],
        "Порог": metrics["threshold"], "Recall": metrics["recall"], "Precision": metrics["precision"],
        "Accuracy": metrics["accuracy"], "Обучение, с": metrics["fit_seconds"],
    })
    st.dataframe(table.round(3), hide_index=True)
    if selection:
        st.markdown(f"**Выбранная модель:** {MODEL_NAMES.get(chosen, chosen)}. Правило выбора задано заранее: "
                    "лучшая ROC-AUC на 2014 годе; логистическая регрессия, если отстаёт меньше чем на 0,005; "
                    "XGBoost, нейросеть или ансамбль — только если лучше на 0,005 и больше.")
    with st.expander("Что означают метрики"):
        st.markdown(METRICS_HELP)
    approval = C.REPORTS / "approval_table.csv"
    if approval.exists():
        st.subheader("Таблица одобрения: дефолтность портфеля, если одобрять долю самых надёжных заявок")
        approval_table = pd.read_csv(approval)
        approval_table["model"] = approval_table["model"].map(lambda m: MODEL_NAMES.get(m, m))
        st.plotly_chart(px.line(approval_table, x=approval_table["approval_rate"] * 100, y="default_rate_pct",
                                color="model", markers=True,
                                labels={"x": "Одобряем, % заявок", "default_rate_pct": "Дефолт, %", "model": ""}))
    censoring = C.REPORTS / "censoring_sensitivity.csv"
    if censoring.exists():
        st.subheader("Чувствительность к цензурированию")
        st.dataframe(pd.read_csv(censoring))
    for name in ["09_roc_pr.png", "11_calibration.png", "13_approval.png", "12_shap_summary.png"]:
        if (C.FIGURES / name).exists():
            st.image(str(C.FIGURES / name))


def main() -> None:
    """Собирает страницу: загружает данные и показывает три вкладки."""
    st.title("LendingClub: анализ и прогноз дефолта")
    if not C.PARQUET.exists():
        st.info(FULL_DATA_HINT)
    try:
        df = get_loans()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    about, analytics, scoring, models = st.tabs(["О проекте", "Аналитика", "Скоринг заявки", "Модели"])
    with about:
        about_tab()
    with analytics:
        analytics_tab(sidebar_filters(df))
    with scoring:
        scoring_tab()
    with models:
        models_tab()


main()
