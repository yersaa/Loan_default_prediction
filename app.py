"""Мини-дашборд Streamlit: аналитика по займам + скоринг заявки.
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
            "verification_status": "Source Verified", "addr_state": "CA",
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
            col.metric(name, f"PD = {res['pd']:.1%}")
            show = col.error if res["decision"] == "отказ" else col.success
            show(f"Решение: {res['decision']} (порог {res['threshold']:.2f})")
        st.caption("Калибровка выполнена на займах на 36 мес., поэтому PD для срока 60 мес. менее надёжна.")


def models_tab() -> None:
    """Вкладка «Модели»: таблицы и графики, сохранённые при обучении (пункт меню 6)."""
    path = C.REPORTS / "model_metrics.csv"
    if not path.exists():
        st.info("Модели ещё не обучены (пункт меню 6).")
        return
    st.subheader("Сравнение моделей на созревших займах 2015 г.")
    st.dataframe(pd.read_csv(path))
    selection = load_json(C.REPORTS / "model_selection.json")
    if selection:
        st.markdown(f"**Выбранная модель:** {selection['selected']}. Правило: {selection['rule']}.")
    approval = C.REPORTS / "approval_table.csv"
    if approval.exists():
        st.subheader("Таблица одобрения")
        st.dataframe(pd.read_csv(approval))
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
    tab1, tab2, tab3 = st.tabs(["Аналитика", "Скоринг заявки", "Модели"])
    with tab1:
        analytics_tab(sidebar_filters(df))
    with tab2:
        scoring_tab()
    with tab3:
        models_tab()


main()
