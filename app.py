"""Мини-дашборд Streamlit: аналитика по займам + скоринг заявки.
Запуск: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C
from src.data_io import load_dataset
from src.features import add_target

st.set_page_config(page_title="LendingClub PD", layout="wide")


@st.cache_data
def get_loans() -> pd.DataFrame:
    cols = ["issue_d", "grade", "purpose", "addr_state", "term", "loan_amnt", "int_rate", "loan_status"]
    df = add_target(load_dataset(cols))
    df = df[df[C.TARGET].notna()].copy()
    df["year"] = df["issue_d"].dt.year
    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Фильтры")
    years = st.sidebar.slider("Год выдачи", int(df.year.min()), int(df.year.max()), (2012, 2015))
    grades = st.sidebar.multiselect("Грейд", list("ABCDEFG"), default=list("ABCDEFG"))
    terms = st.sidebar.multiselect("Срок", sorted(df.term.cat.categories), default=sorted(df.term.cat.categories))
    purposes = st.sidebar.multiselect("Цель кредита", sorted(df.purpose.cat.categories))
    mask = df.year.between(*years) & df.grade.isin(grades) & df.term.isin(terms)
    if purposes:
        mask &= df.purpose.isin(purposes)
    return df[mask]


def analytics_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Нет данных под выбранные фильтры.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Завершённых займов", f"{len(df):,}")
    k2.metric("Дефолтность", f"{df[C.TARGET].mean():.1%}")
    k3.metric("Средняя ставка", f"{df.int_rate.mean():.1f}%")
    k4.metric("Средняя сумма", f"${df.loan_amnt.mean():,.0f}")

    left, right = st.columns(2)
    by_grade = df.groupby("grade", observed=True)[C.TARGET].mean().mul(100).reset_index()
    left.plotly_chart(px.bar(by_grade, x="grade", y=C.TARGET, title="Дефолт по грейду, %",
                             labels={C.TARGET: "Дефолт, %", "grade": "Грейд"}))
    by_year = df.groupby(["year", "term"], observed=True)[C.TARGET].mean().mul(100).reset_index()
    right.plotly_chart(px.line(by_year, x="year", y=C.TARGET, color="term", markers=True,
                               title="Дефолт по году выдачи, %", labels={C.TARGET: "Дефолт, %", "year": "Год"}))
    by_state = df.groupby("addr_state", observed=True)[C.TARGET].agg(["mean", "size"]).reset_index()
    by_state["default_pct"] = by_state["mean"] * 100
    st.plotly_chart(px.choropleth(by_state, locations="addr_state", locationmode="USA-states", scope="usa",
                                  color="default_pct", hover_data=["size"], color_continuous_scale="Reds",
                                  title="Дефолт по штатам, %"))


def scoring_tab() -> None:
    from src.models import predict_pd
    with st.form("application"):
        c1, c2, c3 = st.columns(3)
        raw = {
            "loan_amnt": c1.number_input("Сумма, $", 500, 40000, 15000, step=500),
            "term_months": c1.selectbox("Срок, мес.", [36, 60]),
            "int_rate": c1.number_input("Ставка, %", 5.0, 31.0, 13.0),
            "sub_grade": c1.selectbox("Подгрейд", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)], index=10),
            "annual_inc": c2.number_input("Годовой доход, $", 1000, 5_000_000, 65000, step=1000),
            "dti": c2.number_input("DTI, %", 0.0, 100.0, 18.0),
            "fico": c2.number_input("FICO", 600, 850, 700),
            "emp_years": c2.slider("Стаж, лет", 0, 10, 5),
            "revol_util": c3.number_input("Использование кредитных линий, %", 0.0, 150.0, 50.0),
            "inq_last_6mths": c3.number_input("Запросов в бюро за 6 мес.", 0, 30, 1),
            "credit_hist_months": c3.number_input("Кредитная история, мес.", 0, 900, 180),
            "home_ownership": c3.selectbox("Жильё", ["RENT", "MORTGAGE", "OWN"]),
            "purpose": c3.selectbox("Цель", ["debt_consolidation", "credit_card", "home_improvement",
                                             "small_business", "major_purchase", "medical", "car", "other"]),
            "verification_status": "Source Verified", "addr_state": "CA",
            "initial_list_status": "w", "application_type": "Individual",
        }
        submitted = st.form_submit_button("Оценить")
    if submitted:
        try:
            result = predict_pd(raw)
        except FileNotFoundError as e:
            st.error(str(e))
            return
        cols = st.columns(len(result))
        for col, (name, res) in zip(cols, result.items()):
            col.metric(name, f"PD = {res['pd']:.1%}")
            show = col.error if res["decision"] == "отказ" else col.success
            show(f"Решение: {res['decision']} (порог {res['threshold']:.2f})")


def models_tab() -> None:
    path = C.REPORTS / "model_metrics.csv"
    if not path.exists():
        st.info("Модели ещё не обучены (пункт меню 6).")
        return
    st.subheader("Сравнение моделей на займах 2015 г.")
    st.dataframe(pd.read_csv(path))
    for name in ["09_roc_pr.png", "12_shap_summary.png", "11_calibration.png"]:
        if (C.FIGURES / name).exists():
            st.image(str(C.FIGURES / name))


def main():
    st.title("LendingClub: анализ и прогноз дефолта")
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
