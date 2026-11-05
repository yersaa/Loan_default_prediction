"""Консольное меню проекта: LendingClub — прогноз дефолта по займу."""
from datetime import datetime

import pandas as pd

from src import config as C
from src.data_io import csv_to_parquet, load_dataset, load_json, save_json

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

MENU = {
    "1": "Подготовить данные (CSV.gz -> Parquet чанками)",
    "2": "Аудит качества данных",
    "3": "Создать базу SQLite (нормализованная схема)",
    "4": "Выполнить SQL-запросы",
    "5": "EDA и графики",
    "6": "Обучить и сравнить модели",
    "7": "Оценить заявку (ввод вручную)",
    "8": "Бенчмарки: память, CSV vs Parquet, Pandas vs SQL",
    "9": "Выполнить шаги 1–6 и 8 по порядку",
    "0": "Выход",
}


def mark_done(step: str) -> None:
    """Сохраняет в state.json, какие шаги и когда выполнены."""
    state = load_json(C.STATE_FILE, default={})
    state[step] = datetime.now().isoformat(timespec="seconds")
    save_json(state, C.STATE_FILE)


def step_prepare():
    csv_to_parquet()


def step_audit():
    from src.audit import print_audit, run_audit
    print_audit(run_audit(load_dataset()))


def step_database():
    from src.database import build_database
    print(build_database(load_dataset()))


def step_queries():
    from src.database import run_all_queries
    for name, df in run_all_queries().items():
        print(f"\n== {name}\n{df.head(10).to_string(index=False)}")


def step_eda():
    from src.eda import run_eda
    from src.features import modeling_frame
    df = load_dataset()
    print("Графики:", run_eda(df, modeling_frame(df.copy())))


def step_models():
    from src.explain import run_explain
    from src.features import modeling_frame
    from src.models import train_and_compare, tune
    model_df = modeling_frame(load_dataset())
    train = model_df[model_df["issue_d"].dt.year < C.TEST_YEAR]
    print(f"Train: {len(train):,}  Test: {len(model_df) - len(train):,}")
    best = tune(train[C.FEATURES], train[C.TARGET])
    save_json(best, C.REPORTS / "tuning.json")
    metrics, fitted, preds, test = train_and_compare(model_df, best)
    print(metrics.to_string(index=False))
    result = run_explain(fitted, preds, test, metrics)
    print("\nОшибки по грейдам:\n", result["errors"].to_string(index=False))


def ask_number(prompt: str, default: float, low: float, high: float) -> float:
    """Повторяет ввод, пока не получит число в допустимом диапазоне."""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("    Нужно число.")
            continue
        if low <= value <= high:
            return value
        print(f"    Допустимо от {low} до {high}.")


def ask_choice(prompt: str, options: list, default: str) -> str:
    while True:
        hint = options if len(options) <= 5 else f"{options[0]}…{options[-1]}"
        raw = input(f"  {prompt} {hint} [{default}]: ").strip().upper() or default
        if raw in options:
            return raw
        print("    Выберите значение из списка.")


def step_predict():
    from src.models import predict_pd
    print("Анкета заёмщика (Enter — значение по умолчанию):")
    raw = {
        "loan_amnt": ask_number("Сумма займа, $", 15000, 500, 40000),
        "term_months": ask_number("Срок, мес. (36/60)", 36, 36, 60),
        "int_rate": ask_number("Ставка, %", 13.0, 5, 31),
        "sub_grade": ask_choice("Подгрейд", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)], "C1"),
        "annual_inc": ask_number("Годовой доход, $", 65000, 1000, 10_000_000),
        "dti": ask_number("DTI (долговая нагрузка), %", 18, 0, 100),
        "fico": ask_number("FICO", 700, 600, 850),
        "emp_years": ask_number("Стаж, лет (0–10)", 5, 0, 10),
        "revol_util": ask_number("Использование кредитных линий, %", 50, 0, 150),
        "inq_last_6mths": ask_number("Запросов в бюро за 6 мес.", 1, 0, 30),
        "credit_hist_months": ask_number("Длина кредитной истории, мес.", 180, 0, 900),
        "home_ownership": ask_choice("Жильё", ["MORTGAGE", "RENT", "OWN"], "RENT"),
        "purpose": "debt_consolidation", "verification_status": "Source Verified",
        "addr_state": "CA", "initial_list_status": "w", "application_type": "Individual",
    }
    for name, res in predict_pd(raw).items():
        print(f"  {name:<20} PD = {res['pd']:.1%}  (порог {res['threshold']:.2f}) -> {res['decision']}")


def step_benchmarks():
    from src.benchmark import run_benchmarks
    import json
    print(json.dumps(run_benchmarks(), ensure_ascii=False, indent=2))


STEPS = {"1": step_prepare, "2": step_audit, "3": step_database, "4": step_queries,
         "5": step_eda, "6": step_models, "7": step_predict, "8": step_benchmarks}


def run_step(key: str) -> None:
    try:
        STEPS[key]()
        mark_done(MENU[key])
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано пользователем.")


def main():
    for d in (C.REPORTS, C.FIGURES, C.MODELS):
        d.mkdir(parents=True, exist_ok=True)
    while True:
        print("\n=== LendingClub: прогноз дефолта ===")
        for key, title in MENU.items():
            print(f"  {key} – {title}")
        choice = input("Выбор: ").strip()
        if choice == "0":
            break
        elif choice == "9":
            for key in ["1", "2", "3", "4", "5", "6", "8"]:
                print(f"\n>>> {MENU[key]}")
                run_step(key)
        elif choice in STEPS:
            run_step(choice)
        else:
            print("Нет такого пункта.")


if __name__ == "__main__":
    main()
