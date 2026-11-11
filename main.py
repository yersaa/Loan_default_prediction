"""Консольное меню проекта: LendingClub — прогноз дефолта по займу.

python main.py                   интерактивное меню на полных данных
python main.py --sample          меню на демо-выборке, результаты пишутся в sample_run/
python main.py --run-all         весь пайплайн без меню
python main.py --make-sample     пересобрать демо-выборку из полного Parquet
"""
import argparse
import json
import sqlite3
from datetime import datetime

import pandas as pd

from src import config as C
from src.data_io import csv_to_parquet, load_dataset, load_json, make_sample, save_json

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
    "8": "Бенчмарки: память, CSV vs Parquet, Pandas vs SQL, память моделей",
    "9": "Выполнить шаги 1–6 и 8 по порядку",
    "0": "Выход",
}
PIPELINE = ["1", "2", "3", "4", "5", "6", "8"]
EXIT_INTERRUPTED = 130  # общепринятый код завершения после Ctrl+C


def mark_done(step: str) -> None:
    """Сохраняет в state.json, какие шаги и когда выполнены."""
    state = load_json(C.STATE_FILE, default={})
    state[step] = datetime.now().isoformat(timespec="seconds")
    save_json(state, C.STATE_FILE)


# Каждый пункт меню — отдельная функция step_*. Модули src импортируются внутри функций:
# меню открывается сразу, а тяжёлые библиотеки (LightGBM, SHAP, matplotlib) загружаются только когда нужны.

def step_prepare() -> None:
    """Пункт 1: читает исходный CSV.gz частями и сохраняет нужные колонки в Parquet."""
    if C.SAMPLE_MODE:
        print("Демо-режим: выборка уже в формате Parquet, шаг не нужен.")
        return
    csv_to_parquet()


def step_audit() -> None:
    """Пункт 2: считает пропуски, дубликаты, некорректные значения и выбросы, сохраняет reports/audit.json."""
    from src.audit import print_audit, run_audit
    print_audit(run_audit(load_dataset()))


def step_database() -> None:
    """Пункт 3: раскладывает таблицу займов по нормализованной схеме и загружает её в SQLite."""
    from src.database import build_database
    print(build_database(load_dataset()))


def step_queries() -> None:
    """Пункт 4: выполняет все запросы из sql/queries.sql и печатает первые 10 строк каждого."""
    from src.database import run_all_queries
    for name, df in run_all_queries().items():
        print(f"\n== {name}\n{df.head(10).to_string(index=False)}")


def step_eda() -> None:
    """Пункт 5: строит графики EDA. Часть графиков берёт данные из SQL, поэтому нужна база (пункт 3)."""
    from src.eda import run_eda
    from src.features import modeling_frame
    df = load_dataset()
    print("Графики:", run_eda(df, modeling_frame(df.copy())))


def step_models() -> None:
    """Пункт 6: подбор гиперпараметров, обучение и сравнение моделей, SHAP, ошибки и чувствительность."""
    from src.censoring import censoring_sensitivity
    from src.explain import run_explain
    from src.features import modeling_frame, temporal_split
    from src.models import train_and_compare, tune
    df = load_dataset()
    model_df = modeling_frame(df.copy())
    windows = temporal_split(model_df)
    fit = windows["fit"]
    print(f"Созревшие займы. Train: {len(windows['train']):,}  Test: {len(windows['test']):,}")
    # Гиперпараметры подбираются на 2007–2013: 2014 год не участвует, по нему потом выбирается модель
    best = tune(fit[C.FEATURES], fit[C.TARGET].astype(int))
    save_json(best, C.REPORTS / "tuning.json")
    metrics, fitted, preds, preds_raw, test = train_and_compare(model_df, best)
    print(metrics[["model", "valid_auc", "roc_auc", "gini", "ks", "pr_auc", "brier", "f1"]].to_string(index=False))
    result = run_explain(fitted, preds, preds_raw, test, metrics)
    print("\nОшибки по грейдам:\n", result["errors"].to_string(index=False))
    print("\nЧувствительность к цензурированию:\n", censoring_sensitivity(df, best).to_string(index=False))


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
    """Повторяет ввод, пока значение не окажется в списке (регистр не важен)."""
    while True:
        hint = options if len(options) <= 5 else f"{options[0]}…{options[-1]}"
        raw = input(f"  {prompt} {hint} [{default}]: ").strip().upper() or default
        if raw in options:
            return raw
        print("    Выберите значение из списка.")


def step_predict() -> None:
    """Пункт 7: спрашивает анкету заёмщика и печатает вероятность дефолта и решение каждой модели."""
    from src.models import APPLICATION_LIMITS as L
    from src.models import HOME_OPTIONS, SUB_GRADES, predict_pd
    print("Анкета заёмщика (Enter — значение по умолчанию):")
    raw = {
        "loan_amnt": ask_number("Сумма займа, $", 15000, *L["loan_amnt"]),
        "term_months": int(ask_choice("Срок, мес.", ["36", "60"], "36")),
        "int_rate": ask_number("Ставка, %", 13.0, *L["int_rate"]),
        "sub_grade": ask_choice("Подгрейд", SUB_GRADES, "C1"),
        "annual_inc": ask_number("Годовой доход, $", 65000, *L["annual_inc"]),
        "dti": ask_number("DTI (долговая нагрузка), %", 18, *L["dti"]),
        "fico": ask_number("FICO", 700, *L["fico"]),
        "emp_years": ask_number("Стаж, лет (0–10)", 5, *L["emp_years"]),
        "revol_util": ask_number("Использование кредитных линий, %", 50, *L["revol_util"]),
        "inq_last_6mths": ask_number("Запросов в бюро за 6 мес.", 1, *L["inq_last_6mths"]),
        "credit_hist_months": ask_number("Длина кредитной истории, мес.", 180, *L["credit_hist_months"]),
        "home_ownership": ask_choice("Жильё", HOME_OPTIONS, "RENT"),
        "purpose": "debt_consolidation", "verification_status": "Source Verified",
        "addr_state": "CA", "initial_list_status": "w", "application_type": "Individual",
    }
    for name, res in predict_pd(raw).items():
        print(f"  {name:<26} PD = {res['pd']:.1%}  (порог {res['threshold']:.2f}) -> {res['decision']}")


def step_benchmarks() -> None:
    """Пункт 8: CSV против Parquet, Pandas против SQL, память моделей. Нужны исходный CSV, база и пункт 6."""
    if C.SAMPLE_MODE:
        print("Демо-режим: бенчмарки сравнивают форматы на исходном CSV, поэтому шаг пропущен.")
        return
    from src.benchmark import run_benchmarks
    print(json.dumps(run_benchmarks(), ensure_ascii=False, indent=2))


STEPS = {"1": step_prepare, "2": step_audit, "3": step_database, "4": step_queries,
         "5": step_eda, "6": step_models, "7": step_predict, "8": step_benchmarks}


def run_step(key: str) -> bool:
    """Выполняет шаг меню, не давая ожидаемым ошибкам завершить программу. True — шаг выполнен.

    KeyboardInterrupt здесь не перехватывается: Ctrl+C должен остановить и весь пайплайн.
    """
    try:
        STEPS[key]()
    except (FileNotFoundError, ValueError, sqlite3.Error, pd.errors.DatabaseError) as e:
        print(f"Ошибка: {e}")
        return False
    mark_done(MENU[key])
    return True


def run_pipeline() -> list:
    """Шаги 1–6 и 8 по порядку; неудачный шаг не мешает остальным, Ctrl+C останавливает всё.

    Возвращает номера шагов, которые не удалось выполнить.
    """
    failed = []
    for key in PIPELINE:
        print(f"\n>>> {MENU[key]}")
        if not run_step(key):
            failed.append(key)
    return failed


def menu_loop() -> None:
    """Показывает меню и выполняет выбранный пункт, пока пользователь не введёт 0."""
    while True:
        mode = " (демо-выборка)" if C.SAMPLE_MODE else ""
        print(f"\n=== LendingClub: прогноз дефолта{mode} ===")
        for key, title in MENU.items():
            print(f"  {key} – {title}")
        choice = input("Выбор: ").strip()
        if choice == "0":
            break
        try:
            if choice == "9":
                run_pipeline()
            elif choice in STEPS:
                run_step(choice)
            else:
                print("Нет такого пункта.")
        except KeyboardInterrupt:  # Ctrl+C прерывает шаг и возвращает в меню
            print("\nПрервано пользователем.")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Разбирает флаги командной строки: --sample, --run-all, --make-sample."""
    parser = argparse.ArgumentParser(description="LendingClub: прогноз дефолта по займу")
    parser.add_argument("--sample", action="store_true", help="демо-выборка, результаты в sample_run/")
    parser.add_argument("--run-all", action="store_true", help="выполнить весь пайплайн без меню")
    parser.add_argument("--make-sample", action="store_true", help="пересобрать демо-выборку из полного Parquet")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    """Точка входа. Код завершения: 0 — успех, 1 — часть шагов не выполнена, 130 — прервано Ctrl+C."""
    args = parse_args(argv)
    if args.make_sample:
        print(make_sample())
        return 0
    if args.sample:
        C.activate_sample()
    for folder in (C.REPORTS, C.FIGURES, C.MODELS):
        folder.mkdir(parents=True, exist_ok=True)
    if args.run_all:
        try:
            failed = run_pipeline()
        except KeyboardInterrupt:
            print("\nПрервано пользователем: оставшиеся шаги не выполнены.")
            return EXIT_INTERRUPTED
        print(f"\nНе выполнены шаги: {failed}" if failed else "\nВсе шаги выполнены.")
        return 1 if failed else 0
    try:
        menu_loop()
    except (KeyboardInterrupt, EOFError):
        print("\nВыход.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
