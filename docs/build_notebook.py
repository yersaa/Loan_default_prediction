"""Собирает и выполняет project_walkthrough.ipynb — главный файл проекта для чтения и защиты.

Запуск из корня проекта:  python docs/build_notebook.py          (собрать и выполнить, 25–35 минут на полных данных)
                          python docs/build_notebook.py --no-run (только собрать, без выполнения)

Ноутбук не дублирует конвейер: он вызывает функции из src/ и показывает исходный код ключевых из них.
Графики и анализ результатов написаны прямо в ячейках. У каждой ячейки кода есть ключ metadata["guide"]:
по нему docs/build_guide.py находит ячейку и добавляет к ней подробное объяснение.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "project_walkthrough.ipynb"
cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip()))


def code(key: str, text: str) -> None:
    cell = new_code_cell(text.strip())
    cell.metadata["guide"] = key
    cells.append(cell)


# --------------------------------------------------------------------------------------------- введение
md("""
# LendingClub: прогноз дефолта по займу

**Итоговый проект** по дисциплине *Programming for Data Analysis and Database*, тема №33 «Loan Default Prediction».
Автор: Kilybayev Yersaiyn, AAI-2602M.

**Задача.** По данным, известным *до выдачи* займа, оценить вероятность того, что заёмщик его не вернёт
(PD, probability of default), и проверить, насколько грейды A–G, которые ставит сам LendingClub, отражают реальный риск.

**Данные.** [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
файл `accepted_2007_to_2018Q4.csv.gz`: 2,26 млн займов, 151 колонка, 374 МБ в сжатом виде.

**Как читать ноутбук.** Он идёт по шагам консольного меню `main.py`. Конвейер (чтение, база, обучение) живёт в модулях
папки `src/` — так требует задание (модульность), — а ноутбук вызывает эти функции и показывает исходный код самых
важных из них. Все графики и разбор результатов написаны прямо здесь.

| Раздел | Что внутри |
|---|---|
| 1. Данные | чтение 2,26 млн строк частями, Parquet, экономия памяти |
| 2. Аудит качества | пропуски, некорректные значения, проверка на утечку |
| 3. Цель и разбиение | что считаем дефолтом, почему только «созревшие» займы, разбиение по времени |
| 4. База данных и SQL | нормализованная схема SQLite, 10 запросов |
| 5. Разведочный анализ | 14 графиков: от чего зависит дефолт |
| 6. Модели | 9 моделей, подбор параметров, правило выбора |
| 7. Анализ результатов | ROC и PR, пороги, ошибки, калибровка, важность признаков, таблица одобрения |
| 8. Проверка смещения | насколько незавершённые займы завышают результат |
| 9. Оценка одной заявки | как модель работает на конкретном заёмщике |
| 10. Сравнение способов обработки | CSV и Parquet, Pandas и SQLite: время и память |
| 11. Соответствие требованиям | где выполнен каждый пункт задания |

**Запуск.** С файлом Kaggle в папке `datasets/` ноутбук считает всё на полных данных (около 30 минут).
Без него он сам переключается на демо-выборку из репозитория (50 тыс. займов, 2–3 минуты); числа на выборке
отличаются от полного набора, пояснения в тексте относятся к полному запуску.
Всё, что создаёт ноутбук, пишется в отдельную папку и не меняет опубликованные отчёты в `reports/`.
""")

code("colab", """
# Только для Google Colab: снимите комментарии, чтобы скачать проект и установить библиотеки.
# !git clone https://github.com/yersaa/Loan_default_prediction.git
# %cd Loan_default_prediction
# !pip install -q -r requirements.txt
""")

md("""
## 0. Настройка

Здесь подключаются библиотеки и выбирается режим. Пути к файлам и списки колонок проекта хранятся в `src/config.py`
(модуль импортируется под коротким именем `C`). Функция `show_source` выводит исходный код функции из `src/`,
чтобы читать его, не открывая модули.
""")

code("setup", """
import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Code, Image, display

from src import config as C
from src.data_io import csv_to_parquet, load_dataset, load_json, mb, memory_report
from src.style import AQUA, BLUE, MUTED, ORANGE, RED, VIOLET, YELLOW

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
warnings.filterwarnings("ignore", message="IProgress not found")  # SHAP ищет виджет прогресса, он не нужен

FULL = C.RAW_CSV.exists() or C.PARQUET.exists()  # есть ли полные данные Kaggle
if FULL:
    C.redirect_outputs(C.NOTEBOOK_RUN)  # результаты ноутбука — в notebook_run/
else:
    C.activate_sample()  # демо-выборка, результаты — в sample_run/
for folder in (C.REPORTS, C.FIGURES, C.MODELS):
    folder.mkdir(parents=True, exist_ok=True)


def show_source(function) -> None:
    \"\"\"Показывает исходный код функции из src/ прямо в ноутбуке.\"\"\"
    display(Code(inspect.getsource(function), language="python"))


print("Режим:", "полные данные" if FULL else "демо-выборка")
print("Результаты ноутбука:", C.REPORTS.relative_to(C.ROOT))
""")

# --------------------------------------------------------------------------------------------- 1. данные
md("""
## 1. Данные: чтение частями, Parquet, экономия памяти

Исходный CSV целиком занимает в памяти около 3 ГБ. Поэтому `csv_to_parquet` из `src/data_io.py`:

1. читает файл частями по 250 тыс. строк (`chunksize`) и только нужные колонки (`usecols`);
2. чистит каждую часть: убирает служебные строки-итоги, приводит даты и типы (`clean_chunk`);
3. дописывает части в один файл Parquet и заменяет прежний файл только после успешной записи всех частей.

Parquet — колоночный формат: хранит типы, сжимает данные и позволяет читать только нужные колонки.
""")

code("src_csv_to_parquet", "show_source(csv_to_parquet)")

code("ingest", """
if C.RAW_CSV.exists():
    ingest = csv_to_parquet()
else:
    print("Исходного CSV нет, используется готовый файл:", C.PARQUET.relative_to(C.ROOT))
""")

code("load", """
df = load_dataset()  # Parquet -> DataFrame, сразу с оптимизированными типами
print(f"Займов: {len(df):,}   колонок: {df.shape[1]}   в памяти: {mb(df)} МБ")
df.head(3)
""")

md("""
**Оптимизация типов** (`optimize_dtypes`). Колонки с небольшим числом разных значений (грейд, цель, штат) хранятся как
`category`: каждое значение записано один раз, а в строках — короткие коды. Числа хранятся в `float32` вместо `float64`.
`memory_report` сравнивает три варианта на первых 200 тыс. строках исходного файла и пересчитывает на весь набор.
""")

code("memory", """
if C.RAW_CSV.exists():
    memory = memory_report()["full_dataset_estimate_mb"]
else:
    memory = load_json(C.ROOT / "reports" / "benchmarks.json")["memory"]["full_dataset_estimate_mb"]

labels = ["все 151 колонка", "только нужные\\nколонки", "нужные колонки +\\ncategory/float32"]
values = [memory["all_cols"], memory["selected_cols"], memory["selected_optimized"]]
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(labels, values, color=[MUTED, BLUE, AQUA], width=0.6)
ax.bar_label(bars, fmt="%d МБ", padding=3)
ax.set(title=f"Память для всего набора: в {values[0] / values[2]:.0f} раз меньше", ylabel="МБ")
ax.grid(axis="x", visible=False)
plt.show()
""")

# --------------------------------------------------------------------------------------------- 2. аудит
md("""
## 2. Аудит качества данных

`run_audit` из `src/audit.py` считает пропуски, дубликаты, некорректные значения и выбросы. По его результатам
в `src/features.py` (функция `clean_values`) заданы правила очистки: например, нулевой доход заменяется пропуском.
""")

code("audit", """
from src.audit import run_audit

audit = run_audit(df)
print("Дубликатов id:", audit["duplicate_ids"])
missing = pd.Series(audit["missing_pct"]).head(12).sort_values()

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(missing.index, missing.values, color=BLUE, height=0.7)
ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3, fontsize=9)
ax.set(title="Колонки с наибольшей долей пропусков", xlabel="Пропуски, %")
ax.grid(axis="y", visible=False)
plt.show()
""")

md("""
Больше всего пропусков у полей кредитного отчёта вида «месяцев с последней просрочки» (`mths_since_last_delinq`):
пусто значит, что события не было. Кроме того, LendingClub публикует поля кредитного отчёта только с августа 2012 года.
Пропуски не удаляются: модели заполняют их медианой по обучающим данным или, как бустинг, работают с ними напрямую.
""")

code("invalid", """
pd.Series(audit["invalid_values"], name="строк с нарушением").to_frame()
""")

md("""
**Проверка на утечку из будущего.** Признак можно использовать, только если он известен *в момент заявки*.
Для полей кредитного отчёта `bureau_timing` сравнивает возраст самого старого счёта заёмщика с длиной его кредитной
истории на дату выдачи. Если бы LendingClub обновлял эти поля к дате выгрузки (2018 г.), у старых займов
превышение составляло бы годы. Превышений больше года нет ни одного — значит, поля сняты при заявке.
""")

code("bureau_timing", "audit['bureau_timing']")

# --------------------------------------------------------------------------------------------- 3. цель и разбиение
md("""
## 3. Целевая переменная, «созревшие» займы и разбиение по времени

**Целевая переменная** (`add_target`): `Fully Paid` → 0 (погашен), `Charged Off` / `Default` → 1 (дефолт).
Остальные статусы (`Current`, `Late`) означают, что заём ещё не закончился.

**Почему только «созревшие» займы.** Заём на 60 месяцев, выданный в 2015 году, к концу 2018-го ещё идёт. Среди таких
займов «закончились» только два вида: погашенные досрочно и уже списанные. Если просто убрать статус `Current`,
доля дефолтов среди оставшихся будет завышена. Поэтому в модель идут только займы, срок которых истёк к декабрю 2018:
""")

code("src_is_matured", """
from src.features import is_matured, modeling_frame, temporal_split

show_source(is_matured)
""")

md("""
**Разбиение по времени** (`temporal_split`). Модель учится на прошлом и проверяется на будущем, как в реальном банке.
Случайное перемешивание позволило бы модели «подсмотреть» займы того года, на котором её проверяют.
""")

code("src_temporal_split", "show_source(temporal_split)")

code("windows", """
model_df = modeling_frame(df.copy())  # цель, очистка, новые признаки, только созревшие займы 2007–2015
windows = temporal_split(model_df)

summary = {}
for name, window in windows.items():
    years = sorted(window["issue_d"].dt.year.unique())
    summary[name] = {"годы выдачи": str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}",
                     "займов": len(window), "дефолт, %": round(100 * float(window[C.TARGET].mean()), 2)}
pd.DataFrame(summary).T
""")

md("""
| Окно | Зачем |
|---|---|
| `fit` (2007–2013) | подбор гиперпараметров и «пробная» модель |
| `valid` (2014) | калибровка вероятностей, порог, выбор модели |
| `train` (2007–2014) | обучение финальной модели |
| `test` (2015) | только итоговая оценка, один раз |

**Признаки.** В модель идут только поля, известные до выдачи: анкета, условия займа и 16 полей кредитного отчёта.
Пересечение со списком полей, появляющихся после выдачи (`LEAKAGE_COLS`), должно быть пустым.
""")

code("features", """
print("Признаков:", len(C.FEATURES), "| из них полей кредитного отчёта:", len(C.BUREAU_COLS))
print("Пересечение с полями после выдачи:", set(C.FEATURES) & set(C.LEAKAGE_COLS) or "пусто")
print("Назначает LendingClub, а не заёмщик:", sorted(C.LC_ASSIGNED))
""")

# --------------------------------------------------------------------------------------------- 4. SQL
md("""
## 4. База данных SQLite и SQL-запросы

`build_database` из `src/database.py` раскладывает плоскую таблицу по нормализованной схеме из `sql/schema.sql`:
четыре справочника (грейды, цели, штаты, типы жилья) и три таблицы: `loans` — условия займа, `borrowers` — профиль
заёмщика, `outcomes` — исход. Поля исхода вынесены в отдельную таблицу, чтобы они случайно не попали в признаки.

![ER-диаграмма](docs/er_diagram.png)
""")

code("database", """
from src.database import build_database, load_queries, run_all_queries

build_database(df.copy())
""")

md("""
Запросы лежат в `sql/queries.sql` и выполняются функцией `run_all_queries`; результат каждого — таблица Pandas.
Используются JOIN, GROUP BY и HAVING, CTE, подзапрос и оконные функции `RANK`, `LAG`, `NTILE`, `AVG() OVER`.
Пример — рейтинг штатов по дефолтности в целом и внутри своего региона:
""")

code("query_text", """
queries = load_queries()
print(len(queries), "запросов:", ", ".join(queries))
print()
print(queries["state_rank"])
""")

code("queries", """
sql = run_all_queries()  # словарь {имя запроса: DataFrame}
sql["state_rank"].head(8)
""")

code("sql_grade", """
grade = sql["default_by_grade"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.bar(grade["grade"], grade["default_rate_pct"], color=BLUE, width=0.7)
ax1.bar_label(ax1.containers[0], fmt="%.1f", padding=2, fontsize=9)
ax1.set(title="Дефолтность по грейду (запрос default_by_grade)", xlabel="Грейд", ylabel="Дефолт, %")
ax2.bar(grade["grade"], grade["avg_int_rate"], color=ORANGE, width=0.7)
ax2.bar_label(ax2.containers[0], fmt="%.1f", padding=2, fontsize=9)
ax2.set(title="Средняя ставка по грейду", xlabel="Грейд", ylabel="Ставка, %")
for ax in (ax1, ax2):
    ax.grid(axis="x", visible=False)
plt.show()
""")

md("""
Грейды LendingClub хорошо упорядочивают риск: дефолтность монотонно растёт от A к G, и ставка вместе с ней.
Но внутри одного грейда доход всё ещё различает заёмщиков (запрос `income_vs_grade_avg`): грейд учитывает доход не полностью.
""")

code("sql_income", """
income = sql["income_vs_grade_avg"].pivot(index="grade", columns="income_group", values="default_rate_pct")
ax = income.plot.bar(color=[AQUA, ORANGE], figsize=(8, 3.8), width=0.75, rot=0)
ax.set(title="Дефолтность внутри грейда: доход выше и ниже среднего по подгрейду",
       xlabel="Грейд", ylabel="Дефолт, %")
ax.legend(["доход выше среднего", "доход ниже среднего"])
ax.grid(axis="x", visible=False)
plt.show()
""")

# --------------------------------------------------------------------------------------------- 5. EDA
md("""
## 5. Разведочный анализ (EDA)

Все доли дефолтов ниже посчитаны по созревшим займам (`model_df`), кроме графика цензурирования: он как раз
показывает, что происходит без этого отбора.
""")

code("eda_target", """
counts = model_df[C.TARGET].map({0: "погашен", 1: "дефолт"}).value_counts()
fig, ax = plt.subplots(figsize=(5.5, 3.5))
bars = ax.bar(counts.index, counts.values, color=[AQUA, RED], width=0.55)
ax.bar_label(bars, labels=[f"{v:,} ({v / counts.sum():.1%})" for v in counts.values], padding=3)
ax.set(title="Целевая переменная: классы несбалансированы", ylabel="Займов")
ax.grid(axis="x", visible=False)
plt.show()
""")

md("""
Дефолтов около 15%. Это главная особенность задачи: модель «все вернут» уже даёт 85% верных ответов,
поэтому доля верных ответов (accuracy) плохо подходит для оценки модели (подробнее в разделе 7).
""")

code("eda_volume", """
volume = df.groupby([df["issue_d"].dt.year, "term"], observed=True).size().unstack()
ax = volume.plot.bar(stacked=True, color=[BLUE, ORANGE], figsize=(9, 3.8), width=0.8, rot=0)
ax.set(title="Число выданных займов по годам", xlabel="Год выдачи", ylabel="Займов")
ax.legend(title="Срок")
ax.grid(axis="x", visible=False)
plt.show()
""")

code("eda_censoring", """
vintage = sql["vintage_by_year_term"]
vintage = vintage[vintage["issue_year"] >= 2010]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
for (term, part), color in zip(vintage.groupby("term_months"), [BLUE, ORANGE], strict=False):
    ax1.plot(part["issue_year"], part["resolved_pct"], "o-", color=color, label=f"{term} мес.")
    ax2.plot(part["issue_year"], part["default_rate_pct"], "o-", color=color, label=f"{term} мес.")
ax1.set(title="Доля завершённых займов", xlabel="Год выдачи", ylabel="Завершено, %", ylim=(0, 105))
ax2.set(title="Дефолтность среди завершённых", xlabel="Год выдачи", ylabel="Дефолт, %")
ax1.legend()
ax2.legend()
plt.show()
""")

md("""
**Ловушка цензурирования.** Чем моложе займы, тем меньшая их доля завершена — и тем выше дефолтность среди
завершённых: у 60-месячных займов 2015 года она около 36% против 25% у полностью созревших займов 2013 года.
Это не рост риска, а смещение: досрочно погашенные и списанные заканчиваются раньше остальных. Поэтому — только созревшие займы.
""")

code("eda_categories", """
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, column, title in zip(axes, ["term", "home_ownership", "purpose"], ["Срок", "Жильё", "Цель займа"], strict=True):
    rate = model_df.groupby(column, observed=True)[C.TARGET].agg(["mean", "size"])
    rate = rate[rate["size"] >= 2000]["mean"].mul(100).sort_values()
    ax.barh(rate.index.astype(str), rate.values, color=VIOLET, height=0.7)
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=2, fontsize=8)
    ax.set(title=f"Дефолт, %: {title}")
    ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
""")

md("""
Займы на 60 месяцев дефолтят почти вдвое чаще, чем на 36. Арендаторы рискованнее владельцев жилья с ипотекой.
Самая рискованная цель — малый бизнес, наименее рискованные — покупка автомобиля и крупные покупки.
""")

code("eda_distributions", """
sample = model_df.sample(min(200_000, len(model_df)), random_state=C.RANDOM_STATE)
sample = sample.assign(log_income=np.log10(sample["annual_inc"].clip(lower=1_000)))
panels = [("fico", "FICO при заявке"), ("log_income", "Годовой доход, log10($)"),
          ("int_rate", "Ставка, %"), ("dti", "Долговая нагрузка DTI, %")]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for ax, (column, title) in zip(axes.ravel(), panels, strict=True):
    for target, color, label in [(0, AQUA, "погашен"), (1, RED, "дефолт")]:
        values = sample.loc[sample[C.TARGET] == target, column].dropna()
        ax.hist(values, bins=50, density=True, alpha=0.55, color=color, label=label)
    ax.set(title=title, yticks=[])
    ax.legend()
plt.tight_layout()
plt.show()
""")

md("""
Распределения погашенных и дефолтных займов сильно пересекаются: по одному признаку их не разделить.
Дефолтные займы в среднем имеют ниже FICO, ниже доход, выше ставку и выше долговую нагрузку. Сильнее всего
различается ставка — но её назначает сам LendingClub по своей оценке риска.
""")

code("eda_fico_deciles", """
deciles = pd.qcut(model_df["fico"], 10, duplicates="drop")
rate = model_df.groupby(deciles, observed=True)[C.TARGET].mean().mul(100)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar([f"{interval.left:.0f}–{interval.right:.0f}" for interval in rate.index], rate.values, color=BLUE, width=0.7)
ax.bar_label(ax.containers[0], fmt="%.1f", padding=2, fontsize=9)
ax.set(title="Дефолтность по десяти группам FICO", xlabel="FICO", ylabel="Дефолт, %")
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="x", visible=False)
plt.show()
""")

code("eda_grade_term", """
table = model_df.pivot_table(index="grade", columns="term_months", values=C.TARGET, aggfunc="mean", observed=True)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
sns.heatmap(table.mul(100), annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "Дефолт, %"}, ax=ax)
ax.set(title="Дефолтность: грейд × срок", xlabel="Срок, мес.", ylabel="Грейд")
plt.show()
""")

md("""
При одном и том же грейде заём на 60 месяцев рискованнее, чем на 36. Значит, срок несёт информацию сверх грейда,
и модели полезно видеть оба признака.
""")

code("eda_bureau", """
bureau = model_df.dropna(subset=["acc_open_past_24mths"])
opened = bureau["acc_open_past_24mths"].clip(upper=15)
rate = bureau.groupby(opened)[C.TARGET].agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(rate.index, rate["mean"] * 100, "o-", color=RED)
ax.set(title="Чем больше новых счетов за 2 года, тем выше риск (поле кредитного отчёта)",
       xlabel="Счетов открыто за 24 месяца (15 = 15 и больше)", ylabel="Дефолт, %")
plt.show()
""")

code("eda_correlations", """
columns = ["int_rate", "fico", "dti", "annual_inc", "loan_amnt", "term_months", "revol_util",
           "inq_last_6mths", "acc_open_past_24mths", "loan_to_income", C.TARGET]
corr = sample[columns].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8})
ax.set_title("Корреляции Спирмена между признаками и дефолтом")
plt.show()
""")

md("""
Связь каждого признака с дефолтом по отдельности слабая: сильнее всех ставка (0,20), затем FICO (−0,11). Значит, нужна модель, которая
сочетает много признаков. Ставка и FICO заметно связаны между собой (−0,4…−0,5): ставку назначают в том числе по FICO.
""")

# --------------------------------------------------------------------------------------------- 6. модели
md("""
## 6. Модели: подбор, обучение, выбор

`model_specs` в `src/models.py` описывает девять моделей:

| Модель | Роль |
|---|---|
| Dummy | baseline: всегда предсказывает среднюю долю дефолтов |
| Только подгрейд LC | baseline: логистическая регрессия на одном подгрейде — оценка самого LendingClub |
| Logistic Regression | линейная модель, принятая в банковском скоринге |
| LightGBM | градиентный бустинг деревьев решений |
| XGBoost | другая реализация бустинга (претендент) |
| Нейросеть (MLP) | два скрытых слоя, 64 и 32 нейрона (претендент) |
| Ансамбль | среднее вероятностей LightGBM, XGBoost и нейросети (претендент) |
| LogReg и LightGBM на полях заёмщика | без грейда и ставки, которые назначает LendingClub |

Сначала `tune` подбирает гиперпараметры на окне `fit` перебором по сетке. Кросс-валидация тоже идёт по времени
(`TimeSeriesSplit`): каждая проверочная часть позже обучающей.
""")

code("tune", """
from src.data_io import save_json
from src.models import train_and_compare, tune

fit = windows["fit"]
best = tune(fit[C.FEATURES], fit[C.TARGET].astype(int))
save_json(best, C.REPORTS / "tuning.json")  # пригодится для замера памяти моделей в разделе 10
""")

md("""
**Как обучается каждая модель** (`fit_one`). Шаг 1: «пробная» модель учится на 2007–2013 и делает прогноз на 2014;
по нему подбираются калибровка вероятностей (метод Платта) и порог. Шаг 2: финальная модель учится на 2007–2014
и оценивается на 2015. Тест не участвует ни в одном решении.
""")

code("train", """
metrics, fitted, preds, preds_raw, test = train_and_compare(model_df, best)
y_test = test[C.TARGET].astype(int)
metrics[["model", "valid_auc", "roc_auc", "pr_auc", "ece"]]
""")

md("""
### Какие метрики и зачем

Модель выдаёт **вероятность дефолта** (PD), а банку нужно решение «выдать / отказать». Поэтому метрик две группы.

**Качество вероятностей** — от порога не зависит:

| Метрика | Вопрос | Как читать |
|---|---|---|
| **ROC-AUC** | Насколько хорошо модель упорядочивает заёмщиков от надёжных к рискованным? | вероятность, что у случайного дефолтного займа PD выше, чем у случайного погашенного. 0,5 — угадывание, 1 — идеал. **По ней выбирается модель.** |
| **PR-AUC** | Насколько хорошо модель находит редкий класс — дефолты? | у угадывания равна доле дефолтов (около 0,15) |
| **ECE** | Можно ли верить самой цифре PD? | средний разрыв между прогнозом и фактом; 0,01 = 1 процентный пункт |

**Качество решений при пороге** — заём с PD не ниже порога получает отказ:

| Метрика | Вопрос |
|---|---|
| **Recall** | Какую долю дефолтов модель поймала? (цена пропуска — потерянные деньги) |
| **Precision** | Какая доля отказов пришлась на настоящие дефолты? (цена ложного отказа — упущенный клиент) |
| **F1** | Баланс precision и recall; по его максимуму на 2014 годе выбирается порог |
| **Accuracy** | Доля верных решений — показана, чтобы увидеть, почему она здесь обманчива |

В банковском скоринге часто называют **Gini** — это та же ROC-AUC в другой шкале: Gini = 2 · AUC − 1.
""")

code("threshold_metrics", """
decisions = metrics[["model", "threshold", "accuracy", "precision", "recall", "f1"]].copy()
all_repay = metrics["accuracy_all_repay"].iloc[0]
decisions.loc[len(decisions)] = ["Константа «все вернут»", None, all_repay, None, 0.0, 0.0]
decisions
""")

md("""
**Почему не accuracy.** Константа «все вернут» получает самую высокую accuracy (около 0,85), не найдя *ни одного*
дефолта. Модель, которая ищет дефолты, часть надёжных заёмщиков отправляет в отказ, и её accuracy ниже. Accuracy
наказывает модель за поиск дефолтов, поэтому модели сравниваются по ROC-AUC. (Dummy при пороге 0,05 отказывает всем.)

**Правило выбора модели** задано заранее (`config.SIMPLICITY_MARGIN_AUC = 0.005`) и применяется к валидации 2014 года:
берётся лучшая ROC-AUC, но логистическая регрессия предпочитается, если отстаёт меньше чем на 0,005; претендент
(XGBoost, нейросеть, ансамбль) заменяет выбранную модель, только если лучше её на 0,005 или больше.
""")

code("src_select_model", """
from src.models import select_model

show_source(select_model)
""")

code("selection", """
selection = load_json(C.REPORTS / "model_selection.json")
chosen = selection["selected"]
print("Выбрана модель:", chosen)
selection
""")

code("auc_bars", """
auc = metrics.set_index("model")["roc_auc"].drop("Dummy (prior)").sort_values()
colors = [RED if name == chosen else BLUE for name in auc.index]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(auc.index, auc.values, color=colors, height=0.65)
ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)
ax.axvline(auc["Grade only (LogReg)"], color=MUTED, ls="--", lw=1)
ax.set(title="ROC-AUC на тесте 2015 г. (красным — выбранная модель)", xlim=(0.6, 0.72), xlabel="ROC-AUC")
ax.grid(axis="y", visible=False)
plt.show()
""")

md("""
Все модели лучше baseline «только подгрейд» (пунктир). Сильные модели сошлись на 0,70: LightGBM, XGBoost и
ансамбль различаются в третьем-четвёртом знаке. Это потолок качества на этих данных без утечки, а не слабость алгоритмов.
""")

# --------------------------------------------------------------------------------------------- 7. анализ результатов
md("""
## 7. Анализ результатов

`run_explain` из `src/explain.py` строит и сохраняет итоговые таблицы: ошибки по грейдам, таблицу одобрения,
важность признаков SHAP. Ниже — графики по этим результатам.
""")

code("explain", """
from src.explain import run_explain

explained = run_explain(fitted, preds, preds_raw, test, metrics)
main_models = ["Grade only (LogReg)", "Logistic Regression", "LightGBM", "XGBoost", "Neural network (MLP)"]
colors = dict(zip(main_models, [MUTED, BLUE, RED, ORANGE, VIOLET], strict=True))
""")

code("roc_pr", """
from sklearn.metrics import precision_recall_curve, roc_curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
for name in main_models:
    fpr, tpr, _ = roc_curve(y_test, preds[name])
    precision, recall, _ = precision_recall_curve(y_test, preds[name])
    label = f"{name} ({metrics.set_index('model').loc[name, 'roc_auc']:.3f})"
    ax1.plot(fpr, tpr, color=colors[name], label=label)
    keep = recall >= 0.01  # при крошечном recall precision считается по единицам займов
    ax2.plot(recall[keep], precision[keep], color=colors[name], label=name)
ax1.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="угадывание (0.500)")
ax2.axhline(y_test.mean(), ls="--", color=MUTED, lw=1, label="доля дефолтов")
ax1.set(title="ROC-кривая", xlabel="Доля ложных отказов (FPR)", ylabel="Доля пойманных дефолтов (TPR)")
ax2.set(title="Precision–Recall", xlabel="Recall", ylabel="Precision", ylim=(0, None))
ax1.legend(fontsize=8)
ax2.legend(fontsize=8)
plt.show()
""")

md("""
**ROC-кривая** показывает все пороги сразу: чем выше и левее кривая, тем больше дефолтов модель ловит при той же
доле ложных отказов. Площадь под ней и есть ROC-AUC. **PR-кривая** показывает, насколько чисты отказы: среди самых
рискованных заявок дефолтов больше половины, но чем больше дефолтов мы хотим поймать, тем ниже точность.
""")

code("pd_distribution", """
pd_best = preds[chosen]
fig, ax = plt.subplots(figsize=(9, 3.8))
bins = np.linspace(0, 0.6, 61)
ax.hist(pd_best[y_test == 0], bins=bins, density=True, alpha=0.55, color=AQUA, label="погашен")
ax.hist(pd_best[y_test == 1], bins=bins, density=True, alpha=0.55, color=RED, label="дефолт")
threshold = metrics.set_index("model").loc[chosen, "threshold"]
ax.axvline(threshold, color="black", lw=1.2, label=f"порог {threshold:.2f}")
ax.set(title=f"Распределение PD ({chosen}): дефолтным займам модель в среднем даёт больший риск",
       xlabel="Предсказанная вероятность дефолта", yticks=[])
ax.legend()
plt.show()
""")

md("""
Распределения сдвинуты, но пересекаются. Поэтому любой порог — компромисс: правее порога окажутся и дефолты,
и часть надёжных заёмщиков. Это и видно в следующих графиках.
""")

code("threshold_curve", """
grid = np.linspace(0.05, 0.45, 41)
rows = []
for t in grid:
    refuse = pd_best >= t
    caught = (refuse & (y_test == 1)).sum()
    rows.append({"порог": t, "recall": caught / (y_test == 1).sum(),
                 "precision": caught / max(refuse.sum(), 1), "доля отказов": refuse.mean()})
curve = pd.DataFrame(rows).set_index("порог")
ax = curve.plot(figsize=(9, 3.8), color=[RED, BLUE, MUTED])
ax.axvline(threshold, color="black", lw=1, ls="--")
ax.set(title="Как порог меняет решения (тест 2015 г.)", xlabel="Порог PD", ylabel="Доля")
plt.show()
""")

md("""
Порог — решение кредитора, а не модели. Низкий порог ловит больше дефолтов, но отказывает многим надёжным
заёмщикам; высокий — наоборот. Пунктир — порог по максимуму F1, подобранный на 2014 годе.
""")

code("confusion", """
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(5, 4.3))
ConfusionMatrixDisplay.from_predictions(y_test, pd_best >= threshold, display_labels=["погашен", "дефолт"],
                                        values_format=",", cmap="Blues", ax=ax, colorbar=False)
ax.set(title=f"Матрица ошибок, порог {threshold:.2f}", xlabel="Решение модели", ylabel="Факт")
ax.set_xticklabels(["одобрить", "отказать"])
ax.grid(False)
plt.show()
""")

code("calibration", """
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(6, 5))
for name, color in [(chosen, RED), ("Logistic Regression", BLUE)]:
    fact, forecast = calibration_curve(y_test, preds[name], n_bins=10, strategy="quantile")
    fact_raw, forecast_raw = calibration_curve(y_test, preds_raw[name], n_bins=10, strategy="quantile")
    ax.plot(forecast, fact, "o-", color=color, label=f"{name}, после калибровки")
    ax.plot(forecast_raw, fact_raw, "o--", color=color, alpha=0.4, label=f"{name}, до калибровки")
ax.plot([0, 0.4], [0, 0.4], color=MUTED, lw=1)
ax.set(title="Калибровка: прогноз PD и фактическая доля дефолтов",
       xlabel="Средняя предсказанная PD в группе", ylabel="Фактическая доля дефолтов")
ax.legend(fontsize=8)
plt.show()
""")

md("""
Займы разбиты на 10 групп по предсказанной PD. Если точки лежат на диагонали, цифре PD можно верить. До калибровки
модель немного недооценивает риск; калибровка Платта, подобранная на 2014 годе, сдвигает точки к диагонали и не
меняет порядок заёмщиков, поэтому ROC-AUC остаётся прежней.
""")

code("lift", """
deciles = pd.qcut(pd.Series(pd_best).rank(method="first"), 10, labels=range(1, 11))
lift = pd.DataFrame({"y": y_test.to_numpy(), "decile": deciles.to_numpy()}).groupby("decile", observed=True)["y"].mean()
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(lift.index.astype(str), lift.values * 100, color=[AQUA] * 5 + [ORANGE] * 3 + [RED] * 2, width=0.7)
ax.bar_label(ax.containers[0], fmt="%.1f", padding=2, fontsize=9)
ax.axhline(y_test.mean() * 100, color=MUTED, ls="--", lw=1)
ax.set(title="Фактическая дефолтность по десяти группам риска (1 — самые надёжные)",
       xlabel="Группа по предсказанной PD", ylabel="Дефолт, %")
ax.grid(axis="x", visible=False)
plt.show()
""")

md("""
Модель хорошо разделяет крайние группы: в 10% самых надёжных займов дефолтов в несколько раз меньше, чем в среднем
(пунктир), а в 10% самых рискованных — в два с лишним раза больше.
""")

code("shap", """
display(Image(filename=str(C.FIGURES / "12_shap_summary.png"), width=650))
""")

md("""
**SHAP** показывает вклад каждого признака в прогноз каждого займа. Точка — заём; правее нуля признак повышает риск,
левее — снижает; цвет — значение признака. Главные признаки: подгрейд, число недавно открытых счетов
(`acc_open_past_24mths`), доход, отношение суммы займа к доходу, ставка, DTI и FICO.
""")

code("importance", """
def gain_share(booster_gain: dict) -> pd.Series:
    \"\"\"Важность признаков как доля суммарного прироста качества (gain) по всем деревьям, %.\"\"\"
    gain = pd.Series(booster_gain, dtype=float)
    return gain / gain.sum() * 100


lgbm = fitted["LightGBM"].named_steps["model"].booster_
xgb = fitted["XGBoost"].named_steps["model"].get_booster()
importance = pd.DataFrame({
    "LightGBM": gain_share(dict(zip(lgbm.feature_name(), lgbm.feature_importance("gain"), strict=True))),
    "XGBoost": gain_share(xgb.get_score(importance_type="total_gain")),
}).fillna(0)
top = importance.loc[importance.mean(axis=1).nlargest(15).index[::-1]]
ax = top.plot.barh(color=[RED, ORANGE], figsize=(9, 6), width=0.8)
ax.set(title="Важность признаков (доля gain): два бустинга видят одно и то же", xlabel="Доля, %")
ax.grid(axis="y", visible=False)
plt.show()
""")

code("grade_recovery", """
recovery = explained["recovery"].set_index("grade")
ax = recovery[["actual_default_pct", "borrower_pd_pct", "full_pd_pct"]].plot.bar(
    color=[MUTED, YELLOW, RED], figsize=(9, 3.8), width=0.8, rot=0)
ax.set(title="Факт и средняя PD по грейдам", xlabel="Грейд", ylabel="%")
ax.legend(["фактический дефолт", "PD без грейда и ставки", "PD на всех полях"])
ax.grid(axis="x", visible=False)
plt.show()
""")

md("""
**Что добавляет грейд LendingClub.** Модель только на данных заёмщика (без грейда и ставки) по ROC-AUC уже сильнее
одного подгрейда: поля кредитного отчёта восстанавливают большую часть того, что знает LendingClub. Но без грейда
модель «сжимает» риск: завышает PD надёжным заёмщикам грейда A и занижает рискованным F и G.
""")

code("approval", """
approval = explained["approval"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for name, part in approval.groupby("model", sort=False):
    color = {"Grade only (LogReg)": MUTED, "Logistic Regression": BLUE, "LightGBM": RED}.get(name, YELLOW)
    ax1.plot(part["approval_rate"] * 100, part["default_rate_pct"], "o-", color=color, label=name)
    ax2.plot(part["approval_rate"] * 100, part["net_return_pct"], "o-", color=color, label=name)
ax1.set(title="Дефолтность одобренного портфеля", xlabel="Одобряем, % заявок", ylabel="Дефолт, %")
ax2.set(title="Номинальный возврат сверх выданного", xlabel="Одобряем, % заявок", ylabel="%")
ax1.legend(fontsize=8)
plt.show()
approval.pivot(index="approval_rate", columns="model", values="default_rate_pct").round(1)
""")

md("""
**Таблица одобрения** — как модель используется на практике. Одобряем заданную долю заявок с наименьшей PD и смотрим,
какой будет дефолтность портфеля. Без модели (100%) она около 15%; при одобрении 80% лучших заявок — около 11%.
Доход здесь номинальный: за весь срок, без стоимости денег и дисконтирования, поэтому это сравнение моделей, а не прогноз прибыли.
""")

# --------------------------------------------------------------------------------------------- 8. цензурирование
md("""
## 8. Проверка: насколько незавершённые займы завышают результат

`censoring_sensitivity` обучает те же модели на «всех завершённых» займах, включая незрелые, и сравнивает с основным
вариантом. Общая AUC на смещённой когорте выше, но внутри займов на 36 месяцев (`auc_36m`) она почти та же:
завышение создаёт смешение сроков, а не качество модели.
""")

code("censoring", """
from src.censoring import censoring_sensitivity

censoring = censoring_sensitivity(df, best)
fig, ax = plt.subplots(figsize=(8, 3.6))
pivot = censoring.pivot(index="model", columns="cohort", values="test_auc")
pivot.plot.barh(ax=ax, color=[RED, AQUA], width=0.7)
ax.set(title="ROC-AUC на тесте: смещённая когорта выглядит лучше", xlabel="ROC-AUC", xlim=(0.65, 0.75))
ax.grid(axis="y", visible=False)
plt.show()
censoring
""")

# --------------------------------------------------------------------------------------------- 9. одна заявка
md("""
## 9. Оценка одной заявки

`predict_pd` проверяет анкету (`validate_application`), собирает строку признаков тем же кодом, что при обучении,
и возвращает калиброванную PD и решение каждой модели. Поля, которых нет в анкете (например, данные кредитного
отчёта), заполняются типичными значениями из обучающих данных.
""")

code("predict", """
from src.models import predict_pd

application = {
    "loan_amnt": 15000, "term_months": 36, "int_rate": 13.0, "sub_grade": "C1", "annual_inc": 65000, "dti": 18,
    "fico": 700, "emp_years": 5, "revol_util": 50, "inq_last_6mths": 1, "credit_hist_months": 180,
    "home_ownership": "RENT", "purpose": "debt_consolidation", "verification_status": "Source Verified",
    "initial_list_status": "w", "application_type": "Individual",
}
pd.DataFrame(predict_pd(application)).T
""")

md("Некорректная анкета отклоняется с понятным сообщением — обработка ошибок через `try` / `except`:")

code("predict_error", """
try:
    predict_pd({**application, "fico": 300})
except ValueError as error:
    print("Ошибка:", error)
""")

# --------------------------------------------------------------------------------------------- 10. бенчмарки
md("""
## 10. Сравнение способов обработки данных

`src/benchmark.py`: каждая задача выполняется в **отдельном новом процессе** три раза, в отчёт идёт медиана — так
замеры не влияют друг на друга. Память — прирост пикового объёма памяти процесса. Без исходного CSV показывается
результат полного запуска из репозитория.
""")

code("benchmarks", """
if C.RAW_CSV.exists():
    from src.benchmark import run_benchmarks
    bench = run_benchmarks()
else:
    bench = load_json(C.ROOT / "reports" / "benchmarks.json")
print("Результаты Pandas и SQL совпали:", bench["pandas_vs_sql"]["results_match"])
""")

code("benchmark_plots", """
tasks = {"CSV.gz: чтение\\n6 колонок": bench["csv_vs_parquet"]["csv_gz"],
         "Parquet: чтение\\n6 колонок": bench["csv_vs_parquet"]["parquet"],
         "Pandas:\\nагрегация": bench["pandas_vs_sql"]["pandas_parquet"],
         "SQLite:\\nагрегация": bench["pandas_vs_sql"]["sqlite"]}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
colors_bench = [MUTED, AQUA, BLUE, ORANGE]
ax1.bar(tasks.keys(), [t["seconds"] for t in tasks.values()], color=colors_bench, width=0.6)
ax1.bar_label(ax1.containers[0], fmt="%.2f с", padding=2, fontsize=9)
ax2.bar(tasks.keys(), [t["peak_mb"] for t in tasks.values()], color=colors_bench, width=0.6)
ax2.bar_label(ax2.containers[0], fmt="%.0f МБ", padding=2, fontsize=9)
ax1.set(title="Время, с (медиана 3 запусков)")
ax2.set(title="Пиковая память, МБ")
for ax in (ax1, ax2):
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
plt.show()
""")

md("""
Parquet читает нужные колонки в десятки раз быстрее CSV. При агрегации Pandas быстрее SQLite, зато SQLite почти
не занимает память приложения: данные остаются на диске. Отсюда выбор: Parquet — рабочий формат, Pandas — для анализа
и обучения, SQLite — для нормализованного хранения и запросов, когда памяти мало.
""")

code("model_cost", """
cost = metrics[["model", "fit_seconds", "predict_ms_per_1000", "model_mb"]].copy()
cost.insert(2, "fit_peak_mb", cost["model"].map(lambda name: bench["model_fit_memory"][name]["fit_peak_mb"]))
cost.round(2)
""")

md("""
Цена качества: ансамбль даёт лишние 0,0003 AUC, но обучается в несколько раз дольше и медленнее выдаёт прогноз.
LightGBM при том же качестве обучается быстро и требует меньше памяти, чем логистическая регрессия и нейросеть.
""")

# --------------------------------------------------------------------------------------------- 11. требования
md("""
## 11. Соответствие требованиям задания

| № | Требование | Где выполнено |
|---|---|---|
| 1 | Не менее 5 функций | более 120 функций в `src/`, `main.py`, `app.py`; ключевые показаны в разделах 1, 3, 6 |
| 2 | Списки и словари | `MENU`, `STEPS` в `main.py`; `sql` и `summary` в этом ноутбуке; реестр моделей |
| 3 | `if` / `elif` / `else` | `menu_loop` в `main.py`, выбор режима в разделе 0 |
| 4 | Циклы | меню и повтор ввода в `main.py`; циклы по моделям и графикам здесь |
| 5 | Работа с файлами | CSV, Parquet, JSON, SQLite, joblib (разделы 1, 4, 6) |
| 6 | `try` / `except` | раздел 9; `run_step`, `load_json`, `csv_to_parquet` |
| 7 | Интерфейс | консольное меню `python main.py`, дашборд `streamlit run app.py` |
| 8 | Модульность | 12 модулей в `src/`; `pathlib`, `json`, `datetime`, `sqlite3`, `argparse` |
| 9 | Сравнение подходов с baseline | 9 моделей против Dummy и «только подгрейд» по качеству, времени и памяти (разделы 6, 10); CSV/Parquet и Pandas/SQL (раздел 10) |
| 10 | SQLite, ER-диаграмма, 8+ запросов | раздел 4: 7 таблиц, 10 запросов с JOIN, HAVING, CTE, подзапросом и оконными функциями |
| 11 | Объём данных | 2,26 млн строк; чтение частями, Parquet, оптимизация типов с замером (раздел 1) |
| Тема 33 | Chunked loading + Parquet; признаки только на момент заявки; разбиение по дате выдачи | разделы 1, 2–3, 3 |

## Итог
""")

code("summary", """
best_row = metrics.set_index("model").loc[chosen]
print(f"Выбрана модель: {chosen}")
print(f"  ROC-AUC на тесте {best_row['roc_auc']:.3f} (baseline «только подгрейд» "
      f"{metrics.set_index('model').loc['Grade only (LogReg)', 'roc_auc']:.3f}, угадывание 0.500)")
print(f"  ловит {best_row['recall']:.0%} дефолтов при пороге {best_row['threshold']:.2f}")
print(f"  средняя PD {100 * best_row['mean_pd']:.1f}% "
      f"при фактической дефолтности {100 * best_row['actual_default_rate']:.1f}%")
""")

md("""
- Parquet и оптимизация типов позволяют работать с 2,26 млн займов на обычном компьютере.
- Незавершённые займы завышают оценку качества, поэтому в модель идут только созревшие займы.
- Модель выбирается по правилу, заданному заранее, на валидации; тест используется один раз.
- Данные заёмщика с полями кредитного отчёта сильнее одного грейда LendingClub, а грейд и ставка добавляют к ним немного.
- XGBoost, нейросеть и ансамбль не превзошли LightGBM: качество ограничено данными, а не алгоритмом.

Подробное объяснение каждого шага и кода — в `docs/guide.docx`; официальный отчёт — `docs/report.pdf`.
Интерактивная аналитика и оценка заявки — `streamlit run app.py`.
""")


def build(run: bool = True) -> Path:
    notebook = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    if run:
        NotebookClient(notebook, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, OUTPUT)
    print(f"Готово: {OUTPUT} ({OUTPUT.stat().st_size / 2**20:.2f} МБ)")
    return OUTPUT


if __name__ == "__main__":
    build(run="--no-run" not in sys.argv)
