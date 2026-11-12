"""Пути, списки колонок и константы проекта."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "datasets" / "accepted_2007_to_2018Q4.csv.gz"
PARQUET = ROOT / "data" / "loans.parquet"
SAMPLE_PARQUET = ROOT / "data" / "sample" / "loans_sample.parquet"
DB_PATH = ROOT / "db" / "lending.db"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
SHIPPED_MODELS = ROOT / "models"  # модели из репозитория: для скоринга сразу после клонирования
STATE_FILE = REPORTS / "state.json"
SQL_DIR = ROOT / "sql"
SAMPLE_RUN = ROOT / "sample_run"
NOTEBOOK_RUN = ROOT / "notebook_run"  # результаты ноутбука на полных данных
SAMPLE_MODE = False

CHUNKSIZE = 250_000
RANDOM_STATE = 42
SAMPLE_MATURED_ROWS = 35_000
SAMPLE_OTHER_ROWS = 15_000
EDA_SAMPLE_ROWS = 200_000
SHAP_SAMPLE_ROWS = 5_000
MEMORY_SAMPLE_ROWS = 200_000


def redirect_outputs(folder: Path) -> None:
    """Все результаты (база, отчёты, графики, модели) пишутся в folder, опубликованные reports/ и models/ не меняются.

    Остальные модули читают пути как C.REPORTS, C.MODELS и т. д. в момент использования,
    поэтому достаточно один раз заменить их здесь (для этого нужен global).
    """
    global DB_PATH, REPORTS, FIGURES, MODELS, STATE_FILE
    DB_PATH = folder / "db" / "lending.db"
    REPORTS = folder / "reports"
    FIGURES = REPORTS / "figures"
    MODELS = folder / "models"
    STATE_FILE = REPORTS / "state.json"


def activate_sample() -> None:
    """Режим демонстрации: данные из выборки, результаты в sample_run/."""
    global PARQUET, SAMPLE_MODE
    SAMPLE_MODE = True
    PARQUET = SAMPLE_PARQUET
    redirect_outputs(SAMPLE_RUN)


# Поля кредитного отчёта заёмщика на момент заявки. LendingClub публикует их для займов с августа 2012 г.,
# у более ранних займов здесь пропуски. Аудит проверяет, что поля не обновлялись после выдачи.
BUREAU_COLS = [
    "acc_open_past_24mths",    # счетов открыто за последние 24 месяца
    "num_tl_op_past_12m",      # счетов открыто за последние 12 месяцев
    "mo_sin_rcnt_tl",          # месяцев с открытия самого нового счёта
    "mo_sin_old_rev_tl_op",    # месяцев с открытия самого старого возобновляемого счёта
    "mo_sin_old_il_acct",      # месяцев с открытия самого старого кредита с рассрочкой
    "mths_since_recent_bc",    # месяцев с открытия последней банковской карты
    "mths_since_recent_inq",   # месяцев с последнего запроса в кредитное бюро
    "mths_since_last_delinq",  # месяцев с последней просрочки (пусто — просрочек не было)
    "mths_since_last_record",  # месяцев с последней публичной записи: суд, банкротство (пусто — не было)
    "total_bc_limit",          # суммарный лимит по банковским картам
    "bc_open_to_buy",          # свободный лимит по банковским картам
    "bc_util",                 # использование лимита по банковским картам, %
    "percent_bc_gt_75",        # доля карт, использованных более чем на 75%
    "total_rev_hi_lim",        # суммарный лимит по возобновляемым счетам
    "avg_cur_bal",             # средний текущий остаток по всем счетам
    "total_bal_ex_mort",       # общий долг без ипотеки
]

# Поля анкеты: известны до выдачи займа -> можно использовать в модели
APP_COLS = [
    "id", "issue_d", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose",
    "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "initial_list_status", "application_type", "mort_acc", "pub_rec_bankruptcies",
] + BUREAU_COLS

# Поля исхода: появляются ПОСЛЕ выдачи. Хранятся для аналитики и оценки портфеля, в модель не идут
OUTCOME_COLS = ["loan_status", "total_pymnt", "recoveries", "last_pymnt_d"]

USECOLS = APP_COLS + OUTCOME_COLS

# Поля-утечки: известны только после выдачи. Тест проверяет, что их нет среди признаков
LEAKAGE_COLS = [
    "loan_status", "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "out_prncp", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag",
    "hardship_flag", "settlement_status", "pymnt_plan", "funded_amnt_inv",
]

CATEGORICAL = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership", "verification_status",
    "purpose", "addr_state", "initial_list_status", "application_type", "loan_status",
]

FLOAT32 = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "total_pymnt", "recoveries",
] + BUREAU_COLS

DATE_COLS = ["issue_d", "earliest_cr_line", "last_pymnt_d"]

# Целевая переменная: 1 = дефолт, 0 = полностью погашен, остальное (Current, Late...) — не завершён
STATUS_TO_TARGET = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Последний месяц выдачи в выгрузке. Займ считается созревшим, если его срок истёк к этой дате
SNAPSHOT_YEAR, SNAPSHOT_MONTH = 2018, 12

# Окно моделирования и временной сплит (по дате выдачи)
MODEL_YEARS = (2007, 2015)
TEST_YEAR = 2015   # train: созревшие займы 2007–2014, test: созревшие займы 2015 (только 36 мес.)
VALID_YEAR = 2014  # порог и калибровка: fit на 2007–2013, подбор на 2014

NUM_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "fico",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "credit_hist_months", "term_months", "emp_years",
    "loan_to_income", "installment_to_income",
] + BUREAU_COLS
# addr_state (штат) в модель не входит: на валидации он не улучшает качество, а в реальном скоринге
# такой признак требует проверки на справедливость. Для аналитики по штатам он остаётся в данных.
CAT_FEATURES = [
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose",
    "initial_list_status", "application_type",
]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "is_default"

# Поля, которые назначает сам LendingClub при андеррайтинге, а не указывает заёмщик.
# installment и installment_to_income зависят от назначенной ставки.
LC_ASSIGNED = {
    "grade", "sub_grade", "int_rate", "installment", "installment_to_income",
    "initial_list_status", "verification_status",
}
BORROWER_NUM = [c for c in NUM_FEATURES if c not in LC_ASSIGNED]
BORROWER_CAT = [c for c in CAT_FEATURES if c not in LC_ASSIGNED]

# Правило выбора модели зафиксировано ДО просмотра результатов: лучшая ROC-AUC на валидации 2014 г.;
# если логистическая регрессия отстаёт меньше чем на этот запас, выбираем её как более простую.
SIMPLICITY_MARGIN_AUC = 0.005

APPROVAL_RATES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
