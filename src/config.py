"""Пути, списки колонок и константы проекта."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "datasets" / "accepted_2007_to_2018Q4.csv.gz"
PARQUET = ROOT / "data" / "loans.parquet"
DB_PATH = ROOT / "db" / "lending.db"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
STATE_FILE = REPORTS / "state.json"
SQL_DIR = ROOT / "sql"

CHUNKSIZE = 250_000
RANDOM_STATE = 42

# Поля анкеты: известны на момент подачи заявки -> можно использовать в модели
APP_COLS = [
    "id", "issue_d", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose",
    "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "initial_list_status", "application_type", "mort_acc", "pub_rec_bankruptcies",
]

# Поля исхода: появляются ПОСЛЕ выдачи. Хранятся в БД для анализа, в модель не идут (утечка)
OUTCOME_COLS = ["loan_status", "total_pymnt", "recoveries", "last_pymnt_d"]

USECOLS = APP_COLS + OUTCOME_COLS

# Примеры полей-утечек, сознательно исключённых из модели
LEAKAGE_COLS = [
    "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "out_prncp", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag", "hardship_flag",
    "settlement_status", "pymnt_plan", "funded_amnt_inv",
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
]

DATE_COLS = ["issue_d", "earliest_cr_line", "last_pymnt_d"]

# Целевая переменная: 1 = дефолт, 0 = полностью погашен, остальное (Current, Late...) — не завершён
STATUS_TO_TARGET = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Окно моделирования и временной сплит (по дате выдачи)
MODEL_YEARS = (2007, 2015)  # займы после 2015 ещё в основном не закрыты -> смещение
TEST_YEAR = 2015            # train: 2007–2014, test: 2015
VALID_YEAR = 2014           # для выбора порога: fit на 2007–2013, порог на 2014

NUM_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "fico",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "credit_hist_months", "term_months", "emp_years",
    "loan_to_income", "installment_to_income",
]
CAT_FEATURES = [
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status", "application_type",
]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "is_default"
