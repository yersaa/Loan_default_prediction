-- Нормализованная схема: справочники + займы + профиль заёмщика + исход займа
DROP TABLE IF EXISTS outcomes;
DROP TABLE IF EXISTS borrowers;
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS dim_grade;
DROP TABLE IF EXISTS dim_purpose;
DROP TABLE IF EXISTS dim_state;
DROP TABLE IF EXISTS dim_home;

CREATE TABLE dim_grade (
    sub_grade TEXT PRIMARY KEY,
    grade     TEXT NOT NULL
);

CREATE TABLE dim_purpose (
    purpose_id INTEGER PRIMARY KEY,
    purpose    TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_state (
    state_code TEXT PRIMARY KEY,
    region     TEXT NOT NULL
);

CREATE TABLE dim_home (
    home_id        INTEGER PRIMARY KEY,
    home_ownership TEXT NOT NULL UNIQUE
);

-- Условия займа (известны при выдаче)
CREATE TABLE loans (
    loan_id             INTEGER PRIMARY KEY,
    issue_d             TEXT NOT NULL,
    loan_amnt           REAL NOT NULL,
    term_months         INTEGER NOT NULL,
    int_rate            REAL NOT NULL,
    installment         REAL NOT NULL,
    sub_grade           TEXT NOT NULL REFERENCES dim_grade(sub_grade),
    purpose_id          INTEGER NOT NULL REFERENCES dim_purpose(purpose_id),
    state_code          TEXT NOT NULL REFERENCES dim_state(state_code),
    verification_status TEXT,
    application_type    TEXT,
    initial_list_status TEXT
);

-- Профиль заёмщика на момент заявки (1:1 с loans)
CREATE TABLE borrowers (
    loan_id              INTEGER PRIMARY KEY REFERENCES loans(loan_id),
    home_id              INTEGER REFERENCES dim_home(home_id),
    emp_years            REAL,
    annual_inc           REAL,
    dti                  REAL,
    fico                 REAL,
    credit_hist_months   REAL,
    delinq_2yrs          REAL,
    inq_last_6mths       REAL,
    open_acc             REAL,
    pub_rec              REAL,
    revol_bal            REAL,
    revol_util           REAL,
    total_acc            REAL,
    mort_acc             REAL,
    pub_rec_bankruptcies REAL
);

-- Исход займа: известен только ПОСЛЕ выдачи, в модель не идёт
CREATE TABLE outcomes (
    loan_id      INTEGER PRIMARY KEY REFERENCES loans(loan_id),
    loan_status  TEXT NOT NULL,
    is_default   INTEGER,          -- NULL, если займ ещё не завершён
    total_pymnt  REAL,
    recoveries   REAL,
    last_pymnt_d TEXT
);

CREATE INDEX idx_loans_issue ON loans(issue_d);
CREATE INDEX idx_loans_grade ON loans(sub_grade);
CREATE INDEX idx_loans_state ON loans(state_code);
CREATE INDEX idx_outcomes_default ON outcomes(is_default);
