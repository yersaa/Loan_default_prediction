"""Модели PD: baseline (Dummy, только грейд) vs Logistic Regression vs LightGBM.
Подбор гиперпараметров — TimeSeriesSplit на train, оценка — на займах следующего года."""
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.data_io import load_json, save_json


class Winsorizer(BaseEstimator, TransformerMixin):
    """Обрезает значения по перцентилям, посчитанным только на train."""

    def __init__(self, low=0.01, high=0.99):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        self.bounds_ = np.nanquantile(np.asarray(X, dtype=float), [self.low, self.high], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.bounds_[0], self.bounds_[1])


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Фиксирует категории train и приводит к ним новые данные (для нативных категорий LightGBM)."""

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in C.CAT_FEATURES}
        return self

    def transform(self, X):
        X = X[C.FEATURES].copy()
        for col, cats in self.categories_.items():
            X[col] = pd.Categorical(X[col].astype("object"), categories=cats)
        return X


def logreg_pipeline(C_reg=1.0) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("clip", Winsorizer()),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])
    prep = ColumnTransformer([("num", numeric, C.NUM_FEATURES), ("cat", categorical, C.CAT_FEATURES)])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=C_reg, max_iter=500))])


def grade_only_pipeline() -> Pipeline:
    """Baseline: риск по одному подгрейду, присвоенному самим LendingClub."""
    prep = ColumnTransformer([("grade", OneHotEncoder(handle_unknown="ignore"), ["sub_grade"])])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=500))])


def lgbm_pipeline(**params) -> Pipeline:
    model = LGBMClassifier(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           min_child_samples=200, random_state=C.RANDOM_STATE, verbose=-1, **params)
    return Pipeline([("cast", CategoryCaster()), ("model", model)])


def tune(X, y) -> dict:
    """Подбор гиперпараметров кросс-валидацией по времени (без перемешивания)."""
    cv = TimeSeriesSplit(n_splits=3)
    searches = {
        "Logistic Regression": (logreg_pipeline(), {"model__C": [0.05, 1.0]}),
        "LightGBM": (lgbm_pipeline(), {"model__num_leaves": [31, 63], "model__n_estimators": [300, 600]}),
    }
    best = {}
    for name, (pipe, grid) in searches.items():
        start = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(X, y)
        best[name] = {"params": gs.best_params_, "cv_auc": round(gs.best_score_, 4),
                      "seconds": round(time.time() - start, 1),
                      "grid": pd.DataFrame(gs.cv_results_)[["params", "mean_test_score"]]
                      .assign(params=lambda d: d["params"].astype(str)).to_dict("records")}
        print(f"  {name}: {gs.best_params_}  CV ROC-AUC={gs.best_score_:.4f}")
    return best


def make_models(best: dict) -> dict:
    lr_c = best["Logistic Regression"]["params"]["model__C"]
    lgb = {k.replace("model__", ""): v for k, v in best["LightGBM"]["params"].items()}
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Grade only (LogReg)": grade_only_pipeline(),
        "Logistic Regression": logreg_pipeline(lr_c),
        "LightGBM": lgbm_pipeline(**lgb),
    }


def ks_stat(y, p) -> float:
    """Статистика Колмогорова–Смирнова: max(TPR - FPR) — стандартная метрика скоринга."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.6, 56)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def evaluate(y, p, threshold) -> dict:
    pred = p >= threshold
    auc = roc_auc_score(y, p)
    return {"roc_auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y, p),
            "ks": ks_stat(y, p), "brier": brier_score_loss(y, p), "threshold": threshold,
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def train_and_compare(model_df: pd.DataFrame, best: dict) -> tuple:
    """1) fit на 2007–2013 -> порог по F1 на 2014; 2) fit на 2007–2014 -> метрики на 2015."""
    year = model_df["issue_d"].dt.year
    fit, valid = model_df[year < C.VALID_YEAR], model_df[year == C.VALID_YEAR]
    train, test = model_df[year < C.TEST_YEAR], model_df[year == C.TEST_YEAR]
    C.MODELS.mkdir(exist_ok=True)
    rows, fitted, preds = [], {}, {}
    for name, model in make_models(best).items():
        model.fit(fit[C.FEATURES], fit[C.TARGET])
        thr = best_f1_threshold(valid[C.TARGET], model.predict_proba(valid[C.FEATURES])[:, 1])

        start = time.time()
        model.fit(train[C.FEATURES], train[C.TARGET])
        fit_s = time.time() - start
        start = time.time()
        p = model.predict_proba(test[C.FEATURES])[:, 1]
        pred_s = time.time() - start

        path = C.MODELS / f"{name.split()[0].lower()}.joblib"
        joblib.dump(model, path)
        row = {"model": name, **evaluate(test[C.TARGET], p, thr), "fit_seconds": fit_s,
               "predict_ms_per_1000": 1000 * pred_s / len(test) * 1000,
               "model_mb": path.stat().st_size / 2**20}
        rows.append(row)
        fitted[name], preds[name] = model, p
        print(f"  {name:<22} ROC-AUC={row['roc_auc']:.4f}  KS={row['ks']:.3f}  fit={fit_s:.1f}c")
    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(C.REPORTS / "model_metrics.csv", index=False)
    save_feature_defaults(train)
    save_json({r["model"]: {"file": f"{r['model'].split()[0].lower()}.joblib", "threshold": r["threshold"]}
               for r in rows}, C.MODELS / "registry.json")
    return metrics, fitted, preds, test


def save_feature_defaults(train: pd.DataFrame) -> None:
    """Типичные значения признаков (медиана/мода train) — для полей, не указанных в анкете."""
    defaults = {c: float(train[c].median()) for c in C.NUM_FEATURES}
    defaults.update({c: str(train[c].mode()[0]) for c in C.CAT_FEATURES})
    save_json(defaults, C.MODELS / "feature_defaults.json")


def load_model(name="LightGBM"):
    registry = load_json(C.MODELS / "registry.json")
    if not registry:
        raise FileNotFoundError("Модели не обучены. Сначала выполните пункт меню 6.")
    info = registry[name]
    return joblib.load(C.MODELS / info["file"]), info["threshold"]


def make_input_row(raw: dict) -> pd.DataFrame:
    """Из полей анкеты собирает строку признаков так же, как features.build_features."""
    r = dict(raw)
    rate = r["int_rate"] / 1200
    n = r["term_months"]
    r["installment"] = r["loan_amnt"] * rate / (1 - (1 + rate) ** -n)
    r["grade"] = r["sub_grade"][0]
    r["loan_to_income"] = r["loan_amnt"] / r["annual_inc"]
    r["installment_to_income"] = r["installment"] * 12 / r["annual_inc"]
    defaults = load_json(C.MODELS / "feature_defaults.json", default={})
    for col in C.FEATURES:
        r.setdefault(col, defaults.get(col, np.nan))
    return pd.DataFrame([r])[C.FEATURES]


def predict_pd(raw: dict) -> dict:
    """Вероятность дефолта по анкете от LightGBM и Logistic Regression."""
    row = make_input_row(raw)
    out = {}
    for name in ["LightGBM", "Logistic Regression"]:
        model, thr = load_model(name)
        p = float(model.predict_proba(row)[:, 1][0])
        out[name] = {"pd": p, "threshold": thr, "decision": "отказ" if p >= thr else "одобрить"}
    return out
