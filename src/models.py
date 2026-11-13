"""Модели PD: baseline (Dummy, только грейд) vs LogReg и LightGBM, на всех полях и только на полях заёмщика.
Претенденты на всех полях: XGBoost, нейросеть (MLP) и ансамбль из LightGBM, XGBoost и нейросети.

Подбор гиперпараметров — TimeSeriesSplit на 2007–2013. Порог и калибровка — на 2014 годе (модель обучена на 2007–2013).
Оценка — на созревших займах 2015 года, которые не участвуют ни в одном из этих шагов.
"""
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from lightgbm import __version__ as lightgbm_version
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config as C
from src.data_io import load_json, save_json
from src.features import add_income_ratios, temporal_split
from src.profiling import measure

SCORING_MODELS = ["LightGBM", "Logistic Regression", "LightGBM (borrower only)"]
SUB_GRADES = [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
HOME_OPTIONS = ["MORTGAGE", "RENT", "OWN"]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Обрезает значения по перцентилям, посчитанным только на train."""

    def __init__(self, low: float = 0.01, high: float = 0.99) -> None:
        self.low, self.high = low, high

    def fit(self, X: np.ndarray, y: pd.Series | None = None) -> "Winsorizer":
        """Запоминает границы перцентилей по обучающим данным."""
        self.bounds_ = np.nanquantile(np.asarray(X, dtype=float), [self.low, self.high], axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Обрезает значения по запомненным границам."""
        return np.clip(np.asarray(X, dtype=float), self.bounds_[0], self.bounds_[1])


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Фиксирует категории train и приводит к ним новые данные (для нативных категорий LightGBM)."""

    def __init__(self, num_features: list | None = None, cat_features: list | None = None) -> None:
        self.num_features, self.cat_features = num_features, cat_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoryCaster":
        """Запоминает список категорий каждого признака по обучающим данным."""
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.cat_features}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Оставляет нужные колонки; категории, которых не было в train, становятся пропусками."""
        X = X[list(self.num_features) + list(self.cat_features)].copy()
        for col, cats in self.categories_.items():
            X[col] = pd.Categorical(X[col].astype("object"), categories=cats)
        return X


def scaled_onehot(num: list, cat: list) -> ColumnTransformer:
    """Подготовка данных для моделей, которым нужны числа одного масштаба: регрессии и нейросети.

    Числа: пропуски -> медиана, обрезка выбросов, масштабирование. Категории: one-hot,
    значения реже 50 раз объединяются. Подготовка обучается на тех же данных, что и модель.
    """
    # keep_empty_features: у займов до августа 2012 г. поля кредитного отчёта целиком пустые; колонку не отбрасываем
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("clip", Winsorizer()), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])


def logreg_pipeline(num: list, cat: list, C_reg: float = 1.0) -> Pipeline:
    """Логистическая регрессия с подготовкой данных внутри Pipeline."""
    return Pipeline([("prep", scaled_onehot(num, cat)), ("model", LogisticRegression(C=C_reg, max_iter=1000))])


def mlp_pipeline(num: list, cat: list, **params) -> Pipeline:
    """Нейронная сеть прямого распространения: два скрытых слоя, 64 и 32 нейрона.

    Обучение останавливается, когда качество на отложенных 10% обучающих данных перестаёт расти
    (early_stopping), поэтому число эпох подбирать не нужно. alpha — сила L2-регуляризации.
    """
    model = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, early_stopping=True, n_iter_no_change=3,
                          max_iter=40, random_state=C.RANDOM_STATE, **params)
    return Pipeline([("prep", scaled_onehot(num, cat)), ("model", model)])


def grade_only_pipeline() -> Pipeline:
    """Baseline: риск по одному подгрейду, присвоенному самим LendingClub."""
    prep = ColumnTransformer([("grade", OneHotEncoder(handle_unknown="ignore"), ["sub_grade"])])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])


def lgbm_pipeline(num: list, cat: list, **params) -> Pipeline:
    """LightGBM сам работает с пропусками и категориями, поэтому подготовка — только список категорий."""
    model = LGBMClassifier(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           min_child_samples=200, random_state=C.RANDOM_STATE, verbose=-1, **params)
    return Pipeline([("cast", CategoryCaster(num, cat)), ("model", model)])


def xgb_pipeline(num: list, cat: list, **params) -> Pipeline:
    """XGBoost: другая реализация градиентного бустинга. Как и LightGBM, сам работает с пропусками и категориями.

    Настройки по умолчанию (глубина 6, шаг 0.3) на этих данных переобучаются, поэтому шаг меньше,
    а в листе должно быть не меньше 50 займов.
    """
    model = XGBClassifier(tree_method="hist", enable_categorical=True, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=50, random_state=C.RANDOM_STATE, **params)
    return Pipeline([("cast", CategoryCaster(num, cat)), ("model", model)])


@dataclass(frozen=True)
class ModelSpec:
    """Описание модели: как её создать, что подбирать и какую роль она играет в сравнении."""
    name: str
    slug: str
    build: Callable[..., object] | None = None
    grid: dict = field(default_factory=dict)
    baseline: bool = False
    family: str = ""
    challenger: bool = False  # претендент: заменяет выбранную модель, только если лучше её на запас
    members: tuple = ()       # для ансамбля: названия моделей, вероятности которых усредняются


LR_GRID = {"model__C": [0.05, 1.0]}
LGBM_GRID = {"model__num_leaves": [15, 31], "model__n_estimators": [300, 600]}
XGB_GRID = {"model__max_depth": [4, 6], "model__n_estimators": [300, 600]}
MLP_GRID = {"model__alpha": [0.001, 0.1]}
ENSEMBLE = "Ensemble (LightGBM + XGBoost + MLP)"


def model_specs() -> list:
    """Все сравниваемые модели: два baseline, основные модели на всех полях и только на полях заёмщика
    и три претендента на всех полях: XGBoost, нейросеть и ансамбль."""
    num, cat = C.NUM_FEATURES, C.CAT_FEATURES
    b_num, b_cat = C.BORROWER_NUM, C.BORROWER_CAT
    # Третий аргумент — функция, которая создаёт новую необученную модель. lambda **p передаёт в неё
    # гиперпараметры, например lgbm_pipeline(num, cat, num_leaves=31). Четвёртый — сетка для подбора.
    return [
        ModelSpec("Dummy (prior)", "dummy", lambda: DummyClassifier(strategy="prior"), baseline=True),
        ModelSpec("Grade only (LogReg)", "grade", grade_only_pipeline, baseline=True),
        ModelSpec("Logistic Regression", "logistic", lambda **p: logreg_pipeline(num, cat, **p), LR_GRID,
                  family="logreg"),
        ModelSpec("LightGBM", "lightgbm", lambda **p: lgbm_pipeline(num, cat, **p), LGBM_GRID, family="gbm"),
        ModelSpec("XGBoost", "xgboost", lambda **p: xgb_pipeline(num, cat, **p), XGB_GRID, family="gbm",
                  challenger=True),
        ModelSpec("Neural network (MLP)", "mlp", lambda **p: mlp_pipeline(num, cat, **p), MLP_GRID, family="nn",
                  challenger=True),
        ModelSpec(ENSEMBLE, "ensemble", family="ensemble", challenger=True,
                  members=("LightGBM", "XGBoost", "Neural network (MLP)")),
        ModelSpec("Logistic Regression (borrower only)", "logistic_borrower",
                  lambda **p: logreg_pipeline(b_num, b_cat, **p), LR_GRID, family="logreg"),
        ModelSpec("LightGBM (borrower only)", "lightgbm_borrower",
                  lambda **p: lgbm_pipeline(b_num, b_cat, **p), LGBM_GRID, family="gbm"),
    ]


def _plain_params(params: dict) -> dict:
    """{'model__C': 0.05} -> {'C_reg': 0.05}: аргументы фабрик пайплайнов."""
    renamed = {k.removeprefix("model__"): v for k, v in params.items()}
    if "C" in renamed:
        renamed["C_reg"] = renamed.pop("C")
    return renamed


def tune(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> dict:
    """Подбор гиперпараметров кросс-валидацией по времени (без перемешивания)."""
    best = {}
    for spec in model_specs():
        if not spec.grid:
            continue
        with measure() as m:
            gs = GridSearchCV(spec.build(), spec.grid, cv=TimeSeriesSplit(n_splits=n_splits),
                              scoring="roc_auc", n_jobs=1).fit(X, y)
        grid = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score"]]
        best[spec.name] = {"params": gs.best_params_, "cv_auc": round(gs.best_score_, 4),
                           "seconds": round(m.seconds, 1),
                           "grid": grid.assign(params=grid["params"].astype(str)).to_dict("records")}
        print(f"  {spec.name}: {gs.best_params_}  CV ROC-AUC={gs.best_score_:.4f}")
    return best


def build_model(spec: ModelSpec, best: dict) -> BaseEstimator:
    """Создаёт необученную модель с гиперпараметрами из подбора (или стандартными, если подбора не было).

    Ансамбль собирается из своих моделей: VotingClassifier с voting="soft" усредняет их вероятности.
    """
    if spec.members:
        by_name = {s.name: s for s in model_specs()}
        parts = [(by_name[name].slug, build_model(by_name[name], best)) for name in spec.members]
        return VotingClassifier(parts, voting="soft")
    params = best.get(spec.name, {}).get("params", {})
    return spec.build(**_plain_params(params))


def fit_platt(p: np.ndarray, y: pd.Series) -> tuple:
    """Калибровка Платта: логистическая регрессия на логите прогноза. Порядок заёмщиков не меняется."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Применяет калибровку Платта с коэффициентами a и b к вероятностям p."""
    return expit(a * logit(np.clip(p, 1e-6, 1 - 1e-6)) + b)


def ks_stat(y: pd.Series, p: np.ndarray) -> float:
    """Статистика Колмогорова–Смирнова: max(TPR - FPR) — стандартная метрика скоринга."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y: pd.Series, p: np.ndarray) -> float:
    """Порог вероятности от 0.05 до 0.60 с шагом 0.01, при котором F1 максимальна."""
    grid = np.linspace(0.05, 0.6, 56)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def expected_calibration_error(y: pd.Series, p: np.ndarray, bins: int = 10) -> float:
    """Средневзвешенный модуль разницы между прогнозом и фактом по квантильным корзинам."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": p})
    if frame["p"].nunique() < 2:  # константный прогноз: квантильных границ нет, корзина одна
        return float(abs(frame["y"].mean() - frame["p"].mean()))
    groups = frame.groupby(pd.qcut(frame["p"], bins, duplicates="drop"), observed=True)
    return float(sum(len(g) * abs(g["y"].mean() - g["p"].mean()) for _, g in groups) / len(frame))


def evaluate(y: pd.Series, p: np.ndarray, threshold: float) -> dict:
    """Все метрики качества одной модели на test при заданном пороге."""
    pred = p >= threshold
    auc = roc_auc_score(y, p)
    return {"roc_auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y, p),
            "ks": ks_stat(y, p), "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p),
            "threshold": threshold, "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def fit_one(spec: ModelSpec, best: dict, windows: dict) -> dict:
    """Обучает одну модель и считает её метрики.

    Шаг 1. Пробная модель на fit (2007–2013) делает прогноз на valid (2014): по нему подбираются
           калибровка Платта и порог, и считается valid_auc для выбора модели.
    Шаг 2. Финальная модель обучается на train (2007–2014) и оценивается на test (2015).
           Калибровка и порог берутся из шага 1: test в их подборе не участвует.

    probe_test_auc — пробная модель на test. Она объясняет, почему AUC на test выше, чем на valid:
    разница valid_auc -> probe_test_auc — это другой год, probe_test_auc -> roc_auc — добавленный в обучение 2014 год.
    """
    fit, valid, train, test = windows["fit"], windows["valid"], windows["train"], windows["test"]
    y_valid, y_test = valid[C.TARGET].astype(int), test[C.TARGET].astype(int)

    probe = build_model(spec, best).fit(fit[C.FEATURES], fit[C.TARGET].astype(int))
    p_valid = probe.predict_proba(valid[C.FEATURES])[:, 1]
    a, b = (1.0, 0.0) if spec.slug == "dummy" else fit_platt(p_valid, y_valid)
    threshold = best_f1_threshold(y_valid, apply_platt(p_valid, a, b))
    valid_auc = roc_auc_score(y_valid, p_valid)
    probe_test_auc = roc_auc_score(y_test, probe.predict_proba(test[C.FEATURES])[:, 1])

    model = build_model(spec, best)
    with measure() as fit_cost:
        model.fit(train[C.FEATURES], train[C.TARGET].astype(int))
    with measure() as predict_cost:
        p_raw = model.predict_proba(test[C.FEATURES])[:, 1]
    p = apply_platt(p_raw, a, b)

    path = C.MODELS / f"{spec.slug}.joblib"
    joblib.dump(model, path)
    # Память обучения здесь не измеряется: модели идут подряд в одном процессе и влияют друг на друга.
    # Она замеряется в отдельных процессах в src/benchmark.py (пункт меню 8).
    row = {"model": spec.name, "valid_auc": valid_auc, "probe_test_auc": probe_test_auc,
           **evaluate(y_test, p, threshold),
           "brier_raw": brier_score_loss(y_test, p_raw), "ece_raw": expected_calibration_error(y_test, p_raw),
           "mean_pd": float(p.mean()), "actual_default_rate": float(y_test.mean()),
           "fit_seconds": fit_cost.seconds,
           "predict_ms_per_1000": 1e6 * predict_cost.seconds / len(test), "model_mb": path.stat().st_size / 2**20}
    return {"row": row, "model": model, "p": p, "p_raw": p_raw,
            "registry": {"file": path.name, "threshold": threshold, "platt_a": a, "platt_b": b}}


def select_model(metrics: pd.DataFrame, specs: list) -> dict:
    """Выбор модели по AUC на валидации (не на test), запас задан в config.SIMPLICITY_MARGIN_AUC.

    Шаг 1. Среди основных моделей берётся лучшая; логистическая регрессия предпочитается,
           если отстаёт меньше чем на запас.
    Шаг 2. Претендент (XGBoost, нейросеть, ансамбль) заменяет выбранную модель, только если лучше её
           на запас или больше: ради прироста в пределах шума модель не меняется.
    """
    by_name = {s.name: s for s in specs}
    scores = metrics.set_index("model")["valid_auc"]
    scores = scores[[name for name in scores.index if name in by_name and not by_name[name].baseline]]
    table = scores[[name for name in scores.index if not by_name[name].challenger]]
    challengers = scores[[name for name in scores.index if by_name[name].challenger]]
    best_name = table.idxmax()
    choice = best_name
    logreg = [n for n in table.index if by_name[n].family == "logreg"
              and ("borrower" in n) == ("borrower" in best_name)]
    if logreg and by_name[best_name].family != "logreg" \
            and table[best_name] - table[logreg[0]] < C.SIMPLICITY_MARGIN_AUC:
        choice = logreg[0]
    if len(challengers) and challengers.max() - scores[choice] >= C.SIMPLICITY_MARGIN_AUC:
        choice = challengers.idxmax()
    margin = C.SIMPLICITY_MARGIN_AUC
    return {"rule": f"max valid_auc (2014); логрегрессия, если отстаёт меньше чем на {margin}; "
                    f"претендент заменяет выбранную модель, только если лучше на {margin} или больше",
            "best_by_valid_auc": best_name, "selected": choice, "valid_auc": table.round(4).to_dict(),
            "challengers_valid_auc": challengers.round(4).to_dict()}


def save_metadata(windows: dict) -> None:
    """Записывает, когда, на каких данных и какими версиями библиотек обучены модели."""
    train, test = windows["train"], windows["test"]
    save_json({"trained_at": datetime.now().isoformat(timespec="seconds"), "sample_mode": C.SAMPLE_MODE,
               "python": platform.python_version(), "sklearn": sklearn.__version__,
               "lightgbm": lightgbm_version, "pandas": pd.__version__, "random_state": C.RANDOM_STATE,
               "cohort": "созревшие завершённые займы", "train_rows": len(train), "test_rows": len(test),
               "train_last_issue": str(train["issue_d"].max().date()),
               "test_last_issue": str(test["issue_d"].max().date())},
              C.MODELS / "metadata.json")


def train_and_compare(model_df: pd.DataFrame, best: dict) -> tuple:
    """Обучает все модели, сохраняет метрики и реестр. Возвращает метрики, модели, прогнозы и test."""
    windows = temporal_split(model_df)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    rows, fitted, preds, preds_raw, registry = [], {}, {}, {}, {}
    specs = model_specs()
    for spec in specs:
        res = fit_one(spec, best, windows)
        rows.append(res["row"])
        fitted[spec.name], preds[spec.name], preds_raw[spec.name] = res["model"], res["p"], res["p_raw"]
        registry[spec.name] = res["registry"]
        print(f"  {spec.name:<36} ROC-AUC={res['row']['roc_auc']:.4f}  KS={res['row']['ks']:.3f}  "
              f"fit={res['row']['fit_seconds']:.1f}c")
    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(C.REPORTS / "model_metrics.csv", index=False)
    save_json(select_model(metrics, specs), C.REPORTS / "model_selection.json")
    save_feature_defaults(windows["train"])
    save_json(registry, C.MODELS / "registry.json")
    save_metadata(windows)
    return metrics, fitted, preds, preds_raw, windows["test"]


def save_feature_defaults(train: pd.DataFrame) -> None:
    """Типичные значения признаков (медиана/мода train) — для полей, не указанных в анкете."""
    defaults = {c: float(train[c].median()) for c in C.NUM_FEATURES}
    defaults.update({c: str(train[c].mode()[0]) for c in C.CAT_FEATURES})
    save_json(defaults, C.MODELS / "feature_defaults.json")


def models_dir() -> Path:
    """Обученные в этом запуске модели, а если их нет — поставляемые в репозитории."""
    return C.MODELS if (C.MODELS / "registry.json").exists() else C.SHIPPED_MODELS


def load_model(name: str = "LightGBM") -> tuple:
    """Загружает модель по названию и её запись из реестра: порог и коэффициенты калибровки."""
    folder = models_dir()
    registry = load_json(folder / "registry.json")
    if not registry:
        raise FileNotFoundError("Модели не обучены. Сначала выполните пункт меню 6.")
    info = registry[name]
    return joblib.load(folder / info["file"]), info


APPLICATION_LIMITS = {
    "loan_amnt": (500, 40_000), "int_rate": (5, 31), "annual_inc": (1_000, 10_000_000), "dti": (0, 100),
    "fico": (600, 850), "emp_years": (0, 10), "revol_util": (0, 150), "inq_last_6mths": (0, 30),
    "credit_hist_months": (0, 900),
}


def validate_application(raw: dict) -> None:
    """Отклоняет анкету вне диапазонов, на которых обучалась модель."""
    for name, (low, high) in APPLICATION_LIMITS.items():
        if not low <= raw[name] <= high:
            raise ValueError(f"{name} = {raw[name]}: допустимо от {low} до {high}")
    if raw["term_months"] not in (36, 60):
        raise ValueError(f"term_months = {raw['term_months']}: допустимо 36 или 60")
    if raw["sub_grade"] not in SUB_GRADES:
        raise ValueError(f"sub_grade = {raw['sub_grade']}: неизвестный подгрейд")


def make_input_row(raw: dict) -> pd.DataFrame:
    """Из полей анкеты собирает строку признаков; отношения к доходу считает тот же код, что при обучении."""
    r = dict(raw)
    rate, n = r["int_rate"] / 1200, r["term_months"]
    r["installment"] = r["loan_amnt"] * rate / (1 - (1 + rate) ** -n)
    r["grade"] = r["sub_grade"][0]
    row = add_income_ratios(pd.DataFrame([r]))
    defaults = load_json(models_dir() / "feature_defaults.json", default={})
    for col in C.FEATURES:
        if col not in row:
            row[col] = defaults.get(col, np.nan)
    return row[C.FEATURES]


def predict_pd(raw: dict) -> dict:
    """Калиброванная вероятность дефолта по анкете от каждой модели скоринга."""
    validate_application(raw)
    row = make_input_row(raw)
    out = {}
    for name in SCORING_MODELS:
        model, info = load_model(name)
        p = apply_platt(model.predict_proba(row)[:, 1], info["platt_a"], info["platt_b"])[0]
        out[name] = {"pd": float(p), "threshold": info["threshold"],
                     "decision": "отказ" if p >= info["threshold"] else "одобрить"}
    return out
