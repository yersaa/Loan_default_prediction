"""Анализ чувствительности: насколько цензурирование 60-месячных займов искажает результат.

Сравниваем основной сплит (только созревшие займы) со старым (все завершённые займы 2007–2015).
Старый вариант нужен лишь для того, чтобы показать величину смещения, а не как итоговый результат.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config as C
from src.features import modeling_frame, temporal_split
from src.models import build_model, model_specs

MODELS = ["Logistic Regression", "LightGBM"]


def term_auc(test: pd.DataFrame, p, term: int) -> float | None:
    """ROC-AUC только по займам с заданным сроком; None, если таких займов нет или в них один класс."""
    mask = (test["term_months"] == term).to_numpy()
    y = test.loc[mask, C.TARGET]
    return round(float(roc_auc_score(y, p[mask])), 4) if y.nunique() > 1 else None


def cohort_rows(frame: pd.DataFrame, cohort: str, best: dict) -> list:
    """Обучает модели на train когорты и возвращает по строке метрик на её test для каждой модели."""
    windows = temporal_split(frame)
    train, test = windows["train"], windows["test"]
    specs = {s.name: s for s in model_specs()}
    rows = []
    for name in MODELS:
        model = build_model(specs[name], best).fit(train[C.FEATURES], train[C.TARGET].astype(int))
        p = model.predict_proba(test[C.FEATURES])[:, 1]
        rows.append({"cohort": cohort, "model": name, "train_rows": len(train), "test_rows": len(test),
                     "train_default_pct": round(100 * float(train[C.TARGET].mean()), 2),
                     "test_default_pct": round(100 * float(test[C.TARGET].mean()), 2),
                     "mean_pd_pct": round(100 * float(p.mean()), 2),
                     "test_auc": round(float(roc_auc_score(test[C.TARGET], p)), 4),
                     "auc_36m": term_auc(test, p, 36), "auc_60m": term_auc(test, p, 60)})
    return rows


def censoring_sensitivity(df: pd.DataFrame, best: dict) -> pd.DataFrame:
    """Те же модели и гиперпараметры на двух когортах. df — плоская таблица из load_dataset."""
    matured = modeling_frame(df.copy(), matured_only=True)
    censored = modeling_frame(df.copy(), matured_only=False)
    table = pd.DataFrame(cohort_rows(matured, "созревшие (основной)", best)
                         + cohort_rows(censored, "все завершённые (смещённый)", best))
    table.to_csv(C.REPORTS / "censoring_sensitivity.csv", index=False)
    return table
