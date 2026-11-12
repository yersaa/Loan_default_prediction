"""Интерпретация и анализ ошибок: ROC/PR, калибровка, SHAP, ошибки по грейдам, таблица одобрения."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score, roc_curve

from src import config as C
from src.style import AQUA, AXIS, BLUE, MAGENTA, MUTED, ORANGE, YELLOW, save

PRIMARY = "LightGBM"
BORROWER = "LightGBM (borrower only)"
MODEL_COLORS = {"Grade only (LogReg)": BLUE, "Logistic Regression": ORANGE, PRIMARY: AQUA, BORROWER: YELLOW,
                "Logistic Regression (borrower only)": MAGENTA}
APPROVAL_MODELS = ["Grade only (LogReg)", "Logistic Regression", PRIMARY, BORROWER]
PR_MIN_RECALL = 0.01


def plot_curves(y: pd.Series, preds: dict) -> None:
    """Модель без цвета — Dummy: она совпадает с диагональю ROC и с линией доли дефолтов на PR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, color in MODEL_COLORS.items():
        p = preds[name]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        keep = rec >= PR_MIN_RECALL  # при крошечном recall precision считается по единицам займов
        ax2.plot(rec[keep], prec[keep], color=color, label=name)
    ax1.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1, label="Dummy (AUC=0.500)")
    ax1.set(title="ROC-кривые (test: созревшие займы 2015 г.)", xlabel="FPR", ylabel="TPR")
    ax2.axhline(y.mean(), color=MUTED, ls="--", lw=1, label="доля дефолтов (Dummy)")
    ax2.set(title="Precision–Recall (test)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, None))
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    save(fig, "09_roc_pr.png")


def plot_confusion(y: pd.Series, p: np.ndarray, threshold: float, name: str = PRIMARY) -> None:
    """Матрица ошибок на test при заданном пороге: сколько дефолтов поймано и сколько пропущено."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, p >= threshold, display_labels=["погашен", "дефолт"],
                                            values_format=",", cmap="Blues", ax=ax, colorbar=False)
    ax.grid(False)
    ax.set(title=f"{name}, порог {threshold:.2f}", xlabel="Прогноз", ylabel="Факт")
    save(fig, "10_confusion_matrix.png")


def plot_calibration(y: pd.Series, preds: dict, preds_raw: dict) -> None:
    """Сплошная линия — после калибровки на 2014 годе, пунктир — исходный прогноз."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name in ["Logistic Regression", PRIMARY]:
        color = MODEL_COLORS[name]
        frac, mean_p = calibration_curve(y, preds[name], n_bins=10, strategy="quantile")
        frac_raw, mean_raw = calibration_curve(y, preds_raw[name], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=color, label=f"{name}, калибр.")
        ax.plot(mean_raw, frac_raw, "o--", color=color, alpha=0.45, lw=1.2, ms=4, label=f"{name}, исходный")
    top = float(max(np.max(mean_p), np.max(frac))) * 1.05
    ax.plot([0, top], [0, top], color=AXIS, ls="--", lw=1)
    ax.set(title="Калибровка: прогноз PD и факт (test)", xlabel="Средняя предсказанная PD",
           ylabel="Фактическая доля дефолтов")
    ax.legend(fontsize=8)
    save(fig, "11_calibration.png")


def plot_shap(model, X: pd.DataFrame, filename: str, title: str) -> pd.DataFrame:
    """SHAP для LightGBM на подвыборке test: вклад каждого признака в прогноз."""
    sample = X.sample(min(C.SHAP_SAMPLE_ROWS, len(X)), random_state=C.RANDOM_STATE)
    data = model.named_steps["cast"].transform(sample)
    with warnings.catch_warnings():  # shap предупреждает о формате вывода, он обработан строкой ниже
        warnings.simplefilter("ignore", UserWarning)
        values = shap.TreeExplainer(model.named_steps["model"]).shap_values(data)
    values = values[1] if isinstance(values, list) else values
    plt.figure()
    shap.summary_plot(values, data, max_display=15, show=False)
    plt.title(title)
    save(plt.gcf(), filename)
    importance = pd.DataFrame({"feature": data.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    return importance.sort_values("mean_abs_shap", ascending=False).round(4)


def error_analysis(test: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    """Качество и ошибки по грейдам: где модель различает риск хуже всего."""
    df = test.assign(pd_hat=p, pred=p >= threshold)
    rows = []
    for grade, g in df.groupby("grade", observed=True):
        y = g[C.TARGET]
        rows.append({"grade": grade, "loans": len(g), "actual_default_pct": 100 * float(y.mean()),
                     "predicted_pd_pct": 100 * g["pd_hat"].mean(),
                     "auc_within_grade": roc_auc_score(y, g["pd_hat"]) if y.nunique() > 1 else np.nan,
                     "missed_defaults_pct": 100 * ((~g["pred"]) & (y == 1)).sum() / max(y.sum(), 1),
                     "false_alarms_pct": 100 * (g["pred"] & (y == 0)).sum() / max((y == 0).sum(), 1)})
    return pd.DataFrame(rows).round(2)


def missed_defaults_profile(test: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    """Медианы признаков: пропущенные дефолты vs пойманные дефолты vs хорошие заёмщики."""
    df = test.assign(pred=p >= threshold)
    groups = {"пропущенный дефолт (FN)": (df[C.TARGET] == 1) & ~df["pred"],
              "пойманный дефолт (TP)": (df[C.TARGET] == 1) & df["pred"],
              "погашен (TN+FP)": df[C.TARGET] == 0}
    cols = ["int_rate", "fico", "dti", "annual_inc", "loan_amnt", "revol_util"]
    return pd.DataFrame({k: df.loc[m, cols].median() for k, m in groups.items()}).round(1)


def approval_table(test: pd.DataFrame, preds: dict, rates: tuple | None = None) -> pd.DataFrame:
    """Одобряем заданную долю заявок с наименьшей PD: какая дефолтность и фактическая доходность портфеля.

    total_pymnt известен только после выдачи, поэтому используется лишь для оценки, не как признак.
    Доходность номинальная: без стоимости фондирования, дисконтирования и риска досрочного погашения.
    """
    rows = []
    y = test[C.TARGET].to_numpy(dtype=float)
    paid, lent = test["total_pymnt"].to_numpy(dtype=float), test["loan_amnt"].to_numpy(dtype=float)
    for name, p in preds.items():
        order = np.argsort(p, kind="stable")
        for rate in rates or C.APPROVAL_RATES:
            take = order[:int(round(rate * len(p)))]
            rows.append({"model": name, "approval_rate": rate, "max_pd": float(p[take[-1]]),
                         "default_rate_pct": 100 * y[take].mean(),
                         "net_return_pct": 100 * (paid[take].sum() / lent[take].sum() - 1)})
    return pd.DataFrame(rows).round(3)


def plot_approval(table: pd.DataFrame) -> None:
    """Два графика по таблице одобрения: дефолтность и доходность портфеля при разной доле одобрения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, part in table.groupby("model", sort=False):
        color = MODEL_COLORS[name]
        rate = part["approval_rate"] * 100
        ax1.plot(rate, part["default_rate_pct"], "o-", color=color, label=name)
        ax2.plot(rate, part["net_return_pct"], "o-", color=color, label=name)
    ax1.set(title="Дефолтность одобренного портфеля", xlabel="Доля одобренных заявок, %", ylabel="Дефолт, %")
    ax2.set(title="Номинальная доходность портфеля", xlabel="Доля одобренных заявок, %",
            ylabel="Возврат сверх выданного, %")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    save(fig, "13_approval.png")


def grade_recovery(test: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """Насколько грейд LendingClub восстанавливается из полей заёмщика и что он добавляет."""
    df = test.assign(borrower=preds[BORROWER], full=preds[PRIMARY])
    rows = []
    for grade, g in df.groupby("grade", observed=True):
        y = g[C.TARGET]
        rows.append({"grade": grade, "loans": len(g), "actual_default_pct": 100 * float(y.mean()),
                     "borrower_pd_pct": 100 * g["borrower"].mean(), "full_pd_pct": 100 * g["full"].mean(),
                     "auc_within_borrower": roc_auc_score(y, g["borrower"]) if y.nunique() > 1 else np.nan,
                     "auc_within_full": roc_auc_score(y, g["full"]) if y.nunique() > 1 else np.nan})
    return pd.DataFrame(rows).round(3)


def run_explain(fitted: dict, preds: dict, preds_raw: dict, test: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """Строит все графики и таблицы интерпретации по результатам обучения и сохраняет их в reports/."""
    y = test[C.TARGET].astype(int)
    thr = float(metrics.set_index("model").loc[PRIMARY, "threshold"])
    plot_curves(y, preds)
    plot_confusion(y, preds[PRIMARY], thr)
    plot_calibration(y, preds, preds_raw)
    shap_full = plot_shap(fitted[PRIMARY], test[C.FEATURES], "12_shap_summary.png",
                          "SHAP: влияние признаков на PD (LightGBM, все поля)")
    shap_borrower = plot_shap(fitted[BORROWER], test[C.FEATURES], "12b_shap_borrower.png",
                              "SHAP: влияние признаков на PD (LightGBM, поля заёмщика)")
    errors = error_analysis(test, preds[PRIMARY], thr)
    profile = missed_defaults_profile(test, preds[PRIMARY], thr)
    approval = approval_table(test, {m: preds[m] for m in APPROVAL_MODELS})
    plot_approval(approval)
    recovery = grade_recovery(test, preds)
    shap_full.to_csv(C.REPORTS / "shap_importance.csv", index=False)
    shap_borrower.to_csv(C.REPORTS / "shap_importance_borrower.csv", index=False)
    errors.to_csv(C.REPORTS / "error_by_grade.csv", index=False)
    profile.to_csv(C.REPORTS / "missed_defaults_profile.csv")
    approval.to_csv(C.REPORTS / "approval_table.csv", index=False)
    recovery.to_csv(C.REPORTS / "grade_recovery.csv", index=False)
    return {"shap": shap_full, "shap_borrower": shap_borrower, "errors": errors, "profile": profile,
            "approval": approval, "recovery": recovery}
