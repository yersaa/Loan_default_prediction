"""Интерпретация и анализ ошибок: ROC/PR, матрица ошибок, калибровка, SHAP, ошибки по грейдам."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score, roc_curve

from src import config as C
from src.eda import save


def plot_curves(y, preds: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        if not name.startswith("Dummy"):  # у константного прогноза PR-кривая вырождена
            ax2.plot(rec, prec, label=name)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(title="ROC-кривые (test: 2015)", xlabel="FPR", ylabel="TPR")
    ax2.axhline(y.mean(), color="k", ls="--", lw=0.8, label="доля дефолтов")
    ax2.set(title="Precision–Recall (test: 2015)", xlabel="Recall", ylabel="Precision")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    save(fig, "09_roc_pr.png")


def plot_confusion(y, p, threshold, name="LightGBM") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, p >= threshold, display_labels=["погашен", "дефолт"],
                                            values_format=",", cmap="Blues", ax=ax, colorbar=False)
    ax.grid(False)
    ax.set(title=f"{name}, порог {threshold:.2f}", xlabel="Прогноз", ylabel="Факт")
    save(fig, "10_confusion_matrix.png")


def plot_calibration(y, preds: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["Logistic Regression", "LightGBM"]:
        frac, mean_p = calibration_curve(y, preds[name], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", label=name)
    ax.plot([0, 0.7], [0, 0.7], "k--", lw=0.8)
    ax.set(title="Калибровка: прогноз PD vs факт", xlabel="Средняя предсказанная PD", ylabel="Фактическая доля дефолтов")
    ax.legend()
    save(fig, "11_calibration.png")


def plot_shap(model, X: pd.DataFrame, n=5000) -> pd.DataFrame:
    """SHAP для LightGBM на подвыборке test: вклад каждого признака в прогноз."""
    sample = X.sample(n, random_state=C.RANDOM_STATE)
    data = model.named_steps["cast"].transform(sample)
    values = shap.TreeExplainer(model.named_steps["model"]).shap_values(data)
    values = values[1] if isinstance(values, list) else values
    plt.figure()
    shap.summary_plot(values, data, max_display=15, show=False)
    plt.title("SHAP: влияние признаков на PD (LightGBM)")
    save(plt.gcf(), "12_shap_summary.png")
    importance = pd.DataFrame({"feature": data.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    return importance.sort_values("mean_abs_shap", ascending=False).round(4)


def error_analysis(test: pd.DataFrame, p, threshold) -> pd.DataFrame:
    """Качество и ошибки по грейдам: где модель различает риск хуже всего."""
    df = test.assign(pd_hat=p, pred=p >= threshold)
    rows = []
    for grade, g in df.groupby("grade", observed=True):
        y = g[C.TARGET]
        rows.append({"grade": grade, "loans": len(g), "actual_default_pct": 100 * y.mean(),
                     "predicted_pd_pct": 100 * g["pd_hat"].mean(),
                     "auc_within_grade": roc_auc_score(y, g["pd_hat"]) if y.nunique() > 1 else np.nan,
                     "missed_defaults_pct": 100 * ((~g["pred"]) & (y == 1)).sum() / max(y.sum(), 1),
                     "false_alarms_pct": 100 * (g["pred"] & (y == 0)).sum() / max((y == 0).sum(), 1)})
    return pd.DataFrame(rows).round(2)


def missed_defaults_profile(test: pd.DataFrame, p, threshold) -> pd.DataFrame:
    """Медианы признаков: пропущенные дефолты vs пойманные дефолты vs хорошие заёмщики."""
    df = test.assign(pred=p >= threshold)
    groups = {"пропущенный дефолт (FN)": (df[C.TARGET] == 1) & ~df["pred"],
              "пойманный дефолт (TP)": (df[C.TARGET] == 1) & df["pred"],
              "погашен (TN+FP)": df[C.TARGET] == 0}
    cols = ["int_rate", "fico", "dti", "annual_inc", "loan_amnt", "term_months", "revol_util"]
    return pd.DataFrame({k: df.loc[m, cols].median() for k, m in groups.items()}).round(1)


def run_explain(fitted: dict, preds: dict, test: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    y = test[C.TARGET].astype(int)
    thr = float(metrics.set_index("model").loc["LightGBM", "threshold"])
    plot_curves(y, preds)
    plot_confusion(y, preds["LightGBM"], thr)
    plot_calibration(y, preds)
    shap_imp = plot_shap(fitted["LightGBM"], test[C.FEATURES])
    errors = error_analysis(test, preds["LightGBM"], thr)
    profile = missed_defaults_profile(test, preds["LightGBM"], thr)
    shap_imp.to_csv(C.REPORTS / "shap_importance.csv", index=False)
    errors.to_csv(C.REPORTS / "error_by_grade.csv", index=False)
    profile.to_csv(C.REPORTS / "missed_defaults_profile.csv")
    return {"shap": shap_imp, "errors": errors, "profile": profile}
