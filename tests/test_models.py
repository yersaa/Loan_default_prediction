"""Метрики, калибровка, выбор модели, скоринг и равенство признаков при обучении и скоринге."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config as C
from src.censoring import censoring_sensitivity
from src.data_io import load_json
from src.features import add_target, build_features, modeling_frame, temporal_split
from src.models import (
    ENSEMBLE,
    CategoryCaster,
    Winsorizer,
    apply_platt,
    best_f1_threshold,
    build_model,
    expected_calibration_error,
    fit_platt,
    ks_stat,
    make_input_row,
    model_specs,
    predict_pd,
    select_model,
    train_and_compare,
    tune,
    validate_application,
)

APPLICATION = {
    "loan_amnt": 15000, "term_months": 36, "int_rate": 13.0, "sub_grade": "C1", "annual_inc": 65000, "dti": 18,
    "fico": 700, "emp_years": 5, "revol_util": 50, "inq_last_6mths": 1, "credit_hist_months": 180,
    "home_ownership": "RENT", "purpose": "debt_consolidation", "verification_status": "Source Verified",
    "initial_list_status": "w", "application_type": "Individual",
}


@pytest.mark.unit
def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_stat(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_stat(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


@pytest.mark.unit
def test_best_threshold_separates_classes():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.4] * 50)
    assert 0.1 < best_f1_threshold(y, p) <= 0.4


@pytest.mark.unit
def test_platt_preserves_order_and_fixes_bias():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.5, 20_000)
    y = (rng.random(20_000) < true_p).astype(int)
    biased = true_p * 0.5
    a, b = fit_platt(biased, y)
    fixed = apply_platt(biased, a, b)
    assert roc_auc_score(y, fixed) == pytest.approx(roc_auc_score(y, biased))
    assert expected_calibration_error(y, fixed) < expected_calibration_error(y, biased)
    assert fixed.mean() == pytest.approx(y.mean(), abs=0.01)


@pytest.mark.unit
def test_ece_on_constant_prediction_is_the_mean_gap():
    y = np.array([0, 0, 0, 1])
    assert expected_calibration_error(y, np.full(4, 0.5)) == pytest.approx(0.25)


@pytest.mark.unit
def test_platt_identity_for_baseline():
    p = np.array([0.1, 0.3, 0.7])
    assert apply_platt(p, 1.0, 0.0) == pytest.approx(p)


@pytest.mark.unit
def test_winsorizer_uses_train_bounds_only():
    train = np.arange(101.0).reshape(-1, 1)
    w = Winsorizer(low=0.1, high=0.9).fit(train)
    clipped = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert clipped.ravel().tolist() == [10.0, 50.0, 90.0]


@pytest.mark.unit
def test_category_caster_maps_unseen_to_nan():
    train = pd.DataFrame({"n": [1.0, 2.0], "c": ["a", "b"]})
    caster = CategoryCaster(["n"], ["c"]).fit(train)
    out = caster.transform(pd.DataFrame({"n": [3.0], "c": ["unseen"], "ignored": [1]}))
    assert list(out.columns) == ["n", "c"]
    assert out["c"].isna().all()
    assert list(out["c"].cat.categories) == ["a", "b"]


def _selection(**valid_auc: float) -> dict:
    """Выбор модели при заданных AUC на валидации; не названные модели получают низкое значение."""
    short = {"dummy": "Dummy (prior)", "grade": "Grade only (LogReg)", "logreg": "Logistic Regression",
             "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "Neural network (MLP)", "ensemble": ENSEMBLE,
             "logreg_b": "Logistic Regression (borrower only)", "lgbm_b": "LightGBM (borrower only)"}
    scores = {short[key]: value for key, value in valid_auc.items()}
    specs = model_specs()
    metrics = pd.DataFrame({"model": [s.name for s in specs], "valid_auc": [scores.get(s.name, 0.6) for s in specs]})
    return select_model(metrics, specs)


@pytest.mark.unit
def test_select_model_prefers_simple_logreg_within_margin():
    assert _selection(logreg=0.700, lgbm=0.703)["selected"] == "Logistic Regression"


@pytest.mark.unit
def test_select_model_keeps_gbm_when_gap_is_large():
    assert _selection(logreg=0.690, lgbm=0.710)["selected"] == "LightGBM"


@pytest.mark.unit
def test_challenger_within_margin_does_not_replace_the_choice():
    result = _selection(logreg=0.680, lgbm=0.689, xgb=0.690, ensemble=0.691)
    assert result["selected"] == "LightGBM"
    assert result["challengers_valid_auc"][ENSEMBLE] == 0.691
    assert "XGBoost" not in result["valid_auc"]


@pytest.mark.unit
def test_challenger_replaces_the_choice_when_better_by_the_margin():
    assert _selection(logreg=0.680, lgbm=0.689, xgb=0.690, ensemble=0.695)["selected"] == ENSEMBLE


@pytest.mark.unit
def test_ensemble_averages_the_probabilities_of_its_members(loans):
    frame = modeling_frame(loans.copy())
    train = temporal_split(frame)["train"]
    X, y = train[C.FEATURES], train[C.TARGET].astype(int)
    specs = {s.name: s for s in model_specs()}
    ensemble = build_model(specs[ENSEMBLE], {}).fit(X, y)
    parts = [model.predict_proba(X)[:, 1] for model in ensemble.estimators_]
    assert len(parts) == 3
    assert np.allclose(ensemble.predict_proba(X)[:, 1], np.mean(parts, axis=0))


@pytest.mark.unit
@pytest.mark.parametrize("field, value, message", [
    ("term_months", 45, "term_months"), ("fico", 100, "fico"), ("sub_grade", "Z9", "sub_grade"),
    ("loan_amnt", 10**7, "loan_amnt"),
])
def test_validate_application_rejects_bad_input(field, value, message):
    with pytest.raises(ValueError, match=message):
        validate_application({**APPLICATION, field: value})


@pytest.mark.unit
def test_validate_application_accepts_valid():
    validate_application(APPLICATION)


@pytest.mark.unit
def test_predict_without_models(tmp_project):
    with pytest.raises(FileNotFoundError, match="пункт меню 6"):
        predict_pd(APPLICATION)


@pytest.mark.unit
def test_serving_row_matches_training_features(tmp_project, loans):
    """Строка для скоринга должна совпасть с build_features на тех же исходных данных."""
    source = build_features(add_target(loans.head(1).copy())).iloc[0]
    derived = {"installment", "grade", "loan_to_income", "installment_to_income"}
    raw = {c: source[c] for c in C.FEATURES if c not in derived}
    raw = {k: (str(v) if isinstance(v, str) else v) for k, v in raw.items()}
    row = make_input_row(raw).iloc[0]
    assert row["loan_to_income"] == pytest.approx(source["loan_to_income"], rel=1e-5)
    assert row["installment"] == pytest.approx(source["installment"], rel=1e-5)
    assert row["installment_to_income"] == pytest.approx(source["installment_to_income"], rel=1e-5)
    assert row["grade"] == source["grade"]
    assert list(make_input_row(raw).columns) == C.FEATURES


@pytest.fixture
def trained(tmp_project, loans, fast_best):
    frame = modeling_frame(loans.copy())
    return (*train_and_compare(frame, fast_best), frame)


@pytest.mark.integration
def test_train_and_compare_produces_full_comparison(trained):
    metrics, fitted, preds, preds_raw, test, _ = trained
    names = {s.name for s in model_specs()}
    assert set(metrics["model"]) == names == set(fitted)
    assert len(preds["LightGBM"]) == len(test)
    table = metrics.set_index("model")
    assert table.loc["Dummy (prior)", "roc_auc"] == pytest.approx(0.5)
    assert table.loc["LightGBM", "roc_auc"] > 0.55
    for col in ("fit_seconds", "model_mb", "valid_auc", "probe_test_auc", "ece", "ece_raw", "mean_pd"):
        assert col in metrics
    assert "fit_peak_mb" not in metrics  # память моделей замеряется в отдельных процессах, см. benchmark.py
    assert table.loc["LightGBM", "actual_default_rate"] == pytest.approx(test[C.TARGET].mean(), abs=1e-4)
    assert (C.REPORTS / "model_metrics.csv").exists()


@pytest.mark.integration
def test_calibration_does_not_change_ranking(trained):
    _, _, preds, preds_raw, test, _ = trained
    y = test[C.TARGET].astype(int)
    assert roc_auc_score(y, preds["LightGBM"]) == pytest.approx(roc_auc_score(y, preds_raw["LightGBM"]))


@pytest.mark.integration
def test_borrower_model_ignores_lendingclub_fields(trained):
    _, fitted, _, _, test, _ = trained
    X = test[C.FEATURES].head(200).copy()
    baseline = fitted["LightGBM (borrower only)"].predict_proba(X)[:, 1]
    X["int_rate"] = 30.0
    X["sub_grade"] = pd.Categorical(["A1"] * len(X), categories=X["sub_grade"].cat.categories)
    changed = fitted["LightGBM (borrower only)"].predict_proba(X)[:, 1]
    assert changed == pytest.approx(baseline)
    assert not np.allclose(fitted["LightGBM"].predict_proba(X)[:, 1],
                           fitted["LightGBM"].predict_proba(test[C.FEATURES].head(200))[:, 1])


@pytest.mark.integration
def test_registry_and_scoring_roundtrip(trained):
    registry = load_json(C.MODELS / "registry.json")
    assert {"file", "threshold", "platt_a", "platt_b"} <= set(registry["LightGBM"])
    assert load_json(C.MODELS / "metadata.json")["train_rows"] > 0
    result = predict_pd(APPLICATION)
    assert set(result) == {"LightGBM", "Logistic Regression", "LightGBM (borrower only)"}
    for res in result.values():
        assert 0 < res["pd"] < 1
        assert res["decision"] in {"отказ", "одобрить"}
    risky = predict_pd({**APPLICATION, "sub_grade": "E5", "int_rate": 28.0, "fico": 620, "dti": 45})
    safe = predict_pd({**APPLICATION, "sub_grade": "A1", "int_rate": 6.0, "fico": 800})
    assert risky["LightGBM"]["pd"] > safe["LightGBM"]["pd"]


@pytest.mark.integration
def test_selection_file_written(trained):
    selection = load_json(C.REPORTS / "model_selection.json")
    assert selection["selected"] in {s.name for s in model_specs() if not s.baseline}


@pytest.mark.integration
def test_tune_uses_time_series_folds(loans):
    frame = modeling_frame(loans.copy())
    fit = temporal_split(frame)["fit"]  # 2014 год в подборе не участвует: по нему выбирается модель
    best = tune(fit[C.FEATURES], fit[C.TARGET].astype(int), n_splits=2)
    assert {"Logistic Regression", "LightGBM", "LightGBM (borrower only)"} <= set(best)
    assert 0 <= best["LightGBM"]["cv_auc"] <= 1


@pytest.mark.integration
def test_censoring_sensitivity_shows_inflated_default_rate(tmp_project, loans, fast_best):
    table = censoring_sensitivity(loans.copy(), fast_best)
    assert set(table["cohort"]) == {"созревшие (основной)", "все завершённые (смещённый)"}
    matured = table[table["cohort"].str.startswith("созревшие")].iloc[0]
    assert pd.isna(matured["auc_60m"])  # в тесте созревшей когорты 60-месячных займов нет
    assert (C.REPORTS / "censoring_sensitivity.csv").exists()
