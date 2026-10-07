# Purpose: Run the whole project. It reads the contract data, fits the regression models,
# checks how well they work, and saves the results and charts in the outputs folder.
#
# How to run it:
#   python src/run_model.py --data path/to/All_Contracts_PrimeTransactions_....csv
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (AWARD_VALUE, build_award_table, filter_awards, get_input_ranges,
                      make_x_and_y)
from ols import OLS
import plots


def get_scores(actual, predicted) -> dict:
    """
    Score a set of predictions
    (actual and predicted are on the log scale, the typical miss is turned back into a factor)
    """
    errors = np.asarray(actual) - np.asarray(predicted)
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse_log": float(np.sqrt(np.mean(errors ** 2))),
        "mae_log": float(np.mean(np.abs(errors))),
        # half of the predictions are within this factor of the real value (3.5 = off by 3.5x)
        "typical_factor_error": float(np.exp(np.median(np.abs(errors)))),
    }


def main() -> None:
    """
    Run the project from start to finish

    Fits OLS, Ridge, Lasso and gradient boosting, runs the checks on the OLS model,
    and saves the tables and charts in the output folder.
    """
    # settings that can be changed when running the file
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)   # path to the csv file
    parser.add_argument("--out", default="outputs")   # folder for the results
    parser.add_argument("--seed", type=int, default=42)   # keeps the random split the same each run
    parser.add_argument("--test-size", type=float, default=0.2)   # share of awards held out for testing
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the data and get it ready for the model
    awards = build_award_table(args.data)
    model_awards, award_counts = filter_awards(awards)
    X, y, reference_levels = make_x_and_y(model_awards)
    print(f"Awards: {award_counts}")
    print(f"Model data: {X.shape[0]:,} awards x {X.shape[1]} predictors")

    # 2. Split into training and test awards (the test awards are hidden while fitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)

    # 3. Fit the models
    ols_model = OLS(X.columns).fit(X_train, y_train)

    # ridge and lasso need the columns on the same scale, so the scaler goes first
    ridge_model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X_train, y_train)
    lasso_model = make_pipeline(
        StandardScaler(), LassoCV(cv=5, random_state=args.seed, max_iter=20000)).fit(X_train, y_train)

    # the gradient boosting model is only a benchmark, it isn't easy to explain
    boosting_model = HistGradientBoostingRegressor(random_state=args.seed).fit(X_train, y_train)

    # 4. Predict the test awards (the baseline just guesses the average every time)
    predictions = {
        "Mean-only baseline": np.full(len(y_test), y_train.mean()),
        "OLS": ols_model.predict(X_test),
        "Ridge": ridge_model.predict(X_test),
        "Lasso": lasso_model.predict(X_test),
        "Gradient boosting (benchmark)": boosting_model.predict(X_test),
    }

    # 5. Score every model on the test awards
    test_scores = {name: get_scores(y_test, predicted) for name, predicted in predictions.items()}

    # the dashboard also needs the slider ranges and how far off the OLS predictions usually are
    ols_errors = np.abs(y_test.to_numpy() - predictions["OLS"])
    model_inputs = get_input_ranges(model_awards)
    model_inputs["error_factor_half"] = float(np.exp(np.median(ols_errors)))   # half are within this factor
    model_inputs["error_factor_90"] = float(np.exp(np.quantile(ols_errors, 0.9)))   # nine in ten are

    # cross validation on the training awards, to see if the OLS score is steady
    cv_scores = cross_val_score(LinearRegression(), X_train, y_train, scoring="r2",
                                cv=KFold(5, shuffle=True, random_state=args.seed))

    # 6. Check the OLS against scikit-learn (the coefficients should match almost exactly)
    sklearn_model = LinearRegression().fit(X_train, y_train)
    sklearn_coefs = np.append(sklearn_model.intercept_, sklearn_model.coef_)
    max_coef_difference = float(np.max(np.abs(sklearn_coefs - ols_model.coefs)))

    # how much penalty ridge and lasso picked, and how many columns lasso kept
    lasso_coefs = lasso_model.named_steps["lassocv"].coef_
    penalty_info = {
        "ridge_alpha": float(ridge_model.named_steps["ridgecv"].alpha_),
        "lasso_alpha": float(lasso_model.named_steps["lassocv"].alpha_),
        "lasso_nonzero": int((lasso_coefs != 0).sum()),
        "lasso_total": int(len(lasso_coefs)),
    }

    # 7. Checks on the OLS model (using the training awards)
    vif_values = OLS.vif(X_train)
    cooks_values = ols_model.cooks_distance()
    diagnostics = {
        "train_r2": float(ols_model.r2), "train_adj_r2": float(ols_model.adj_r2),
        "n_train": int(ols_model.n_rows),
        "breusch_pagan": ols_model.breusch_pagan(), "jarque_bera": ols_model.jarque_bera(),
        "vif_max": float(vif_values.max()), "vif_max_term": str(vif_values.idxmax()),
        "n_vif_above_10": int((vif_values > 10).sum()),
        # a common cutoff: flag awards with a Cook's distance above 4/n (they may change the model a lot)
        "n_cooks_above_4_over_n": int((cooks_values > 4 / ols_model.n_rows).sum()),
        "max_cooks": float(cooks_values.max()),
        "max_leverage": float(ols_model.leverage.max()),
    }

    # 8. Save the tables
    coef_table = ols_model.coef_table()

    # % effect only makes sense for the 0/1 columns, so the log columns are left blank
    is_zero_one = coef_table["term"].str.contains("=") | coef_table["term"].isin(
        ["offers_missing", "foreign_entity"])
    coef_table["pct_effect"] = np.where(
        is_zero_one, (np.exp(coef_table["coef"]) - 1) * 100, np.nan)
    coef_table["reference_level"] = ""
    for column, reference in reference_levels.items():
        is_this_category = coef_table["term"].str.startswith(column + "=")
        coef_table.loc[is_this_category, "reference_level"] = reference
    coef_table.to_csv(output_dir / "coefficients.csv", index=False)
    vif_values.to_csv(output_dir / "vif.csv")

    metrics = {
        "award_counts": award_counts, "n_predictors": int(X.shape[1]), "seed": args.seed,
        "test_scores": test_scores, "ols_cv_r2_mean": float(cv_scores.mean()),
        "ols_cv_r2_sd": float(cv_scores.std()),
        "ols_vs_sklearn_max_abs_coef_diff": max_coef_difference, "regularization": penalty_info,
        "diagnostics": diagnostics, "reference_levels": reference_levels,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "model_inputs.json").write_text(json.dumps(model_inputs, indent=2))

    # 9. Save the charts
    plots.plot_target_distribution(model_awards[AWARD_VALUE].to_numpy(),
                                   output_dir / "target_distribution.png")
    plots.plot_residuals(ols_model.fitted, ols_model.resid,
                         output_dir / "residual_diagnostics.png")
    plots.plot_predicted_vs_actual(y_test.to_numpy(), predictions["OLS"],
                                   test_scores["OLS"]["r2"],
                                   output_dir / "predicted_vs_actual.png")
    plots.plot_effects(coef_table, output_dir / "coefficient_effects.png")

    # 10. Print a summary
    print("\nHeld-out test performance (log-dollar scale)")
    print(pd.DataFrame(test_scores).T.round(3).to_string())
    print(f"\nOLS 5-fold CV R2 on training set: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"OLS vs scikit-learn max |coef diff|: {max_coef_difference:.2e}")
    print(f"Regularization: {penalty_info}")
    print("\nDiagnostics:", json.dumps(diagnostics, indent=2))
    print("\nTop VIFs:\n", vif_values.head(6).round(2).to_string())
    print("\nLargest effects (|t|), % difference in award value vs reference:")
    category_effects = coef_table[coef_table["term"].str.contains("=")].copy()
    category_effects["abs_t"] = category_effects["t"].abs()
    category_effects = category_effects.sort_values("abs_t", ascending=False).head(12)
    print(category_effects[["term", "pct_effect", "t", "reference_level"]].round(1).to_string(index=False))
    print("\nNumeric terms:")
    print(coef_table[~coef_table["term"].str.contains("=")][
        ["term", "coef", "se_hc3", "t", "p_value"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
