# Purpose: Check that the dashboard's estimator gives the same answers as the model.
# It rebuilds real awards from the raw data, runs them through the estimator, and compares the
# results to the model's own columns. The differences should be almost 0.
#
# How to run it (run src/run_model.py first):
#   python src/check_estimator.py --data path/to/All_Contracts_PrimeTransactions_....csv
from __future__ import annotations

import argparse

import numpy as np

from estimator import Estimator
from features import build_award_table, filter_awards, make_features, make_x_and_y
from plots import CATEGORY_LABELS


def main() -> None:
    """Compare the estimator to the model on a random sample of real awards"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)   # path to the csv file
    parser.add_argument("--out", default="outputs")   # folder with the saved results
    parser.add_argument("--rows", type=int, default=2000)   # how many awards to check
    args = parser.parse_args()
    estimator = Estimator(args.out)

    # 1. Rebuild the awards and the model's own columns
    model_awards, _ = filter_awards(build_award_table(args.data))
    features = make_features(model_awards)
    X, _, _ = make_x_and_y(model_awards)
    sample = X.sample(n=min(args.rows, len(X)), random_state=1).index

    # 2. Get the model's predictions (the intercept plus each column times its coefficient)
    model_coefs = np.array([estimator.coefs[column] for column in X.columns])
    model_predictions = estimator.coefs["Intercept"] + X.loc[sample].to_numpy() @ model_coefs

    # 3. Get the estimator's predictions for the same awards (it only sees the choices)
    estimator_predictions = []
    for row in sample:
        choices = {column: features.loc[row, column] for column in estimator.levels}
        choices["duration_days"] = float(np.expm1(features.loc[row, "log_duration_days"]))
        choices["offers"] = float(np.exp(features.loc[row, "log_offers"]))
        choices["offers_unknown"] = bool(features.loc[row, "offers_missing"] == 1)
        choices["foreign_entity"] = bool(features.loc[row, "foreign_entity"] == 1)
        estimator_predictions.append(estimator.predict_log(choices))
    max_difference = float(np.max(np.abs(model_predictions - np.array(estimator_predictions))))
    print(f"Awards checked: {len(sample):,}")
    print(f"Biggest difference in the predicted log value: {max_difference:.2e}")

    # 4. Check that the pieces in the breakdown add up to the estimate
    choices = estimator.default_choices()
    choices["duration_days"] = 900
    choices["foreign_entity"] = True
    breakdown = estimator.breakdown(choices, CATEGORY_LABELS)
    start_log = estimator.predict_log(estimator.default_choices())
    breakdown_difference = abs(start_log + breakdown["log_change"].sum() - estimator.predict_log(choices))
    print(f"Breakdown adds up to the estimate (difference {breakdown_difference:.2e})")

    # 5. Check that a contract longer than the cap is treated the same as one at the cap
    long_choices = dict(choices, duration_days=estimator.inputs["duration_days_cap"] * 5)
    cap_choices = dict(choices, duration_days=estimator.inputs["duration_days_cap"])
    same_as_cap = estimator.predict_log(long_choices) == estimator.predict_log(cap_choices)
    print(f"Very long contract is capped like the model does: {same_as_cap}")

    passed = max_difference < 1e-8 and breakdown_difference < 1e-8 and same_as_cap
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
