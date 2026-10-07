# Purpose: Turn the saved model results into estimates for the dashboard. It doesn't need the
# raw contract data, only coefficients.csv, metrics.json and model_inputs.json from the
# outputs folder. check_estimator.py makes sure it gives the same answers as the model.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# names shown for the number inputs (the category names come from plots.CATEGORY_LABELS)
NUMBER_LABELS = {
    "duration": "Contract length", "offers": "Number of offers", "foreign": "Foreign-owned company",
}


def format_dollars(value: float) -> str:
    """Write a dollar amount in short form, like $450K or $1.2M"""
    for size, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(value) >= size * 0.9995:   # so $999,999 shows as $1.0M and not $1,000.0K
            return f"${value / size:,.1f}{suffix}"
    return f"${value:,.0f}"


class Estimator:
    """Estimate the value of an award from the saved model results"""

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        self.coef_table = pd.read_csv(output_dir / "coefficients.csv")
        self.metrics = json.loads((output_dir / "metrics.json").read_text())
        self.inputs = json.loads((output_dir / "model_inputs.json").read_text())
        self.coefs = dict(zip(self.coef_table["term"], self.coef_table["coef"]))
        self.reference_levels = self.metrics["reference_levels"]

        # the levels of each category, with the reference level first
        self.levels = {}
        for column, reference in self.reference_levels.items():
            others = [term.split("=", 1)[1] for term in self.coefs if term.startswith(column + "=")]
            self.levels[column] = [reference] + sorted(others)

    def default_choices(self) -> dict:
        """Start from the reference level of every category and the typical numbers"""
        choices = dict(self.reference_levels)
        choices["duration_days"] = int(round(self.inputs["duration_days_median"]))
        choices["offers"] = max(1, int(round(self.inputs["offers_median"])))
        choices["offers_unknown"] = False
        choices["foreign_entity"] = False
        return choices

    def feature_values(self, choices: dict) -> dict:
        """Turn the choices into the same columns the model was fit on"""
        # 1. Cap and log the contract length (the same way as in features.py)
        duration_days = min(max(choices["duration_days"], 0), self.inputs["duration_days_cap"])
        values = {"log_duration_days": float(np.log1p(duration_days))}

        # 2. Cap and log the number of offers (unknown = 1 offer plus the missing flag, like the model)
        if choices["offers_unknown"]:
            values["offers_missing"], values["log_offers"] = 1.0, 0.0
        else:
            offers = min(max(choices["offers"], 1), self.inputs["offers_cap"])
            values["offers_missing"], values["log_offers"] = 0.0, float(np.log(offers))
        values["foreign_entity"] = float(choices["foreign_entity"])

        # 3. Set each category column to 1 for the level that was picked and 0 for the rest
        for term in self.coefs:
            if "=" in term:
                column, level = term.split("=", 1)
                values[term] = float(choices[column] == level)
        return values

    def predict_log(self, choices: dict) -> float:
        """Predict the log of the award value"""
        values = self.feature_values(choices)
        return self.coefs["Intercept"] + sum(
            self.coefs[term] * value for term, value in values.items())

    def estimate(self, choices: dict) -> dict:
        """
        Estimate the award value in dollars, with two ranges around it
        (half of the test awards were within the first range, nine in ten within the second)
        """
        dollars = float(np.exp(self.predict_log(choices)))
        half, ninety = self.inputs["error_factor_half"], self.inputs["error_factor_90"]
        return {
            "dollars": dollars,
            "half_low": dollars / half, "half_high": dollars * half,
            "ninety_low": dollars / ninety, "ninety_high": dollars * ninety,
        }

    def breakdown(self, choices: dict, category_labels: dict) -> pd.DataFrame:
        """
        Show how much each choice moves the estimate compared to the starting award
        (the starting award uses the reference levels and the typical numbers)
        """
        values = self.feature_values(choices)
        start_values = self.feature_values(self.default_choices())

        # 1. Group the model terms by the choice they belong to
        groups = {}
        for column in self.levels:
            groups[column] = [f"{column}={level}" for level in self.levels[column][1:]]
        groups["duration"] = ["log_duration_days"]
        groups["offers"] = ["log_offers", "offers_missing"]
        groups["foreign"] = ["foreign_entity"]

        # 2. Write each choice the way the user sees it
        shown = {column: choices[column] for column in self.levels}
        shown["duration"] = f"{choices['duration_days']:,} days"
        shown["offers"] = "Unknown" if choices["offers_unknown"] else str(choices["offers"])
        shown["foreign"] = "Yes" if choices["foreign_entity"] else "No"
        labels = dict(category_labels)
        labels.update(NUMBER_LABELS)

        # 3. Add up the change in the log value for each choice, then turn it into a % change
        rows = []
        for name, terms in groups.items():
            log_change = sum(
                self.coefs[term] * (values[term] - start_values[term]) for term in terms)
            multiplier = float(np.exp(log_change))
            rows.append({"input": labels[name], "choice": shown[name], "log_change": log_change,
                         "multiplier": multiplier, "pct_change": (multiplier - 1) * 100})

        # 4. Put the biggest changes first
        result = pd.DataFrame(rows)
        order = result["log_change"].abs().sort_values(ascending=False).index
        return result.loc[order].reset_index(drop=True)

    def category_effects(self, column: str) -> pd.DataFrame:
        """Get the % difference for each level of one category (compared to its reference level)"""
        rows = self.coef_table[self.coef_table["term"].str.startswith(column + "=")].copy()
        rows["level"] = rows["term"].str.split("=", n=1).str[1]
        rows["effect_pct"] = (np.exp(rows["coef"]) - 1) * 100
        rows["effect_low"] = (np.exp(rows["ci95_low"]) - 1) * 100
        rows["effect_high"] = (np.exp(rows["ci95_high"]) - 1) * 100
        rows["clear_difference"] = (rows["ci95_low"] > 0) | (rows["ci95_high"] < 0)   # the interval doesn't cross 0
        columns = ["level", "effect_pct", "effect_low", "effect_high", "clear_difference"]
        return rows.sort_values("effect_pct")[columns].reset_index(drop=True)
