# Purpose: Make the charts for the project. All the charts share the same colors and fonts.
# Purple is the main color and red is used for negative values (the two were picked so they
# are easy to tell apart with the common types of color blindness).
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")   # draw to files only, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# colors used in the charts
BACKGROUND = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"   # for axis labels and reference lines
GRID_COLOR = "#e6e5e1"
MAIN_COLOR = "#5b3f9a"   # purple
RED = "#e34948"
MAIN_SCALE = LinearSegmentedColormap.from_list(
    "main_scale", ["#e8e1f3", "#c4b6e2", "#9a85c8", "#5b3f9a", "#33205f"])

# divide a natural log by this to get log10 (easier to read as powers of 10)
LN_OF_10 = np.log(10)

# nicer names for the categories on the effects chart
CATEGORY_LABELS = {
    "sub_agency": "Sub-agency", "award_type": "Award type", "pricing": "Pricing",
    "competition": "Competition", "set_aside": "Set-aside", "business_size": "Business size",
    "pbsa": "Performance-based", "sector": "Sector",
}


def set_plot_style():
    """Set the colors and fonts once so every chart matches"""
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "text.color": TEXT_COLOR, "axes.labelcolor": MUTED_COLOR,
        "xtick.color": MUTED_COLOR, "ytick.color": MUTED_COLOR,
        "axes.edgecolor": GRID_COLOR, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 150,
    })


def plot_target_distribution(award_values, path):
    """Plot award values in dollars next to award values on the log scale"""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # left: plain dollars (cut off at the 99th percentile or the big awards squash everything)
    cutoff_99th = np.quantile(award_values, 0.99)
    axes[0].hist(award_values[award_values <= cutoff_99th] / 1e6, bins=50, color=MAIN_COLOR,
                 edgecolor=BACKGROUND, linewidth=0.5)
    axes[0].set(title="Award value, $ millions (top 1% left out)", xlabel="$ millions",
                ylabel="Awards")

    # right: log10 dollars, all awards (much closer to a bell shape)
    axes[1].hist(np.log10(award_values), bins=50, color=MAIN_COLOR, edgecolor=BACKGROUND,
                 linewidth=0.5)
    axes[1].set(title="Award value on a log scale (all awards)",
                xlabel="log10 of award value in dollars (5 = 100K, 6 = 1M, 7 = 10M)",
                ylabel="Awards")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(fitted, resid, path):
    """
    Plot the three residual checks for the model
    (fitted and resid are on the natural log scale, the chart converts them to log10)
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    fitted_log10, resid_log10 = fitted / LN_OF_10, resid / LN_OF_10

    # 1. Plot the residuals vs the fitted values (we want a flat line around 0)
    axes[0].scatter(fitted_log10, resid_log10, s=3, alpha=0.12, color=MAIN_COLOR, linewidths=0)

    # black line = average residual in each of 20 equal-sized groups of fitted values
    group_number = pd.qcut(fitted_log10, 20, labels=False)
    axes[0].plot([fitted_log10[group_number == i].mean() for i in range(20)],
                 [resid_log10[group_number == i].mean() for i in range(20)],
                 color=TEXT_COLOR, linewidth=1.6)
    axes[0].axhline(0, color=MUTED_COLOR, linewidth=0.8)
    axes[0].set(title="Residuals vs fitted", xlabel="Fitted log10($)",
                ylabel="Residual (log10 units)")

    # 2. Plot the Q-Q plot (points close to the line mean the residuals are close to normal)
    qq_result = stats.probplot(resid_log10, dist="norm")
    theory_quantiles, sample_quantiles = qq_result[0]   # the points
    slope, intercept, _ = qq_result[1]   # the line
    axes[1].scatter(theory_quantiles, sample_quantiles, s=3, alpha=0.25, color=MAIN_COLOR,
                    linewidths=0)
    axes[1].plot(theory_quantiles, slope * theory_quantiles + intercept, color=MUTED_COLOR,
                 linewidth=1)
    axes[1].set(title="Normal Q-Q of residuals", xlabel="Theoretical quantiles",
                ylabel="Residual quantiles")

    # 3. Plot a histogram of the residuals
    axes[2].hist(resid_log10, bins=60, color=MAIN_COLOR, edgecolor=BACKGROUND, linewidth=0.5)
    axes[2].set(title="Residual distribution", xlabel="Residual (log10 units)",
                ylabel="Awards")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_predicted_vs_actual(actual, predicted, r2, path):
    """
    Plot the predicted award values against the actual ones for the test awards
    (actual and predicted are natural logs, r2 is just shown in the title)
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    actual_log10, predicted_log10 = actual / LN_OF_10, predicted / LN_OF_10

    # hexagons instead of dots because there are too many awards to see each one
    hexbins = ax.hexbin(predicted_log10, actual_log10, gridsize=45, mincnt=1,
                        cmap=MAIN_SCALE, linewidths=0.2, bins="log")

    # the line where predicted = actual (a perfect model would sit right on it)
    low = min(actual_log10.min(), predicted_log10.min())
    high = max(actual_log10.max(), predicted_log10.max())
    ax.plot([low, high], [low, high], color=MUTED_COLOR, linewidth=1)
    ax.set(title=f"Held-out test awards: predicted vs actual (R² = {r2:.2f})",
           xlabel="Predicted log10($)", ylabel="Actual log10($)")
    color_bar = fig.colorbar(hexbins, ax=ax, pad=0.02)
    color_bar.set_label("Awards (log scale)", color=MUTED_COLOR)
    color_bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_effects(coef_table, path, n_effects=16):
    """
    Plot the strongest category effects as a % difference in award value
    (each dot is compared to the reference level of its category, the lines are 95% intervals)
    """
    set_plot_style()

    # 1. Keep the category terms and turn the coefficients into % differences
    effects = coef_table[coef_table["term"].str.contains("=")].copy()
    effects["effect_pct"] = (np.exp(effects["coef"]) - 1) * 100
    effects["effect_low"] = (np.exp(effects["ci95_low"]) - 1) * 100
    effects["effect_high"] = (np.exp(effects["ci95_high"]) - 1) * 100

    # 2. Keep the strongest effects (biggest |t|), then sort them for the chart
    effects["abs_t"] = effects["t"].abs()
    effects = effects.sort_values("abs_t", ascending=False).head(n_effects)
    effects = effects.sort_values("effect_pct")
    labels = [f"{CATEGORY_LABELS[term.split('=')[0]]}: {term.split('=', 1)[1]}"
              for term in effects["term"]]
    positions = np.arange(len(effects))
    colors = [MAIN_COLOR if value >= 0 else RED for value in effects["effect_pct"]]

    # 3. Build the chart
    fig, ax = plt.subplots(figsize=(8.6, 0.38 * len(effects) + 1.6))
    ax.hlines(positions, effects["effect_low"], effects["effect_high"], color=colors,
              linewidth=1.5)
    ax.scatter(effects["effect_pct"], positions, color=colors, s=36, zorder=3,
               edgecolor=BACKGROUND, linewidth=1.2)
    ax.axvline(0, color=MUTED_COLOR, linewidth=0.9)   # 0 = same as the reference level
    ax.set_yticks(positions, labels)
    ax.grid(axis="y", visible=False)
    ax.set(title="Strongest effects on award value (holding other factors fixed)",
           xlabel="% difference in award value vs the reference level (95% interval)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_category_effects_figure(effects, category_label, reference_level):
    """
    Make the chart for one category on the dashboard
    (each dot is a level compared to the reference level, the lines are 95% intervals)
    """
    set_plot_style()
    positions = np.arange(len(effects))
    colors = [MAIN_COLOR if value >= 0 else RED for value in effects["effect_pct"]]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(effects) + 1.6))
    ax.hlines(positions, effects["effect_low"], effects["effect_high"], color=colors,
              linewidth=1.8)
    ax.scatter(effects["effect_pct"], positions, color=colors, s=44, zorder=3,
               edgecolor=BACKGROUND, linewidth=1.2)
    ax.axvline(0, color=MUTED_COLOR, linewidth=0.9)   # 0 = same as the reference level
    ax.set_yticks(positions, effects["level"])
    ax.grid(axis="y", visible=False)
    ax.set(title=f"{category_label}: difference in award value vs {reference_level}",
           xlabel="% difference in award value (95% interval)")
    fig.tight_layout()
    return fig


def make_breakdown_figure(breakdown):
    """
    Make the chart of what moves an estimate
    (each bar is the % change in the estimate caused by one choice, compared to the starting award)
    """
    set_plot_style()
    breakdown = breakdown.iloc[::-1]   # barh draws from the bottom, so flip it to put the biggest on top
    labels = [f"{name}: {choice}" for name, choice in zip(breakdown["input"], breakdown["choice"])]
    colors = [MAIN_COLOR if value >= 0 else RED for value in breakdown["pct_change"]]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(breakdown) + 1.6))
    ax.barh(labels, breakdown["pct_change"], color=colors, height=0.6)
    ax.axvline(0, color=MUTED_COLOR, linewidth=0.9)
    ax.grid(axis="y", visible=False)
    ax.set(title="What moves this estimate (vs the starting award)",
           xlabel="% change in the estimated value")
    fig.tight_layout()
    return fig
