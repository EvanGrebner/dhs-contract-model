# Purpose: The Streamlit dashboard for the project. It reads the results saved in the outputs
# folder (run src/run_model.py first) and has four pages: an overview, the effect of each
# category on award value, an estimator for an imaginary award, and the model checks.
#
# How to run it:
#   streamlit run app.py
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# so python can find the files in the src folder
sys.path.insert(0, str(Path(__file__).parent / "src"))
from estimator import Estimator, format_dollars
from plots import CATEGORY_LABELS, make_breakdown_figure, make_category_effects_figure

OUTPUT_DIR = Path(__file__).parent / "outputs"

# a short explanation under each category so the names make sense
CATEGORY_HELP = {
    "sub_agency": "The DHS component that awarded the contract (CBP, FEMA, TSA, USCG, ...).",
    "award_type": "The kind of contract: delivery order, purchase order, definitive contract or BPA call.",
    "pricing": "How the government pays: a fixed price, cost reimbursement, or time and materials.",
    "competition": "How much competition there was when the contract was awarded.",
    "set_aside": "Whether the contract was reserved for a type of business (small business, 8(a), ...).",
    "business_size": "Whether the contracting officer decided the winning company is small.",
    "pbsa": "Whether the contract pays for results (performance-based) instead of effort.",
    "sector": "The type of work, from the first two digits of the NAICS industry code.",
}


def get_label(column: str) -> str:
    """Get the readable name of a category column"""
    return CATEGORY_LABELS[column]


def safe(text: str) -> str:
    """Put a backslash in front of dollar signs so streamlit doesn't treat them as math"""
    return text.replace("$", "\\$")


@st.cache_resource
def load_estimator():
    """Load the saved results once so each click doesn't reread the files"""
    return Estimator(OUTPUT_DIR)


def show_overview(estimator):
    """Show the main numbers and the two overview charts"""
    counts = estimator.metrics["award_counts"]
    ols_scores = estimator.metrics["test_scores"]["OLS"]
    boosting_scores = estimator.metrics["test_scores"]["Gradient boosting (benchmark)"]

    st.title("What drives the value of DHS contracts?")
    st.write(
        "This dashboard looks at what goes along with the value of Department of Homeland "
        "Security contract awards. It uses public FY2025 data from USAspending.gov, and only "
        "things that are known when an award starts (the sub-agency, how it is priced, how much "
        "competition there was, and so on).")

    # 1. Show the main numbers
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Awards analyzed", f"{counts['after_value_at_least_1_dollar']:,}")
    col2.metric("R² on new awards", f"{ols_scores['r2']:.2f}",
                help="How much of the differences between award values the model explains, on awards it never "
                     "saw. 1.00 would be perfect and 0 means no better than guessing the average.")
    col3.metric("Typical miss", f"{ols_scores['typical_factor_error']:.1f}x",
                help="Half of the estimates are within this many times of the real value "
                     "(3.5x means up to 3.5 times too high or too low).")
    col4.metric("Benchmark R²", f"{boosting_scores['r2']:.2f}",
                help="The score of a harder to explain model (gradient boosting). It shows how much the "
                     "simple model leaves out.")

    # 2. Show the charts
    st.subheader("Award values are very uneven")
    st.write(
        "A few huge awards make the dollar scale hard to read (left), so the model works with the "
        "log of the value instead (right).")
    st.image(str(OUTPUT_DIR / "target_distribution.png"))

    st.subheader("How close are the predictions?")
    st.write(
        "Each hexagon is a group of awards the model had not seen. The closer they sit to the "
        "line, the better the prediction. There is a lot of spread because the data doesn't say "
        "what is being bought.")
    st.image(str(OUTPUT_DIR / "predicted_vs_actual.png"))


def show_effects(estimator):
    """Show how each level of a category compares to its reference level"""
    st.title("What drives award value")
    st.write(
        "Pick a category. Each dot shows how awards at that level compare to awards at the "
        "reference level (the level everything else is compared to), with the other factors "
        "held the same. A line that crosses 0 means the difference could just be chance.")

    # 1. Pick the category and draw its chart
    column = st.selectbox("Category", list(estimator.levels),
                          format_func=get_label)
    st.caption(CATEGORY_HELP[column])
    reference_level = estimator.reference_levels[column]
    effects = estimator.category_effects(column)
    st.pyplot(make_category_effects_figure(effects, CATEGORY_LABELS[column], reference_level))

    # 2. Say the biggest difference in plain words
    biggest = effects.loc[effects["effect_pct"].abs().idxmax()]
    direction = "larger" if biggest["effect_pct"] > 0 else "smaller"
    st.write(safe(
        f"The biggest difference is **{biggest['level']}**: those awards are about "
        f"**{abs(biggest['effect_pct']):.0f}% {direction}** than {reference_level} awards, on average."))

    # 3. Show the table behind the chart
    table = effects.rename(columns={
        "level": "Level", "effect_pct": "% difference", "effect_low": "Low (95%)",
        "effect_high": "High (95%)", "clear_difference": "Clearly different?"})
    table["Clearly different?"] = table["Clearly different?"].map({True: "Yes", False: "No"})
    st.dataframe(table.round(0), hide_index=True)

    # 4. Show the number columns (they work differently, so they get their own section)
    st.subheader("Contract length, offers and foreign ownership")
    st.write(
        "These are not categories. For the first two, a 10% change in the number goes with the "
        "% change in award value shown here (roughly).")
    coefs = estimator.coefs
    duration_change = (1.1 ** coefs["log_duration_days"] - 1) * 100
    offers_change = (1.1 ** coefs["log_offers"] - 1) * 100
    foreign_change = (np.exp(coefs["foreign_entity"]) - 1) * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("10% longer contract", f"{duration_change:+.1f}%")
    col2.metric("10% more offers", f"{offers_change:+.1f}%")
    col3.metric("Foreign-owned company (vs not)", f"{foreign_change:+.0f}%")


def show_estimator(estimator):
    """Let the user build an imaginary award and see the estimated value"""
    st.title("Estimate an award")
    st.write(
        "Choose the features of an imaginary contract award. The model estimates what a typical "
        "award like it is worth. It is a rough guide, so look at the ranges and not just the "
        "single number.")
    defaults = estimator.default_choices()
    inputs = estimator.inputs
    choices = {}

    # 1. Add a dropdown for each category (split over two columns)
    left, right = st.columns(2)
    for position, column in enumerate(estimator.levels):
        container = left if position % 2 == 0 else right
        levels = estimator.levels[column]
        choices[column] = container.selectbox(
            CATEGORY_LABELS[column], levels, index=levels.index(defaults[column]),
            help=CATEGORY_HELP[column])

    # 2. Add the number choices (the sliders stop at the same caps the model used)
    choices["duration_days"] = left.slider(
        "Contract length (days)", 0, int(inputs["duration_days_cap"]), defaults["duration_days"])
    choices["offers_unknown"] = right.checkbox(
        "Number of offers unknown", help="About 30% of awards don't report this.")
    choices["offers"] = right.slider(
        "Number of offers", 1, int(inputs["offers_cap"]), defaults["offers"],
        disabled=choices["offers_unknown"])
    choices["foreign_entity"] = right.checkbox("Foreign-owned company")

    # 3. Show the estimate and the two ranges around it
    result = estimator.estimate(choices)
    st.divider()
    st.metric("Estimated award value", format_dollars(result["dollars"]),
              help="This is the middle (median) estimate, not the average.")
    st.write(safe(
        f"Half of awards like this land between **{format_dollars(result['half_low'])}** and "
        f"**{format_dollars(result['half_high'])}**."))
    st.write(safe(
        f"Nine in ten land between **{format_dollars(result['ninety_low'])}** and "
        f"**{format_dollars(result['ninety_high'])}**."))

    # 4. Show what moved the estimate
    st.subheader("What moves this estimate")
    st.write(
        "Each bar is how much one choice changes the estimate compared to a starting award "
        "(the first option in each dropdown, typical length, typical number of offers).")
    breakdown = estimator.breakdown(choices, CATEGORY_LABELS)
    st.pyplot(make_breakdown_figure(breakdown))
    table = breakdown[["input", "choice", "multiplier", "pct_change"]].rename(columns={
        "input": "Choice", "choice": "Your pick", "multiplier": "Multiplier",
        "pct_change": "% change"})
    st.dataframe(table.round(2), hide_index=True)

    with st.expander("How to read this estimate"):
        st.write(safe(
            "- The model was fit on FY2025 DHS awards with a place of performance in the US.\n"
            "- It shows what goes along with award value, not what would happen if a buying "
            "office changed one choice.\n"
            "- The model can't see what is being bought, so the ranges are wide on purpose. "
            "The ranges come from how far off the model was on awards it had not seen.\n"
            "- The estimate is the middle value, so the average for awards like this would be higher."))


def show_checks(estimator):
    """Show the checks on the model in plain words"""
    metrics = estimator.metrics
    diagnostics = metrics["diagnostics"]
    breusch_pagan = diagnostics["breusch_pagan"]
    jarque_bera = diagnostics["jarque_bera"]
    st.title("Model checks")
    st.write(
        "These checks show how far the OLS model can be trusted. The model is a linear regression "
        "on the log of the award value.")

    # 1. Show the main numbers
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² on training awards", f"{diagnostics['train_r2']:.2f}")
    col2.metric("R² on new awards", f"{metrics['test_scores']['OLS']['r2']:.2f}")
    col3.metric("R² across 5 folds", f"{metrics['ols_cv_r2_mean']:.2f}",
                help="Average over 5 splits of the training awards, to see if the score is steady.")
    col4.metric("Highest VIF", f"{diagnostics['vif_max']:.1f}",
                help="VIF shows how much a predictor overlaps with the others. Above 10 means too much overlap.")

    # 2. Explain what each check says
    st.subheader("What the checks say")
    if diagnostics["n_vif_above_10"] == 0:
        overlap = "No predictor has a VIF above 10, so the predictors don't repeat each other too much."
    else:
        overlap = f"{diagnostics['n_vif_above_10']} predictors have a VIF above 10, so some of them overlap."
    # show tiny p-values as "p < 0.001" instead of a long number
    if breusch_pagan["p_value"] < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {breusch_pagan['p_value']:.3f}"
    st.write(
        f"- **Overlap between predictors:** {overlap}\n"
        "- **Spread of the errors:** the model misses by more on some awards than on others "
        f"(Breusch-Pagan test, {p_text}). Because of that, the "
        "standard errors use a method (called HC3) that still works when the misses are uneven.\n"
        "- **Shape of the errors:** they are close to a bell shape. Skew is "
        f"{jarque_bera['skew']:.2f} and kurtosis is {jarque_bera['excess_kurtosis']:.2f} "
        "(a perfect bell curve has 0 for both), so a few awards are missed by more than usual.\n"
        f"- **Influence:** the largest Cook's distance is {diagnostics['max_cooks']:.3f}. "
        "Cook's distance shows how much one award changes the model, so no single award "
        "drives the results.")
    st.image(str(OUTPUT_DIR / "residual_diagnostics.png"))
    st.caption("In the charts, a residual is how far off the model was (actual minus predicted).")

    # 3. List what the model can't do
    st.subheader("Limits to keep in mind")
    st.write(
        "- The model shows patterns across awards, not cause and effect.\n"
        "- It only covers one fiscal year (FY2025) and awards with a place of performance in the US.\n"
        "- It can't see what is being bought, only the sector, which limits how accurate it can be.\n"
        "- Many awards come from the same parent contract or company, and the test awards were "
        "picked at random, so some test awards look a lot like awards the model trained on. "
        "The test score may be a little too high.")


st.set_page_config(page_title="DHS Contract Value Explorer", layout="wide")

# stop with a message if the model hasn't been run yet
try:
    estimator = load_estimator()
except FileNotFoundError:
    st.error("The results files weren't found. Run src/run_model.py first.")
    st.stop()

page = st.sidebar.radio("Page", [
    "Overview", "What drives award value", "Estimate an award", "Model checks"])
st.sidebar.caption("Data: USAspending.gov, DHS contract awards for FY2025")

# show the page that was picked
if page == "Overview":
    show_overview(estimator)
elif page == "What drives award value":
    show_effects(estimator)
elif page == "Estimate an award":
    show_estimator(estimator)
else:
    show_checks(estimator)
