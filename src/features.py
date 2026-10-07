# Purpose: Turn the raw USAspending contract data into one row per award, then build
# the columns (features) that the regression model uses.
# The raw file has one row for every change (modification) to a contract, so the same
# award shows up many times. We combine those rows first. We also only keep things
# that are known when the award starts, so the model isn't using information from the future.
from __future__ import annotations

import numpy as np
import pandas as pd

# column names we use a lot
AWARD_KEY = "contract_award_unique_key"
AWARD_VALUE = "current_total_value_of_award"   # what the model is trying to predict

# only read the columns we need so the file loads faster
COLUMNS_TO_READ = [
    AWARD_KEY, "action_date", AWARD_VALUE, "awarding_sub_agency_name", "award_type",
    "type_of_contract_pricing", "extent_competed", "type_of_set_aside",
    "contracting_officers_determination_of_business_size",
    "performance_based_service_acquisition", "naics_code", "number_of_offers_received",
    "period_of_performance_start_date", "period_of_performance_current_end_date",
    "primary_place_of_performance_country_name", "domestic_or_foreign_entity",
]

# short names for the DHS sub-agencies so the tables and charts are easier to read
AGENCY_ABBREVIATIONS = {
    "U.S. Coast Guard": "USCG",
    "Federal Emergency Management Agency": "FEMA",
    "U.S. Customs and Border Protection": "CBP",
    "Office of Procurement Operations": "OPO (HQ)",
    "U.S. Immigration and Customs Enforcement": "ICE",
    "Transportation Security Administration": "TSA",
    "Federal Law Enforcement Training Center": "FLETC",
    "U.S. Secret Service": "USSS",
    "U.S. Citizenship and Immigration Services": "USCIS",
    "Office of the Inspector General": "OIG",
}

# first two digits of the NAICS code and the type of work they stand for
NAICS_SECTORS = {
    "54": "Professional & technical svcs", "31": "Manufacturing", "32": "Manufacturing",
    "33": "Manufacturing", "56": "Admin & support svcs", "23": "Construction",
    "51": "Information", "53": "Real estate & leasing", "81": "Other services",
    "48": "Transportation", "49": "Transportation", "72": "Accommodation & food",
    "61": "Educational svcs", "42": "Wholesale trade", "44": "Retail trade",
    "45": "Retail trade", "62": "Health care", "92": "Public administration",
}

# columns that are categories (these get turned into 0/1 columns later)
CATEGORY_COLUMNS = ["sub_agency", "award_type", "pricing", "competition", "set_aside",
                    "business_size", "pbsa", "sector"]

# columns that are already numbers
NUMBER_COLUMNS = ["log_duration_days", "log_offers", "offers_missing", "foreign_entity"]

# the level left out of each category, so every other level is compared to it
# it's the most common level unless it is listed here ("No set-aside" is easier to explain
# than "not reported")
REFERENCE_OVERRIDE = {"set_aside": "No set-aside"}

# the name given to the small levels (under 1% of awards) that get combined together
RARE_LEVEL_LABEL = {
    "business_size": "Unknown", "sector": "Other sector", "set_aside": "Other set-aside",
    "pricing": "Other / unreported", "competition": "Other competed",
}


def build_award_table(path: str) -> pd.DataFrame:
    """
    Read the raw csv and combine it into one row per award

    Args:
        path (str): The file path for the USAspending csv

    Returns:
        A data frame with one row for each award
    """
    # 1. Read the file and turn the date columns into real dates
    transactions = pd.read_csv(path, usecols=COLUMNS_TO_READ, low_memory=False)
    for column in ["action_date", "period_of_performance_start_date",
                   "period_of_performance_current_end_date"]:
        transactions[column] = pd.to_datetime(transactions[column], errors="coerce")

    # 2. Put each award's rows in date order (rows with the same date stay in file order)
    transactions = transactions.sort_values([AWARD_KEY, "action_date"], kind="mergesort")
    by_award = transactions.groupby(AWARD_KEY, sort=False)

    # 3. Combine the rows for each award
    awards = by_award.first()   # first value of each column = how the award started
    awards[AWARD_VALUE] = by_award[AWARD_VALUE].last()   # value and end date use the latest change
    awards["period_of_performance_current_end_date"] = by_award[
        "period_of_performance_current_end_date"].last()
    awards["n_transactions"] = by_award.size()   # not used in the model, saved for later
    return awards.reset_index()


def filter_awards(awards: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Keep only the awards we want to model

    Args:
        awards (data frame): The award table from build_award_table()

    Returns:
        The awards that are kept, and a dictionary with how many awards are left
        after each filter
    """
    award_counts = {"awards_total": len(awards)}

    # 1. US place of performance only
    us_awards = awards[awards["primary_place_of_performance_country_name"] == "UNITED STATES"]
    award_counts["after_us_place_of_performance"] = len(us_awards)

    # 2. Drop awards under $1 (they look like data errors, and we take the log of the value later)
    kept_awards = us_awards[us_awards[AWARD_VALUE] >= 1.0]
    award_counts["after_value_at_least_1_dollar"] = len(kept_awards)
    return kept_awards.reset_index(drop=True), award_counts


def merge_rare_levels(column: pd.Series, min_share: float = 0.01,
                      new_label: str = "Other") -> pd.Series:
    """
    Combine the small levels of a category into one level
    (levels under min_share of the awards get renamed to new_label)
    """
    share = column.value_counts(normalize=True)
    levels_to_keep = share[share >= min_share].index
    return column.where(column.isin(levels_to_keep), new_label)


def clean_text(column: pd.Series) -> pd.Series:
    """Make a text column upper case and fill the missing values with an empty string"""
    return column.fillna("").str.upper()


# The group_ functions below turn the long text from USAspending into a few simple levels.
# Each one takes the text of one award. The checks run from top to bottom and the first one
# that matches is used.

def group_pricing(text: str) -> str:
    """Group one contract pricing type into a level"""
    if "TIME AND MATERIALS" in text or "LABOR HOURS" in text:
        return "T&M / labor hours"
    if text.startswith("COST"):
        return "Cost-reimbursable"
    if text.startswith("FIRM FIXED"):
        return "Firm fixed price"
    if text.startswith("FIXED PRICE"):
        return "Other fixed-price"
    return "Other / unreported"


def group_competition(text: str) -> str:
    """Group how much competition there was into a level"""
    if text == "":
        return "Unreported"
    if "NOT COMPETED" in text or "NON-COMPETITIVE" in text or "NOT AVAILABLE" in text:
        return "Not competed"
    if "SAP" in text and "NOT" not in text:
        return "Competed (simplified)"
    if "FULL AND OPEN" in text:
        return "Full & open"
    return "Other competed"


def group_set_aside(text: str) -> str:
    """Group one set-aside type into a level"""
    if text == "":
        return "Not reported"
    if "NO SET ASIDE" in text:
        return "No set-aside"
    if "WOMEN" in text:
        return "Woman-owned"
    if "SERVICE DISABLED" in text or "SDVOSB" in text:
        return "SDVOSB"
    if "HUBZONE" in text:
        return "HUBZone"
    if "8(A)" in text or "8A" in text:
        return "8(a)"
    if "SMALL BUSINESS SET ASIDE" in text or "RESERVED FOR SMALL BUSINESS" in text:
        return "Small business"
    return "Other set-aside"


def group_yes_no(text: str) -> str:
    """Turn one yes/no text into Yes, No, or N/A"""
    if text.startswith("YES"):
        return "Yes"
    if text.startswith("NO"):
        return "No"
    return "N/A"


def group_sector(naics_code: pd.Series) -> pd.Series:
    """Turn the NAICS code into a type of work using the first two digits"""
    code = pd.to_numeric(naics_code, errors="coerce")
    first_two_digits = code.astype("Int64").astype(str).str[:2]   # 541519 -> "54"
    sector = first_two_digits.map(NAICS_SECTORS).fillna("Other sector")
    sector[code.isna()] = "Unknown"
    return sector


def get_duration_days(awards: pd.DataFrame) -> pd.Series:
    """Get the length of each contract in days (a negative length is counted as 0)"""
    return (awards["period_of_performance_current_end_date"]
            - awards["period_of_performance_start_date"]).dt.days.clip(lower=0)


def get_input_ranges(awards: pd.DataFrame) -> dict:
    """
    Get the caps and typical values for the two number columns
    (the dashboard uses these for its sliders and its starting values)
    """
    offers = pd.to_numeric(awards["number_of_offers_received"], errors="coerce")
    return {
        "duration_days_cap": float(get_duration_days(awards).quantile(0.995)),
        "duration_days_median": float(get_duration_days(awards).median()),
        "offers_cap": float(offers.quantile(0.99)),
        "offers_median": float(offers.median()),
    }


def make_features(awards: pd.DataFrame) -> pd.DataFrame:
    """Build the columns the model uses from the filtered award table (one row per award)"""
    features = pd.DataFrame(index=awards.index)

    # 1. Build the category columns
    features["sub_agency"] = awards["awarding_sub_agency_name"].map(AGENCY_ABBREVIATIONS).fillna(
        awards["awarding_sub_agency_name"])
    features["award_type"] = awards["award_type"].str.title()
    features["pricing"] = clean_text(awards["type_of_contract_pricing"]).map(group_pricing)
    features["competition"] = clean_text(awards["extent_competed"]).map(group_competition)
    features["set_aside"] = clean_text(awards["type_of_set_aside"]).map(group_set_aside)
    features["business_size"] = awards[
        "contracting_officers_determination_of_business_size"].str.title().fillna("Unknown")
    features["pbsa"] = clean_text(awards["performance_based_service_acquisition"]).map(group_yes_no)
    features["sector"] = group_sector(awards["naics_code"])

    # 2. Get the length of the contract in days
    # very long contracts are capped at the 99.5th percentile so a few don't pull the model around
    duration_days = get_duration_days(awards)
    duration_days = duration_days.clip(
        upper=duration_days.quantile(0.995)).fillna(duration_days.median())
    features["log_duration_days"] = np.log1p(duration_days)   # log1p so a 0 day contract still works

    # 3. Get the number of offers
    # about 30% are missing (often orders under a bigger contract), so a missing value
    # is counted as 1 offer and a 0/1 column keeps track of which ones were missing
    offers = pd.to_numeric(awards["number_of_offers_received"], errors="coerce")
    features["offers_missing"] = offers.isna().astype(float)
    offers_cap = offers.quantile(0.99)
    features["log_offers"] = np.log(offers.fillna(1).clip(lower=1, upper=offers_cap))

    # 4. Mark foreign owned companies (1 if foreign owned, 0 if not)
    features["foreign_entity"] = awards["domestic_or_foreign_entity"].fillna(
        "").str.upper().str.contains("FOREIGN").astype(float)

    # 5. Combine the small levels (under 1% of awards) so no 0/1 column is based on just a few awards
    for column, label in RARE_LEVEL_LABEL.items():
        features[column] = merge_rare_levels(features[column], new_label=label)
    return features


def make_x_and_y(awards: pd.DataFrame):
    """
    Build the predictors (X) and the target (y) for the model

    Args:
        awards (data frame): The filtered awards from filter_awards()

    Returns:
        X (the predictor columns), y (the log of award value), and a dictionary with
        the reference level that was left out of each category
    """
    features = make_features(awards)
    y = np.log(awards[AWARD_VALUE].astype(float))

    # turn each category into 0/1 columns, and leave out the reference level
    pieces = [features[NUMBER_COLUMNS].astype(float)]
    reference_levels = {}
    for column in CATEGORY_COLUMNS:
        reference = REFERENCE_OVERRIDE.get(column, features[column].value_counts().idxmax())
        reference_levels[column] = reference
        dummies = pd.get_dummies(features[column], prefix=column, prefix_sep="=", dtype=float)
        pieces.append(dummies.drop(columns=f"{column}={reference}"))
    return pd.concat(pieces, axis=1), y, reference_levels
