"""
Real-data-based ACS microdata analysis for hybrid tree-boosted entropy balancing.

Data: OpenIntro acs12 sample (2000 observations from the 2012 ACS). The script
uses adults age >= 18 as a finite pseudo-population, repeatedly draws biased
source samples using a nonlinear covariate-dependent selection rule, and compares
unweighted, main-effect EB, fixed pairwise EB, and EB-offset hybrid
tree-boosted EB.  The main-offset hybrid uses the fitted main-effect EB dual
parameters as the base offset; the pairwise-offset hybrid uses the fitted
pairwise EB dual parameters as the base offset and projects back to the same
main+pairwise hard constraints after each tree correction.

The data are downloaded separately to /mnt/data/acs12.csv.
"""

import os
import numpy as np
import pandas as pd

from entropy_common import (
    stable_softmax,
    standardize_train_target,
    drop_zero_variance,
    pairwise_products,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "acs12.csv")
OUT_DIR = BASE_DIR
SEED = 20260625


# -----------------------------
# ACS Feature Engineering
# -----------------------------

def make_features(df):
    """Create raw feature matrices for main effects, compact pairwise basis, and tree search."""
    d = df.copy()
    # Numeric preprocessing for covariates only. Outcome is not used for weighting.
    d["hrs_work_imp"] = d["hrs_work"].fillna(0.0)
    d["hrs_work_missing"] = d["hrs_work"].isna().astype(float)
    d["time_to_work_imp"] = d["time_to_work"].fillna(0.0)
    d["time_to_work_missing"] = d["time_to_work"].isna().astype(float)
    d["lang_missing"] = d["lang"].isna().astype(float)
    d["edu_missing"] = d["edu"].isna().astype(float)
    d["lang"] = d["lang"].fillna("missing")
    d["edu"] = d["edu"].fillna("missing")

    # Include all covariate main effects. Drop first category for each factor.
    cat_cols = ["employment", "race", "gender", "citizen", "lang", "married", "edu", "disability", "birth_qrtr"]
    num_cols = ["age", "hrs_work_imp", "hrs_work_missing", "time_to_work_imp", "time_to_work_missing", "lang_missing", "edu_missing"]
    X_cat = pd.get_dummies(d[cat_cols], drop_first=True, dtype=float)
    X_num = d[num_cols].astype(float)
    X_main_df = pd.concat([X_num, X_cat], axis=1)

    # Compact features for pairwise products and tree search. Keep interpretable signals.
    compact = pd.DataFrame({
        "age": d["age"].astype(float),
        "hrs_work": d["hrs_work_imp"].astype(float),
        "commute": d["time_to_work_imp"].astype(float),
        "employed": (d["employment"] == "employed").astype(float),
        "male": (d["gender"] == "male").astype(float),
        "college": (d["edu"].isin(["college", "grad"])).astype(float),
        "grad": (d["edu"] == "grad").astype(float),
        "nonwhite": (d["race"] != "white").astype(float),
        "citizen": (d["citizen"] == "yes").astype(float),
        "english": (d["lang"] == "english").astype(float),
        "married": (d["married"] == "yes").astype(float),
        "disabled": (d["disability"] == "yes").astype(float),
    })
    return X_main_df, compact


# -----------------------------
# ACS Data Preparation
# -----------------------------

def ensure_acs_data():
    """Download the OpenIntro acs12 CSV if it is not already present."""
    if os.path.exists(DATA_PATH):
        return
    import urllib.request
    url = "https://www.openintro.org/data/csv/acs12.csv"
    urllib.request.urlretrieve(url, DATA_PATH)


def prepare_acs_inputs():
    """
    Prepare ACS data for analysis.
    
    Returns:
        dict: A dictionary containing:
            - N: population size
            - X_main_std: standardized main effect features
            - X_comp_std: standardized compact features
            - pair_all: main effects + pairwise products
            - mu_main: target mean for main effects
            - mu_pair: target mean for main + pairwise
            - wt: uniform target population weights
            - y: log income vector
            - y_income: raw income vector
            - target_log: target mean log income
            - target_income: target mean income
            - probs: biased sampling probabilities
            - main_names: names of main effect features
            - comp_names: names of compact features
    """
    ensure_acs_data()
    df = pd.read_csv(DATA_PATH)
    df = df[(df["age"] >= 18) & df["income"].notna()].copy().reset_index(drop=True)
    df["log_income"] = np.log1p(df["income"].astype(float))
    N = len(df)

    X_main_df, X_compact_df = make_features(df)
    X_main_raw = X_main_df.to_numpy(dtype=float)
    X_comp_raw = X_compact_df.to_numpy(dtype=float)

    X_main_std, _, _, _ = standardize_train_target(X_main_raw, X_main_raw)
    X_main_std, keep_main = drop_zero_variance(X_main_std)
    main_names = [c for c, keep in zip(X_main_df.columns, keep_main) if keep]
    X_comp_std, _, _, _ = standardize_train_target(X_comp_raw, X_comp_raw)
    X_comp_std, keep_comp = drop_zero_variance(X_comp_std)
    comp_names = [c for c, keep in zip(X_compact_df.columns, keep_comp) if keep]

    mu_main = X_main_std.mean(axis=0)
    pair_prod = pairwise_products(X_comp_std[:, :8])
    pair_all = np.hstack([X_main_std, pair_prod])
    pair_all, keep_pair = drop_zero_variance(pair_all)
    mu_pair = pair_all.mean(axis=0)
    wt = np.ones(N) / N
    y = df["log_income"].to_numpy(dtype=float)
    y_income = df["income"].to_numpy(dtype=float)
    target_log = float(np.mean(y))
    target_income = float(np.mean(y_income))

    c = X_compact_df
    age_scaled = (c["age"].values - c["age"].mean()) / c["age"].std()
    hours_scaled = (c["hrs_work"].values - c["hrs_work"].mean()) / (c["hrs_work"].std() + 1e-8)
    commute_scaled = (c["commute"].values - c["commute"].mean()) / (c["commute"].std() + 1e-8)
    employed = c["employed"].values
    male = c["male"].values
    college = c["college"].values
    nonwhite = c["nonwhite"].values
    english = c["english"].values
    married = c["married"].values
    disabled = c["disabled"].values
    score = (
        0.12 * employed +
        0.10 * college +
        0.06 * male +
        0.06 * married -
        0.05 * disabled +
        0.04 * age_scaled +
        0.05 * hours_scaled -
        0.04 * commute_scaled +
        2.20 * (age_scaled > 0.70) * college +
        1.70 * (hours_scaled > 0.60) * employed +
        1.20 * (commute_scaled < -0.50) * male +
        1.00 * (age_scaled < -0.75) * (1 - english) -
        1.20 * nonwhite * (1 - english) +
        0.75 * employed * college * male
    )
    probs = stable_softmax(score)
    return {
        "N": N,
        "X_main_std": X_main_std,
        "X_comp_std": X_comp_std,
        "pair_all": pair_all,
        "mu_main": mu_main,
        "mu_pair": mu_pair,
        "wt": wt,
        "y": y,
        "y_income": y_income,
        "target_log": target_log,
        "target_income": target_income,
        "probs": probs,
        "main_names": main_names,
        "comp_names": comp_names,
    }


if __name__ == "__main__":
    # Example: prepare and display ACS data
    acs_data = prepare_acs_inputs()
    print(f"Population size: {acs_data['N']}")
    print(f"Main effect features: {len(acs_data['main_names'])}")
    print(f"Compact features: {len(acs_data['comp_names'])}")
    print(f"Target mean log income: {acs_data['target_log']:.4f}")
    print(f"Target mean income: {acs_data['target_income']:.2f}")
