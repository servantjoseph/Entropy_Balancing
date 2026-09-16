# ============================================================
# Load Entropy_Balancing repository
# ============================================================

import os
import sys

REPO_URL = "https://github.com/servantjoseph/Entropy_Balancing.git"
REPO_DIR = "/content/Entropy_Balancing"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL}

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


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

The EB fitting, pairwise-product construction, CART tree fitting, and EB-offset boosting routines are shared with the Kang--Schafer simulation implementation.
"""

import os
import math
import json
# from ACS_EBW_Boosted_code import BASE_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.getcwd()  # Use current working directory as base if __file__ is not defined
#BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' n globals() else os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "acs12.csv")
OUT_DIR = BASE_DIR
SEED = 20260625


from entropy_common import (
    normalize_weights, effective_sample_size, sigmoid, eb_fit, eb_weights,
    pairwise_products, compact_leaf_ids_for_two, props, fit_balance_tree,
    hybrid, cell_props,validation_leaf_imbalance,summarize_results)

# -----------------------------
# Utility functions
# -----------------------------

def stable_softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    p = np.exp(z)
    return p / p.sum()


def weighted_mean(w, A):
    return np.asarray(w) @ np.asarray(A)


def normalize_weights(w):
    w = np.maximum(np.asarray(w, float), 1e-300)
    return w / w.sum()


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    return 1.0 / np.sum(w * w)


def standardize_train_target(source_raw, target_raw):
    """Standardize source and target using target mean/sd."""
    mu = target_raw.mean(axis=0)
    sd = target_raw.std(axis=0)
    sd[sd < 1e-10] = 1.0
    return (source_raw - mu) / sd, (target_raw - mu) / sd, mu, sd


def drop_zero_variance(A, tol=1e-12):
    sd = A.std(axis=0)
    keep = sd > tol
    return A[:, keep], keep


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


# --------------------------------
# Main analysis (last on 9/3/2026)
# --------------------------------

ACS_STATE = {}

def ensure_acs_data():
    """Download the OpenIntro acs12 CSV if it is not already present."""
    if os.path.exists(DATA_PATH):
        return

    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url = "https://www.openintro.org/data/csv/acs12.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, verify=False)
    response.raise_for_status()

    with open(DATA_PATH, "wb") as f:
        f.write(response.content)

    print("Saved:", DATA_PATH)

# def ensure_acs_data():
#     """Download the OpenIntro acs12 CSV if it is not already present."""
#     if os.path.exists(DATA_PATH):
#         return
#     import urllib.request
#     url = "https://www.openintro.org/data/csv/acs12.csv"
#     urllib.request.urlretrieve(url, DATA_PATH)


def _prepare_acs_inputs():
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


def _acs_worker_init(state):
    ACS_STATE.clear()
    ACS_STATE.update(state) #make ACS_STATE == state


# def _acs_one_rep(args):
    # r, n_source = args
state,R,n_source = _prepare_acs_inputs(),1000,450
_acs_worker_init(state) #state is the same as ACS_STATE
#Check state==ACS_STATE

args = [(r, n_source) for r in range(R)]
r, n_source = args[0]
    s = ACS_STATE
    rng = np.random.default_rng(SEED + r)
    N = s["N"]
    idx = rng.choice(N, size=n_source, replace=False, p=s["probs"])
    Xs_main = s["X_main_std"][idx, :]
    Xs_comp = s["X_comp_std"][idx, :]
    ys = s["y"][idx]
    ys_income = s["y_income"][idx]
    mu_m = s["mu_main"]
    failures = {"main": 0, "pairwise": 0, "hybrid_main_init": 0, "hybrid_pairwise_init": 0}
    methods = []
    w_un = np.ones(n_source) / n_source
    q = w_un.copy()
    methods.append(("Biased source", w_un, None))
    try:
        w_main, lam_main = eb_fit(Xs_main, mu_m, q=q, max_iter=80, tol=1e-8)#, ridge=1e-7 #same as KS
    except Exception:
        failures["main"] += 1
        w_main = w_un.copy()
    methods.append(("Main-effect EB", w_main, None))
    Xs_pair = s["pair_all"][idx, :]
    try:
        w_pair, lam_pair = eb_fit(Xs_pair, s["mu_pair"], q=q, max_iter=100, tol=1e-8)#, ridge=1e-7 #same as KS
    except Exception:
        failures["pairwise"] += 1
        w_pair = w_main.copy()
    methods.append(("Fixed pairwise EB", w_pair, None))
    # try:
    #     rs = int(rng.integers(0, 2**31 - 1))
    #     w_hyb, nt = hybrid(Xs_comp, s["X_comp_std"], Xs_main, mu_m, q0=w_main, B=100, nu=0.10, min_mass=0.001, interaction_depth=2, random_state=rs, min_ess_frac=0.10, score_tol=0.05)





