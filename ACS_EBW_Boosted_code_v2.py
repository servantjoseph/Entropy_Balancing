"""
ACS real-data ACS microdata analysis using shared eb_common utilities.
This file mirrors ACS_EBW_Boosted_code.py but imports EB routines from
eb_common.py so the codebase avoids duplicating EB implementations.
"""

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eb_common as ebc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "acs12.csv")
OUT_DIR = BASE_DIR
SEED = 20260625


# -----------------------------
# Feature engineering (same as original ACS script)
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
        "college": (d["edu"].isin(["college", "grad"])) .astype(float),
        "grad": (d["edu"] == "grad").astype(float),
        "nonwhite": (d["race"] != "white").astype(float),
        "citizen": (d["citizen"] == "yes").astype(float),
        "english": (d["lang"] == "english").astype(float),
        "married": (d["married"] == "yes").astype(float),
        "disabled": (d["disability"] == "yes").astype(float),
    })
    return X_main_df, compact


# -----------------------------
# Data preparation and simulation helpers (use eb_common where appropriate)
# -----------------------------

ACS_STATE = {}


def ensure_acs_data():
    """Download the OpenIntro acs12 CSV if it is not already present."""
    if os.path.exists(DATA_PATH):
        return
    import urllib.request
    url = "https://www.openintro.org/data/csv/acs12.csv"
    urllib.request.urlretrieve(url, DATA_PATH)


def _prepare_acs_inputs():
    ensure_acs_data()
    df = pd.read_csv(DATA_PATH)
    df = df[(df["age"] >= 18) & df["income"].notna()].copy().reset_index(drop=True)
    df["log_income"] = np.log1p(df["income"].astype(float))
    N = len(df)

    X_main_df, X_compact_df = make_features(df)
    X_main_raw = X_main_df.to_numpy(dtype=float)
    X_comp_raw = X_compact_df.to_numpy(dtype=float)

    # Use eb_common standardization helpers
    X_main_std, _, _, _ = ebc.standardize_train_target(X_main_raw, X_main_raw)
    X_main_std, keep_main = ebc.drop_zero_variance(X_main_std)
    main_names = [c for c, keep in zip(X_main_df.columns, keep_main) if keep]
    X_comp_std, _, _, _ = ebc.standardize_train_target(X_comp_raw, X_comp_raw)
    X_comp_std, keep_comp = ebc.drop_zero_variance(X_comp_std)
    comp_names = [c for c, keep in zip(X_compact_df.columns, keep_comp) if keep]

    mu_main = X_main_std.mean(axis=0)
    pair_prod = ebc.pairwise_products(X_comp_std[:, :8])
    pair_all = np.hstack([X_main_std, pair_prod])
    pair_all, keep_pair = ebc.drop_zero_variance(pair_all)
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
    probs = ebc.stable_softmax(score)
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
    ACS_STATE.update(state)


def _acs_one_rep(args):
    r, n_source = args
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
    methods.append(("Biased source", w_un, None))

    # Main-effect EB
    try:
        w_main, lam_main, info = ebc.eb_fit(Xs_main, mu_m, tol=5e-8, ridge=1e-8)
    except Exception:
        failures["main"] += 1
        w_main = w_un.copy()
    methods.append(("Main-effect EB", w_main, None))

    # Fixed pairwise EB
    Xs_pair = s["pair_all"][idx, :]
    try:
        w_pair, lam_pair, info = ebc.eb_fit(Xs_pair, s["mu_pair"], tol=1e-7, ridge=1e-7, max_iter=100)
    except Exception:
        failures["pairwise"] += 1
        w_pair = w_main.copy()
    methods.append(("Fixed pairwise EB", w_pair, None))

    # EB-offset hybrid starting from main EB
    try:
        w_hyb, nt = ebc.hybrid_boosted_eb(Xs_main, mu_m, Xs_comp, s["X_comp_std"], s["wt"], B=12, nu=0.60, min_mass=0.025, q_start=w_main)
        selected = nt
    except Exception:
        failures["hybrid_main_init"] += 1
        w_hyb = w_main.copy()
        selected = 0
    methods.append(("EB-offset hybrid: main", w_hyb, selected))

    # EB-offset hybrid starting from pairwise EB
    try:
        w_hyb_pair, ntp = ebc.hybrid_boosted_eb(Xs_pair, s["mu_pair"], Xs_comp, s["X_comp_std"], s["wt"], B=12, nu=0.60, min_mass=0.025, q_start=w_pair)
        selected_pair = ntp
    except Exception:
        failures["hybrid_pairwise_init"] += 1
        w_hyb_pair = w_pair.copy()
        selected_pair = 0
    methods.append(("EB-offset hybrid: pairwise", w_hyb_pair, selected_pair))

    rows = []
    for method, w, extra in methods:
        est = float(w @ ys)
        est_income = float(w @ ys_income)
        main_l2 = float(np.linalg.norm(Xs_main.T @ w - mu_m))
        validation_tv = ebc.validation_leaf_imbalance(Xs_comp, s["X_comp_std"], w, s["wt"])
        pair_l2 = float(np.linalg.norm(Xs_pair.T @ w - s["mu_pair"]))
        rows.append({
            "rep": r,
            "method": method,
            "estimate": est,
            "error": est - s["target_log"],
            "estimate_income": est_income,
            "error_income": est_income - s["target_income"],
            "main_l2": main_l2,
            "pair_l2": pair_l2,
            "validation_tv": validation_tv,
            "ess": ebc.effective_sample_size(w),
            "max_weight": float(np.max(w)),
            "num_trees": int(extra) if extra is not None else np.nan,
        })
    return rows, failures


def run_analysis(R=1000, n_source=450, n_jobs=None):
    import multiprocessing as mp
    state = _prepare_acs_inputs()
    if n_jobs is None:
        n_jobs = min(8, max(1, (os.cpu_count() or 2) - 1))
    args = [(r, n_source) for r in range(R)]
    raw_rows = []
    failures = {"main": 0, "pairwise": 0, "hybrid_main_init": 0, "hybrid_pairwise_init": 0}
    if n_jobs <= 1:
        _acs_worker_init(state)
        iterable = map(_acs_one_rep, args)
    else:
        pool = mp.Pool(processes=n_jobs, initializer=_acs_worker_init, initargs=(state,))
        iterable = pool.imap(_acs_one_rep, args, chunksize=5)
    try:
        for rows, fail in iterable:
            raw_rows.extend(rows)
            for k, v in fail.items():
                failures[k] += v
    finally:
        if n_jobs > 1:
            pool.close(); pool.join()
    raw = pd.DataFrame(raw_rows)
    summary = ebc.summarize_results(raw)
    income_rows = []
    for method, g in raw.groupby("method"):
        e = g["error_income"].values
        income_rows.append({
            "method": method,
            "bias_income": e.mean(),
            "rmse_income": math.sqrt(np.mean(e**2)),
            "mae_income": np.mean(np.abs(e)),
        })
    income_summary = pd.DataFrame(income_rows).sort_values("rmse_income")
    summary = summary.merge(income_summary, on="method", how="left")

    raw.to_csv(os.path.join(OUT_DIR, "acs_realdata_raw_rows_v2.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "acs_realdata_summary_v2.csv"), index=False)
    meta = {
        "seed": SEED,
        "R": R,
        "n_source": n_source,
        "N_adult_pseudopopulation": state["N"],
        "target_mean_log_income": state["target_log"],
        "target_mean_income": state["target_income"],
        "main_feature_count": int(state["X_main_std"].shape[1]),
        "compact_feature_count": int(state["X_comp_std"].shape[1]),
        "failures": failures,
        "main_feature_names": state["main_names"],
        "compact_feature_names": state["comp_names"],
    }
    with open(os.path.join(OUT_DIR, "acs_realdata_metadata_v2.json"), "w") as f:
        json.dump(meta, f, indent=2)

    order = ["Biased source", "Main-effect EB", "Fixed pairwise EB", "EB-offset hybrid: main", "EB-offset hybrid: pairwise"]
    summary_ordered = summary.set_index("method").loc[order].reset_index()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["rmse"])
    plt.ylabel("RMSE for mean log income")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_rmse_log_income_v2.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["rmse_income"])
    plt.ylabel("RMSE for mean income")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_rmse_income_v2.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["validation_tv_mean"])
    plt.ylabel("Mean validation leaf total variation")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_validation_tv_v2.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["ess_mean"])
    plt.ylabel("Mean effective sample size")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_ess_v2.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    data = [raw.loc[raw["method"] == m, "error"].values for m in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Error in mean log income")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_error_boxplot_v2.pdf"))
    plt.close()

    print("Summary")
    print(summary_ordered.to_string(index=False))
    print("Metadata")
    print(json.dumps(meta, indent=2)[:2000])
    return raw, summary, meta


if __name__ == "__main__":
    run_analysis(R=1000, n_source=450, n_jobs=8)
