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
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "acs12.csv")
OUT_DIR = BASE_DIR
SEED = 20260625

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


def eb_fit(X, mu_target, q=None, max_iter=80, tol=1e-9, ridge=1e-8, verbose=False):
    """
    Entropy balancing fit with base weights q.
    Solves min sum w log(w/q) s.t. sum w = 1 and X^T w = mu_target.
    Returns normalized weights and estimated dual tilt parameters.
    """
    X = np.asarray(X, dtype=float)
    mu_target = np.asarray(mu_target, dtype=float)
    n, p = X.shape
    if q is None:
        q = np.ones(n) / n
    else:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 1e-300)
        q = q / q.sum()

    lam = np.zeros(p)
    for it in range(max_iter):
        eta = X @ lam
        m = np.max(eta)
        a = q * np.exp(eta - m)
        denom = a.sum()
        if not np.isfinite(denom) or denom <= 0:
            raise RuntimeError("Nonfinite denominator in EB")
        w = a / denom
        g = X.T @ w - mu_target
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return w, lam, {"converged": True, "iter": it, "grad_norm": gnorm}
        Xc = X - (w @ X)
        H = (Xc.T * w) @ Xc + ridge * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]

        alpha = 1.0
        improved = False
        for _ in range(25):
            lam_new = lam - alpha * step
            eta2 = X @ lam_new
            m2 = np.max(eta2)
            a2 = q * np.exp(eta2 - m2)
            w2 = a2 / a2.sum()
            g2 = X.T @ w2 - mu_target
            n2 = float(np.linalg.norm(g2))
            if np.isfinite(n2) and n2 <= gnorm * (1 - 1e-4 * alpha) + 1e-12:
                lam = lam_new
                improved = True
                break
            alpha *= 0.5
        if not improved:
            lam = lam - 0.1 * step
    eta = X @ lam
    a = q * np.exp(eta - np.max(eta))
    w = a / a.sum()
    return w, lam, {"converged": False, "iter": max_iter, "grad_norm": float(np.linalg.norm(X.T @ w - mu_target))}


def eb_weights(X, mu_target, q=None, max_iter=80, tol=1e-9, ridge=1e-8, verbose=False):
    w, _, info = eb_fit(X, mu_target, q=q, max_iter=max_iter, tol=tol, ridge=ridge, verbose=verbose)
    return w, info

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



def pairwise_products(compact_std):
    """Pairwise products among compact standardized features, excluding duplicated main effects."""
    n, p = compact_std.shape
    cols = []
    for j in range(p):
        for k in range(j + 1, p):
            cols.append((compact_std[:, j] * compact_std[:, k])[:, None])
    if not cols:
        return np.zeros((n, 0))
    return np.hstack(cols)


def leaf_ids_for_tree(X, tree):
    """
    Tree format: ((v1,t1), left_child, right_child), where child is None or (v,t).
    Depth is at most 2. Leaf ids are 0,1,2,3 depending on path.
    """
    (v1, t1), left, right = tree
    root_left = X[:, v1] <= t1
    ids = np.zeros(X.shape[0], dtype=int)
    # left side
    if left is None:
        ids[root_left] = 0
        offset_right = 1
    else:
        v, t = left
        ids[root_left & (X[:, v] <= t)] = 0
        ids[root_left & (X[:, v] > t)] = 1
        offset_right = 2
    # right side
    if right is None:
        ids[~root_left] = offset_right
    else:
        v, t = right
        ids[(~root_left) & (X[:, v] <= t)] = offset_right
        ids[(~root_left) & (X[:, v] > t)] = offset_right + 1
    # Compress labels if any unused label gaps.
    unique = np.unique(ids)
    mapper = {old: new for new, old in enumerate(unique)}
    return np.array([mapper[z] for z in ids], dtype=int)


def cell_props(ids, weights, n_leaves=None):
    ids = np.asarray(ids, dtype=int)
    if n_leaves is None:
        n_leaves = int(ids.max()) + 1
    out = np.bincount(ids, weights=weights, minlength=n_leaves).astype(float)
    return out / out.sum()


def score_tree(Xs, Xt, ws, wt, tree, min_mass=0.02):
    ids_s = leaf_ids_for_tree(Xs, tree)
    ids_t = leaf_ids_for_tree(Xt, tree)
    J = max(ids_s.max(), ids_t.max()) + 1
    ps = cell_props(ids_s, ws, J)
    pt = cell_props(ids_t, wt, J)
    if np.min(ps) < min_mass or np.min(pt) < min_mass:
        return -np.inf
    return float(np.sum((ps - pt) ** 2 / np.maximum(pt, 1e-6)))


def candidate_splits(Xt, max_quantiles=6):
    splits = []
    for j in range(Xt.shape[1]):
        vals = Xt[:, j]
        uniq = np.unique(vals)
        if len(uniq) <= 2:
            if np.min(vals) < np.max(vals):
                splits.append((j, 0.5 * (np.min(vals) + np.max(vals))))
        else:
            qs = np.linspace(0.15, 0.85, max_quantiles)
            ths = np.unique(np.quantile(vals, qs))
            for t in ths:
                if np.min(vals) < t < np.max(vals):
                    splits.append((j, float(t)))
    return splits



def fit_best_tree(Xs, Xt, ws, wt, min_mass=0.02):
    """Greedy depth-2 tree search to keep repeated simulations fast."""
    splits = candidate_splits(Xt, max_quantiles=4)
    best_root = None
    best_score = -np.inf
    for root in splits:
        tr = (root, None, None)
        sc = score_tree(Xs, Xt, ws, wt, tr, min_mass=min_mass)
        if sc > best_score:
            best_score = sc
            best_root = root
    if best_root is None or not np.isfinite(best_score):
        return None, -np.inf
    best_tree = (best_root, None, None)
    # Add one left-child split if it improves the full leaf imbalance score.
    cur_score = best_score
    best_left = None
    for sp in splits:
        if sp[0] == best_root[0]:
            continue
        tr = (best_root, sp, None)
        sc = score_tree(Xs, Xt, ws, wt, tr, min_mass=min_mass)
        if sc > cur_score:
            cur_score = sc
            best_left = sp
    best_tree = (best_root, best_left, None)
    # Add one right-child split if it improves the full leaf imbalance score.
    best_right = None
    for sp in splits:
        if sp[0] == best_root[0]:
            continue
        tr = (best_root, best_left, sp)
        sc = score_tree(Xs, Xt, ws, wt, tr, min_mass=min_mass)
        if sc > cur_score:
            cur_score = sc
            best_right = sp
    best_tree = (best_root, best_left, best_right)
    return best_tree, cur_score

def hybrid_boosted_eb(X_hard_s, mu_hard, X_tree_s, X_tree_t, wt, B=12, nu=0.60,
                      min_mass=0.025, q_start=None):
    """
    EB-offset hybrid boosted EB.

    The starting base measure q_start is typically the fitted classical EB
    weight vector obtained from the same hard constraint matrix X_hard_s. The
    boosted tree step then adds a small leaf-wise multiplicative correction,
    and each update is projected back to the same hard constraints. If
    X_hard_s contains main effects, the main effects remain balanced. If it
    contains main effects plus pairwise products, both main and pairwise moments
    remain balanced after every boosted tree correction.
    """
    w, info0 = eb_weights(X_hard_s, mu_hard, q=q_start, tol=5e-8, ridge=1e-8)
    selected = []
    for b in range(B):
        tree, score = fit_best_tree(X_tree_s, X_tree_t, w, wt, min_mass=min_mass)
        if tree is None or not np.isfinite(score) or score < 1e-8:
            break
        ids_s = leaf_ids_for_tree(X_tree_s, tree)
        ids_t = leaf_ids_for_tree(X_tree_t, tree)
        J = max(ids_s.max(), ids_t.max()) + 1
        ps = cell_props(ids_s, w, J)
        pt = cell_props(ids_t, wt, J)
        ratio = np.power(np.maximum(pt, 1e-12) / np.maximum(ps, 1e-12), nu)
        q_temp = w * ratio[ids_s]
        q_temp = q_temp / q_temp.sum()
        # Hard projection back to the same EB-offset constraint set.
        w, infob = eb_weights(X_hard_s, mu_hard, q=q_temp, tol=5e-8, ridge=1e-8, max_iter=70)
        selected.append((tree, score, infob.get("grad_norm", np.nan)))
    return w, selected


def validation_leaf_imbalance(Xs, Xt, ws, wt):
    """Average total variation across fixed two-variable median partitions."""
    p = Xs.shape[1]
    # Use first 10 compact features to avoid too many cells.
    p_use = min(10, p)
    vals = []
    thresholds = np.median(Xt[:, :p_use], axis=0)
    for j in range(p_use):
        for k in range(j + 1, p_use):
            ids_s = (Xs[:, j] > thresholds[j]).astype(int) + 2 * (Xs[:, k] > thresholds[k]).astype(int)
            ids_t = (Xt[:, j] > thresholds[j]).astype(int) + 2 * (Xt[:, k] > thresholds[k]).astype(int)
            ps = cell_props(ids_s, ws, 4)
            pt = cell_props(ids_t, wt, 4)
            vals.append(0.5 * np.sum(np.abs(ps - pt)))
    return float(np.mean(vals))


def summarize_results(raw):
    rows = []
    for method, g in raw.groupby("method"):
        err = g["error"].values
        rows.append({
            "method": method,
            "mean_estimate": g["estimate"].mean(),
            "bias": err.mean(),
            "abs_bias": abs(err.mean()),
            "rmse": math.sqrt(np.mean(err ** 2)),
            "mae": np.mean(np.abs(err)),
            "main_l2_mean": g["main_l2"].mean(),
            "pair_l2_mean": g["pair_l2"].mean(),
            "validation_tv_mean": g["validation_tv"].mean(),
            "ess_mean": g["ess"].mean(),
            "max_weight_mean": g["max_weight"].mean(),
        })
    return pd.DataFrame(rows).sort_values("rmse")



# -----------------------------
# Main analysis
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
    try:
        w_main, lam_main, info = eb_fit(Xs_main, mu_m, tol=5e-8, ridge=1e-8)
    except Exception:
        failures["main"] += 1
        w_main = w_un.copy()
    methods.append(("Main-effect EB", w_main, None))
    Xs_pair = s["pair_all"][idx, :]
    try:
        w_pair, lam_pair, info = eb_fit(Xs_pair, s["mu_pair"], tol=1e-7, ridge=1e-7, max_iter=100)
    except Exception:
        failures["pairwise"] += 1
        w_pair = w_main.copy()
    methods.append(("Fixed pairwise EB", w_pair, None))
    try:
        w_hyb, selected = hybrid_boosted_eb(Xs_main, mu_m, Xs_comp, s["X_comp_std"], s["wt"], B=12, nu=0.60, min_mass=0.025, q_start=w_main)
    except Exception:
        failures["hybrid_main_init"] += 1
        w_hyb = w_main.copy()
        selected = []
    methods.append(("EB-offset hybrid: main", w_hyb, selected))
    try:
        w_hyb_pair, selected_pair = hybrid_boosted_eb(Xs_pair, s["mu_pair"], Xs_comp, s["X_comp_std"], s["wt"], B=12, nu=0.60, min_mass=0.025, q_start=w_pair)
    except Exception:
        failures["hybrid_pairwise_init"] += 1
        w_hyb_pair = w_pair.copy()
        selected_pair = []
    methods.append(("EB-offset hybrid: pairwise", w_hyb_pair, selected_pair))
    rows = []
    for method, w, extra in methods:
        est = float(w @ ys)
        est_income = float(w @ ys_income)
        main_l2 = float(np.linalg.norm(Xs_main.T @ w - mu_m))
        validation_tv = validation_leaf_imbalance(Xs_comp, s["X_comp_std"], w, s["wt"])
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
            "ess": effective_sample_size(w),
            "max_weight": float(np.max(w)),
            "num_trees": len(extra) if extra is not None else np.nan,
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
    summary = summarize_results(raw)
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

    raw.to_csv(os.path.join(OUT_DIR, "acs_realdata_raw_rows.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "acs_realdata_summary.csv"), index=False)
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
    with open(os.path.join(OUT_DIR, "acs_realdata_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    order = ["Biased source", "Main-effect EB", "Fixed pairwise EB", "EB-offset hybrid: main", "EB-offset hybrid: pairwise"]
    summary_ordered = summary.set_index("method").loc[order].reset_index()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["rmse"])
    plt.ylabel("RMSE for mean log income")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_rmse_log_income.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["rmse_income"])
    plt.ylabel("RMSE for mean income")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_rmse_income.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["validation_tv_mean"])
    plt.ylabel("Mean validation leaf total variation")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_validation_tv.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    plt.bar(summary_ordered["method"], summary_ordered["ess_mean"])
    plt.ylabel("Mean effective sample size")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_ess.pdf"))
    plt.close()

    plt.figure(figsize=(7.4, 4.2))
    data = [raw.loc[raw["method"] == m, "error"].values for m in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Error in mean log income")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "acs_fig_error_boxplot.pdf"))
    plt.close()

    print("Summary")
    print(summary_ordered.to_string(index=False))
    print("Metadata")
    print(json.dumps(meta, indent=2)[:2000])
    return raw, summary, meta

if __name__ == "__main__":
    run_analysis(R=1000, n_source=450, n_jobs=8)
