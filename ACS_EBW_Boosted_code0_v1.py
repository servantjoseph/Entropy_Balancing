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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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


def eb_fit(X, mu, q=None, max_iter=80, tol=1e-8, ridge=1e-7):
    """Classical entropy-balancing fit returning weights and dual parameters."""
    X=np.asarray(X,float); mu=np.asarray(mu,float); n,p=X.shape
    q=np.ones(n)/n if q is None else normalize_weights(q)
    lam=np.zeros(p)
    for _ in range(max_iter):
        eta=X@lam; a=q*np.exp(eta-np.max(eta)); w=a/a.sum(); g=X.T@w-mu
        if np.linalg.norm(g)<tol: return w,lam
        m=w@X; Xc=X-m; H=(Xc.T*w)@Xc + ridge*np.eye(p)
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: step=np.linalg.lstsq(H,g,rcond=None)[0]
        alpha=1.0; ng=np.linalg.norm(g)
        for _ in range(25):
            lam2=lam-alpha*step; a2=q*np.exp(X@lam2-np.max(X@lam2)); w2=a2/a2.sum()
            if np.linalg.norm(X.T@w2-mu)<=ng+1e-12:
                lam=lam2; break
            alpha*=0.5
        else: lam=lam-0.05*step
    eta=X@lam; a=q*np.exp(eta-np.max(eta)); return a/a.sum(),lam


def eb_weights(X, mu, q=None, max_iter=80, tol=1e-8, ridge=1e-7):
    w,_=eb_fit(X,mu,q=q,max_iter=max_iter,tol=tol,ridge=ridge)
    return w

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




def pairwise_products(X):
    cols=[]
    for j in range(X.shape[1]):
        for k in range(j+1,X.shape[1]): cols.append((X[:,j]*X[:,k])[:,None])
    return np.hstack(cols) if cols else np.zeros((X.shape[0],0))

def compact_leaf_ids_for_two(clf,Xs,Xt):
    raw_s=clf.apply(Xs); raw_t=clf.apply(Xt); vals=np.unique(np.r_[raw_s,raw_t]); mp={old:i for i,old in enumerate(vals)}
    return np.array([mp[z] for z in raw_s],int), np.array([mp[z] for z in raw_t],int)

def props(ids,w,J=None):
    if J is None: J=int(ids.max())+1
    p=np.bincount(ids,weights=normalize_weights(w),minlength=J).astype(float); return p/p.sum()

def fit_balance_tree(Xs,Xt,ws,wt,interaction_depth=3,min_mass=0.001,random_state=0):
    ws=normalize_weights(ws); wt=normalize_weights(wt); ns=Xs.shape[0]; nt=Xt.shape[0]
    X_aug=np.vstack([Xs,Xt]); y=np.r_[np.zeros(ns,dtype=int),np.ones(nt,dtype=int)]; sw=np.r_[0.5*ws,0.5*wt]
    best=None
    # Try a small grid of CART leaf-size controls and keep the highest valid leaf discrepancy.
    for min_frac in (0.01,0.02,0.05,0.10):
        clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=interaction_depth,min_weight_fraction_leaf=min_frac,random_state=random_state)
        clf.fit(X_aug,y,sample_weight=sw)
        ids_s,ids_t=compact_leaf_ids_for_two(clf,Xs,Xt); J=max(ids_s.max(),ids_t.max())+1
        ps=props(ids_s,ws,J); pt=props(ids_t,wt,J)
        if np.min(ps)<min_mass or np.min(pt)<min_mass or J<2: continue
        score=float(np.sum((ps-pt)**2/pt))
        if best is None or score>best[0]: best=(score,clf,ids_s,ids_t,ps,pt)
    if best is None: return None,None,None,-np.inf,None,None
    score,clf,ids_s,ids_t,ps,pt=best
    return clf,ids_s,ids_t,score,ps,pt

def hybrid(Xs,Xt,Xhard,mu_hard,q0=None,B=100,nu=0.10,min_mass=0.001,interaction_depth=3,random_state=0,min_ess_frac=0.10,score_tol=0.05):
    wt=np.ones(Xt.shape[0])/Xt.shape[0]
    w=eb_weights(Xhard,mu_hard,q=q0,tol=1e-8)
    nt=0
    for b in range(B):
        clf,ids_s,ids_t,score,ps,pt=fit_balance_tree(Xs,Xt,w,wt,interaction_depth=interaction_depth,min_mass=min_mass,random_state=random_state+b)
        if clf is None or not np.isfinite(score) or score<score_tol: break
        ratio=np.power(np.maximum(pt,1e-12)/np.maximum(ps,1e-12),nu)
        qtemp=normalize_weights(w*ratio[ids_s])
        w_new=eb_weights(Xhard,mu_hard,q=qtemp,tol=1e-8,max_iter=60)
        if effective_sample_size(w_new)<min_ess_frac*Xs.shape[0]: break
        w=w_new; nt+=1
    return w,nt

def cell_props(ids, weights, n_leaves=None):
    ids = np.asarray(ids, dtype=int)
    if n_leaves is None:
        n_leaves = int(ids.max()) + 1
    out = np.bincount(ids, weights=weights, minlength=n_leaves).astype(float)
    return out / out.sum()




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
    q = w_un.copy()
    methods.append(("Biased source", w_un, None))
    try:
        w_main, lam_main = eb_fit(Xs_main, mu_m, q=q, max_iter=80, tol=1e-8, ridge=1e-7)
    except Exception:
        failures["main"] += 1
        w_main = w_un.copy()
    methods.append(("Main-effect EB", w_main, None))
    Xs_pair = s["pair_all"][idx, :]
    try:
        w_pair, lam_pair = eb_fit(Xs_pair, s["mu_pair"], q=q, max_iter=100, tol=1e-8, ridge=1e-7)
    except Exception:
        failures["pairwise"] += 1
        w_pair = w_main.copy()
    methods.append(("Fixed pairwise EB", w_pair, None))
    try:
        rs = int(rng.integers(0, 2**31 - 1))
        w_hyb, nt = hybrid(Xs_comp, s["X_comp_std"], Xs_main, mu_m, q0=w_main, B=100, nu=0.10, min_mass=0.001, interaction_depth=3, random_state=rs, min_ess_frac=0.10, score_tol=0.05)
    except Exception:
        failures["hybrid_main_init"] += 1
        w_hyb = w_main.copy()
        nt = 0
    methods.append(("EB-offset hybrid: main", w_hyb, nt))
    try:
        w_hyb_pair, ntp = hybrid(Xs_comp, s["X_comp_std"], Xs_pair, s["mu_pair"], q0=w_pair, B=100, nu=0.10, min_mass=0.001, interaction_depth=3, random_state=rs + 10000, min_ess_frac=0.10, score_tol=0.05)
    except Exception:
        failures["hybrid_pairwise_init"] += 1
        w_hyb_pair = w_pair.copy()
        ntp = 0
    methods.append(("EB-offset hybrid: pairwise", w_hyb_pair, ntp))
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
            "num_trees": (extra if isinstance(extra, (int, np.integer)) else len(extra)) if extra is not None else np.nan,
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
