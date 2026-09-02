"""
Common entropy-balancing utilities and helper functions.
"""
import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier


def normalize_weights(w):
    w=np.maximum(np.asarray(w,float),1e-300); return w/w.sum()

def effective_sample_size(w):
    w=normalize_weights(w); return 1.0/np.sum(w**2)

def sigmoid(z):
    z=np.asarray(z,float); out=np.empty_like(z); pos=z>=0; out[pos]=1/(1+np.exp(-z[pos])); ez=np.exp(z[~pos]); out[~pos]=ez/(1+ez); return out

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

#Below functions were used for the ACS analysis
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

