"""
Shared EB utilities for Entropy_Balancing repository.
Provides: normalize_weights, effective_sample_size, stable_softmax,
          pairwise_products, eb_fit, eb_weights

Designed to be imported by KS_EBW_Boosted_code_v2.py and
ACS_EBW_Boosted_code_v2.py so both use the same EB routines.
"""

import numpy as np


def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 1e-300)
    return w / float(w.sum())


def effective_sample_size(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w * w)


def stable_softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    p = np.exp(z)
    return p / p.sum()


def pairwise_products(X):
    X = np.asarray(X, dtype=float)
    cols = []
    p = X.shape[1] if X.ndim == 2 else 0
    for j in range(p):
        for k in range(j + 1, p):
            cols.append((X[:, j] * X[:, k])[:, None])
    if not cols:
        return np.zeros((X.shape[0], 0))
    return np.hstack(cols)


def eb_fit(X, mu_target, q=None, max_iter=80, tol=1e-9, ridge=1e-8, verbose=False):
    """
    Entropy balancing fit with base weights q.
    Solves min sum w log(w/q) s.t. sum w = 1 and X^T w = mu_target.
    Returns normalized weights, estimated dual tilt parameters, and info dict.
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
    w, lam, info = eb_fit(X, mu_target, q=q, max_iter=max_iter, tol=tol, ridge=ridge, verbose=verbose)
    return w, info
