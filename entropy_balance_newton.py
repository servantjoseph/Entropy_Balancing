import numpy as np

def entropy_balance_newton(
    x,
    mu,
    q=None,
    tolerance=1e-8,
    max_iterations=100,
    lambda0=None,
    ridge=0.0,
    verbose=False,
):
    """
    Entropy Balancing (EBW) via Newton–Raphson using NumPy only.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Feature matrix. Each row is an observation; columns are the balancing features.
    mu : ndarray, shape (n_features,)
        Target means for the features.
    q : ndarray, shape (n_samples,), optional
        Base weights. If None, defaults to uniform weights (1/n). Will be normalized to sum to 1.
    tolerance : float, optional
        Convergence threshold on the gradient norm.
    max_iterations : int, optional
        Maximum number of Newton–Raphson iterations.
    lambda0 : ndarray, shape (n_features,), optional
        Initial value for Lagrange multipliers. Defaults to zeros.
    ridge : float, optional
        Non-negative ridge term added to the Hessian diagonal to improve numerical stability.
        Set > 0 if Hessian is ill-conditioned (e.g., 1e-10 to 1e-6).
    verbose : bool, optional
        If True, prints iteration logs and final summary.

    Returns
    -------
    final_w : ndarray, shape (n_samples,)
        Final EBW weights that approximately match target means `mu`. Sum to 1.
    current_lambda : ndarray, shape (n_features,)
        Final Lagrange multipliers.
    info : dict
        Diagnostics:
          - "iterations": int
          - "converged": bool
          - "gradient_norms": ndarray
          - "mean_diffs": ndarray (norm of weighted mean minus mu per iteration)
    """
    # --- Shape checks ---
    if x.ndim != 2:
        raise ValueError("`x` must be a 2D array of shape (n_samples, n_features).")
    n_samples, n_features = x.shape

    mu = np.asarray(mu, dtype=float)
    if mu.shape != (n_features,):
        raise ValueError(f"`mu` must have shape ({n_features},). Got {mu.shape}.")

    # --- Base weights q ---
    if q is None:
        q = np.full(n_samples, 1.0 / n_samples, dtype=float)
    else:
        q = np.asarray(q, dtype=float)
        if q.shape != (n_samples,):
            raise ValueError(f"`q` must have shape ({n_samples},). Got {q.shape}.")
        if np.any(q < 0):
            raise ValueError("`q` must be non-negative.")
        s = q.sum()
        if s <= 0:
            raise ValueError("Sum of `q` must be positive.")
        q = q / s  # normalize to sum to 1

    # --- Initialize lambda ---
    if lambda0 is None:
        current_lambda = np.zeros(n_features, dtype=float)
    else:
        lambda0 = np.asarray(lambda0, dtype=float)
        if lambda0.shape != (n_features,):
            raise ValueError(f"`lambda0` must have shape ({n_features},). Got {lambda0.shape}.")
        current_lambda = lambda0.copy()

    # --- Iteration state ---
    gradient_norms = []
    mean_diffs = []
    iter_count = 0

    if verbose:
        print("Starting Entropy Balancing Optimization (Newton–Raphson):")

    # --- Newton–Raphson loop ---
    while iter_count < max_iterations:
        # (a) Compute -lambda^T x_i for each row
        lambda_dot_x = -x @ current_lambda  # shape (n_samples,)

        # (b) Log-sum-exp trick for numerical stability
        c = np.max(lambda_dot_x)
        exp_terms = np.exp(lambda_dot_x - c)          # shape (n_samples,)
        Z_lambda = np.sum(q * exp_terms)              # scalar

        # (c) Current weights
        current_w = (q * exp_terms) / Z_lambda        # shape (n_samples,)

        # (d) Gradient: g(lambda) = -E_w[x] + mu
        weighted_mean_x = current_w @ x               # shape (n_features,)
        current_g = -weighted_mean_x + mu             # shape (n_features,)

        # (e) Hessian: H = E_w[xx^T] - E_w[x] E_w[x]^T (weighted covariance)
        # E_w[xx^T] = sum_i w_i x_i x_i^T
        # Broadcasting (x.T * current_w) multiplies each column by w_i, then @ x sums over samples.
        weighted_x_outer_sum = (x.T * current_w) @ x  # shape (n_features, n_features)
        H = weighted_x_outer_sum - np.outer(weighted_mean_x, weighted_mean_x)

        # Optional ridge regularization
        if ridge > 0.0:
            H = H + ridge * np.eye(n_features)

        # (f) Newton step: H * delta = g
        try:
            delta_lambda = np.linalg.solve(H, current_g)
        except np.linalg.LinAlgError:
            # Fallback if H is singular/ill-conditioned
            delta_lambda = np.linalg.pinv(H) @ current_g

        # (g) Update lambda
        current_lambda = current_lambda - delta_lambda

        # (h) Diagnostics
        gnorm = float(np.linalg.norm(current_g))
        mdiff = float(np.linalg.norm(weighted_mean_x - mu))
        gradient_norms.append(gnorm)
        mean_diffs.append(mdiff)
        iter_count += 1

        # (i) Convergence check
        if gnorm < tolerance:
            if verbose:
                print(f"Converged after {iter_count} iterations. Gradient norm: {gnorm:.2e}")
            break

    if iter_count == max_iterations and verbose:
        print(f"Maximum iterations ({max_iterations}) reached. "
              f"Last gradient norm: {gradient_norms[-1]:.2e}")

    # --- Final weights with the converged lambda ---
    lambda_dot_x = -x @ current_lambda
    c = np.max(lambda_dot_x)
    exp_terms = np.exp(lambda_dot_x - c)
    Z_lambda = np.sum(q * exp_terms)
    final_w = (q * exp_terms) / Z_lambda

    if verbose:
        weighted_mean_final = final_w @ x
        print("\nFinal Results:")
        print(f"Final lambda: {current_lambda}")
        print(f"Final weighted mean (sum(w_i * x_i)): {weighted_mean_final}")
        print(f"Target mean (mu): {mu}")
        print(f"Sum of final weights: {np.sum(final_w):.4f}")
        print(f"Norm of difference between final weighted mean and mu: "
              f"{np.linalg.norm(weighted_mean_final - mu):.2e}")

    info = {
        "iterations": iter_count,
        "converged": (len(gradient_norms) > 0 and gradient_norms[-1] < tolerance),
        "gradient_norms": np.array(gradient_norms),
        "mean_diffs": np.array(mean_diffs),
    }
    return final_w, current_lambda, info
