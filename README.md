# Entropy Balancing

Methods, simulations, and supporting materials for estimating entropy-balancing weights and evaluating their performance in observational and missing-data settings.

The repository combines classical entropy balancing with boosted, tree-based balance corrections. It includes reusable Python utilities, executable simulation scripts and notebooks, documentation, and supporting artifacts for reproducible analyses.

> **Status:** Research code. APIs, defaults, and output formats may change as the analyses develop. Review the source and notebook assumptions before using the code for production or applied inference.

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Methods](#methods)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Running the analyses](#running-the-analyses)
- [Core API](#core-api)
- [Data and outputs](#data-and-outputs)
- [Interpreting diagnostics](#interpreting-diagnostics)
- [Reproducibility](#reproducibility)
- [Notes and limitations](#notes-and-limitations)
- [Contributing](#contributing)
- [License and citation](#license-and-citation)

## Overview

Entropy balancing constructs nonnegative weights for one sample so that weighted covariate moments match specified target moments from another sample or population. In this repository, the workflow is organized around a reusable weighting engine, simulation scripts, and analysis notebooks for comparing balance diagnostics and estimation performance.

- survey or sample reweighting;
- covariate shift and distributional adjustment;
- missing-data and response-bias problems; and
- simulation-based comparisons of weighting and prediction estimators.

The main implementation fits the dual parameters of the entropy-balancing optimization problem with a Newton-style iteration. A ridge-stabilized Hessian and a least-squares fallback are used when the default update is unstable or poorly conditioned.

The repository also implements a hybrid approach: classical entropy balancing enforces selected hard moment constraints, while shallow CART models identify remaining distributional imbalance and inform iterative corrections.

## Repository layout

```text
.
├── entropy_common.py                 # Reusable weighting, diagnostics, and hybrid methods
├── KS_EBW_Boosted_code.py            # Kang–Schafer-style missing-data simulation
├── ACS_EBW_Boosted_code_paper.ipynb  # ACS analysis notebook
├── KS_EBW_Boosted_code_paper.ipynb   # KS analysis notebook
├── acs12.csv                        # Example ACS data used by the analysis
├── Doc/                             # Supporting notes, math, and documentation
│   ├── KS_EBW_Boosted_code_note.md  # Numerical-method notes
│   └── MathEBW.tex                  # Mathematical documentation
├── OutputData/                      # Generated summaries, raw rows, and figures
└── Old/                             # Archived historical notebook/script versions
```

The repository currently keeps only the two active analysis notebooks at the top level. Earlier or superseded notebook versions and scripts are archived under `Old/`, while supporting notes and mathematical write-ups live in `Doc/`.

## Methods

### Classical entropy balancing

Given source covariates `X`, target moments `mu`, and optional base weights `q`, the implementation finds weights of the form

```text
w_i ∝ q_i exp(X_i λ)
```

subject to the target moment condition

```text
Xᵀw = mu.
```

Weights are normalized to sum to one. If no base weights are supplied, a uniform base distribution is used. The fitted dual parameters are returned by `eb_fit`; `eb_weights` is a convenience wrapper that returns only the weights.

### Pairwise moments

`pairwise_products` augments a covariate matrix with all pairwise products. This allows the pairwise EB analysis to balance first-order and selected second-order structure.

### Boosted balance corrections

The hybrid procedure repeatedly:

1. fits a shallow decision tree to distinguish source and target samples;
2. measures weighted leaf-probability discrepancy;
3. updates source weights using a learning rate; and
4. projects the updated weights back onto the requested hard moment constraints with entropy balancing.

The procedure stops when no valid tree is found, the discrepancy is below the score tolerance, the maximum number of iterations is reached, or the effective sample size becomes too small.

## Requirements

The code is Python-based and uses the following packages:

- Python 3.9 or newer recommended;
- NumPy;
- pandas;
- scikit-learn; and
- Matplotlib for the analysis scripts and figures.

A minimal environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib jupyter
```

No package metadata file is currently provided, so install dependencies explicitly or add them to a project-specific environment before running the analyses.

## Quick start

From the repository root:

```python
import numpy as np
from entropy_common import eb_fit, eb_weights, effective_sample_size

rng = np.random.default_rng(2026)
source = rng.normal(size=(500, 3))
target_moments = np.array([0.0, 0.2, -0.1])

weights, dual_parameters = eb_fit(source, target_moments)
print("sum(weights):", weights.sum())
print("weighted moments:", source.T @ weights)
print("effective sample size:", effective_sample_size(weights))

# Equivalent convenience call when dual parameters are not needed:
weights_only = eb_weights(source, target_moments)
```

For a complete worked analysis, open either notebook in Jupyter:

```bash
jupyter notebook ACS_EBW_Boosted_code_paper.ipynb
# or
jupyter notebook KS_EBW_Boosted_code_paper.ipynb
```

Run notebook cells in order. Check the input paths and output locations if you launch a notebook from a directory other than the repository root.

## Running the analyses

### Kang–Schafer-style simulation

The script generates a synthetic latent-variable design, creates nonlinear observed covariates and a response indicator, then compares several estimators:

- naive respondent mean;
- OLS prediction;
- main-effect entropy balancing;
- pairwise entropy balancing;
- EB-offset hybrid with main-effect constraints; and
- EB-offset hybrid with main and pairwise constraints.

Run it with:

```bash
python KS_EBW_Boosted_code.py
```

The checked-in script uses a fixed base seed, 1,000 replications, sample size 1,000, up to 100 tree corrections, learning rate `0.10`, and multiprocessing by default. The run can be computationally expensive for the default settings, so smaller values are often used for exploratory checks.

For example:

```python
from KS_EBW_Boosted_code import run

raw, summary = run(R=10, n=500, B=20, nu=0.10, n_jobs=1)
print(summary)
```

The script writes simulation tables and PDF figures to the directory containing the script. When run from the repository root, these are placed in the repository root; the checked-in `OutputData/` directory contains saved artifacts for the published-style summary tables and figures.

### ACS analysis

`ACS_EBW_Boosted_code_paper.ipynb` contains the American Community Survey-oriented analysis using `acs12.csv`. Because it is a notebook, the exact execution sequence and data preparation steps are documented in the notebook itself.

## Core API

`entropy_common.py` provides the primary reusable functions:

| Function | Purpose |
| --- | --- |
| `normalize_weights(w)` | Clips extremely small values and normalizes weights to sum to one. |
| `effective_sample_size(w)` | Computes `1 / sum(normalized_weights²)`. |
| `sigmoid(z)` | Numerically stable logistic transform. |
| `eb_fit(X, mu, q=None, ...)` | Fits classical entropy-balancing weights and returns `(weights, dual_parameters)`. |
| `eb_weights(X, mu, q=None, ...)` | Returns only entropy-balancing weights. |
| `pairwise_products(X)` | Builds all pairwise product features. |
| `fit_balance_tree(...)` | Fits a weighted CART classifier and scores source-target leaf imbalance. |
| `hybrid(...)` | Applies boosted tree corrections followed by entropy-balancing projections. |
| `validation_leaf_imbalance(...)` | Measures average total variation across fixed two-variable median partitions. |
| `summarize_results(raw)` | Aggregates simulation estimates, bias, RMSE, imbalance, ESS, and weight diagnostics. |

Important `eb_fit` controls include `max_iter`, `tol`, and `ridge`. Important `hybrid` controls include `B`, `nu`, `min_mass`, `interaction_depth`, `min_ess_frac`, and `score_tol`.

## Data and outputs

`acs12.csv` is the primary example input file for the ACS notebook. Treat it as analysis data rather than a general-purpose benchmark dataset; inspect the notebook for variable construction, scaling, and estimation choices before reusing it for a different application.

`OutputData/` contains previously generated artifacts, including:

- raw simulation and real-data rows (`*_raw_rows.csv`);
- grouped summaries (`*_summary.csv`);
- metadata (`acs_realdata_metadata.json`); and
- PDF figures for RMSE, effective sample size, error, validation imbalance, and latent-covariate balance.

These files are useful for inspecting prior results and reproducing plots, but they are not guaranteed to be regenerated identically across operating systems, dependency versions, or changes to the underlying code.

## Interpreting diagnostics

- **Moment imbalance:** Compare `X.T @ w` with the requested target moments. Small numerical discrepancies are expected; large discrepancies indicate infeasible targets, insufficient convergence, or poor conditioning.
- **Effective sample size (ESS):** Higher ESS generally indicates less concentrated weights. A low ESS means that a small number of observations dominate the estimate and uncertainty may be large.
- **Maximum weight:** Inspect alongside ESS. Extreme weights can signal limited overlap or overly ambitious balance constraints.
- **Leaf discrepancy and validation total variation:** These assess distributional balance beyond the explicitly constrained moments. They should be interpreted with the chosen tree depth, minimum leaf mass, and score tolerance.
- **Bias, MAE, and RMSE:** In simulations, use the raw truth and repeated-sample summaries together rather than relying on a single metric.

Entropy balancing does not create overlap where none exists. Before relying on estimates, inspect covariate support and the resulting weight distribution.

## Reproducibility

- The KS simulation defines a fixed base seed (`SEED`) and derives replicate seeds from it.
- Record Python and dependency versions when producing publishable results.
- Preserve the input data, notebook execution order, parameter values, and generated output tables.
- For parallel execution, compare a small single-process run (`n_jobs=1`) with the parallel configuration when validating a new environment.
- Avoid editing generated output files by hand; regenerate them from the corresponding script or notebook.

## Notes and limitations

- This repository is primarily research and teaching code, not a packaged library.
- The default solver uses iterative numerical optimization. Convergence should be checked rather than assumed.
- Target moments must be compatible with the source data's support; infeasible constraints can produce unstable or highly variable weights.
- Pairwise products increase the number of constraints and can worsen conditioning or weight concentration.
- The hybrid method uses heuristic tree selection, learning-rate updates, stopping rules, and ESS safeguards. Treat these as methodological choices that should be justified for a specific application.
- The scripts may use multiprocessing and can require substantial memory and compute time for the default simulation settings.
- The repository does not currently define a formal test suite or continuous-integration workflow. Validate changes with small deterministic examples and the supplied analyses.

## Contributing

Suggestions, bug reports, and improvements are welcome through GitHub issues and pull requests. When proposing a change, please include:

1. a concise description of the methodological or implementation change;
2. a reproducible example or test case;
3. any effects on numerical convergence, ESS, or output schema; and
4. updated documentation or notebook notes when behavior changes.

## License and citation

No license is currently declared for this repository. Contact the repository owner before redistributing the code or using it in a product.

If you use this work, cite the repository and the associated paper or working documentation when a formal publication becomes available. The mathematical development currently included with the project is in `Doc/MathEBW.tex` and the supporting notes are in `Doc/KS_EBW_Boosted_code_note.md`.

## Acknowledgments

This project builds on the entropy-balancing literature and related work on calibration weighting, missing-data adjustment, covariate shift, and tree-based distributional balancing. Please consult the references in the notebooks and supporting notes for methodological background.
