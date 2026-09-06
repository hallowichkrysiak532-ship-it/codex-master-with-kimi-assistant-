"""Shrinkage Gaussian-copula correlation primitives.

Standalone numerical module: fitting, scoring, blending, and blending-parameter
selection for shrinkage Gaussian-copula correlation models. Only NumPy and
scikit-learn are used. No I/O, plotting, global mutable state, or randomness.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

__all__ = [
    "fit_shrunk_correlation",
    "gaussian_copula_log_density",
    "blend_correlations",
    "select_global_blend_alpha",
]

_DIAG_TOL = 1e-8
_SYMM_TOL = 1e-10


def _as_finite_2d(z, name, min_rows, min_cols):
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        raise ValueError(
            f"{name} must be a 2D array with at least "
            f"{min_rows} rows and {min_cols} columns; got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_correlation_matrix(matrix, name):
    """Return a validated copy of a correlation matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"{name} must be a square 2D matrix of size >= 2.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(arr, arr.T, atol=_SYMM_TOL, rtol=0.0):
        raise ValueError(f"{name} must be symmetric.")
    if not np.allclose(np.diag(arr), 1.0, atol=_DIAG_TOL, rtol=0.0):
        raise ValueError(f"{name} must have a unit diagonal.")
    # Strictly positive definite: Cholesky fails on non-PD input.
    try:
        np.linalg.cholesky(arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite.") from exc
    return arr


def fit_shrunk_correlation(z):
    """Fit a shrinkage correlation matrix via Ledoit-Wolf on normal scores.

    Parameters
    ----------
    z : (n, d) array_like of finite normal scores, n >= 4, d >= 2,
        with no constant columns.

    Returns
    -------
    (d, d) symmetric correlation matrix with an exact unit diagonal,
    strictly positive definite. The input is not mutated.
    """
    arr = _as_finite_2d(z, "z", min_rows=4, min_cols=2)
    # Reject raw constant columns up front: Ledoit-Wolf shrinkage would give
    # them a positive fitted variance, so the post-fit variance check cannot
    # catch them.
    if np.any(np.ptp(arr, axis=0) == 0.0) or np.any(np.var(arr, axis=0) == 0.0):
        raise ValueError(
            "z must not contain constant columns (zero range or zero variance)."
        )
    lw = LedoitWolf(assume_centered=False)
    lw.fit(arr)
    cov = np.asarray(lw.covariance_, dtype=float)
    var = np.diag(cov).copy()
    if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
        raise ValueError(
            "Ledoit-Wolf covariance has zero or nonfinite variances; "
            "cannot form a correlation matrix."
        )
    inv_sd = 1.0 / np.sqrt(var)
    corr = cov * np.outer(inv_sd, inv_sd)
    # Symmetrize numerical noise, then set the diagonal exactly to one.
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    # Guard: the shrunk covariance is PD, but enforce the contract and fail
    # loudly rather than silently clipping eigenvalues or adding jitter.
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Fitted correlation matrix is not positive definite.") from exc
    return corr


def gaussian_copula_log_density(z, correlation):
    """Gaussian-copula log density of normal scores under correlation R.

    Returns one value per row of ``z``::

        -0.5 * logdet(R) - 0.5 * z @ (inv(R) - I) @ z

    evaluated with a Cholesky factorization instead of an explicit inverse.
    """
    arr = _as_finite_2d(z, "z", min_rows=1, min_cols=2)
    r = _validate_correlation_matrix(correlation, "correlation")
    if arr.shape[1] != r.shape[0]:
        raise ValueError(
            f"z has {arr.shape[1]} columns but correlation has size {r.shape[0]}."
        )
    chol = np.linalg.cholesky(r)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    sol = np.linalg.solve(r, arr.T)  # (d, n); solves R X = z^T
    quadratic = np.einsum("ij,ji->i", arr, sol)
    d = r.shape[0]
    return -0.5 * logdet - 0.5 * (quadratic - np.einsum("ij,ij->i", arr, arr))


def blend_correlations(shared, specific, alpha):
    """Return ``(1 - alpha) * shared + alpha * specific``.

    Both matrices must be valid correlation matrices and ``alpha`` must lie
    in [0, 1]. The blend is symmetrized, given an exact unit diagonal, and
    must be strictly positive definite.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be a finite scalar in [0, 1]; got {alpha!r}.")
    s = _validate_correlation_matrix(shared, "shared")
    p = _validate_correlation_matrix(specific, "specific")
    if s.shape != p.shape:
        raise ValueError(
            f"shared has shape {s.shape} but specific has shape {p.shape}."
        )
    blended = (1.0 - alpha) * s + alpha * p
    blended = 0.5 * (blended + blended.T)
    np.fill_diagonal(blended, 1.0)
    try:
        np.linalg.cholesky(blended)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Blended correlation matrix is not positive definite.") from exc
    return blended


def select_global_blend_alpha(shared, specific_by_source, validation_by_source, alpha_grid):
    """Select the global blend weight by validation log density.

    For each candidate alpha, blends every source-specific fitted correlation
    matrix with ``shared`` and computes the sample-weighted mean validation
    Gaussian-copula log density across sources. Returns the alpha with the
    largest score; ties choose the smaller alpha.

    Returns
    -------
    (best_alpha, diagnostics) where diagnostics contains the ordered alpha
    grid, the mean score for every alpha, the total validation rows, and the
    selected score.
    """
    s = _validate_correlation_matrix(shared, "shared")
    d = s.shape[0]

    if not isinstance(specific_by_source, dict) or not specific_by_source:
        raise ValueError("specific_by_source must be a non-empty dict.")
    if not isinstance(validation_by_source, dict) or not validation_by_source:
        raise ValueError("validation_by_source must be a non-empty dict.")
    if set(specific_by_source) != set(validation_by_source):
        raise ValueError(
            "specific_by_source and validation_by_source must have identical keys."
        )

    grid = np.asarray(alpha_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha_grid must be a non-empty 1D array.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("alpha_grid must contain only finite values.")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("alpha_grid must be strictly increasing with no duplicates.")
    if grid[0] < 0.0 or grid[-1] > 1.0:
        raise ValueError("alpha_grid values must lie in [0, 1].")

    val = {}
    total_rows = 0
    for key, zv in validation_by_source.items():
        arr = _as_finite_2d(zv, f"validation_by_source[{key!r}]", min_rows=1, min_cols=2)
        if arr.shape[1] != d:
            raise ValueError(
                f"validation_by_source[{key!r}] has {arr.shape[1]} columns; "
                f"expected {d}."
            )
        specific = _validate_correlation_matrix(
            specific_by_source[key], f"specific_by_source[{key!r}]"
        )
        if specific.shape != (d, d):
            raise ValueError(
                f"specific_by_source[{key!r}] has shape {specific.shape}; "
                f"expected {(d, d)}."
            )
        val[key] = arr
        total_rows += arr.shape[0]

    mean_scores = np.empty(grid.size, dtype=float)
    for i, alpha in enumerate(grid):
        log_density_sum = 0.0
        for key, arr in val.items():
            blended = blend_correlations(s, specific_by_source[key], float(alpha))
            log_density_sum += float(np.sum(gaussian_copula_log_density(arr, blended)))
        mean_scores[i] = log_density_sum / float(total_rows)

    best_index = int(np.argmax(mean_scores))  # first max: ties pick smaller alpha
    best_alpha = float(grid[best_index])
    diagnostics = {
        "alpha_grid": grid.copy(),
        "mean_scores": mean_scores.copy(),
        "total_validation_rows": total_rows,
        "selected_score": float(mean_scores[best_index]),
    }
    return best_alpha, diagnostics
