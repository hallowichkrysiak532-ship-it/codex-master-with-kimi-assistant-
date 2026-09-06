"""Shrinkage Gaussian-copula correlation primitives.

Standalone numerical module: fitting, scoring, blending, and blending-parameter
selection. There is no I/O, plotting, mutable global state, or randomness.
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
            f"{name} must be a 2D array with at least {min_rows} rows and "
            f"{min_cols} columns; got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_correlation_matrix(matrix, name):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"{name} must be a square 2D matrix of size >= 2.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(arr, arr.T, atol=_SYMM_TOL, rtol=0.0):
        raise ValueError(f"{name} must be symmetric.")
    if not np.allclose(np.diag(arr), 1.0, atol=_DIAG_TOL, rtol=0.0):
        raise ValueError(f"{name} must have a unit diagonal.")
    try:
        np.linalg.cholesky(arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite.") from exc
    return arr


def fit_shrunk_correlation(z):
    """Fit a Ledoit-Wolf covariance and return its correlation matrix."""
    arr = _as_finite_2d(z, "z", min_rows=4, min_cols=2)
    # Shrinkage gives a raw constant column positive fitted variance, so raw
    # degeneracy must be rejected before fitting.
    if np.any(np.ptp(arr, axis=0) == 0.0) or np.any(np.var(arr, axis=0) == 0.0):
        raise ValueError("z must not contain constant columns.")
    fitted = LedoitWolf(assume_centered=False).fit(arr)
    covariance = np.asarray(fitted.covariance_, dtype=float)
    variance = np.diag(covariance).copy()
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("Fitted covariance has invalid variances.")
    inverse_sd = 1.0 / np.sqrt(variance)
    correlation = covariance * np.outer(inverse_sd, inverse_sd)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Fitted correlation is not positive definite.") from exc
    return correlation


def gaussian_copula_log_density(z, correlation):
    """Return row-wise Gaussian-copula log density for normal scores."""
    arr = _as_finite_2d(z, "z", min_rows=1, min_cols=2)
    corr = _validate_correlation_matrix(correlation, "correlation")
    if arr.shape[1] != corr.shape[0]:
        raise ValueError(
            f"z has {arr.shape[1]} columns but correlation has size {corr.shape[0]}."
        )
    cholesky = np.linalg.cholesky(corr)
    logdet = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    solved = np.linalg.solve(corr, arr.T)
    quadratic = np.einsum("ij,ji->i", arr, solved)
    identity_quadratic = np.einsum("ij,ij->i", arr, arr)
    return -0.5 * logdet - 0.5 * (quadratic - identity_quadratic)


def blend_correlations(shared, specific, alpha):
    """Return the positive-definite convex blend of two correlations."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1].")
    common = _validate_correlation_matrix(shared, "shared")
    source = _validate_correlation_matrix(specific, "specific")
    if common.shape != source.shape:
        raise ValueError("shared and specific must have equal shapes.")
    result = (1.0 - alpha) * common + alpha * source
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 1.0)
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Blended correlation is not positive definite.") from exc
    return result


def select_global_blend_alpha(
    shared, specific_by_source, validation_by_source, alpha_grid
):
    """Select one blend weight by sample-weighted validation log density."""
    common = _validate_correlation_matrix(shared, "shared")
    dimension = common.shape[0]
    if not isinstance(specific_by_source, dict) or not specific_by_source:
        raise ValueError("specific_by_source must be a non-empty dict.")
    if not isinstance(validation_by_source, dict) or not validation_by_source:
        raise ValueError("validation_by_source must be a non-empty dict.")
    if set(specific_by_source) != set(validation_by_source):
        raise ValueError("specific and validation source keys must match.")
    grid = np.asarray(alpha_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha_grid must be a non-empty 1D array.")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0.0):
        raise ValueError("alpha_grid must be finite and strictly increasing.")
    if grid[0] < 0.0 or grid[-1] > 1.0:
        raise ValueError("alpha_grid values must lie in [0, 1].")
    validated, total_rows = {}, 0
    for source, values in validation_by_source.items():
        value = _as_finite_2d(
            values, f"validation_by_source[{source!r}]", min_rows=1, min_cols=2
        )
        if value.shape[1] != dimension:
            raise ValueError("Validation dimension does not match shared correlation.")
        specific = _validate_correlation_matrix(
            specific_by_source[source], f"specific_by_source[{source!r}]"
        )
        if specific.shape != common.shape:
            raise ValueError("Specific dimension does not match shared correlation.")
        validated[source] = value
        total_rows += len(value)
    scores = np.empty(len(grid), dtype=float)
    for index, alpha in enumerate(grid):
        total = 0.0
        for source, values in validated.items():
            correlation = blend_correlations(
                common, specific_by_source[source], float(alpha)
            )
            total += float(np.sum(gaussian_copula_log_density(values, correlation)))
        scores[index] = total / float(total_rows)
    selected_index = int(np.argmax(scores))
    selected = float(grid[selected_index])
    diagnostics = {
        "alpha_grid": grid.copy(),
        "mean_scores": scores.copy(),
        "total_validation_rows": total_rows,
        "selected_score": float(scores[selected_index]),
    }
    return selected, diagnostics
