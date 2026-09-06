"""Empirical copula transforms: pseudo-observations, reference ECDF mapping
and dependence summaries. Standalone NumPy/SciPy module; no I/O, no
randomness, no global state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import rankdata

__all__ = [
    "pseudo_observations",
    "fit_reference_ecdf",
    "apply_reference_ecdf",
    "normal_scores",
    "offdiagonal_dependence",
]

FloatArray = NDArray[np.float64]

_MIN_ROWS = 4


def _as_2d_finite(x: object, name: str) -> FloatArray:
    """Coerce to float ndarray and require finite 2D layout."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float array: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_sample(x: object, name: str = "x") -> FloatArray:
    """Validate a sample-by-coordinate array: 2D, finite, >=4 rows, >=1 col."""
    arr = _as_2d_finite(x, name)
    n, d = arr.shape
    if n < _MIN_ROWS:
        raise ValueError(f"{name} must have at least {_MIN_ROWS} rows, got {n}")
    if d < 1:
        raise ValueError(f"{name} must have at least 1 column, got {d}")
    return arr


def pseudo_observations(x: object) -> FloatArray:
    """Rank-normalize each coordinate to pseudo-observations in (0, 1).

    Per coordinate ``u = (rankdata(x[:, j], method='average') - 0.5) / n``
    so ties receive equal midranks and tie-free sorted values equal
    ``(arange(n) + 0.5) / n``.
    """
    arr = _validate_sample(x)
    n = arr.shape[0]
    u = np.empty_like(arr)
    for j in range(arr.shape[1]):
        u[:, j] = (rankdata(arr[:, j], method="average") - 0.5) / n
    return u


def fit_reference_ecdf(reference: object) -> FloatArray:
    """Fit a reference ECDF by sorting each coordinate independently.

    Accepts only the reference sample (single argument) so evaluation data
    cannot leak into the fitted reference. Returns a new array with values
    sorted nondecreasing per column; shape is unchanged.
    """
    arr = _validate_sample(reference, name="reference")
    return np.sort(arr, axis=0)


def _validate_sorted_reference(sorted_reference: object) -> FloatArray:
    ref = _as_2d_finite(sorted_reference, "sorted_reference")
    n, d = ref.shape
    if n < _MIN_ROWS:
        raise ValueError(
            f"sorted_reference must have at least {_MIN_ROWS} rows, got {n}"
        )
    if d < 1:
        raise ValueError(f"sorted_reference must have at least 1 column, got {d}")
    if not np.all(np.diff(ref, axis=0) >= 0.0):
        raise ValueError("sorted_reference must be nondecreasing in every coordinate")
    return ref


def apply_reference_ecdf(evaluation: object, sorted_reference: object) -> FloatArray:
    """Map evaluation values through the fitted reference ECDF.

    For each value ``x`` and coordinate with sorted reference ``r`` of length
    ``m``: ``u = (count(r < x) + 0.5 * count(r == x) + 0.5) / (m + 1)``,
    computed with searchsorted (no loop over reference rows). Results are
    strictly inside (0, 1) by construction; ``sorted_reference`` is never
    modified.
    """
    eval_arr = _as_2d_finite(evaluation, "evaluation")
    if eval_arr.shape[0] < 1:
        raise ValueError("evaluation must have at least 1 row")
    ref = _validate_sorted_reference(sorted_reference)
    if eval_arr.shape[1] != ref.shape[1]:
        raise ValueError(
            "evaluation and sorted_reference must share the coordinate count: "
            f"{eval_arr.shape[1]} vs {ref.shape[1]}"
        )
    m = ref.shape[0]
    denom = m + 1.0
    u = np.empty_like(eval_arr)
    for j in range(ref.shape[1]):
        r = ref[:, j]
        x = eval_arr[:, j]
        left = np.searchsorted(r, x, side="left")
        right = np.searchsorted(r, x, side="right")
        u[:, j] = (left + 0.5 * (right - left) + 0.5) / denom
    return u


def normal_scores(u: object) -> FloatArray:
    """Map pseudo-observations in (0, 1) to standard normal scores via ndtri."""
    arr = _as_2d_finite(u, "u")
    if arr.size == 0:
        raise ValueError("u must be nonempty")
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError("u must lie strictly inside (0, 1); boundaries are invalid")
    scores = ndtri(arr)
    if not np.all(np.isfinite(scores)):
        raise ValueError("normal scores are not finite")
    return scores


def offdiagonal_dependence(x: object) -> dict[str, FloatArray]:
    """Upper-triangle Pearson and Spearman correlations across coordinates.

    Returns a dict with keys ``pearson`` and ``spearman``, each a 1D array
    of the upper-triangle (k=1) entries of the across-sample correlation
    matrix. Constant coordinates are rejected.
    """
    arr = _validate_sample(x)
    n, d = arr.shape
    if d < 2:
        raise ValueError(f"x must have at least 2 coordinates, got {d}")
    spread = np.ptp(arr, axis=0)
    if np.any(spread == 0.0):
        raise ValueError("constant coordinates are not allowed")
    pearson = np.corrcoef(arr, rowvar=False)
    ranked = np.empty_like(arr)
    for j in range(d):
        ranked[:, j] = rankdata(arr[:, j], method="average")
    spearman = np.corrcoef(ranked, rowvar=False)
    iu = np.triu_indices(d, k=1)
    return {
        "pearson": np.asarray(pearson[iu], dtype=np.float64),
        "spearman": np.asarray(spearman[iu], dtype=np.float64),
    }
