"""Frozen shared-copula transfer primitives.

Standalone numerical module used by a larger experiment. NumPy only; no I/O,
plotting, randomness, or mutable global state. No caller-provided array or
mapping is mutated.
"""

from collections.abc import Mapping

import numpy as np

__all__ = [
    "low_rank_correlation",
    "select_rank_from_log_density",
    "correlation_pattern_metrics",
]


def _is_integer_rank(rank):
    return isinstance(rank, (int, np.integer)) and not isinstance(
        rank, (bool, np.bool_)
    )


def _validate_correlation_matrix(matrix, name="correlation"):
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {arr.shape}.")
    arr = arr.astype(float, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    diag = np.diag(arr)
    if not np.allclose(diag, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must have a (close to) unit diagonal.")
    try:
        np.linalg.cholesky(arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite.") from exc
    return arr


def low_rank_correlation(correlation, rank):
    """Keep the ``rank`` dominant modes of ``correlation`` and replace the rest.

    The matrix is eigen-decomposed; modes are ordered by descending
    ``abs(eigenvalue - 1)`` with deterministic index tie breaking. Omitted
    eigenvalues are replaced by one, the matrix is reconstructed, symmetrized,
    and renormalized to a unit diagonal. Rank 0 returns the identity; rank d
    returns a numerical copy of the input. Eigenvalues are never clipped and no
    jitter is added.
    """
    matrix = _validate_correlation_matrix(correlation, name="correlation")
    d = matrix.shape[0]

    if not _is_integer_rank(rank):
        raise TypeError(f"rank must be an integer, got {type(rank).__name__}.")
    rank = int(rank)
    if not 0 <= rank <= d:
        raise ValueError(f"rank must lie in [0, {d}], got {rank}.")

    if rank == 0:
        return np.eye(d)
    if rank == d:
        return matrix.copy()

    eigvals, eigvecs = np.linalg.eigh(matrix)  # ascending eigenvalues
    deviation = np.abs(eigvals - 1.0)
    order = np.argsort(-deviation, kind="stable")  # descending; ties keep index
    keep = order[:rank]

    new_eigvals = np.ones(d)
    new_eigvals[keep] = eigvals[keep]

    reconstructed = (eigvecs * new_eigvals) @ eigvecs.T
    reconstructed = 0.5 * (reconstructed + reconstructed.T)

    scales = np.sqrt(np.diag(reconstructed))
    return reconstructed / np.outer(scales, scales)


def select_rank_from_log_density(log_density_by_rank):
    """Select the rank maximizing the arithmetic mean of per-sample log densities.

    Exact ties resolve to the smaller rank. Returns ``(rank, diagnostics)``
    where diagnostics contains the sorted candidate ranks, their mean scores,
    the (shared) sample count, and the selected score.
    """
    if not isinstance(log_density_by_rank, Mapping):
        raise TypeError("log_density_by_rank must be a mapping from rank to samples.")
    if len(log_density_by_rank) == 0:
        raise ValueError("log_density_by_rank must be nonempty.")

    means = {}
    sample_counts = set()
    for key, samples in log_density_by_rank.items():
        if not _is_integer_rank(key):
            raise TypeError(f"rank keys must be integers, got {key!r}.")
        key = int(key)
        if key < 0:
            raise ValueError(f"rank keys must be nonnegative, got {key}.")
        arr = np.asarray(samples)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"samples for rank {key} must be a nonempty 1D array.")
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"samples for rank {key} must be numeric."
            ) from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"samples for rank {key} must be finite.")
        means[key] = float(np.mean(arr))
        sample_counts.add(arr.size)

    if len(sample_counts) != 1:
        raise ValueError(
            "all ranks must have equal sample counts, got "
            f"{sorted(sample_counts)}."
        )

    ranks = sorted(means)
    scores = [means[r] for r in ranks]
    selected = max(ranks, key=lambda r: means[r])  # sorted: ties -> smaller rank

    diagnostics = {
        "candidate_ranks": ranks,
        "mean_scores": scores,
        "n_samples": sample_counts.pop(),
        "selected_score": means[selected],
    }
    return selected, diagnostics


def _upper_triangle(matrix):
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def correlation_pattern_metrics(reference, candidate):
    """Upper-triangle pattern correlation, RMSE, and sign agreement.

    ``pattern_r`` is the Pearson correlation of the strict upper triangles;
    it is undefined for a constant reference triangle and raises ValueError.
    ``sign_agreement`` is the fraction of entries with equal ``np.sign``.
    """
    ref = _validate_correlation_matrix(reference, name="reference")
    cand = _validate_correlation_matrix(candidate, name="candidate")
    if ref.shape != cand.shape:
        raise ValueError(
            f"reference and candidate shapes must match, got {ref.shape} and {cand.shape}."
        )
    d = ref.shape[0]
    if d < 3:
        raise ValueError(f"dimension must be at least three, got {d}.")

    ref_tri = _upper_triangle(ref)
    cand_tri = _upper_triangle(cand)

    if np.all(ref_tri == ref_tri[0]) or np.all(cand_tri == cand_tri[0]):
        raise ValueError("pattern correlation is undefined for a constant upper triangle.")

    pattern_r = float(np.corrcoef(ref_tri, cand_tri)[0, 1])
    rmse = float(np.sqrt(np.mean((ref_tri - cand_tri) ** 2)))
    sign_agreement = float(np.mean(np.sign(ref_tri) == np.sign(cand_tri)))

    return {
        "pattern_r": pattern_r,
        "rmse": rmse,
        "sign_agreement": sign_agreement,
    }
