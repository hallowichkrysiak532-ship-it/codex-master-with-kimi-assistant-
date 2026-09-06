"""Focused tests for source_empirical_copula."""

import inspect

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import rankdata, spearmanr

from source_empirical_copula import (
    apply_reference_ecdf,
    fit_reference_ecdf,
    normal_scores,
    offdiagonal_dependence,
    pseudo_observations,
)


def _sample() -> np.ndarray:
    """Deterministic 8x3 sample, columns 1 and 2 strictly positive."""
    return np.array(
        [
            [0.5, 1.0, 2.0],
            [1.5, 2.0, 1.0],
            [2.5, 1.5, 3.0],
            [3.5, 3.0, 2.5],
            [4.5, 2.5, 4.0],
            [5.5, 4.0, 1.5],
            [6.5, 3.5, 3.5],
            [7.5, 5.0, 5.0],
        ]
    )


# --------------------------------------------------------------------------
# pseudo_observations
# --------------------------------------------------------------------------


def test_pseudo_observations_exact_grid_tie_free():
    x = np.arange(8.0).reshape(-1, 1) * 1.7 + 0.3
    u = pseudo_observations(x)
    expected = (np.arange(8) + 0.5) / 8.0
    np.testing.assert_allclose(np.sort(u[:, 0]), expected, rtol=0, atol=1e-15)
    assert u.shape == x.shape
    assert np.all(u > 0.0) and np.all(u < 1.0)
    assert np.all(np.isfinite(u))


def test_pseudo_observations_tie_midranks():
    x = np.array([[1.0], [2.0], [2.0], [3.0]])
    u = pseudo_observations(x)
    np.testing.assert_allclose(
        u[:, 0], np.array([0.125, 0.5, 0.5, 0.875]), rtol=0, atol=1e-15
    )


def test_pseudo_observations_invariance_increasing_transforms():
    x = _sample()
    transformed = np.column_stack(
        [3.0 * x[:, 0] + 1.0, np.exp(x[:, 1]), x[:, 2] ** 3]
    )
    np.testing.assert_allclose(
        pseudo_observations(x), pseudo_observations(transformed), atol=1e-12
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(5),  # 1D
        np.ones((2, 3, 4)),  # 3D
        np.ones((3, 2)),  # fewer than 4 rows
        np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
        np.array([[np.inf, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
    ],
)
def test_pseudo_observations_rejects_invalid(bad):
    with pytest.raises((ValueError, TypeError)):
        pseudo_observations(bad)


# --------------------------------------------------------------------------
# fit_reference_ecdf / apply_reference_ecdf
# --------------------------------------------------------------------------


def test_fit_reference_ecdf_single_argument_returns_sorted_copy():
    sig = inspect.signature(fit_reference_ecdf)
    assert len(sig.parameters) == 1
    ref = np.array([[3.0, 5.0], [1.0, 2.0], [4.0, 4.0], [2.0, 3.0]])
    out = fit_reference_ecdf(ref)
    assert out is not ref
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, np.sort(ref, axis=0))
    # Mutating the fitted output must not touch the caller's reference.
    out[0, 0] = -999.0
    np.testing.assert_array_equal(ref, np.array([[3.0, 5.0], [1.0, 2.0], [4.0, 4.0], [2.0, 3.0]]))


def test_fit_reference_ecdf_rejects_invalid():
    with pytest.raises(ValueError):
        fit_reference_ecdf(np.ones((3, 2)))
    with pytest.raises(ValueError):
        fit_reference_ecdf(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))


def test_apply_reference_ecdf_hand_computed():
    # Reference column: [1, 2, 2, 5] (n=4, denominator n+1=5).
    ref = fit_reference_ecdf(
        np.tile(np.array([5.0, 2.0, 1.0, 2.0]).reshape(-1, 1), (1, 2))
    )
    eval_x = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [6.0, 6.0]])
    u = apply_reference_ecdf(eval_x, ref)
    # below min: (0+0+0.5)/5; at tied 2: (1+1+0.5)/5; between: (3+0.5)/5;
    # above max: (4+0.5)/5.
    expected = np.tile(np.array([0.1, 0.5, 0.7, 0.9]).reshape(-1, 1), (1, 2))
    np.testing.assert_allclose(u, expected, rtol=0, atol=1e-15)
    assert np.all(u > 0.0) and np.all(u < 1.0)


def test_apply_reference_ecdf_affine_invariance():
    rng_ref = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 3.0]])
    rng_eval = np.array([[0.5, 1.5], [2.5, 2.5], [6.0, 0.0]])
    a, b = 2.0, 3.0  # power-of-two scale and integer shift are float-exact
    ref = fit_reference_ecdf(rng_ref)
    u1 = apply_reference_ecdf(rng_eval, ref)
    ref2 = fit_reference_ecdf(a * rng_ref + b)
    u2 = apply_reference_ecdf(a * rng_eval + b, ref2)
    np.testing.assert_allclose(u1, u2, rtol=0, atol=1e-15)


def test_apply_reference_ecdf_does_not_mutate_reference():
    ref = fit_reference_ecdf(_sample()[:, :1])
    before = ref.copy()
    apply_reference_ecdf(np.array([[0.1], [1.0], [50.0]]), ref)
    np.testing.assert_array_equal(ref, before)


@pytest.mark.parametrize(
    "eval_x, ref",
    [
        (np.ones((2, 3)), np.ones((5, 2))),  # coordinate mismatch
        (np.ones((2, 2)), np.array([[2.0, 1.0], [1.0, 2.0], [3.0, 3.0], [4.0, 4.0]])),  # unsorted
        (np.ones((2, 2)), np.ones((3, 2))),  # fewer than 4 reference rows
        (
            np.array([[np.nan, 1.0], [2.0, 3.0]]),
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        ),  # nonfinite evaluation
        (
            np.ones((2, 2)),
            np.array([[1.0, np.inf], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        ),  # nonfinite reference
        (np.ones(3), np.ones((4, 3))),  # evaluation not 2D
        (np.ones((2, 2)), np.ones(4)),  # reference not 2D
    ],
)
def test_apply_reference_ecdf_rejects_invalid(eval_x, ref):
    with pytest.raises((ValueError, TypeError)):
        apply_reference_ecdf(eval_x, ref)


# --------------------------------------------------------------------------
# normal_scores
# --------------------------------------------------------------------------


def test_normal_scores_match_ndtri_and_finite():
    u = np.array([[0.05, 0.25], [0.5, 0.75], [0.9, 0.999], [0.001, 0.6]])
    scores = normal_scores(u)
    np.testing.assert_allclose(scores, ndtri(u), rtol=0, atol=1e-15)
    assert scores.shape == u.shape
    assert np.all(np.isfinite(scores))


def test_normal_scores_rejects_boundaries_and_nonfinite():
    with pytest.raises(ValueError):
        normal_scores(np.array([[0.0, 0.5], [0.25, 0.75]]))
    with pytest.raises(ValueError):
        normal_scores(np.array([[1.0, 0.5], [0.25, 0.75]]))
    with pytest.raises(ValueError):
        normal_scores(np.array([[np.nan, 0.5], [0.25, 0.75]]))
    with pytest.raises(ValueError):
        normal_scores(np.ones(4) * 0.5)  # 1D


# --------------------------------------------------------------------------
# offdiagonal_dependence
# --------------------------------------------------------------------------


def test_offdiagonal_dependence_matches_scipy_with_ties():
    x = np.array(
        [
            [1.0, 2.0, 5.0],
            [1.0, 4.0, 3.0],  # tie in column 0
            [3.0, 1.0, 4.0],
            [2.0, 3.0, 5.0],  # tie in column 2
            [4.0, 5.0, 1.0],
            [0.0, 2.0, 2.0],
        ]
    )
    out = offdiagonal_dependence(x)
    pearson_full = np.corrcoef(x, rowvar=False)
    spearman_full = spearmanr(x).correlation
    iu = np.triu_indices(3, k=1)
    np.testing.assert_allclose(out["pearson"], pearson_full[iu], atol=1e-12)
    np.testing.assert_allclose(out["spearman"], spearman_full[iu], atol=1e-12)
    # Spearman with ties must equal Pearson on average ranks.
    ranked = np.column_stack([rankdata(x[:, j], method="average") for j in range(3)])
    np.testing.assert_allclose(
        out["spearman"], np.corrcoef(ranked, rowvar=False)[iu], atol=1e-12
    )
    assert out["pearson"].shape == (3,)
    assert out["spearman"].shape == (3,)


def test_offdiagonal_dependence_rejects_constant_coordinate():
    x = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
    with pytest.raises(ValueError):
        offdiagonal_dependence(x)


def test_offdiagonal_dependence_rejects_invalid():
    with pytest.raises(ValueError):
        offdiagonal_dependence(np.ones((5, 1)))  # only one coordinate
    with pytest.raises(ValueError):
        offdiagonal_dependence(np.ones((3, 2)))  # too few rows
    with pytest.raises(ValueError):
        offdiagonal_dependence(
            np.array([[1.0, 2.0], [3.0, np.inf], [4.0, 5.0], [6.0, 7.0]])
        )
