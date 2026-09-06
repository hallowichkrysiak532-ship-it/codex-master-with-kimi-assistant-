"""Focused tests for gaussian_copula_models. Run with: python -m pytest -q"""

import numpy as np
import pytest

from gaussian_copula_models import (
    blend_correlations,
    fit_shrunk_correlation,
    gaussian_copula_log_density,
    select_global_blend_alpha,
)


def _deterministic_scores(n, d, rng):
    """Full-rank deterministic normal-score-like data via a seeded RNG."""
    z = rng.standard_normal((n, d))
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    return z


def _corr(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


# ---------------------------------------------------------------- fit

def test_fit_shrunk_correlation_properties():
    rng = np.random.default_rng(0)
    z = _deterministic_scores(50, 4, rng)
    corr = fit_shrunk_correlation(z)
    assert corr.shape == (4, 4)
    assert np.allclose(corr, corr.T, atol=1e-12)
    assert np.array_equal(np.diag(corr), np.ones(4))
    eig = np.linalg.eigvalsh(corr)
    assert np.all(eig > 0.0)


def test_fit_rejects_nonfinite_and_constant_column():
    rng = np.random.default_rng(1)
    z = _deterministic_scores(10, 3, rng)
    bad = z.copy()
    bad[2, 1] = np.nan
    with pytest.raises(ValueError):
        fit_shrunk_correlation(bad)
    constant = z.copy()
    constant[:, 0] = 1.5
    with pytest.raises(ValueError):
        fit_shrunk_correlation(constant)


def test_fit_rejects_bad_shape():
    with pytest.raises(ValueError):
        fit_shrunk_correlation(np.ones((3, 3)))
    with pytest.raises(ValueError):
        fit_shrunk_correlation(np.ones((10, 1)))


def test_fit_does_not_mutate_input():
    rng = np.random.default_rng(2)
    z = _deterministic_scores(20, 3, rng)
    before = z.copy()
    fit_shrunk_correlation(z)
    assert np.array_equal(z, before)


# ---------------------------------------------------------- log density

def test_log_density_identity_is_zero():
    rng = np.random.default_rng(3)
    z = _deterministic_scores(12, 3, rng)
    out = gaussian_copula_log_density(z, np.eye(3))
    assert out.shape == (12,)
    assert np.allclose(out, 0.0, atol=1e-12)


def test_log_density_matches_direct_formula():
    r = _corr(0.5)
    z = np.array([[0.3, -1.2], [2.0, 0.4], [-0.7, 0.9]])
    inv = np.linalg.inv(r)
    d = r.shape[0]
    expected = -0.5 * np.linalg.slogdet(r)[1] - 0.5 * np.einsum(
        "ij,jk,ik->i", z, inv - np.eye(d), z
    )
    assert np.allclose(gaussian_copula_log_density(z, r), expected, atol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[1.0, 0.9], [0.1, 1.0]]),  # nonsymmetric
        np.array([[1.5, 0.1], [0.1, 1.0]]),  # non-unit diagonal
        np.array([[1.0, 1.2], [1.2, 1.0]]),  # not positive definite
        np.array([[1.0, np.nan], [np.nan, 1.0]]),  # nonfinite
    ],
)
def test_log_density_rejects_invalid_correlation(bad):
    z = np.array([[0.1, 0.2], [0.3, -0.4]])
    with pytest.raises(ValueError):
        gaussian_copula_log_density(z, bad)


def test_log_density_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        gaussian_copula_log_density(np.ones((3, 3)), np.eye(2))


# ---------------------------------------------------------------- blend

def test_blend_endpoints():
    a, b = _corr(0.2), _corr(0.8)
    assert np.allclose(blend_correlations(a, b, 0.0), a)
    assert np.allclose(blend_correlations(a, b, 1.0), b)
    mid = blend_correlations(a, b, 0.5)
    assert np.allclose(mid, 0.5 * a + 0.5 * b)
    assert np.all(np.linalg.eigvalsh(mid) > 0.0)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf])
def test_blend_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        blend_correlations(np.eye(2), np.eye(2), alpha)


def test_blend_rejects_invalid_matrices():
    good = _corr(0.3)
    with pytest.raises(ValueError):
        blend_correlations(good, np.array([[1.0, 0.9], [0.1, 1.0]]), 0.5)
    with pytest.raises(ValueError):
        blend_correlations(good, np.eye(3), 0.5)  # shape mismatch
    with pytest.raises(ValueError):
        blend_correlations(good, np.array([[1.0, 1.1], [1.1, 1.0]]), 0.5)  # non-PD specific


# -------------------------------------------------------- alpha selector

def _selector_fixture():
    rng = np.random.default_rng(7)
    specific = {"A": _corr(0.7), "B": _corr(-0.6)}
    val_a = _deterministic_scores(30, 2, rng)
    val_b = _deterministic_scores(30, 2, rng)
    # Corrupt the validation scores with each source's true dependence.
    val_a = val_a + np.column_stack([0.7 * val_a[:, 1], 0.7 * val_a[:, 0]])
    val_b = val_b + np.column_stack([-0.6 * val_b[:, 1], -0.6 * val_b[:, 0]])
    val_a = (val_a - val_a.mean(0)) / val_a.std(0)
    val_b = (val_b - val_b.mean(0)) / val_b.std(0)
    return np.eye(2), specific, {"A": val_a, "B": val_b}


def test_selector_prefers_nonzero_alpha_for_specific_dependence():
    shared, specific, val = _selector_fixture()
    best, diag = select_global_blend_alpha(shared, specific, val, [0.0, 0.5, 1.0])
    assert best > 0.0
    assert diag["mean_scores"][1] > diag["mean_scores"][0]
    assert diag["alpha_grid"].tolist() == [0.0, 0.5, 1.0]
    assert diag["total_validation_rows"] == 60
    assert diag["selected_score"] == diag["mean_scores"][list(diag["alpha_grid"]).index(best)]


def test_selector_tie_picks_smaller_alpha():
    shared = np.eye(2)
    specific = {"A": np.eye(2), "B": np.eye(2)}
    val = {"A": np.array([[0.1, 0.2], [0.3, -0.1]]),
           "B": np.array([[-0.4, 0.5]])}
    best, diag = select_global_blend_alpha(shared, specific, val, [0.2, 0.5, 0.9])
    assert best == 0.2
    assert len(set(diag["mean_scores"].round(12))) == 1


@pytest.mark.parametrize(
    "grid",
    [[], [0.5, 0.5], [0.7, 0.2], [-0.1, 0.5], [0.5, 1.2], [np.nan, 0.5]],
)
def test_selector_rejects_bad_grid(grid):
    shared = np.eye(2)
    specific = {"A": np.eye(2)}
    val = {"A": np.array([[0.1, 0.2], [0.3, -0.1]])}
    with pytest.raises(ValueError):
        select_global_blend_alpha(shared, specific, val, grid)


def test_selector_rejects_key_mismatch_and_bad_dims():
    shared = np.eye(2)
    specific = {"A": np.eye(2), "B": np.eye(2)}
    val = {"A": np.array([[0.1, 0.2], [0.3, -0.1]]),
           "C": np.array([[0.1, 0.2], [0.3, -0.1]])}
    with pytest.raises(ValueError):
        select_global_blend_alpha(shared, specific, val, [0.0, 1.0])
    with pytest.raises(ValueError):
        select_global_blend_alpha(
            np.eye(3), {"A": np.eye(2)},
            {"A": np.array([[0.1, 0.2], [0.3, -0.1]])}, [0.0, 1.0],
        )
    with pytest.raises(ValueError):
        select_global_blend_alpha(
            np.eye(2), {"A": np.eye(3)},
            {"A": np.array([[0.1, 0.2], [0.3, -0.1]])}, [0.0, 1.0],
        )
    with pytest.raises(ValueError):
        select_global_blend_alpha(
            np.eye(2), {"A": np.eye(2)},
            {"A": np.ones((2, 3))}, [0.0, 1.0],
        )
