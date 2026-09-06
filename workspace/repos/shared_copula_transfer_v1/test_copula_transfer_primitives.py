"""Focused tests for copula_transfer_primitives."""

import math

import numpy as np
import pytest

from copula_transfer_primitives import (
    correlation_pattern_metrics,
    low_rank_correlation,
    select_rank_from_log_density,
)


def equicorr(d, rho):
    return rho * np.ones((d, d)) + (1.0 - rho) * np.eye(d)


def ar1(d, rho):
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


# ---------------------------------------------------------------------------
# low_rank_correlation
# ---------------------------------------------------------------------------

class TestLowRankCorrelation:
    def test_rank_zero_returns_identity(self):
        result = low_rank_correlation(equicorr(4, 0.5), 0)
        np.testing.assert_array_equal(result, np.eye(4))

    def test_rank_d_returns_numerical_copy(self):
        corr = ar1(3, 0.5)
        result = low_rank_correlation(corr, 3)
        np.testing.assert_allclose(result, corr, rtol=0, atol=1e-15)
        assert result is not corr
        result[0, 1] = -99.0
        assert corr[0, 1] == 0.5

    def test_intermediate_rank_hand_computed(self):
        # equicorr(4, 0.5) has eigenvalues 2.5 and 0.5 (x3); keeping only the
        # 2.5 mode and renormalizing gives off-diagonal 3/11.
        result = low_rank_correlation(equicorr(4, 0.5), 1)
        expected = (3.0 / 11.0) * np.ones((4, 4)) + (8.0 / 11.0) * np.eye(4)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_intermediate_rank_invariants(self):
        result = low_rank_correlation(ar1(4, 0.5), 2)
        np.testing.assert_allclose(result, result.T, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diag(result), np.ones(4), rtol=0, atol=1e-12)
        np.linalg.cholesky(result)  # raises if not positive definite

    def test_numpy_integer_rank_accepted(self):
        result = low_rank_correlation(equicorr(4, 0.5), np.int64(1))
        expected = (3.0 / 11.0) * np.ones((4, 4)) + (8.0 / 11.0) * np.eye(4)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_tie_breaking_is_deterministic(self):
        # Three modes of equicorr(4, 0.5) tie on abs(eigenvalue - 1); the
        # stable index tie break must give identical output across calls.
        corr = equicorr(4, 0.5)
        first = low_rank_correlation(corr, 2)
        second = low_rank_correlation(corr.copy(), 2)
        np.testing.assert_array_equal(first, second)

    def test_does_not_mutate_input(self):
        corr = ar1(3, 0.5)
        snapshot = corr.copy()
        low_rank_correlation(corr, 1)
        low_rank_correlation(corr, 2)
        np.testing.assert_array_equal(corr, snapshot)

    @pytest.mark.parametrize("bad_rank", [True, False, np.bool_(True)])
    def test_bool_rank_rejected(self, bad_rank):
        with pytest.raises(TypeError):
            low_rank_correlation(equicorr(3, 0.5), bad_rank)

    @pytest.mark.parametrize("bad_rank", [1.5, 2.0, "1", None, [1]])
    def test_noninteger_rank_rejected(self, bad_rank):
        with pytest.raises(TypeError):
            low_rank_correlation(equicorr(3, 0.5), bad_rank)

    @pytest.mark.parametrize("bad_rank", [-1, 4, 100])
    def test_out_of_range_rank_rejected(self, bad_rank):
        with pytest.raises(ValueError):
            low_rank_correlation(equicorr(3, 0.5), bad_rank)

    @pytest.mark.parametrize(
        "bad_matrix",
        [
            np.ones((2, 3)),                                # non-square
            np.array([1.0, 0.5, 0.25]),                     # 1D
            np.array([[1.0, 0.5], [0.4, 1.0]]),             # asymmetric
            np.array([[1.0, np.nan], [np.nan, 1.0]]),       # non-finite
            np.diag([2.0, 1.0]),                            # non-unit diagonal
            np.ones((3, 3)),                                # only semidefinite
            np.array([[1.0, 0.9, 0.9],                      # indefinite
                      [0.9, 1.0, -0.9],
                      [0.9, -0.9, 1.0]]),
        ],
    )
    def test_malformed_matrix_rejected(self, bad_matrix):
        with pytest.raises(ValueError):
            low_rank_correlation(bad_matrix, 1)


# ---------------------------------------------------------------------------
# select_rank_from_log_density
# ---------------------------------------------------------------------------

class TestSelectRankFromLogDensity:
    def test_selects_largest_mean(self):
        rank, diag = select_rank_from_log_density(
            {0: np.zeros(3), 3: np.full(3, 2.0), 1: np.ones(3)}
        )
        assert rank == 3
        assert diag == {
            "candidate_ranks": [0, 1, 3],
            "mean_scores": [0.0, 1.0, 2.0],
            "n_samples": 3,
            "selected_score": 2.0,
        }

    def test_exact_tie_prefers_smaller_rank(self):
        rank, diag = select_rank_from_log_density(
            {2: np.ones(4), 1: np.ones(4), 5: np.ones(4)}
        )
        assert rank == 1
        assert diag["candidate_ranks"] == [1, 2, 5]
        assert diag["selected_score"] == 1.0

    def test_single_candidate(self):
        rank, diag = select_rank_from_log_density({7: np.array([0.5, -1.5])})
        assert rank == 7
        assert diag["mean_scores"] == [-0.5]
        assert diag["n_samples"] == 2

    def test_numpy_integer_key_accepted(self):
        rank, _ = select_rank_from_log_density({np.int64(2): np.ones(2)})
        assert rank == 2

    def test_does_not_mutate_input(self):
        samples = {1: np.array([1.0, 2.0]), 0: np.array([0.0, 0.0])}
        snapshots = {k: v.copy() for k, v in samples.items()}
        select_rank_from_log_density(samples)
        for key, value in samples.items():
            np.testing.assert_array_equal(value, snapshots[key])

    def test_unequal_sample_counts_rejected(self):
        with pytest.raises(ValueError):
            select_rank_from_log_density(
                {1: np.ones(3), 2: np.ones(4)}
            )

    def test_empty_mapping_rejected(self):
        with pytest.raises(ValueError):
            select_rank_from_log_density({})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError):
            select_rank_from_log_density([(1, np.ones(2))])

    @pytest.mark.parametrize("bad_key", [True, 1.5, "2", None])
    def test_malformed_keys_rejected(self, bad_key):
        with pytest.raises(TypeError):
            select_rank_from_log_density({bad_key: np.ones(2)})

    def test_negative_rank_rejected(self):
        with pytest.raises(ValueError):
            select_rank_from_log_density({-1: np.ones(2)})

    @pytest.mark.parametrize(
        "bad_samples",
        [
            np.array([]),                       # empty
            np.ones((2, 2)),                    # not 1D
            np.array([1.0, np.nan]),            # non-finite
            np.array([1.0, np.inf]),            # non-finite
        ],
    )
    def test_malformed_samples_rejected(self, bad_samples):
        with pytest.raises(ValueError):
            select_rank_from_log_density({1: bad_samples})


# ---------------------------------------------------------------------------
# correlation_pattern_metrics
# ---------------------------------------------------------------------------

# Hand-computed pair: strict upper triangles (0.5, -0.25, 0.5) and
# (0.5, -0.25, -0.5) give pattern_r = 1/sqrt(13), rmse = sqrt(1/3),
# sign_agreement = 2/3.
REF = np.array([
    [1.0, 0.5, -0.25],
    [0.5, 1.0, 0.5],
    [-0.25, 0.5, 1.0],
])
CAND = np.array([
    [1.0, 0.5, -0.25],
    [0.5, 1.0, -0.5],
    [-0.25, -0.5, 1.0],
])


class TestCorrelationPatternMetrics:
    def test_hand_computed_metrics(self):
        metrics = correlation_pattern_metrics(REF, CAND)
        assert metrics["pattern_r"] == pytest.approx(1.0 / math.sqrt(13.0), abs=1e-12)
        assert metrics["rmse"] == pytest.approx(math.sqrt(1.0 / 3.0), abs=1e-12)
        assert metrics["sign_agreement"] == pytest.approx(2.0 / 3.0, abs=1e-12)

    def test_identical_matrices(self):
        metrics = correlation_pattern_metrics(REF, REF.copy())
        assert metrics["pattern_r"] == pytest.approx(1.0, abs=1e-12)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["sign_agreement"] == pytest.approx(1.0, abs=1e-12)

    def test_metrics_ignore_diagonal_and_lower_triangle(self):
        # Permuting variable order permutes both triangles consistently; the
        # metrics are invariant under simultaneous row/column permutation.
        perm = [2, 0, 1]
        ref_p = REF[np.ix_(perm, perm)]
        cand_p = CAND[np.ix_(perm, perm)]
        base = correlation_pattern_metrics(REF, CAND)
        permuted = correlation_pattern_metrics(ref_p, cand_p)
        assert permuted["pattern_r"] == pytest.approx(base["pattern_r"], abs=1e-12)
        assert permuted["rmse"] == pytest.approx(base["rmse"], abs=1e-12)
        assert permuted["sign_agreement"] == pytest.approx(
            base["sign_agreement"], abs=1e-12
        )

    def test_does_not_mutate_inputs(self):
        ref, cand = REF.copy(), CAND.copy()
        correlation_pattern_metrics(ref, cand)
        np.testing.assert_array_equal(ref, REF)
        np.testing.assert_array_equal(cand, CAND)

    def test_dimension_two_rejected(self):
        eye2 = np.eye(2)
        with pytest.raises(ValueError):
            correlation_pattern_metrics(eye2, eye2)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            correlation_pattern_metrics(REF, equicorr(4, 0.3))

    def test_constant_reference_triangle_rejected(self):
        with pytest.raises(ValueError):
            correlation_pattern_metrics(equicorr(3, 0.4), CAND)

    def test_constant_candidate_triangle_rejected(self):
        with pytest.raises(ValueError):
            correlation_pattern_metrics(REF, equicorr(3, 0.4))

    def test_malformed_matrix_rejected(self):
        with pytest.raises(ValueError):
            correlation_pattern_metrics(np.ones((3, 3)), CAND)
