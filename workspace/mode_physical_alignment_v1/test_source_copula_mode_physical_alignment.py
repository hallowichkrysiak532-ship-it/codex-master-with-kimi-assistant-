import json

import numpy as np
import pandas as pd
import pytest

import source_copula_mode_physical_alignment as alignment


def test_mode_and_band_attributions_reconstruct_direct_density():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(120, 6))
    correlation = np.corrcoef(raw, rowvar=False)
    z = rng.normal(size=(25, 6))
    result = alignment.copula_mode_band_attributions(z, correlation)
    direct = alignment.gaussian_copula_log_density(z, correlation)
    np.testing.assert_allclose(result["mode_contribution"].sum(axis=1), direct, atol=1e-11)
    np.testing.assert_allclose(result["band_contribution"].sum(axis=1), direct, atol=1e-11)
    assert np.all(np.diff(np.abs(result["eigenvalue"] - 1.0)) <= 1e-14)


def test_identity_attributions_are_zero_and_inputs_are_not_mutated():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(10, 4))
    snapshot = z.copy()
    result = alignment.copula_mode_band_attributions(z, np.eye(4))
    np.testing.assert_array_equal(z, snapshot)
    np.testing.assert_allclose(result["mode_contribution"], 0.0, atol=1e-15)
    np.testing.assert_allclose(result["band_contribution"], 0.0, atol=1e-15)


def test_physical_window_weights_use_inclusive_overlap_and_normalize():
    windows = [
        {"name": "a", "start_nm": 400, "end_nm": 424},
        {"name": "b", "start_nm": 412, "end_nm": 449},
    ]
    weights, overlap = alignment.physical_window_weights(windows, band_count=2)
    np.testing.assert_array_equal(overlap[0], [25, 0])
    np.testing.assert_array_equal(overlap[1], [13, 25])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[1], [13 / 38, 25 / 38])


def test_registered_physical_mapping_has_ten_complete_windows():
    manifest = json.loads(
        (alignment.EXP28A2 / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    weights, overlap = alignment.physical_window_weights(manifest["physical_windows"])
    assert weights.shape == overlap.shape == (10, 80)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert (overlap.sum(axis=1) == np.array([100, 100, 100, 100, 300, 200, 300, 300, 50, 325])).all()


def test_exp28a2_target_metrics_reproduce_state_gain_formula():
    rows = []
    for case in range(3):
        for model, error in (
            ("matched_M1", 0.4), ("matched_M3", 0.2),
            ("shuffled_M3", 0.5), ("median_M3", 0.8),
        ):
            rows.append(
                {
                    "selected_case_index": case, "case_id": case, "arm": "a",
                    "effect": "e", "magnitude": 0.1, "target_node": "node",
                    "model": model, "relative_error": error,
                }
            )
    result = alignment.exp28a2_target_metrics(pd.DataFrame(rows)).iloc[0]
    assert result.median_state_gain == pytest.approx(0.6)
    assert result.median_global_gain == pytest.approx(0.75)
    assert result.median_nonlinear_gain == pytest.approx(0.5)
    assert result.median_log_state_advantage == pytest.approx(np.log(2.5))


def test_rank_permutation_reports_perfect_alignment():
    x = np.arange(10, dtype=float)
    rho, p, low, high = alignment.rank_permutation_test(x, x, seed=2, permutations=999)
    assert rho == pytest.approx(1.0)
    assert p <= 0.01
    assert low <= high


def test_exp28c_edges_are_averaged_over_incident_nodes():
    rows = []
    for projection, gain in (("matched", 0.5), ("shuffled", 0.2), ("global", 0.3)):
        rows.append(
            {
                "domain": "d", "domain_role": "r", "component": "normal",
                "channel": "shape", "node_left": "a", "node_right": "b",
                "projection": projection, "pairing_gain": gain,
            }
        )
    result = alignment.exp28c_node_metrics(pd.DataFrame(rows))
    assert set(result.node) == {"a", "b"}
    np.testing.assert_allclose(result.matched_pairing_gain, 0.5)
    np.testing.assert_allclose(result.state_specificity_gain, 0.2)

