"""Synthetic tests for the historical residual location/scale study.

All tests use independently generated data and closed-form expected values;
none of them reads the research data bank.
"""
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial.distance import cdist

from source_signed_residual_location_scale import (
    ROLES, correlation_vectors, eval_descriptor, fit_reference,
    joint_energy_v, quarters_for_ids, transform_eval)
from source_signed_residual_reproducibility import compare, describe


def test_quarters_disjoint_complete_equal_and_order_invariant():
    ids = [f'sample_{i:03d}' for i in range(16)]
    q = quarters_for_ids(ids, 0)
    assert set(q) == set(ROLES)
    sizes = {len(v) for v in q.values()}
    assert sizes == {4}
    union = np.concatenate([q[r] for r in ROLES])
    assert sorted(union.tolist()) == list(range(16))  # complete, disjoint
    mapping = {ids[i]: role for role in ROLES for i in q[role]}
    shuffled = list(reversed(ids))
    q2 = quarters_for_ids(shuffled, 0)
    mapping2 = {shuffled[i]: role for role in ROLES for i in q2[role]}
    assert mapping == mapping2  # invariant to input row ordering
    q3 = quarters_for_ids(ids, 1)
    mapping3 = {ids[i]: role for role in ROLES for i in q3[role]}
    assert mapping3 != mapping  # different repeat, different assignment


def test_quarters_reject_duplicates_and_bad_counts():
    with pytest.raises(ValueError):
        quarters_for_ids(['a', 'b', 'a', 'c'], 0)
    with pytest.raises(ValueError):
        quarters_for_ids(['a', 'b', 'c'], 0)  # not divisible by four
    with pytest.raises(ValueError):
        quarters_for_ids([], 0)


def test_fit_reference_only_accepts_reference():
    sig = inspect.signature(fit_reference)
    assert len(sig.parameters) == 1  # evaluation data can never enter the fit


def test_affine_transforms_exactly_removed():
    rng = np.random.default_rng(11)
    ref = rng.normal(size=(32, 6))
    ev = rng.normal(size=(32, 6))
    shift = rng.normal(size=6) * 4.0
    scale = rng.uniform(0.4, 2.5, size=6)
    ref_t, ev_t = ref * scale + shift, ev * scale + shift
    params = fit_reference(ref_t)
    before = {k: v.copy() for k, v in params.items()}
    checks = {
        'center_median': (ev - np.median(ref, axis=0)) * scale,
        'center_mean': (ev - ref.mean(axis=0)) * scale,
        'scale_median_iqr': (ev - np.median(ref, axis=0)) / (np.quantile(ref, .75, axis=0) - np.quantile(ref, .25, axis=0)),
        'scale_mean_sd': (ev - ref.mean(axis=0)) / ref.std(axis=0),
    }
    for branch, expected in checks.items():
        z = transform_eval(ev_t, params, branch)
        assert np.allclose(z, expected, rtol=1e-12, atol=1e-12)
        # transforming a DIFFERENT evaluation block must not touch the fit
        assert all(np.array_equal(params[k], before[k]) for k in params)
    assert np.array_equal(transform_eval(ev_t, params, 'raw'), ev_t)


def test_correlation_invariance_under_positive_affine():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(48, 5))
    shift = rng.normal(size=5) * 3.0
    scale = rng.uniform(0.3, 4.0, size=5)
    z = x * scale + shift
    pearson_x = pd.DataFrame(x).corr().to_numpy()
    pearson_z = pd.DataFrame(z).corr().to_numpy()
    assert np.allclose(pearson_x, pearson_z, atol=1e-13)
    spearman_x = pd.DataFrame(x).corr(method='spearman').to_numpy()
    spearman_z = pd.DataFrame(z).corr(method='spearman').to_numpy()
    assert np.allclose(spearman_x, spearman_z, atol=1e-13)


def test_joint_energy_matches_direct_formula_and_scipy():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(20, 3))
    y = rng.normal(size=(17, 3)) + 0.4
    direct_xy = cdist(x, y) / np.sqrt(3)
    direct_xx = cdist(x, x) / np.sqrt(3)
    direct_yy = cdist(y, y) / np.sqrt(3)
    expected = 2 * direct_xy.mean() - direct_xx.mean() - direct_yy.mean()
    assert np.isclose(joint_energy_v(x, y), expected, rtol=1e-12, atol=1e-12)
    x1 = rng.normal(size=25)
    y1 = rng.normal(size=25) + 0.7
    v = joint_energy_v(x1.reshape(-1, 1), y1.reshape(-1, 1))
    assert np.isclose(v, stats.energy_distance(x1, y1) ** 2, rtol=1e-10, atol=1e-12)
    assert v >= 0  # V-statistic, not its square root


def test_zero_marginal_w1_with_positive_joint_energy():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(32, 3))
    y = x[rng.permutation(32)]  # identical per-coordinate marginals, permuted pairing
    w1 = float(np.mean(np.abs(np.sort(x, axis=0) - np.sort(y, axis=0))))
    assert w1 == 0.0
    assert joint_energy_v(x, y) > 1e-6


def test_degenerate_reference_scale_rejected_without_flooring():
    rng = np.random.default_rng(9)
    ref = rng.normal(size=(32, 4))
    ref[:, 2] = 1.5  # constant band -> zero IQR and zero SD
    params = fit_reference(ref)  # fitting itself must not crash
    with pytest.raises(ValueError):
        transform_eval(rng.normal(size=(8, 4)), params, 'scale_median_iqr')
    with pytest.raises(ValueError):
        transform_eval(rng.normal(size=(8, 4)), params, 'scale_mean_sd')
    near = ref.copy()
    near[:, 1] = 1.0 + np.linspace(0, 5e-13, 32)  # IQR and SD below 1e-12
    params_near = fit_reference(near)
    with pytest.raises(ValueError):
        transform_eval(rng.normal(size=(8, 4)), params_near, 'scale_median_iqr')
    with pytest.raises(ValueError):
        transform_eval(rng.normal(size=(8, 4)), params_near, 'scale_mean_sd')


def test_nonfinite_inputs_rejected():
    ref = np.zeros((8, 3))
    with pytest.raises(ValueError):
        fit_reference(ref)
    params = fit_reference(np.arange(24.0).reshape(8, 3) + 0.5)
    bad = np.ones((4, 3))
    bad[1, 1] = np.nan
    with pytest.raises(ValueError):
        transform_eval(bad, params, 'center_mean')
    with pytest.raises(ValueError):
        joint_energy_v(bad, np.ones((4, 3)))


def test_eval_descriptor_reuses_supplied_correlations():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(32, 5))
    p, s = correlation_vectors(x)
    d = eval_descriptor(x, corr=(p, s))
    assert d['pearson'] is p and d['spearman'] is s
    c = compare(describe(x), d)  # identical descriptors must give zero distances
    assert c['marginal_w1'] == 0.0
    assert c['pearson_corr_rmse'] == 0.0
