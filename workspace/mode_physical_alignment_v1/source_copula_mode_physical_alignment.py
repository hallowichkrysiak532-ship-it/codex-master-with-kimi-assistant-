"""Decompose frozen shared-copula scores and align them to PP relations.

The score decomposition is exact in the frozen 80-band Gaussian-copula model.
Physical-window alignment is descriptive: Exp28A.2 is simulator-only and has
no observed-sample IDs. Exp28C supplies the frozen observed-side matched,
shuffled, and global projector controls. No model, marginal, rank, or relation
is refit to a validation or test dataset.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from threadpoolctl import threadpool_limits

from gaussian_copula_models import gaussian_copula_log_density
from source_distribution_step1 import ROOT, bh_adjust, sha256, write_json
from source_shared_copula_transfer import (
    COMPONENTS, DIMENSION, EXTERNAL_TEST, TRAIN, VALIDATION,
    load_full_bank, transform_frozen,
)


TRANSFER = ROOT / "output_source_signed_residual_distribution/shared_transfer_v1"
EXP28A2 = ROOT / "experiment_v3_20260724/exp28a2_state_conditioned_cross_band_relation_v1"
EXP28C = ROOT / "experiment_v3_20260724/exp28c_tangent_normal_pairing_localization_v1"
OUTPUT = ROOT / "output_source_signed_residual_distribution/mode_physical_alignment_v1"
DATASETS = (*VALIDATION, *EXTERNAL_TEST)
PERMUTATIONS = 9999
EPS = 1e-10


def copula_mode_band_attributions(z, correlation):
    """Return exact eigenmode and symmetric band allocations of log c_R(z)."""
    z = np.asarray(z, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    if z.ndim != 2 or correlation.shape != (z.shape[1], z.shape[1]):
        raise ValueError("Expected sample-by-band scores and a matching correlation.")
    if not np.isfinite(z).all() or not np.isfinite(correlation).all():
        raise ValueError("Attribution inputs must be finite.")
    np.linalg.cholesky(correlation)
    eigenvalue, eigenvector = np.linalg.eigh(correlation)
    order = np.argsort(-np.abs(eigenvalue - 1.0), kind="stable")
    eigenvalue = eigenvalue[order]
    eigenvector = eigenvector[:, order]
    projection = z @ eigenvector
    mode = (
        -0.5 * np.log(eigenvalue)[None, :]
        -0.5 * (1.0 / eigenvalue - 1.0)[None, :] * projection**2
    )
    inverse_minus_identity = (
        eigenvector * (1.0 / eigenvalue - 1.0)[None, :]
    ) @ eigenvector.T
    log_diagonal = (eigenvector**2) @ np.log(eigenvalue)
    band = -0.5 * log_diagonal[None, :] - 0.5 * z * (z @ inverse_minus_identity)
    direct = gaussian_copula_log_density(z, correlation)
    mode_error = float(np.max(np.abs(mode.sum(axis=1) - direct)))
    band_error = float(np.max(np.abs(band.sum(axis=1) - direct)))
    tolerance = 2e-9 * max(1.0, float(np.max(np.abs(direct))))
    if mode_error > tolerance or band_error > tolerance:
        raise RuntimeError("Copula attribution does not reconstruct the frozen score.")
    return {
        "eigenvalue": eigenvalue,
        "eigenvector": eigenvector,
        "mode_contribution": mode,
        "band_contribution": band,
        "direct_log_density": direct,
        "mode_max_abs_error": mode_error,
        "band_max_abs_error": band_error,
    }


def physical_window_weights(windows, band_count=DIMENSION):
    """Map 25 nm bands to windows by wavelength overlap, normalized per node."""
    weights = np.zeros((len(windows), band_count), dtype=float)
    overlap_nm = np.zeros_like(weights)
    for node_index, window in enumerate(windows):
        start, end = int(window["start_nm"]), int(window["end_nm"])
        if not 400 <= start <= end <= 2399:
            raise ValueError("Physical window is outside the residual wavelength grid.")
        for band in range(band_count):
            band_start = 400 + 25 * band
            band_end = band_start + 24
            overlap = max(0, min(end, band_end) - max(start, band_start) + 1)
            overlap_nm[node_index, band] = overlap
        if overlap_nm[node_index].sum() != end - start + 1:
            raise ValueError("Window overlap does not cover every registered wavelength.")
        weights[node_index] = overlap_nm[node_index] / overlap_nm[node_index].sum()
    return weights, overlap_nm


def mode_physical_loadings(eigenvectors, windows, token_bank):
    """Project piecewise-constant mode loadings into frozen Legendre windows."""
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    if eigenvectors.shape != (DIMENSION, DIMENSION):
        raise ValueError("Expected 80 ordered shared-copula eigenvectors.")
    rows = []
    for mode in range(DIMENSION):
        expanded = np.repeat(eigenvectors[:, mode], 25)
        total_energy = float(np.sum(expanded**2))
        for window in windows:
            node = str(window["name"])
            start, end = int(window["start_nm"]), int(window["end_nm"])
            value = expanded[start - 400 : end - 399]
            basis = np.asarray(token_bank[f"basis_{node}"], dtype=float)
            if basis.shape != (len(value), 4):
                raise ValueError(f"Unexpected frozen basis for {node}.")
            token = value @ basis
            window_energy = float(np.sum(value**2))
            captured = float(np.sum(token**2))
            rows.append(
                {
                    "mode_rank": mode + 1,
                    "node": node,
                    "window_start_nm": start,
                    "window_end_nm": end,
                    "window_loading_rms": float(np.sqrt(window_energy / len(value))),
                    "token_loading_energy": captured,
                    "token_energy_over_full_mode": captured / total_energy,
                    "within_window_token_capture": captured / max(window_energy, EPS),
                    **{f"token_{index}": float(token[index]) for index in range(4)},
                }
            )
    return pd.DataFrame(rows)


def exp28a2_target_metrics(prediction):
    """Summarize frozen simulator current-state advantages by target window."""
    task_index = [
        "selected_case_index", "case_id", "arm", "effect", "magnitude",
        "target_node",
    ]
    error = prediction.pivot_table(
        index=task_index, columns="model", values="relative_error"
    ).reset_index()
    required = {"matched_M1", "matched_M3", "shuffled_M3", "median_M3"}
    if not required.issubset(error.columns):
        raise ValueError("Exp28A.2 prediction table is missing frozen controls.")
    error["state_gain"] = (
        error.shuffled_M3 - error.matched_M3
    ) / np.maximum(error.shuffled_M3, EPS)
    error["global_gain"] = (
        error.median_M3 - error.matched_M3
    ) / np.maximum(error.median_M3, EPS)
    error["nonlinear_gain"] = (
        error.matched_M1 - error.matched_M3
    ) / np.maximum(error.matched_M1, EPS)
    error["log_state_advantage"] = np.log(
        np.maximum(error.shuffled_M3, EPS) / np.maximum(error.matched_M3, EPS)
    )
    error["log_global_advantage"] = np.log(
        np.maximum(error.median_M3, EPS) / np.maximum(error.matched_M3, EPS)
    )
    result = error.groupby("target_node", as_index=False).agg(
        task_count=("case_id", "size"),
        median_matched_error=("matched_M3", "median"),
        median_shuffled_error=("shuffled_M3", "median"),
        median_median_field_error=("median_M3", "median"),
        median_state_gain=("state_gain", "median"),
        median_global_gain=("global_gain", "median"),
        median_nonlinear_gain=("nonlinear_gain", "median"),
        median_log_state_advantage=("log_state_advantage", "median"),
        median_log_global_advantage=("log_global_advantage", "median"),
    )
    return result.rename(columns={"target_node": "node"})


def rank_permutation_test(x, y, seed, permutations=PERMUTATIONS):
    """Two-sided node-label permutation test for Spearman correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 5:
        raise ValueError("Rank permutation inputs must be same-length vectors.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Rank permutation inputs must be finite.")
    xr = stats.rankdata(x)
    yr = stats.rankdata(y)
    xr -= xr.mean()
    yr -= yr.mean()
    denominator = float(np.linalg.norm(xr) * np.linalg.norm(yr))
    if denominator <= EPS:
        raise ValueError("Rank permutation input is constant.")
    observed = float(np.dot(xr, yr) / denominator)
    rng = np.random.default_rng(seed)
    orders = np.vstack([rng.permutation(len(x)) for _ in range(permutations)])
    null = (yr[orders] @ xr) / denominator
    p = float((1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15)) / (permutations + 1))
    return observed, p, float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))


def exp28c_node_metrics(edges):
    """Convert frozen observed pairing edges into incident-node strengths."""
    index = ["domain", "domain_role", "component", "channel", "node_left", "node_right"]
    wide = edges.pivot_table(index=index, columns="projection", values="pairing_gain").reset_index()
    if not {"matched", "shuffled", "global"}.issubset(wide.columns):
        raise ValueError("Exp28C pairing table is missing projector controls.")
    wide["matched_pairing_gain"] = wide.matched
    wide["state_specificity_gain"] = wide.matched - wide[["shuffled", "global"]].max(axis=1)
    rows = []
    for row in wide.itertuples(index=False):
        for node in (row.node_left, row.node_right):
            rows.append(
                {
                    "dataset": row.domain,
                    "exp28c_role": row.domain_role,
                    "component": row.component,
                    "channel": row.channel,
                    "node": node,
                    "matched_pairing_gain": row.matched_pairing_gain,
                    "state_specificity_gain": row.state_specificity_gain,
                }
            )
    return pd.DataFrame(rows).groupby(
        ["dataset", "exp28c_role", "component", "channel", "node"], as_index=False
    ).mean(numeric_only=True)


def _validated_artifacts(directory, status_name, expected_status):
    status_path = directory / status_name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != expected_status:
        raise ValueError(f"Unexpected frozen status: {status_path}")
    hashes = {str(status_path): sha256(status_path)}
    for name, expected in status["artifact_hashes"].items() if "artifact_hashes" in status else status["artifacts"].items():
        path = directory / name
        if sha256(path) != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {path}")
        hashes[str(path)] = expected
    return status, hashes


def _alignment_rows(node_attribution, simulator_metrics):
    rows = []
    role_nodes = node_attribution.groupby(
        ["component", "role", "node"], as_index=False
    ).mean(numeric_only=True)
    for (component, role), group in role_nodes.groupby(["component", "role"]):
        merged = group.merge(simulator_metrics, on="node", validate="one_to_one")
        for attribution in ("mean_penalty", "mean_absolute"):
            for target in ("median_state_gain", "median_log_state_advantage"):
                seed = 20260906 + COMPONENTS.index(component) * 100 + (0 if role == "validation" else 20) + (0 if attribution == "mean_penalty" else 4) + (0 if target == "median_state_gain" else 1)
                rho, p, low, high = rank_permutation_test(
                    merged[attribution], merged[target], seed
                )
                rows.append(
                    {
                        "component": component, "role": role,
                        "attribution": attribution, "simulator_metric": target,
                        "node_count": len(merged), "rho": rho,
                        "p_two_sided": p, "null_q025": low, "null_q975": high,
                    }
                )
    result = pd.DataFrame(rows)
    result["q_bh"] = bh_adjust(result.p_two_sided.to_numpy())
    return result


def _observed_alignment_rows(node_attribution, observed_nodes):
    rows = []
    for (dataset, component), group in node_attribution.groupby(["dataset", "component"]):
        observed = observed_nodes[
            (observed_nodes.dataset == dataset) & (observed_nodes.component == component)
        ]
        # Exp28C freezes tangent and normal atlases only; the undecomposed
        # residual has no observed-side projector-control counterpart.
        if observed.empty:
            continue
        for channel in ("shape", "amplitude"):
            channel_rows = observed[observed.channel == channel]
            merged = group.merge(channel_rows, on=["dataset", "component", "node"], validate="one_to_one")
            if len(merged) != 10:
                raise ValueError("Observed alignment requires all ten physical nodes.")
            for attribution in ("mean_penalty", "mean_absolute"):
                for reference in ("matched_pairing_gain", "state_specificity_gain"):
                    seed = (
                        20261000 + DATASETS.index(dataset) * 1000
                        + COMPONENTS.index(component) * 100
                        + (0 if channel == "shape" else 20)
                        + (0 if attribution == "mean_penalty" else 4)
                        + (0 if reference == "matched_pairing_gain" else 1)
                    )
                    rho, p, low, high = rank_permutation_test(
                        merged[attribution], merged[reference], seed
                    )
                    rows.append(
                        {
                            "dataset": dataset, "role": group.role.iloc[0],
                            "exp28c_role": merged.exp28c_role.iloc[0],
                            "component": component, "channel": channel,
                            "attribution": attribution, "observed_reference": reference,
                            "node_count": len(merged), "rho": rho,
                            "p_two_sided": p, "null_q025": low, "null_q975": high,
                        }
                    )
    result = pd.DataFrame(rows)
    result["q_bh"] = bh_adjust(result.p_two_sided.to_numpy())
    return result


def _domain_shift_rows(node_attribution, margins):
    domain_score = node_attribution.groupby(
        ["component", "role", "dataset"], as_index=False
    ).agg(mean_score_per_band=("mean_signed", "sum"))
    # Node windows overlap, so reconstruct score from the frozen dataset table,
    # not from overlapping node averages.
    summary = pd.read_csv(TRANSFER / "frozen_dataset_summary.csv")
    summary = summary[summary.model == "validation_selected"][
        ["component", "role", "dataset", "mean_log_density_per_dim"]
    ]
    domain_score = domain_score.drop(columns="mean_score_per_band").merge(
        summary, on=["component", "role", "dataset"], validate="one_to_one"
    ).merge(margins, on=["component", "role", "dataset"], validate="one_to_one")
    rows = []
    for component, group in domain_score.groupby("component"):
        for diagnostic in ("outside_train_range_fraction", "normal_score_rms"):
            rho, p, low, high = rank_permutation_test(
                group[diagnostic], group.mean_log_density_per_dim,
                20262000 + COMPONENTS.index(component) * 10 + (0 if diagnostic.startswith("outside") else 1),
            )
            rows.append(
                {
                    "component": component, "diagnostic": diagnostic,
                    "dataset_count": len(group), "rho_with_mean_density": rho,
                    "p_two_sided": p, "null_q025": low, "null_q975": high,
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh"] = bh_adjust(result.p_two_sided.to_numpy())
    return result


def make_figures(output, band_summary, mode_summary, node_attribution, simulator_metrics, observed_summary):
    order = list(DATASETS)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for axis, component in zip(axes, COMPONENTS):
        frame = band_summary[band_summary.component == component].pivot(
            index="dataset", columns="band_index", values="mean_penalty"
        ).loc[order]
        image = axis.imshow(frame, aspect="auto", cmap="magma", origin="upper")
        axis.set_yticks(np.arange(len(order)), order)
        axis.set_title(f"{component}: mean negative copula contribution", loc="left")
        fig.colorbar(image, ax=axis, pad=0.01, label="Penalty per sample")
    axes[-1].set_xticks(np.arange(0, 80, 8), [f"{400 + 25*i}" for i in range(0, 80, 8)])
    axes[-1].set_xlabel("25 nm band start wavelength")
    fig.suptitle("Exact frozen-score band attribution")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "band_penalty_atlas.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    role_nodes = node_attribution.groupby(["component", "role", "node"], as_index=False).mean(numeric_only=True)
    for axis, component in zip(axes, COMPONENTS):
        for role, marker, color in (("validation", "o", "#D68910"), ("external_test", "s", "#2874A6")):
            frame = role_nodes[(role_nodes.component == component) & (role_nodes.role == role)].merge(
                simulator_metrics, on="node", validate="one_to_one"
            )
            axis.scatter(frame.median_log_state_advantage, frame.mean_penalty, marker=marker, color=color, label=role)
            for row in frame.itertuples():
                axis.annotate(row.node.replace("_", "\n"), (row.median_log_state_advantage, row.mean_penalty), fontsize=6, alpha=0.8)
        axis.set(title=component, xlabel="Exp28A.2 log matched/shuffled advantage", ylabel="Mean frozen density penalty")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Simulator state-conditioned targets versus observed frozen-score penalties")
    fig.tight_layout()
    fig.savefig(output / "simulator_state_alignment.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    labels = []
    values = []
    for row in observed_summary.itertuples(index=False):
        labels.append(f"{row.component}\n{row.channel}\n{row.observed_reference.replace('_gain','')}")
        values.append(row.rho_median)
    axis.bar(np.arange(len(values)), values, color=["#2874A6" if "matched" in label else "#D68910" for label in labels])
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(np.arange(len(values)), labels, rotation=40, ha="right")
    axis.set_ylabel("Median node-pattern Spearman across datasets")
    axis.set_title("Frozen density penalty alignment with Exp28C observed pairing controls", loc="left")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "observed_pairing_alignment.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, component in zip(axes, COMPONENTS):
        frame = mode_summary[(mode_summary.component == component) & (mode_summary.role == "external_test")]
        frame = frame.groupby("mode_rank", as_index=False).mean(numeric_only=True).sort_values("mean_contribution").head(12)
        axis.barh(frame.mode_rank.astype(str), frame.mean_contribution, color="#C0392B")
        axis.set(title=component, xlabel="External mean mode contribution", ylabel="Mode rank")
        axis.axvline(0, color="black", lw=0.8)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Most negative frozen copula eigenmode contributions")
    fig.tight_layout()
    fig.savefig(output / "negative_mode_contributions.png", dpi=180)
    plt.close(fig)


def make_report(output, reconstruction, mode_metadata, mode_summary, band_summary, simulator_alignment, observed_summary, shift):
    lines = [
        "# Frozen copula mode and physical-window alignment",
        "",
        "All copula models, marginal transforms, ranks, Exp28A.2 physical bases, "
        "and Exp28C pairing controls were read frozen. No source-specific or "
        "target-specific model was fitted.",
        "",
        "## Exact attribution",
        "",
        f"Maximum mode reconstruction error: {reconstruction.mode_max_abs_error.max():.3e}; "
        f"maximum band reconstruction error: {reconstruction.band_max_abs_error.max():.3e}.",
        "",
    ]
    for component in COMPONENTS:
        meta = mode_metadata[mode_metadata.component == component]
        top = meta.head(3)
        description = ", ".join(
            f"m{int(row.mode_rank)} lambda={row.eigenvalue:.3f} peak={int(row.peak_start_nm)}-{int(row.peak_end_nm)}"
            for row in top.itertuples()
        )
        lines.append(f"- {component} leading |lambda-1| modes: {description}.")
    lines.extend(["", "## Domain-shift decomposition", ""])
    for row in shift.itertuples(index=False):
        lines.append(
            f"- {row.component} density vs {row.diagnostic}: rho={row.rho_with_mean_density:+.3f}, "
            f"BH q={row.q_bh:.4f}."
        )
    lines.extend(["", "## Exp28A.2 simulator-window alignment", ""])
    primary = simulator_alignment[
        (simulator_alignment.attribution == "mean_penalty")
        & (simulator_alignment.simulator_metric == "median_log_state_advantage")
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"- {row.component}/{row.role}: penalty vs simulator state advantage "
            f"rho={row.rho:+.3f}, BH q={row.q_bh:.4f}."
        )
    lines.extend(["", "## Exp28C observed-side control", ""])
    for row in observed_summary.itertuples(index=False):
        lines.append(
            f"- {row.component}/{row.channel}/{row.observed_reference}: median rho "
            f"{row.rho_median:+.3f}; {int(row.q_lt_005_count)}/{int(row.comparison_count)} "
            "dataset alignments have BH q<.05."
        )
    lines.extend(
        [
            "",
            "Exp28A.2 contains simulator anchors and perturbations, not observed "
            "sample IDs or observed theta. Its alignment is therefore across frozen "
            "physical windows. Exp28C is the observed-side state-control reference. "
            "All node-label tests are exploratory and share only ten partly "
            "overlapping physical windows.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=False)
    transfer_status, transfer_hashes = _validated_artifacts(
        TRANSFER, "status.json", "complete_frozen_shared_copula_transfer"
    )
    exp28a2_status, exp28a2_hashes = _validated_artifacts(
        EXP28A2, "exp28a2_status.json", "complete_frozen_state_conditioned_cross_band_relation_audit"
    )
    exp28c_status, exp28c_hashes = _validated_artifacts(
        EXP28C, "exp28c_status.json", "complete_frozen_tangent_normal_pairing_localization"
    )
    bank, bank_hashes = load_full_bank()
    all_inputs = {**transfer_hashes, **exp28a2_hashes, **exp28c_hashes, **bank_hashes}

    a2_manifest = json.loads((EXP28A2 / "protocol_manifest.json").read_text(encoding="utf-8"))
    windows = a2_manifest["physical_windows"]
    weights, overlap_nm = physical_window_weights(windows)
    a2_prediction = pd.read_csv(EXP28A2 / "cross_band_prediction_samples.csv")
    simulator_metrics = exp28a2_target_metrics(a2_prediction)
    with np.load(EXP28A2 / "cross_band_token_bank.npz", allow_pickle=False) as token_file:
        token_bank = {name: token_file[name] for name in token_file.files if name.startswith("basis_")}
    with np.load(TRANSFER / "frozen_shared_models.npz", allow_pickle=False) as model_file:
        models = {name: model_file[name] for name in model_file.files}

    dataset_vector = np.concatenate([
        np.repeat(dataset, len(bank[dataset]["ids"])) for dataset in DATASETS
    ]).astype("U32")
    role_vector = np.concatenate([
        np.repeat("validation" if dataset in VALIDATION else "external_test", len(bank[dataset]["ids"]))
        for dataset in DATASETS
    ]).astype("U16")
    sample_vector = np.concatenate([bank[dataset]["ids"] for dataset in DATASETS]).astype("U64")
    attribution_payload = {
        "dataset": dataset_vector, "role": role_vector, "sample_id": sample_vector,
    }
    reconstruction_rows, mode_rows, mode_summary_rows = [], [], []
    band_summary_rows, node_rows, loading_frames = [], [], []

    for component in COMPONENTS:
        correlation = models[f"{component}__selected_correlation"]
        reference = models[f"{component}__pooled_train_sorted_ecdf"]
        z = np.concatenate([
            transform_frozen(bank[dataset][component], reference)[0]
            for dataset in DATASETS
        ])
        attribution = copula_mode_band_attributions(z, correlation)
        attribution_payload[f"{component}__z"] = z.astype(np.float32)
        attribution_payload[f"{component}__mode_contribution"] = attribution["mode_contribution"].astype(np.float32)
        attribution_payload[f"{component}__band_contribution"] = attribution["band_contribution"].astype(np.float32)
        attribution_payload[f"{component}__eigenvalue"] = attribution["eigenvalue"]
        attribution_payload[f"{component}__eigenvector"] = attribution["eigenvector"]
        reconstruction_rows.append(
            {
                "component": component,
                "rows": len(z),
                "mode_max_abs_error": attribution["mode_max_abs_error"],
                "band_max_abs_error": attribution["band_max_abs_error"],
            }
        )

        eigenvalue = attribution["eigenvalue"]
        eigenvector = attribution["eigenvector"]
        for mode in range(DIMENSION):
            peak = int(np.argmax(np.abs(eigenvector[:, mode])))
            mode_rows.append(
                {
                    "component": component, "mode_rank": mode + 1,
                    "eigenvalue": float(eigenvalue[mode]),
                    "abs_eigenvalue_minus_one": float(abs(eigenvalue[mode] - 1.0)),
                    "volume_contribution": float(-0.5 * np.log(eigenvalue[mode])),
                    "peak_band_index": peak, "peak_start_nm": 400 + 25 * peak,
                    "peak_end_nm": 424 + 25 * peak,
                }
            )
        loading = mode_physical_loadings(eigenvector, windows, token_bank)
        loading.insert(0, "component", component)
        loading_frames.append(loading)

        offset = 0
        for dataset in DATASETS:
            count = len(bank[dataset]["ids"])
            role = "validation" if dataset in VALIDATION else "external_test"
            mode_value = attribution["mode_contribution"][offset : offset + count]
            band_value = attribution["band_contribution"][offset : offset + count]
            for mode in range(DIMENSION):
                value = mode_value[:, mode]
                mode_summary_rows.append(
                    {
                        "component": component, "role": role, "dataset": dataset,
                        "mode_rank": mode + 1, "eigenvalue": float(eigenvalue[mode]),
                        "mean_contribution": float(value.mean()),
                        "median_contribution": float(np.median(value)),
                        "mean_absolute_contribution": float(np.mean(np.abs(value))),
                        "mean_penalty": float(np.mean(np.maximum(-value, 0))),
                        "negative_fraction": float(np.mean(value < 0)),
                    }
                )
            band_mean = band_value.mean(axis=0)
            band_abs = np.abs(band_value).mean(axis=0)
            band_penalty = np.maximum(-band_value, 0).mean(axis=0)
            band_reward = np.maximum(band_value, 0).mean(axis=0)
            for band in range(DIMENSION):
                band_summary_rows.append(
                    {
                        "component": component, "role": role, "dataset": dataset,
                        "band_index": band, "start_nm": 400 + 25 * band,
                        "end_nm": 424 + 25 * band,
                        "mean_signed": float(band_mean[band]),
                        "mean_absolute": float(band_abs[band]),
                        "mean_penalty": float(band_penalty[band]),
                        "mean_reward": float(band_reward[band]),
                    }
                )
            for node_index, window in enumerate(windows):
                node_rows.append(
                    {
                        "component": component, "role": role, "dataset": dataset,
                        "node": window["name"],
                        "start_nm": int(window["start_nm"]), "end_nm": int(window["end_nm"]),
                        "mean_signed": float(weights[node_index] @ band_mean),
                        "mean_absolute": float(weights[node_index] @ band_abs),
                        "mean_penalty": float(weights[node_index] @ band_penalty),
                        "mean_reward": float(weights[node_index] @ band_reward),
                    }
                )
            offset += count

    reconstruction = pd.DataFrame(reconstruction_rows)
    mode_metadata = pd.DataFrame(mode_rows)
    mode_summary = pd.DataFrame(mode_summary_rows)
    band_summary = pd.DataFrame(band_summary_rows)
    node_attribution = pd.DataFrame(node_rows)
    mode_loading = pd.concat(loading_frames, ignore_index=True)
    simulator_alignment = _alignment_rows(node_attribution, simulator_metrics)
    c_edges = pd.read_csv(EXP28C / "pairing_edges.csv")
    observed_nodes = exp28c_node_metrics(c_edges)
    observed_alignment = _observed_alignment_rows(node_attribution, observed_nodes)
    observed_summary = observed_alignment.groupby(
        ["component", "channel", "attribution", "observed_reference"], as_index=False
    ).agg(
        comparison_count=("rho", "size"),
        rho_median=("rho", "median"),
        rho_q25=("rho", lambda value: float(np.quantile(value, 0.25))),
        rho_q75=("rho", lambda value: float(np.quantile(value, 0.75))),
        positive_fraction=("rho", lambda value: float(np.mean(np.asarray(value) > 0))),
        q_lt_005_count=("q_bh", lambda value: int(np.sum(np.asarray(value) < 0.05))),
    )
    observed_plot_summary = observed_summary[observed_summary.attribution == "mean_penalty"].copy()
    margins = pd.read_csv(TRANSFER / "marginal_shift_diagnostics.csv")
    shift = _domain_shift_rows(node_attribution, margins)

    np.savez_compressed(output / "sample_attribution_bank.npz", **attribution_payload)
    np.savez_compressed(
        output / "physical_window_mapping.npz",
        node=np.asarray([window["name"] for window in windows], dtype="U32"),
        weights=weights, overlap_nm=overlap_nm,
    )
    outputs = {
        "attribution_reconstruction.csv": reconstruction,
        "mode_metadata.csv": mode_metadata,
        "dataset_mode_attribution.csv.gz": mode_summary,
        "dataset_band_attribution.csv.gz": band_summary,
        "dataset_node_attribution.csv": node_attribution,
        "mode_physical_loading.csv.gz": mode_loading,
        "exp28a2_target_state_metrics.csv": simulator_metrics,
        "exp28a2_node_alignment.csv": simulator_alignment,
        "exp28c_observed_node_metrics.csv": observed_nodes,
        "exp28c_observed_alignment.csv.gz": observed_alignment,
        "exp28c_observed_alignment_summary.csv": observed_summary,
        "domain_shift_score_association.csv": shift,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    make_figures(
        output, band_summary, mode_summary, node_attribution, simulator_metrics,
        observed_plot_summary,
    )
    make_report(
        output, reconstruction, mode_metadata, mode_summary, band_summary,
        simulator_alignment, observed_plot_summary, shift,
    )

    scripts = {}
    for name in (
        "source_copula_mode_physical_alignment.py",
        "source_shared_copula_transfer.py", "gaussian_copula_models.py",
    ):
        path = ROOT / name
        scripts[str(path)] = sha256(path)
    manifest = {
        "experiment": "frozen_copula_mode_physical_alignment_v1",
        "datasets": list(DATASETS), "components": list(COMPONENTS),
        "score_decomposition": "exact eigenmode and symmetric band allocation of frozen Gaussian-copula log density",
        "physical_mapping": "wavelength-overlap mean into ten frozen Exp28A.2 windows; overlapping nodes retained",
        "simulator_alignment": "node-level residual attribution versus frozen Exp28A.2 target state advantage",
        "observed_control": "node-level attribution versus Exp28C matched pairing and matched-minus-best(shuffled,global) specificity",
        "observed_sample_state_join": False,
        "reason_no_sample_join": "Exp28A.2 is simulator-only and contains no observed sample IDs or observed theta",
        "source_specific_fit": False, "target_calibration": False,
        "labels_read": False, "test_used_for_selection": False,
        "permutations": PERMUTATIONS,
        "parent_routes": {
            "shared_transfer": transfer_status["status"],
            "exp28a2": exp28a2_status["route_decision"],
            "exp28c": exp28c_status["route_decision"],
        },
        "input_sha256": all_inputs, "code_sha256": scripts,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
        },
    }
    write_json(output / "protocol_manifest.json", manifest)
    for path, expected in {**all_inputs, **scripts}.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Input or code changed during analysis: {path}")
    artifacts = {
        path.name: sha256(path) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    }
    write_json(
        output / "status.json",
        {
            "status": "complete_frozen_copula_mode_physical_alignment",
            "datasets": len(DATASETS), "samples": len(dataset_vector),
            "components": len(COMPONENTS), "modes": DIMENSION,
            "physical_nodes": len(windows),
            "source_specific_fit": False, "target_calibration": False,
            "test_used_for_selection": False, "labels_read": False,
            "observed_sample_state_join": False,
            "input_code_reverified": True, "artifacts": artifacts,
        },
    )
    print((output / "report.md").read_text(encoding="utf-8"), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(output=args.output)
