"""Frozen shared-copula transfer from training sources to validation and test.

All marginal transforms and dependence matrices are fitted on the four training
sources.  Two named same-origin validation datasets select one rank per
component.  External datasets are scored only after those ranks are frozen.
No dataset-specific correction or target-data marginal calibration is fitted.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from copula_transfer_primitives import (
    low_rank_correlation,
    select_rank_from_log_density,
)
from gaussian_copula_models import (
    fit_shrunk_correlation,
    gaussian_copula_log_density,
)
from source_distribution_step1 import ROOT, sha256, write_json
from source_empirical_copula import (
    apply_reference_ecdf,
    fit_reference_ecdf,
    normal_scores,
    pseudo_observations,
)
from source_signed_residual_distribution import blocks


PARENT = ROOT / "experiment_v3_20260724/exp20a_jacobian_geometry_v1"
OUTPUT = ROOT / "output_source_signed_residual_distribution/shared_transfer_v1"
TRAIN = ("Lopex", "Angers", "IFGG_train", "UW_train")
VALIDATION = ("IFGG_val", "UW_val")
EXTERNAL_TEST = (
    "Bel", "Beljap", "Bnl", "Cedar", "FFT", "NGEE_arctic",
    "NGEE_panama", "QGRAD",
)
COMPONENTS = ("residual", "tangent", "normal")
RANKS = (0, 2, 5, 10, 20, 40, 80)
VALIDATION_REPEATS = 100
DIMENSION = 80


def balanced_half_indices(ids, repeat):
    """Return an order-invariant deterministic balanced split of sample IDs."""
    ids = list(map(str, ids))
    if len(ids) < 8 or len(ids) % 2 or len(set(ids)) != len(ids):
        raise ValueError("Unique IDs and an even sample count of at least eight are required.")
    keys = [
        hashlib.sha256(
            f"20260906|shared_copula_transfer_validation_v1|{repeat}|{item}".encode()
        ).digest()
        for item in ids
    ]
    order = np.array(sorted(range(len(ids)), key=lambda index: keys[index]))
    middle = len(order) // 2
    return order[:middle], order[middle:]


def _role(dataset):
    if dataset in TRAIN:
        return "train"
    if dataset in VALIDATION:
        return "validation"
    if dataset in EXTERNAL_TEST:
        return "external_test"
    raise ValueError(f"Unregistered dataset: {dataset}")


def load_full_bank(parent=PARENT):
    """Load and audit all registered Exp20A domains without reading labels."""
    status_path = parent / "exp20a_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    bank_path = parent / "empirical_normal_bank.npz"
    expected = status["artifact_hashes"][bank_path.name]
    if sha256(bank_path) != expected:
        raise ValueError("Historical empirical normal bank hash mismatch.")
    registered = set(TRAIN) | set(VALIDATION) | set(EXTERNAL_TEST)
    bank = {}
    with np.load(bank_path, allow_pickle=False) as z:
        if set(z.files) != {
            "normal_whitened", "residual_whitened", "dataset", "sample_index"
        }:
            raise ValueError("Unexpected empirical bank schema.")
        domains = z["dataset"].astype(str)
        if set(domains) != registered:
            raise ValueError("Empirical bank domains do not match the frozen role registry.")
        for dataset in (*TRAIN, *VALIDATION, *EXTERNAL_TEST):
            mask = domains == dataset
            indices = z["sample_index"][mask].astype(int)
            if len(indices) != 128 or len(set(indices.tolist())) != 128:
                raise ValueError(f"Unexpected sample registry for {dataset}.")
            order = np.argsort(indices)
            residual = z["residual_whitened"][mask][order].astype(float)
            normal = z["normal_whitened"][mask][order].astype(float)
            tangent = residual - normal
            denominator = np.linalg.norm(tangent, axis=1) * np.linalg.norm(normal, axis=1)
            cosine = np.abs(np.sum(tangent * normal, axis=1)) / denominator
            if not np.isfinite(cosine).all() or cosine.max() > 1e-4:
                raise ValueError(f"Tangent/normal orthogonality failed for {dataset}.")
            bank[dataset] = {
                "residual": blocks(residual),
                "tangent": blocks(tangent),
                "normal": blocks(normal),
                "ids": np.array([f"{dataset}:{value}" for value in indices[order]]),
            }
    return bank, {str(status_path): sha256(status_path), str(bank_path): expected}


def fit_frozen_component(bank, component):
    """Fit one pooled-train marginal transform and shared correlation."""
    pooled = np.concatenate([bank[dataset][component] for dataset in TRAIN], axis=0)
    sorted_reference = fit_reference_ecdf(pooled)
    train_z = normal_scores(pseudo_observations(pooled))
    shared = fit_shrunk_correlation(train_z)
    models = {rank: low_rank_correlation(shared, rank) for rank in RANKS}
    return sorted_reference, shared, models


def transform_frozen(values, sorted_reference):
    values = np.asarray(values, dtype=float)
    u = apply_reference_ecdf(values, sorted_reference)
    z = normal_scores(u)
    below = values < sorted_reference[0]
    above = values > sorted_reference[-1]
    diagnostics = {
        "outside_train_range_fraction": float(np.mean(below | above)),
        "below_train_range_fraction": float(np.mean(below)),
        "above_train_range_fraction": float(np.mean(above)),
        "normal_score_mean": float(np.mean(z)),
        "normal_score_rms": float(np.sqrt(np.mean(z * z))),
    }
    return z, diagnostics


def estimable_pattern_diagnostics(z, shared):
    """Compare correlations on coordinates that vary in the target dataset.

    A frozen train ECDF maps an entirely out-of-range target coordinate to one
    constant boundary score. Such coordinates are retained for frozen-model
    density scoring but cannot define a target correlation and are excluded
    only from this descriptive pattern diagnostic.
    """
    z = np.asarray(z, dtype=float)
    shared = np.asarray(shared, dtype=float)
    if z.ndim != 2 or shared.shape != (z.shape[1], z.shape[1]):
        raise ValueError("Pattern inputs must use common sample-by-coordinate axes.")
    varying = np.ptp(z, axis=0) > 0.0
    if int(varying.sum()) < 3:
        raise ValueError("Fewer than three varying target coordinates.")
    empirical = np.corrcoef(z[:, varying], rowvar=False)
    empirical = 0.5 * (empirical + empirical.T)
    np.fill_diagonal(empirical, 1.0)
    reference = shared[np.ix_(varying, varying)]
    reference_edges = reference[np.triu_indices(len(reference), 1)]
    empirical_edges = empirical[np.triu_indices(len(empirical), 1)]
    if np.ptp(reference_edges) <= 1e-14 or np.ptp(empirical_edges) <= 1e-14:
        raise ValueError("Correlation edge pattern is numerically constant.")
    return {
        "pattern_band_count": int(varying.sum()),
        "constant_band_count": int((~varying).sum()),
        "pattern_r": float(np.corrcoef(reference_edges, empirical_edges)[0, 1]),
        "rmse": float(np.sqrt(np.mean((reference_edges - empirical_edges) ** 2))),
        "sign_agreement": float(
            np.mean(np.sign(reference_edges) == np.sign(empirical_edges))
        ),
    }


def validation_rank_selection(bank, component, sorted_reference, models):
    """Select a rank on validation only and audit balanced heldout halves."""
    scores = {}
    z_by_dataset = {}
    for dataset in VALIDATION:
        z, _ = transform_frozen(bank[dataset][component], sorted_reference)
        z_by_dataset[dataset] = z
        for rank, correlation in models.items():
            scores[dataset, rank] = gaussian_copula_log_density(z, correlation) / DIMENSION
    combined = {
        rank: np.concatenate([scores[dataset, rank] for dataset in VALIDATION])
        for rank in RANKS
    }
    selected, diagnostic = select_rank_from_log_density(combined)
    candidate_rows = [
        {
            "component": component,
            "rank": int(rank),
            "validation_rows": int(diagnostic["n_samples"]),
            "mean_log_density_per_dim": float(score),
            "selected": int(rank == selected),
        }
        for rank, score in zip(diagnostic["candidate_ranks"], diagnostic["mean_scores"])
    ]

    split_rows = []
    for repeat in range(VALIDATION_REPEATS):
        halves = {
            dataset: balanced_half_indices(bank[dataset]["ids"], repeat)
            for dataset in VALIDATION
        }
        for direction, (selection_half, evaluation_half) in enumerate(((0, 1), (1, 0))):
            selection_scores = {
                rank: np.concatenate(
                    [scores[dataset, rank][halves[dataset][selection_half]] for dataset in VALIDATION]
                )
                for rank in RANKS
            }
            fold_rank, fold_diagnostic = select_rank_from_log_density(selection_scores)
            heldout_selected = np.concatenate(
                [scores[dataset, fold_rank][halves[dataset][evaluation_half]] for dataset in VALIDATION]
            )
            heldout_full = np.concatenate(
                [scores[dataset, DIMENSION][halves[dataset][evaluation_half]] for dataset in VALIDATION]
            )
            split_rows.append(
                {
                    "component": component,
                    "repeat": repeat,
                    "direction": direction,
                    "selection_rows": int(fold_diagnostic["n_samples"]),
                    "evaluation_rows": int(len(heldout_selected)),
                    "selected_rank": int(fold_rank),
                    "selection_score_per_dim": float(fold_diagnostic["selected_score"]),
                    "heldout_selected_log_density_per_dim": float(np.mean(heldout_selected)),
                    "heldout_full_log_density_per_dim": float(np.mean(heldout_full)),
                    "heldout_selected_minus_full_per_dim": float(
                        np.mean(heldout_selected) - np.mean(heldout_full)
                    ),
                }
            )
    return selected, candidate_rows, split_rows


def summarize_scores(sample_scores):
    rows = []
    for key, group in sample_scores.groupby(
        ["component", "role", "dataset", "model", "rank"], sort=False, dropna=False
    ):
        component, role, dataset, model, rank = key
        values = group.log_density_per_dim.to_numpy(float)
        rows.append(
            {
                "component": component,
                "role": role,
                "dataset": dataset,
                "model": model,
                "rank": int(rank),
                "rows": len(values),
                "mean_log_density_per_dim": float(np.mean(values)),
                "median_log_density_per_dim": float(np.median(values)),
                "q05_log_density_per_dim": float(np.quantile(values, 0.05)),
                "q95_log_density_per_dim": float(np.quantile(values, 0.95)),
                "positive_fraction": float(np.mean(values > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _balanced_probe_folds(ids, folds=5):
    ids = list(map(str, ids))
    if len(ids) < folds or len(set(ids)) != len(ids):
        raise ValueError("Probe folds require enough unique sample IDs.")
    keys = [hashlib.sha256(f"20260906|source_probe_v1|{item}".encode()).digest() for item in ids]
    order = np.array(sorted(range(len(ids)), key=lambda index: keys[index]))
    assignment = np.empty(len(ids), dtype=int)
    assignment[order] = np.arange(len(ids)) % folds
    return assignment


def source_identity_probe(sample_scores):
    """Cross-validated nearest-centroid probe on three frozen density scores."""
    selected = sample_scores[sample_scores.model == "validation_selected"]
    features = selected.pivot(
        index=["role", "dataset", "sample_id"],
        columns="component", values="log_density_per_dim",
    ).reset_index()
    if set(COMPONENTS) - set(features.columns):
        raise ValueError("Missing frozen component score for source probe.")
    features["fold"] = -1
    for dataset, index in features.groupby("dataset").groups.items():
        positions = np.asarray(list(index), dtype=int)
        features.loc[positions, "fold"] = _balanced_probe_folds(
            features.loc[positions, "sample_id"].astype(str).to_numpy()
        )
    scopes = {
        "validation_only": set(VALIDATION),
        "external_test_only": set(EXTERNAL_TEST),
        "validation_and_external": set(VALIDATION) | set(EXTERNAL_TEST),
    }
    prediction_rows, summary_rows = [], []
    for scope, datasets in scopes.items():
        frame = features[features.dataset.isin(datasets)].copy()
        classes = sorted(datasets)
        for fold in range(5):
            train = frame[frame.fold != fold]
            test = frame[frame.fold == fold]
            x_train = train[list(COMPONENTS)].to_numpy(float)
            x_test = test[list(COMPONENTS)].to_numpy(float)
            mean = x_train.mean(axis=0)
            scale = x_train.std(axis=0, ddof=0)
            if np.any(scale <= 1e-12):
                raise ValueError("Degenerate source-probe feature scale.")
            x_train = (x_train - mean) / scale
            x_test = (x_test - mean) / scale
            centroids = np.vstack(
                [x_train[train.dataset.to_numpy() == label].mean(axis=0) for label in classes]
            )
            distance = np.sum((x_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            prediction = np.asarray(classes)[np.argmin(distance, axis=1)]
            for row, predicted in zip(test.itertuples(index=False), prediction):
                prediction_rows.append(
                    {
                        "scope": scope, "fold": fold, "role": row.role,
                        "dataset": row.dataset, "sample_id": row.sample_id,
                        "predicted_dataset": predicted,
                        "correct": int(predicted == row.dataset),
                    }
                )
        scope_predictions = pd.DataFrame(
            [row for row in prediction_rows if row["scope"] == scope]
        )
        recalls = scope_predictions.groupby("dataset").correct.mean()
        summary_rows.append(
            {
                "scope": scope,
                "classes": len(classes),
                "rows": len(scope_predictions),
                "chance_accuracy": 1.0 / len(classes),
                "accuracy": float(scope_predictions.correct.mean()),
                "balanced_accuracy": float(recalls.mean()),
                "minimum_class_recall": float(recalls.min()),
                "maximum_class_recall": float(recalls.max()),
                "features": "+".join(COMPONENTS) + " frozen log-density scores",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def make_figures(output, dataset_summary, rank_selection, split_audit):
    selected = dataset_summary[dataset_summary.model == "validation_selected"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    order = list(VALIDATION) + list(EXTERNAL_TEST)
    colors = {"validation": "#D68910", "external_test": "#2874A6"}
    for ax, component in zip(axes, COMPONENTS):
        frame = selected[selected.component == component].set_index("dataset").loc[order]
        mid = frame.mean_log_density_per_dim.to_numpy()
        low = frame.q05_log_density_per_dim.to_numpy()
        high = frame.q95_log_density_per_dim.to_numpy()
        ax.errorbar(
            np.arange(len(order)), mid, yerr=[mid - low, high - mid], fmt="none",
            ecolor="#777777", capsize=3,
        )
        ax.scatter(
            np.arange(len(order)), mid,
            c=[colors[value] for value in frame.role], s=42, zorder=3,
        )
        ax.axhline(0, color="black", lw=0.8, ls="--")
        rank = int(frame["rank"].iloc[0])
        ax.set_ylabel("Log copula density / band")
        ax.set_title(f"{component} | validation-selected rank {rank}", loc="left")
        ax.grid(alpha=0.2)
    axes[-1].set_xticks(np.arange(len(order)), order, rotation=30, ha="right")
    fig.suptitle("Frozen pooled-train copula on validation and external test\nPoints are dataset means; bars are sample 5--95% ranges")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "frozen_transfer_scores.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, component in zip(axes, COMPONENTS):
        frame = rank_selection[rank_selection.component == component]
        ax.plot(frame["rank"], frame.mean_log_density_per_dim, "o-", color="#2874A6")
        chosen = frame[frame.selected == 1].iloc[0]
        ax.scatter([chosen["rank"]], [chosen.mean_log_density_per_dim], color="#C0392B", s=70, zorder=3)
        counts = Counter(
            split_audit[split_audit.component == component].selected_rank.astype(int)
        )
        subtitle = ", ".join(f"r{k}:{v/200:.0%}" for k, v in sorted(counts.items()))
        ax.set(title=f"{component}\nvalidation-half choices {subtitle}", xlabel="Candidate rank", ylabel="Validation log density / band")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.grid(alpha=0.2)
    fig.suptitle("Rank selected only from IFGG_val and UW_val")
    fig.tight_layout()
    fig.savefig(output / "validation_rank_selection.png", dpi=180)
    plt.close(fig)


def make_report(
    output, rank_selection, split_audit, dataset_summary, patterns, margins,
    probe_summary,
):
    chosen = rank_selection[rank_selection.selected == 1].set_index("component")
    selected = dataset_summary[dataset_summary.model == "validation_selected"]
    lines = [
        "# Frozen shared-copula transfer",
        "",
        "All transforms and dependence matrices were fitted on Lopex, Angers, "
        "IFGG_train, and UW_train. Rank was selected jointly on IFGG_val and "
        "UW_val. No validation/test marginal calibration or dataset-specific "
        "correction was fitted.",
        "",
        "## Validation selection",
        "",
    ]
    for component in COMPONENTS:
        row = chosen.loc[component]
        splits = split_audit[split_audit.component == component]
        most_common = Counter(splits.selected_rank.astype(int)).most_common(1)[0]
        lines.append(
            f"- {component}: full-validation rank {int(row['rank'])}, mean log "
            f"density/band {row.mean_log_density_per_dim:.4f}; balanced-half modal "
            f"rank {most_common[0]} ({most_common[1] / len(splits):.1%}), median "
            f"heldout selected-minus-full {splits.heldout_selected_minus_full_per_dim.median():+.4f}."
        )
    lines.extend(["", "## Frozen-score source identity probe", ""])
    for row in probe_summary.itertuples(index=False):
        lines.append(
            f"- {row.scope}: {row.classes}-class 5-fold nearest-centroid accuracy "
            f"{row.accuracy:.1%} (chance {row.chance_accuracy:.1%}); balanced "
            f"accuracy {row.balanced_accuracy:.1%}."
        )
    lines.extend(["", "## Frozen dataset results", ""])
    for component in COMPONENTS:
        frame = selected[selected.component == component].set_index("dataset")
        validation_mean = frame.loc[list(VALIDATION)].mean_log_density_per_dim.mean()
        external_mean = frame.loc[list(EXTERNAL_TEST)].mean_log_density_per_dim.mean()
        positive_external = int((frame.loc[list(EXTERNAL_TEST)].mean_log_density_per_dim > 0).sum())
        lines.append(
            f"- {component}: validation dataset-mean {validation_mean:.4f}; external "
            f"dataset-mean {external_mean:.4f}; {positive_external}/{len(EXTERNAL_TEST)} "
            "external datasets have positive mean gain over independence."
        )
    lines.extend(["", "## Domain diagnostics", ""])
    for component in COMPONENTS:
        p = patterns[(patterns.component == component) & (patterns.role == "external_test")]
        m = margins[(margins.component == component) & (margins.role == "external_test")]
        lines.append(
            f"- {component}: external median shared-correlation pattern r "
            f"{p.pattern_r.median():.3f}; median out-of-train-range coordinate "
            f"fraction {m.outside_train_range_fraction.median():.3%}."
        )
    lines.extend(
        [
            "",
            "Balanced validation halves describe rank stability in this finite bank; "
            "they are not independent replications or confidence intervals. This is "
            "one historical Exp20A checkpoint, not a multi-seed confirmation.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(parent=PARENT, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=False)
    bank, inputs = load_full_bank(parent)
    role_rows = [
        {"dataset": dataset, "role": _role(dataset), "rows": len(bank[dataset]["ids"])}
        for dataset in (*TRAIN, *VALIDATION, *EXTERNAL_TEST)
    ]
    rank_rows, split_rows, sample_rows, pattern_rows, margin_rows = [], [], [], [], []
    matrices = {}
    selections = {}

    # The complete validation selection is executed before any external-test
    # transformation or scoring below.
    frozen = {}
    for component in COMPONENTS:
        reference, shared, models = fit_frozen_component(bank, component)
        selected_rank, component_rank_rows, component_split_rows = validation_rank_selection(
            bank, component, reference, models
        )
        selections[component] = selected_rank
        rank_rows.extend(component_rank_rows)
        split_rows.extend(component_split_rows)
        frozen[component] = (reference, shared, models)
        matrices[f"{component}__shared_correlation"] = shared
        matrices[f"{component}__selected_correlation"] = models[selected_rank]
        matrices[f"{component}__pooled_train_sorted_ecdf"] = reference

    # Validation and external test now receive only the frozen selected and full
    # models. External candidate ranks are never computed or used for selection.
    for component in COMPONENTS:
        reference, shared, models = frozen[component]
        selected_rank = selections[component]
        for dataset in (*VALIDATION, *EXTERNAL_TEST):
            z, diagnostics = transform_frozen(bank[dataset][component], reference)
            role = _role(dataset)
            margin_rows.append({"component": component, "role": role, "dataset": dataset, **diagnostics})
            pattern_rows.append(
                {
                    "component": component,
                    "role": role,
                    "dataset": dataset,
                    **estimable_pattern_diagnostics(z, shared),
                }
            )
            model_registry = (
                ("independence", 0, np.zeros(len(z), dtype=float)),
                (
                    "validation_selected", selected_rank,
                    gaussian_copula_log_density(z, models[selected_rank]) / DIMENSION,
                ),
                (
                    "full_shared", DIMENSION,
                    gaussian_copula_log_density(z, models[DIMENSION]) / DIMENSION,
                ),
            )
            for model, rank, values in model_registry:
                for sample_id, value in zip(bank[dataset]["ids"], values):
                    sample_rows.append(
                        {
                            "component": component,
                            "role": role,
                            "dataset": dataset,
                            "sample_id": sample_id,
                            "model": model,
                            "rank": int(rank),
                            "log_density_per_dim": float(value),
                            "residual_shape_atypicality": float(-value),
                        }
                    )

    roles = pd.DataFrame(role_rows)
    rank_selection = pd.DataFrame(rank_rows)
    split_audit = pd.DataFrame(split_rows)
    sample_scores = pd.DataFrame(sample_rows)
    dataset_summary = summarize_scores(sample_scores)
    patterns = pd.DataFrame(pattern_rows)
    margins = pd.DataFrame(margin_rows)
    probe_summary, probe_predictions = source_identity_probe(sample_scores)
    for frame in (
        rank_selection, split_audit, sample_scores, dataset_summary, patterns,
        margins, probe_summary, probe_predictions,
    ):
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Nonfinite numerical output.")

    roles.to_csv(output / "dataset_roles.csv", index=False)
    rank_selection.to_csv(output / "validation_rank_selection.csv", index=False)
    split_audit.to_csv(output / "validation_half_split_audit.csv.gz", index=False)
    sample_scores.to_csv(output / "frozen_sample_scores.csv.gz", index=False)
    dataset_summary.to_csv(output / "frozen_dataset_summary.csv", index=False)
    patterns.to_csv(output / "correlation_pattern_diagnostics.csv", index=False)
    margins.to_csv(output / "marginal_shift_diagnostics.csv", index=False)
    probe_summary.to_csv(output / "source_identity_probe.csv", index=False)
    probe_predictions.to_csv(output / "source_identity_probe_predictions.csv.gz", index=False)
    np.savez_compressed(output / "frozen_shared_models.npz", **matrices)
    make_figures(output, dataset_summary, rank_selection, split_audit)
    make_report(
        output, rank_selection, split_audit, dataset_summary, patterns, margins,
        probe_summary,
    )

    scripts = {}
    for name in (
        "source_shared_copula_transfer.py", "copula_transfer_primitives.py",
        "gaussian_copula_models.py", "source_empirical_copula.py",
    ):
        path = ROOT / name
        scripts[str(path)] = sha256(path)
    manifest = {
        "experiment": "frozen_shared_copula_transfer_v1",
        "train": list(TRAIN),
        "validation": list(VALIDATION),
        "external_test": list(EXTERNAL_TEST),
        "components": list(COMPONENTS),
        "candidate_ranks": list(RANKS),
        "rank_selection": selections,
        "marginal_fit": "one pooled empirical CDF per component using train only",
        "dependence_fit": "one pooled Ledoit-Wolf Gaussian-copula correlation per component using train only",
        "low_rank_rule": "retain modes ordered by descending abs(eigenvalue-1), omitted eigenvalues set to one, then correlation normalize",
        "validation_selection": "sample-weighted mean log copula density across IFGG_val and UW_val",
        "external_test_policy": "score only after validation rank freeze; no candidate-rank test scores",
        "dataset_specific_fit": False,
        "labels_read": False,
        "dataset_identity_probe": "post-freeze five-fold nearest-centroid diagnostic on three selected density scores",
        "input_sha256": inputs,
        "code_sha256": scripts,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json(output / "protocol_manifest.json", manifest)
    for path, expected in {**inputs, **scripts}.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Input or code changed during analysis: {path}")
    artifacts = {
        path.name: sha256(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    }
    write_json(
        output / "status.json",
        {
            "status": "complete_frozen_shared_copula_transfer",
            "train_rows": 512,
            "validation_rows": 256,
            "external_test_rows": 1024,
            "source_specific_fit": False,
            "target_marginal_calibration": False,
            "test_used_for_selection": False,
            "labels_read": False,
            "single_historical_checkpoint": True,
            "selected_ranks": selections,
            "input_code_reverified": True,
            "artifacts": artifacts,
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
