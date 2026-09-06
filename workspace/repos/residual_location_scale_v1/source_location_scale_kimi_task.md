# Bounded implementation task: historical residual location/scale study

Implement a local scientific analysis in exactly two new files:
1. `source_signed_residual_location_scale.py`
2. `test_source_signed_residual_location_scale.py`

Do not modify the provided helpers, summaries, or any other files. Do not execute commands. Return a compressed handoff in Chinese with file/line references, assumptions, limitations and tests added. Main agent will review and execute everything. Your workspace contains copies of the three existing helper scripts and earlier summary CSVs; actual data stays in the research project and is not needed for code authoring. Do not access any path outside the assigned workspace.

## Research objective

Determine whether empirical source differences persist after removing per-band source location and scale estimated ONLY from independent reference samples. Normal is the main component, tangent and residual are parallel descriptive controls. No neural training, labels, classifiers, risk endpoints, Gaussian fitting or permutation p-values. This is descriptive finite-bank sensitivity, not a test proving common shape or causal attribution.

## Reuse and input

Import ROOT, sha256, write_json from source_distribution_step1; import PARENT, SOURCES, COMPONENTS, blocks from source_signed_residual_distribution; import PRIOR, load_bank, describe, compare, summarize from source_signed_residual_reproducibility.
load_bank(PARENT, PRIOR) already validates historical Exp20A bank and prior pilot hashes, selects four source domains (128 unique samples each), returns bank[source][component] and bank[source]['ids'] plus path-to-hash dictionary. Mixed-domain archive is opened but only source rows are analyzed. No target labels exist in selected inputs. Also validate ALL artifacts and status from output_source_signed_residual_distribution/reproducibility_v1, record hashes and recheck at completion.
Use fixed 80 signed nonoverlapping 25 nm averages from the 400--2399 nm full vectors. Do not fit PCA or new geometry. This step makes no assertion about fine-scale standardized full-2000D distributions.

## Prespecified sampling and transformations

200 repeats. For each source/repeat, SHA256 sort exact ID strings using key `20260905|location_scale_v1|repeat|sample_id`, then split 128 into four disjoint blocks of 32: ref_A, eval_A, ref_B, eval_B. Preserve this role assignment across all components and stages. Reject duplicate IDs or unexpected counts. Provide a helper quarters_for_ids(ids, repeat) that works with any unique count divisible by 4 for tests. Assignments must be invariant to input row ordering.

For each branch, fit on ref only, transform corresponding eval only:
- raw: unchanged.
- center_median: subtract reference band medians.
- scale_median_iqr: subtract reference band medians and divide by reference band IQR. This is the primary standardization.
- center_mean: subtract reference band means.
- scale_mean_sd: subtract reference band means and divide by reference population SD (ddof=0). This is prespecified sensitivity to the choice of estimator, not an alternative selected after seeing results.

Do not clip, winsorize, add offsets, or normalize by individual sample norms. Reject nonfinite values or nonpositive/near-zero band scales (<=1e-12) with a clear error; do not silently floor. Save reference location/scale diagnostics (min, median, max, count<=1e-12), and save all per-band reference parameters in gzip CSV for reproducibility. Fit function must not accept evaluation samples. Affine changes preserve each eval group's Pearson/Spearman correlations; audit this explicitly, do not present unchanged correlation metrics across stages as new independent findings.

## Distance comparison

Every compared group has 32 evaluated samples, and EVERY transformed group has its OWN independent 32-sample reference set.
For each source/component/stage/repeat:
- compute compare(describe(eval_A), describe(eval_B)) for marginal_w1, median_mae, iqr_mae, tail_mae and Pearson/Spearman correlation RMSE/pattern_r;
- add joint energy V-statistic 2 mean(dXY)-mean(dXX)-mean(dYY), diagonals included, d=Euclidean/sqrt(80). Do not take the square root. Use scipy cdist/pdist direct sums rather than computing permutation nulls.

For each of six source pairs/component/stage/repeat:
- cross distance = average of all four cross-source branch comparisons (A/A, A/B, B/A, B/B).
- within distance = mean of the two sources' own A/B distances.
- save cross, within, cross/within for marginal_w1, median_mae, iqr_mae, tail_mae, joint_energy. Reject near-zero denominators.
- Store cross/within correlation RMSE only for raw stage in a separate table or clearly mark affine invariance. Main results focus on marginal_w1 and joint_energy ratios.

Summarize each source and pair metric with median and 5th/95th repeat quantiles. These are repeated-partition ranges, NOT confidence intervals; 200 repeats reuse a finite bank and are not independent replications. No significant/non-significant classification. Ratio near 1 does not prove equality, particularly with n_ref=n_eval=32.

Do NOT compare raw W1 units against standardized W1 as percentages of physical variance explained. Ratios cross/within are descriptive source contrast relative to calibrated internal variation, and changes in these ratios can reflect both contrast reduction and calibration-noise inflation. Always retain numerator and denominator separately. Median/IQR and mean/SD may give different answers; report both, do not pick one.

## Output contract

DEFAULT_OUTPUT = ROOT/'output_source_signed_residual_distribution/location_scale_v1'. Refuse an existing directory. CLI --output permits a fresh directory; parent paths may remain defaults.
Write protocol_manifest.json BEFORE numeric computation: all sampling keys, input hashes, three helper hashes plus own script hash, versions, stages, comparison formulas, n_ref/n_eval, scope and limitations.
Artifacts:
- split_assignments.csv.gz (source, repeat, sample_id, role)
- reference_parameters.csv.gz (source, component, repeat, branch, band, start_nm, median, iqr, mean, sd)
- reference_diagnostics.csv
- within_metrics.csv.gz and between_metrics.csv.gz
- source_summary.csv and pair_summary.csv
- affine_correlation_audit.csv with max abs Pearson/Spearman difference per source/component/stage over branches/repeats
- normal_stage_contrasts.png: six source pairs, raw/center_median/scale_median_iqr W1 ratio median and split ranges; second panel joint_energy.
- normal_estimator_sensitivity.png: median/IQR vs mean/SD standardized ratios, W1 and energy.
- report.md in Chinese: method, dynamic numeric normal pair tables raw->centered->scaled, both estimators, scope, within-baseline interpretation and no p-values. Include short full-component summaries. Do not hard-code optimistic interpretations. Do not claim correlations disappeared after per-band affine transformation.
- status.json with complete_historical_residual_location_scale, sample count, input/code reverified and artifact hashes (exclude status itself).

Use Agg plotting and threadpool_limits(1) in __main__. Print concise progress once per component/stage or source. Runtime should be minutes on local CPU. Avoid redundant Pearson/Spearman fitting when stage changes cannot alter those values, but preserve the audit. No dependencies beyond installed numpy/scipy/pandas/matplotlib/threadpoolctl and existing helpers.

## Meaningful tests

1. Four-way splits are disjoint, complete, equal size and ID-order invariant.
2. Reference fit + eval transform exactly remove known positive per-band affine transforms (use corresponding affine-related ref/eval arrays); changing eval cannot alter reference parameters.
3. Pearson/Spearman eval correlation matrices are invariant under those transformations.
4. joint energy implementation matches direct formula and one-dimensional scipy.stats.energy_distance squared.
5. Marginal W1 can be zero for equal marginals with changed coordinate pairing while joint energy is positive.
6. Degenerate reference scale rejected without flooring; duplicate sample IDs rejected.

Use independent expected values or controls, not assertions that merely mirror implementation. Main agent will add/run a small synthetic integration control after reviewing your code.
