"""Historical signed residual location/scale sensitivity study.

For each source/component and 200 ID-hash-fixed repeats, 128 samples are split
into ref_A / eval_A / ref_B / eval_B (32 each). Per-band reference location and
scale are estimated ONLY from the corresponding 32-sample reference block, then
applied to the paired evaluation block. Source differences are re-quantified
per stage as cross/within distance ratios. This is descriptive finite-bank
sensitivity: repeated partitions are not independent replications, ratio ~1 is
not equality, and no p-values, Gaussian fitting or causal claims are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits

from source_distribution_step1 import ROOT, sha256, write_json
from source_signed_residual_distribution import PARENT, SOURCES, COMPONENTS, blocks
from source_signed_residual_reproducibility import PRIOR, load_bank, describe, compare, summarize

REPRO = ROOT / 'output_source_signed_residual_distribution/reproducibility_v1'
DEFAULT_OUTPUT = ROOT / 'output_source_signed_residual_distribution/location_scale_v1'
SPLIT_REPEATS = 200
SPLIT_SALT = '20260905|location_scale_v1'
ROLES = ('ref_A', 'eval_A', 'ref_B', 'eval_B')
STAGES = ('raw', 'center_median', 'scale_median_iqr', 'center_mean', 'scale_mean_sd')
SCALE_FLOOR = 1e-12
AUDIT_REPEATS = 3  # per stage, correlations are recomputed on the first repeats as an invariance spot-check
DIST4 = ('marginal_w1', 'median_mae', 'iqr_mae', 'tail_mae')
ENERGY_METRICS = (*DIST4, 'joint_energy')
CORR_METRICS = ('pearson_corr_rmse', 'spearman_corr_rmse')
PAIR_NAMES = [f'{a} / {b}' for a, b in combinations(SOURCES, 2)]


def quarters_for_ids(ids, repeat):
    """Order-invariant disjoint quarter assignment fixed only by IDs and repeat.

    Works for any unique count divisible by four. Returns {role: index array}
    with roles ref_A, eval_A, ref_B, eval_B of equal size.
    """
    ids = [str(s) for s in ids]
    n = len(ids)
    if n == 0 or n % 4:
        raise ValueError('Sample count must be positive and divisible by four.')
    if len(set(ids)) != n:
        raise ValueError('Duplicate sample IDs are not allowed.')
    keys = [hashlib.sha256(f'{SPLIT_SALT}|{repeat}|{s}'.encode()).hexdigest() for s in ids]
    order = np.argsort(keys, kind='stable')
    buckets = {role: [] for role in ROLES}
    for rank, idx in enumerate(order):
        buckets[ROLES[rank % 4]].append(int(idx))
    return {role: np.asarray(ix, dtype=int) for role, ix in buckets.items()}


def fit_reference(ref):
    """Per-band location/scale estimated ONLY from the reference block.

    Deliberately accepts no evaluation data, so evaluation values can never
    influence the fitted parameters.
    """
    ref = np.asarray(ref, dtype=float)
    if ref.ndim != 2 or len(ref) < 2 or not np.isfinite(ref).all():
        raise ValueError('Reference matrix must be 2D, finite, with at least two rows.')
    q25, q75 = np.quantile(ref, [.25, .75], axis=0)
    return {'median': np.median(ref, axis=0), 'iqr': q75 - q25,
            'mean': ref.mean(axis=0), 'sd': ref.std(ddof=0)}


def transform_eval(eval_x, params, branch):
    """Apply the reference-only location/scale to the evaluation block.

    No clipping, winsorizing, offsets or per-sample norm normalization.
    Scaling branches refuse nonpositive/near-zero (<=1e-12) reference bands
    instead of silently flooring.
    """
    x = np.asarray(eval_x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Evaluation matrix must be 2D and finite.')
    if branch == 'raw':
        return x
    if branch not in STAGES:
        raise ValueError(f'Unknown branch: {branch}')
    loc_key = 'median' if 'median' in branch else 'mean'
    out = x - np.asarray(params[loc_key], dtype=float)
    if branch.startswith('scale'):
        scale_key = 'iqr' if 'iqr' in branch else 'sd'
        scale = np.asarray(params[scale_key], dtype=float)
        bad = int((scale <= SCALE_FLOOR).sum())
        if bad:
            raise ValueError(f'Reference {scale_key} has {bad} nonpositive/near-zero '
                             f'(<=1e-12) bands; refusing to floor or divide.')
        out = out / scale
    return out


def correlation_vectors(x):
    """Upper-triangle Pearson and Spearman correlation vectors of a sample matrix."""
    x = np.asarray(x, dtype=float)
    upper = np.triu_indices(x.shape[1], 1)
    return (np.corrcoef(x, rowvar=False)[upper],
            np.corrcoef(stats.rankdata(x, axis=0), rowvar=False)[upper])


def eval_descriptor(x, corr=None):
    """Descriptor compatible with compare(); correlations reused when supplied.

    Per-band affine transformations cannot change Pearson/Spearman values, so
    non-raw stages pass the raw-stage correlation vectors instead of refitting.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(x.std(axis=0) <= SCALE_FLOOR):
        raise ValueError('Evaluation descriptor requires finite, nondegenerate per-band variation.')
    q = np.quantile(x, [.05, .25, .5, .75, .95], axis=0)
    d = {'sorted': np.sort(x, axis=0), 'median': q[2], 'iqr': q[3] - q[1], 'tails': q[[0, 4]]}
    if corr is None:
        d['pearson'], d['spearman'] = correlation_vectors(x)
    else:
        d['pearson'], d['spearman'] = corr
    return d


def joint_energy_v(x, y):
    """Biased joint energy V-statistic with diagonals included, no square root.

    V = 2 mean(dXY) - mean(dXX) - mean(dYY), d = Euclidean / sqrt(n_coordinates).
    Computed with direct cdist/pdist sums; no permutation nulls.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(x) < 2 or len(y) < 2:
        raise ValueError('Expected two matrices with >=2 rows and a shared coordinate count.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Joint energy requires finite inputs.')
    n, m, scale = len(x), len(y), np.sqrt(x.shape[1])
    dxy = cdist(x, y) / scale
    dxx = pdist(x) / scale
    dyy = pdist(y) / scale
    return float(2 * dxy.sum() / (n * m) - 2 * dxx.sum() / (n * n) - 2 * dyy.sum() / (m * m))


def reference_diagnostics(params):
    """Min/median/max and count of nonpositive bands per reference statistic."""
    rows = []
    for stat in ('median', 'iqr', 'mean', 'sd'):
        v = np.asarray(params[stat], dtype=float)
        rows.append({'stat': stat, 'min': float(v.min()), 'median': float(np.median(v)),
                     'max': float(v.max()), 'count_nonpos': int((v <= SCALE_FLOOR).sum())})
    return rows


def validate_reproducibility_run(repro):
    """Hash every artifact recorded by the reproducibility_v1 status file."""
    repro = Path(repro)
    status_path = repro / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('status') != 'complete_historical_signed_residual_reproducibility':
        raise ValueError(f'Unexpected reproducibility_v1 status in {status_path}')
    hashes = {}
    for name, expected in sorted(status['artifacts'].items()):
        path = repro / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'reproducibility_v1 artifact hash mismatch: {path}')
        hashes[f'reproducibility_v1/{name}'] = actual
    hashes['reproducibility_v1/status.json'] = sha256(status_path)
    return hashes


def make_figures(output, pair_summary):
    sub = pair_summary[pair_summary.component == 'normal'].set_index(['pair', 'branch'])
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    for ax, metric, title in ((axes[0], 'marginal_w1', 'Marginal W1 ratio'),
                              (axes[1], 'joint_energy', 'Joint energy V-ratio')):
        for k, branch in enumerate(('raw', 'center_median', 'scale_median_iqr')):
            rows = pd.DataFrame([sub.loc[(pair, branch)] for pair in PAIR_NAMES])
            mid = rows[f'{metric}_ratio_median'].to_numpy()
            lo, hi = rows[f'{metric}_ratio_q05'].to_numpy(), rows[f'{metric}_ratio_q95'].to_numpy()
            ax.errorbar(np.arange(6) + (k - 1) * .21, mid, yerr=[mid - lo, hi - mid],
                        fmt='o', capsize=3, label=branch)
        ax.axhline(1, color='#777777', ls='--', lw=1)
        ax.set_xticks(range(6), [p.replace('_train', '') for p in PAIR_NAMES], rotation=18)
        ax.set(title=f'{title} | median and 5--95% repeated-partition range',
               ylabel='cross-source / within-source')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, title='stage')
    fig.suptitle('Normal residual: removing per-band reference location/scale (n_ref = n_eval = 32)\n'
                 'Ranges describe 200 partitions of one finite bank, NOT confidence intervals; no p-values', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(output / 'normal_stage_contrasts.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5))
    for ax, metric, title in ((axes[0], 'marginal_w1', 'Marginal W1 ratio'),
                              (axes[1], 'joint_energy', 'Joint energy V-ratio')):
        for k, branch in enumerate(('scale_median_iqr', 'scale_mean_sd')):
            rows = pd.DataFrame([sub.loc[(pair, branch)] for pair in PAIR_NAMES])
            mid = rows[f'{metric}_ratio_median'].to_numpy()
            lo, hi = rows[f'{metric}_ratio_q05'].to_numpy(), rows[f'{metric}_ratio_q95'].to_numpy()
            ax.errorbar(np.arange(6) + (k - .5) * .22, mid, yerr=[mid - lo, hi - mid],
                        fmt='s', capsize=3, label=branch)
        ax.axhline(1, color='#777777', ls='--', lw=1)
        ax.set_xticks(range(6), [p.replace('_train', '') for p in PAIR_NAMES], rotation=18)
        ax.set(title=f'{title} | estimator sensitivity', ylabel='cross-source / within-source')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, title='standardization')
    fig.suptitle('Normal residual: median/IQR vs mean/SD reference standardization\n'
                 'Prespecified estimator sensitivity; ranges are repeated-partition ranges, not confidence intervals', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(output / 'normal_estimator_sensitivity.png', dpi=170)
    plt.close(fig)


def ratio_cell(row, metric):
    return (f"{row[f'{metric}_ratio_median']:.2f} "
            f"[{row[f'{metric}_ratio_q05']:.2f}, {row[f'{metric}_ratio_q95']:.2f}]")


def report(output, pair_summary):
    normal = pair_summary[pair_summary.component == 'normal'].set_index(['pair', 'branch'])
    lines = ['# 有符号残差：去除 source 内位置/尺度后的 source 差异（历史 bank 敏感性）', '',
        '**基座限定：Exp20A 单个历史 checkpoint，四个 source 各 128 条完整残差曲线，400–2399 nm。** 本轮没有新训练、没有读取性状标签；混合域归档仅为筛选 source 而打开。坐标沿用 80 个固定 25 nm 有符号均值，不拟合 PCA 或新几何。', '',
        '## 本轮做了什么', '',
        '- 每 source、每分量进行 200 次按样本 ID 哈希固定的四分组划分：ref_A / eval_A / ref_B / eval_B，各 32 条。角色在 residual / tangent / normal 及各阶段间保持一致。',
        '- 每波段的参考位置与尺度**只**由配对的 32 条参考样本估计，再作用于对应的 32 条评估样本：raw（不变）、center_median（减参考中位数）、scale_median_iqr（减参考中位数再除以参考 IQR，主标准化）、center_mean（减参考均值）、scale_mean_sd（减参考均值再除以参考总体 SD，ddof=0，事先指定的估计量敏感性）。不使用裁剪、winsorize、偏移或按样本范数归一化；参考尺度 <=1e-12 的波段直接报错，不做隐式下限。',
        '- 每 source 内距离 = 同一次划分下两个评估组的距离；source 间距离 = 四个跨 source 分支比较（A/A、A/B、B/A、B/B）的均值。比值为 cross/within，分子分母分别保留在 between_metrics.csv.gz。比值接近 1 不证明分布相等（n_ref = n_eval = 32 时尤其如此）；比值变化可同时反映对比度下降与标定噪声放大，两者无法在本设计中分离。',
        '- 联合分布距离为 energy V 统计量 2 mean(dXY)−mean(dXX)−mean(dYY)（含对角、不开平方，d 为 Euclidean/√80），直接用 cdist/pdist 求和，不做置换零分布、不产出 p 值。',
        '- Pearson/Spearman 波段相关在逐波段仿射变换下数学不变；各阶段的相关 RMSE 只在 raw 阶段保存，affine_correlation_audit.csv 记录独立重算与前几个重复的最大绝对差（应约为 0）。不把各阶段重复的相关数值当作新的独立发现，也不声称相关结构在仿射变换后“消失”。', '',
        '下列 5–95% 范围是 200 次重复划分的范围，来自同一有限 bank 的复用样本，**不是置信区间，也不是独立重复**；本轮不做显著/不显著分类，不给出 p 值。', '',
        '## 法向分量：source 对比度随位置/尺度去除的变化', '']
    for metric, zh, table_name in (('marginal_w1', '边缘 W1', '法向 source 对：边缘 W1 的 cross/within 比值'),
                                   ('joint_energy', '联合 energy V', '法向 source 对：联合 energy V 的 cross/within 比值')):
        lines += [f'### {table_name}', '',
                  '| Source 对 | raw | center_median | scale_median_iqr（主） | center_mean | scale_mean_sd |',
                  '|---|---:|---:|---:|---:|---:|']
        for pair in PAIR_NAMES:
            cells = [ratio_cell(normal.loc[(pair, branch)], metric) for branch in STAGES]
            lines.append(f"| {pair.replace('_train', '')} | " + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['读表：比值 > 1 表示该次 source 间距离超过两 source 各自的内部评估组距离均值；数值下降说明去除参考位置/尺度后差异相对内部波动变小，但**不**说明差异被完全移除，也不说明标准化“解释了”多少物理方差——raw 的 W1 单位与标准化后的 W1 单位不可互相换算。median/IQR 与 mean/SD 两种估计量可能给出不同答案，两者并列呈现、不择优挑选。', '',
              '## source 内评估基线（within）的解释', '',
              'within 距离是两组各 32 条评估样本在同一 source、同一标准化阶段下的距离，代表该 source 在该标定下的内部波动水平。cross/within 是相对这个内部基线的描述性对比，不是分类准确率，也不是机制归因；它没有控制叶片/采集批次等潜在分组。', '',
              '## 全分量简述（residual / tangent 平行对照）', '']
    for component in COMPONENTS:
        sub = pair_summary[pair_summary.component == component]
        raw = sub[sub.branch == 'raw']
        scaled = sub[sub.branch == 'scale_median_iqr']
        w1_raw = raw.marginal_w1_ratio_median
        w1_scaled = scaled.marginal_w1_ratio_median
        e_raw = raw.joint_energy_ratio_median
        e_scaled = scaled.joint_energy_ratio_median
        tag = '主分量' if component == 'normal' else '平行对照'
        lines.append(f'- {component}（{tag}）：W1 比值中位数跨 source 对范围 raw {w1_raw.min():.2f}–{w1_raw.max():.2f}，'
                     f'scale_median_iqr 后 {w1_scaled.min():.2f}–{w1_scaled.max():.2f}；'
                     f'energy 比值 raw {e_raw.min():.2f}–{e_raw.max():.2f}，'
                     f'scale_median_iqr 后 {e_scaled.min():.2f}–{e_scaled.max():.2f}。')
    lines += ['', '## 范围与边界', '',
        '- 本步骤只对 80 个 25 nm 有符号均值坐标断言；不对细尺度标准化后的完整 2000 维分布作任何断言。',
        '- 200 次重复复用同一 128 条 bank；source 行可能存在叶片/采集批次内依赖，当前无注册表可查。划分范围不可解释为跨批次或跨仪器推广。',
        '- 单个历史模型、无 seed 确认；结论不延伸到真实生化误差或预测风险。比值接近 1 不能用来宣布“共同形状”。', '',
        '## 图与复现', '',
        '![normal_stage_contrasts](normal_stage_contrasts.png)', '',
        '![normal_estimator_sensitivity](normal_estimator_sensitivity.png)', '',
        '运行入口：`python source_signed_residual_location_scale.py --output <新的结果目录>`。现有目录不覆盖；协议、输入与代码 SHA-256 见 protocol_manifest.json；评估组变换所需的全部参考参数见 reference_parameters.csv.gz。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(parent=PARENT, prior=PRIOR, repro=REPRO, output=DEFAULT_OUTPUT):
    parent, prior, repro, output = (Path(p).resolve() for p in (parent, prior, repro, output))
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    bank, inputs = load_bank(parent, prior)
    repro_hashes = validate_reproducibility_run(repro)
    output.mkdir(parents=True)
    scripts = {str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
               str(ROOT / 'source_distribution_step1.py'): sha256(ROOT / 'source_distribution_step1.py'),
               str(ROOT / 'source_signed_residual_distribution.py'): sha256(ROOT / 'source_signed_residual_distribution.py'),
               str(ROOT / 'source_signed_residual_reproducibility.py'): sha256(ROOT / 'source_signed_residual_reproducibility.py')}
    write_json(output / 'protocol_manifest.json', {
        'schema': 'historical_signed_residual_location_scale_v1', 'exploratory': True,
        'parent': str(parent), 'prior': str(prior), 'reproducibility_run': str(repro),
        'input_sha256': {**inputs, **repro_hashes}, 'code_sha256': scripts,
        'sources': list(SOURCES), 'n_per_source': 128, 'components': list(COMPONENTS),
        'sampling': {'repeats': SPLIT_REPEATS,
                     'key': f'SHA256({SPLIT_SALT}|{{repeat}}|{{sample_id}}), lexicographic sort of hex digests',
                     'roles': list(ROLES), 'order_invariant': True, 'role_persistence': 'identical across components and stages'},
        'n_ref': 32, 'n_eval': 32,
        'stages': {'raw': 'unchanged evaluation vectors',
                   'center_median': 'eval - reference band medians',
                   'scale_median_iqr': '(eval - reference medians) / reference IQR (primary standardization)',
                   'center_mean': 'eval - reference band means',
                   'scale_mean_sd': '(eval - means) / reference population SD, ddof=0 (prespecified estimator sensitivity)'},
        'reference_policy': 'parameters fit ONLY on the paired 32-sample reference block; each transformed group has its own independent reference; no clipping/winsorizing/offsets/norm normalization; bands with scale <= 1e-12 raise instead of flooring; reference parameter/diagnostic values are branch-independent (the branch column marks consuming stages)',
        'coordinates': '80 fixed signed nonoverlapping 25 nm means of the 400--2399 nm vectors; no PCA, no new geometry',
        'within_distance': 'compare(describe(eval_A), describe(eval_B)) per source: marginal_w1, median_mae, iqr_mae, tail_mae, Pearson/Spearman correlation RMSE and pattern_r; plus joint energy V between eval_A and eval_B',
        'cross_distance': 'mean of the four cross-source branch comparisons (A/A, A/B, B/A, B/B) between independently reference-transformed 32-sample evaluation groups',
        'ratio': 'cross / mean(two sources\' own A/B distances); near-zero denominators rejected; numerator and denominator both stored',
        'joint_energy': 'V = 2 mean(dXY) - mean(dXX) - mean(dYY), diagonals included, d = Euclidean/sqrt(80), no square root, direct cdist/pdist sums, no permutation nulls',
        'correlation_affine_invariance': 'per-band positive affine transforms cannot change Pearson/Spearman; correlation RMSE stored only for the raw stage and independently recomputed on early repeats as an audit',
        'intervals': '5--95 percentiles of 200 repeated partitions of one finite bank; NOT confidence intervals, NOT independent replications',
        'no_p_values': True, 'no_gaussian_fitting': True, 'no_permutation_tests': True,
        'mixed_domain_archive_opened': True, 'target_rows_analyzed': False, 'labels_read': False,
        'neural_training': False, 'limits': ['single historical model, no seed confirmation',
            'source rows may have acquisition/leaf dependence without a registry',
            'ratio near 1 does not prove common shape, especially at n_ref = n_eval = 32',
            'ratio changes can reflect contrast reduction and/or calibration-noise inflation',
            'raw and standardized W1 units are not comparable as physical variance explained',
            'descriptive finite-bank sensitivity, not causal attribution'],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'pandas': pd.__version__,
                     'matplotlib': matplotlib.__version__}
    })
    quarters = {s: [quarters_for_ids(bank[s]['ids'], r) for r in range(SPLIT_REPEATS)] for s in SOURCES}
    assignments = [(source, repeat, str(bank[source]['ids'][i]), role)
                   for source in SOURCES for repeat, q in enumerate(quarters[source])
                   for role in ROLES for i in q[role]]
    pd.DataFrame(assignments, columns=['source', 'repeat', 'sample_id', 'role']
                 ).to_csv(output / 'split_assignments.csv.gz', index=False)

    within_rows, between_rows = [], []
    ref_records, diag_records = [], []
    audit = {}
    printed = set()
    for component in COMPONENTS:
        xs = {s: blocks(bank[s][component]) for s in SOURCES}
        for repeat in range(SPLIT_REPEATS):
            raw_corr, y_cache, d_cache, within_store = {}, {}, {}, {}
            for source in SOURCES:
                x = xs[source]
                ix = quarters[source][repeat]
                ref_a, ev_a = x[ix['ref_A']], x[ix['eval_A']]
                ref_b, ev_b = x[ix['ref_B']], x[ix['eval_B']]
                params_a, params_b = fit_reference(ref_a), fit_reference(ref_b)
                for ref_name, params in (('ref_A', params_a), ('ref_B', params_b)):
                    for stat in ('median', 'iqr', 'mean', 'sd'):
                        for row in reference_diagnostics({stat: params[stat]}):
                            diag_records.append((source, component, repeat, ref_name, row['stat'],
                                                 row['min'], row['median'], row['max'], row['count_nonpos']))
                    for band in range(params_a['median'].shape[0]):
                        ref_records.append((source, component, repeat, ref_name, band, 400 + 25 * band,
                                            params['median'][band], params['iqr'][band],
                                            params['mean'][band], params['sd'][band]))
                y_cache[source], d_cache[source] = {}, {}
                for branch in STAGES:
                    key = (component, branch)
                    if key not in printed:
                        print(f'Location/scale sensitivity: {component} | {branch}', flush=True)
                        printed.add(key)
                    ya = transform_eval(ev_a, params_a, branch)
                    yb = transform_eval(ev_b, params_b, branch)
                    y_cache[source][branch] = (ya, yb)
                    if branch == 'raw':
                        da = describe(ya)
                        db = describe(yb)
                        raw_corr[source] = {'pearson_A': da['pearson'], 'spearman_A': da['spearman'],
                                            'pearson_B': db['pearson'], 'spearman_B': db['spearman']}
                    else:
                        da = eval_descriptor(ya, (raw_corr[source]['pearson_A'], raw_corr[source]['spearman_A']))
                        db = eval_descriptor(yb, (raw_corr[source]['pearson_B'], raw_corr[source]['spearman_B']))
                        if repeat < AUDIT_REPEATS:
                            for half, y in (('A', ya), ('B', yb)):
                                fresh_p, fresh_s = correlation_vectors(y)
                                base_p = raw_corr[source][f'pearson_{half}']
                                base_s = raw_corr[source][f'spearman_{half}']
                                slot = audit.setdefault((source, component, branch), [0.0, 0.0])
                                slot[0] = max(slot[0], float(np.max(np.abs(fresh_p - base_p))))
                                slot[1] = max(slot[1], float(np.max(np.abs(fresh_s - base_s))))
                    d_cache[source][branch] = (da, db)
                    value = compare(da, db)
                    energy = joint_energy_v(ya, yb)
                    row = {'source': source, 'component': component, 'branch': branch, 'repeat': repeat,
                           **{m: value[m] for m in DIST4}, 'joint_energy': energy}
                    if branch == 'raw':
                        row.update({m: value[m] for m in CORR_METRICS})
                        row['pearson_pattern_r'] = value['pearson_pattern_r']
                        row['spearman_pattern_r'] = value['spearman_pattern_r']
                    within_rows.append(row)
                    within_store[(source, branch)] = row
            for sa, sb in combinations(SOURCES, 2):
                for branch in STAGES:
                    comps = ((d_cache[sa][branch][0], d_cache[sb][branch][0]),
                             (d_cache[sa][branch][0], d_cache[sb][branch][1]),
                             (d_cache[sa][branch][1], d_cache[sb][branch][0]),
                             (d_cache[sa][branch][1], d_cache[sb][branch][1]))
                    pairs_y = ((y_cache[sa][branch][0], y_cache[sb][branch][0]),
                               (y_cache[sa][branch][0], y_cache[sb][branch][1]),
                               (y_cache[sa][branch][1], y_cache[sb][branch][0]),
                               (y_cache[sa][branch][1], y_cache[sb][branch][1]))
                    cross_vals = {m: [] for m in ENERGY_METRICS}
                    corr_cross = {m: [] for m in CORR_METRICS}
                    for (da, db), (xa, xb) in zip(comps, pairs_y):
                        c = compare(da, db)
                        for m in DIST4:
                            cross_vals[m].append(c[m])
                        for m in CORR_METRICS:
                            corr_cross[m].append(c[m])
                        cross_vals['joint_energy'].append(joint_energy_v(xa, xb))
                    row = {'pair': f'{sa} / {sb}', 'source_a': sa, 'source_b': sb,
                           'component': component, 'branch': branch, 'repeat': repeat}
                    for m in ENERGY_METRICS:
                        cross = float(np.mean(cross_vals[m]))
                        base = (within_store[(sa, branch)][m] + within_store[(sb, branch)][m]) / 2
                        if base <= SCALE_FLOOR:
                            raise ValueError('Near-zero within-source distance; refusing ratio.')
                        row[f'{m}_cross'], row[f'{m}_within'] = cross, base
                        row[f'{m}_ratio'] = cross / base
                    if branch == 'raw':
                        for m in CORR_METRICS:
                            cross = float(np.mean(corr_cross[m]))
                            base = (within_store[(sa, branch)][m] + within_store[(sb, branch)][m]) / 2
                            if base <= SCALE_FLOOR:
                                raise ValueError('Near-zero within-source correlation distance; refusing ratio.')
                            row[f'{m}_cross'], row[f'{m}_within'] = cross, base
                            row[f'{m}_ratio'] = cross / base
                    between_rows.append(row)
    print('Location/scale sensitivity: writing tables and figures', flush=True)
    within = pd.DataFrame(within_rows)
    between = pd.DataFrame(between_rows)
    within.to_csv(output / 'within_metrics.csv.gz', index=False)
    between.to_csv(output / 'between_metrics.csv.gz', index=False)
    ref_base = pd.DataFrame(ref_records, columns=['source', 'component', 'repeat', 'ref', 'band',
                                                  'start_nm', 'median', 'iqr', 'mean', 'sd'])
    ref_parts = []
    for branch in STAGES:
        part = ref_base.copy()
        part.insert(3, 'branch', branch)
        ref_parts.append(part)
    pd.concat(ref_parts, ignore_index=True).to_csv(output / 'reference_parameters.csv.gz', index=False)
    pd.DataFrame(diag_records, columns=['source', 'component', 'repeat', 'ref', 'stat',
                                        'min', 'median', 'max', 'count_nonpos']
                 ).to_csv(output / 'reference_diagnostics.csv', index=False)

    ratio_cols = [f'{m}_{suffix}' for m in ENERGY_METRICS for suffix in ('cross', 'within', 'ratio')]
    source_summary = summarize(within, ['source', 'component', 'branch'], [*DIST4, 'joint_energy'])
    raw_within = within[within.branch == 'raw']
    corr_src = summarize(raw_within, ['source', 'component'],
                         [*CORR_METRICS, 'pearson_pattern_r', 'spearman_pattern_r'])
    corr_src['branch'] = 'raw'
    source_summary = source_summary.merge(corr_src, on=['source', 'component', 'branch'], how='left')
    pair_summary = summarize(between, ['pair', 'component', 'branch'], ratio_cols)
    raw_between = between[between.branch == 'raw']
    corr_pair = summarize(raw_between, ['pair', 'component'],
                          [f'{m}_{suffix}' for m in CORR_METRICS for suffix in ('cross', 'within', 'ratio')])
    corr_pair['branch'] = 'raw'
    pair_summary = pair_summary.merge(corr_pair, on=['pair', 'component', 'branch'], how='left')
    source_summary.to_csv(output / 'source_summary.csv', index=False)
    pair_summary.to_csv(output / 'pair_summary.csv', index=False)
    audit_rows = [{'source': source, 'component': component, 'stage': branch,
                   'max_abs_pearson_diff': 0.0 if branch == 'raw' else audit[(source, component, branch)][0],
                   'max_abs_spearman_diff': 0.0 if branch == 'raw' else audit[(source, component, branch)][1]}
                  for source in SOURCES for component in COMPONENTS for branch in STAGES]
    pd.DataFrame(audit_rows).to_csv(output / 'affine_correlation_audit.csv', index=False)
    make_figures(output, pair_summary)
    report(output, pair_summary)
    for path, expected in {**inputs, **repro_hashes, **scripts}.items():
        if sha256(path) != expected:
            raise RuntimeError(f'Input or code changed during analysis: {path}')
    write_json(output / 'status.json', {
        'status': 'complete_historical_residual_location_scale', 'source_count': len(SOURCES),
        'samples': sum(len(bank[s]['ids']) for s in SOURCES),
        'split_repeats': SPLIT_REPEATS, 'n_ref': 32, 'n_eval': 32,
        'target_rows_analyzed': False, 'labels_read': False, 'neural_training': False,
        'input_hashes_reverified': True, 'code_hashes_reverified': True,
        'artifact_sha256': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                            if p.is_file() and p.name != 'status.json'}
    })
    cols = ['pair', 'branch', 'marginal_w1_ratio_median', 'joint_energy_ratio_median']
    print(pair_summary[(pair_summary.component == 'normal')][cols].to_string(index=False), flush=True)
    print(f'Complete: {output / "report.md"}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--repro', type=Path, default=REPRO)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.parent, args.prior, args.repro, args.output)
