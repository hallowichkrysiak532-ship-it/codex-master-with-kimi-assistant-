"""Repeated source-internal splits of the frozen historical signed residual bank.

All resampling units are complete sample vectors. Repeated partitions describe
this finite bank; they are neither independent replications nor confidence sets.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits

from source_distribution_step1 import ROOT, sha256, write_json, bh_adjust, keyed_rng
from source_signed_residual_distribution import PARENT, SOURCES, COMPONENTS, blocks
import matplotlib.pyplot as plt

PRIOR = ROOT / 'output_source_signed_residual_distribution/historical_v1'
OUTPUT = ROOT / 'output_source_signed_residual_distribution/reproducibility_v1'
SPLITS = 200
PERMUTATIONS = 9999
DISTANCES = ('marginal_w1', 'median_mae', 'iqr_mae', 'tail_mae',
             'pearson_corr_rmse', 'spearman_corr_rmse')


def half_indices(ids, repeat):
    """A balanced, order-invariant partition determined only by IDs and repeat."""
    ids = list(map(str, ids))
    if len(ids) % 2 or len(set(ids)) != len(ids):
        raise ValueError('Unique IDs and an even sample count are required.')
    hashes = [hashlib.sha256(f'20260905|signed_split_v1|{repeat}|{s}'.encode()).digest()
              for s in ids]
    order = np.array(sorted(range(len(ids)), key=lambda i: hashes[i]))
    return order[:len(ids)//2], order[len(ids)//2:]


def describe(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(x.std(axis=0) <= 1e-12):
        raise ValueError('Expected finite, nondegenerate sample-by-coordinate matrix.')
    q = np.quantile(x, [.05, .25, .5, .75, .95], axis=0)
    upper = np.triu_indices(x.shape[1], 1)
    return {'sorted': np.sort(x, axis=0), 'median': q[2], 'iqr': q[3]-q[1],
            'tails': q[[0, 4]], 'quantiles': q,
            'pearson': np.corrcoef(x, rowvar=False)[upper],
            'spearman': np.corrcoef(stats.rankdata(x, axis=0), rowvar=False)[upper]}


def compare(a, b):
    if a['sorted'].shape != b['sorted'].shape:
        raise ValueError('The sorted-sample W1 formula requires equal sample sizes.')
    result = {'marginal_w1': float(np.mean(np.abs(a['sorted']-b['sorted']))),
              'median_mae': float(np.mean(np.abs(a['median']-b['median']))),
              'iqr_mae': float(np.mean(np.abs(a['iqr']-b['iqr']))),
              'tail_mae': float(np.mean(np.abs(a['tails']-b['tails'])))}
    for name in ('pearson', 'spearman'):
        result[name+'_corr_rmse'] = float(np.sqrt(np.mean((a[name]-b[name])**2)))
        result[name+'_pattern_r'] = float(np.corrcoef(a[name], b[name])[0, 1])
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Undefined comparison statistic.')
    return result


def energy_vstat(distance, signs):
    """2 mean(dXY)-mean(dXX)-mean(dYY), diagonals included; equal group sizes.

    This is the squared empirical energy distance, without the square root.
    Input distances are Euclidean/sqrt(number of coordinates).
    """
    signs = np.atleast_2d(np.asarray(signs, dtype=float))
    if distance.shape != (signs.shape[1], signs.shape[1]):
        raise ValueError('Distance/assignment dimensions do not match.')
    if not np.all(np.abs(signs) == 1) or not np.all(signs.sum(axis=1) == 0):
        raise ValueError('Expected balanced +/-1 assignments.')
    n = signs.shape[1]//2
    return -np.sum((signs @ distance)*signs, axis=1)/(n*n)


def permutation_signs(n, repeats, key):
    rng = keyed_rng('signed_energy_v1', key)
    result = -np.ones((repeats, 2*n), dtype=np.int8)
    for row in result:
        row[rng.permutation(2*n)[:n]] = 1
    return result


def energy_test(x, y, signs):
    if x.shape != y.shape:
        raise ValueError('Expected equal-sized groups in common coordinates.')
    d = squareform(pdist(np.concatenate([x, y])))/np.sqrt(x.shape[1])
    observed = float(energy_vstat(d, np.r_[np.ones(len(x)), -np.ones(len(y))])[0])
    null = np.concatenate([energy_vstat(d, batch) for batch in np.array_split(signs, 20)])
    tolerance = max(1e-14, abs(observed)*1e-14)
    p = float((1+np.count_nonzero(null >= observed-tolerance))/(len(null)+1))
    return {'energy_vstat': observed, 'p': p,
            'null_median': float(np.median(null)), 'null_q95': float(np.quantile(null, .95))}, null


def summarize(frame, keys, metrics):
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        for metric in metrics:
            value = group[metric].to_numpy()
            for suffix, q in (('q05', .05), ('median', .5), ('q95', .95)):
                row[metric+'_'+suffix] = float(np.quantile(value, q))
        rows.append(row)
    return pd.DataFrame(rows)


def load_bank(parent, prior):
    status = json.loads((parent/'exp20a_status.json').read_text(encoding='utf-8'))
    prior_status = json.loads((prior/'status.json').read_text(encoding='utf-8'))
    inputs = {}
    for name in ('protocol_manifest.json', 'empirical_normal_bank.npz', 'source_wavelength_scale.npy'):
        p = parent/name
        if sha256(p) != status['artifact_hashes'][name]:
            raise ValueError(f'Historical parent hash mismatch: {p}')
        inputs[str(p)] = sha256(p)
    for name, expected in prior_status['artifacts'].items():
        p = prior/name
        if sha256(p) != expected:
            raise ValueError(f'Prior pilot hash mismatch: {p}')
        inputs[str(p)] = expected
    for p in (parent/'exp20a_status.json', prior/'status.json'):
        inputs[str(p)] = sha256(p)
    prior_manifest = json.loads((prior/'protocol_manifest.json').read_text(encoding='utf-8'))
    for name, expected in prior_manifest['input_sha256'].items():
        if sha256(parent/name) != expected:
            raise ValueError('Prior pilot and current bank do not match.')
    bank = {}
    with np.load(parent/'empirical_normal_bank.npz', allow_pickle=False) as z:
        domains = z['dataset'].astype(str)
        for source in SOURCES:
            mask = domains == source
            ids = np.array([f'{source}:{i}' for i in z['sample_index'][mask]])
            if len(ids) != 128 or len(set(ids)) != 128:
                raise ValueError('Unexpected source registry.')
            order = np.argsort(ids)
            residual = z['residual_whitened'][mask][order].astype(float)
            normal = z['normal_whitened'][mask][order].astype(float)
            tangent = residual-normal
            cosine = np.abs(np.sum(tangent*normal, axis=1))/(np.linalg.norm(tangent, axis=1)*np.linalg.norm(normal, axis=1))
            if not np.isfinite(cosine).all() or cosine.max() > 1e-4:
                raise ValueError('Orthogonality audit failed.')
            bank[source] = {'residual': residual, 'tangent': tangent, 'normal': normal, 'ids': ids[order]}
    return bank, inputs


def figures(output, bank, within, between, source_summary, pair_summary):
    colors = {'residual': '#555555', 'tangent': '#D68910', 'normal': '#2874A6'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric, title in zip(axes,
            ('marginal_w1_over_iqr', 'pearson_pattern_r', 'spearman_pattern_r'),
            ('Marginal W1 / mean source IQR (lower better)',
             'Pearson matrix pattern agreement', 'Spearman matrix pattern agreement')):
        for k, component in enumerate(COMPONENTS):
            sub = source_summary[source_summary.component == component].set_index('source').loc[list(SOURCES)]
            mid = sub[metric+'_median'].to_numpy()
            lo, hi = sub[metric+'_q05'].to_numpy(), sub[metric+'_q95'].to_numpy()
            ax.errorbar(np.arange(4)+(k-1)*.17, mid, yerr=[mid-lo, hi-mid], fmt='o',
                        capsize=3, label=component, color=colors[component])
        ax.set_xticks(range(4), [s.replace('_train', '') for s in SOURCES])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('200 balanced 64/64 splits | median and 5--95% split range, NOT confidence intervals')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output/'within_source_reproducibility.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sub = pair_summary[pair_summary.component == 'normal']
    for ax, metric, title in zip(axes, ('marginal_w1', 'pearson_corr_rmse', 'spearman_corr_rmse'),
                                ('Marginal distribution', 'Pearson correlation structure', 'Spearman correlation structure')):
        column = metric+'_between_over_within'
        mid = sub[column+'_median'].to_numpy()
        lo, hi = sub[column+'_q05'].to_numpy(), sub[column+'_q95'].to_numpy()
        ax.errorbar(mid, np.arange(6), xerr=[mid-lo, hi-mid], fmt='o', capsize=3, color='#2874A6')
        ax.axvline(1, color='#777777', ls='--')
        ax.set_yticks(range(6), [s.replace('_train', '') for s in sub.pair])
        ax.set(xlabel='Between-source / within-source distance', title=title)
        ax.grid(alpha=.2)
    fig.suptitle('Normal residual: matched 64-sample comparisons | 5--95% split range, NOT confidence intervals')
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output/'normal_between_within.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    wavelengths = np.arange(400, 2400, 25)+12
    for ax, source in zip(axes.ravel(), SOURCES):
        a, b = half_indices(bank[source]['ids'], 0)
        x = blocks(bank[source]['normal'])
        for ix, color, label in ((a, '#2874A6', 'Half A'), (b, '#D68910', 'Half B')):
            q = np.quantile(x[ix], [.05, .5, .95], axis=0)
            ax.plot(wavelengths, q[1], color=color, label=label+' median')
            ax.plot(wavelengths, q[0], color=color, ls=':', lw=1)
            ax.plot(wavelengths, q[2], color=color, ls=':', lw=1)
        ax.axhline(0, color='#777777', lw=.6)
        ax.set(title=source, xlabel='Wavelength (nm)', ylabel='Signed source-scaled normal residual')
        ax.legend(fontsize=8)
    fig.suptitle('Prespecified split 0 | solid: median; dotted: pointwise 5th/95th sample quantiles\nDisplay split is not selected for agreement; these are not confidence bands')
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output/'normal_example_split.png', dpi=170)
    plt.close(fig)


def report(output, source_summary, pair_summary, energy):
    lines = ['# 有符号残差分布：同 source 重复性与 source 间差异', '',
        '基座为 Exp20A 单个历史 checkpoint，四个 source 各 128 条完整残差曲线，400–2399 nm。沿用上一轮共同 source scale；本轮没有新训练，没有读取性状标签。混合域归档为筛选 source 而打开，其余域行不进入统计。', '',
        '## 本轮检验对象', '',
        '- 每 source 200 次不重叠 64/64 划分；同一次划分在 residual/tangent/normal 间共用。划分按样本 ID 哈希固定。不同重复会复用样本，不是 200 个独立数据集。',
        '- 80 个固定 25 nm 有符号均值坐标。边缘 W1 为逐坐标经验一阶 Wasserstein 距离的平均；另分别报告中位数、IQR、5%/95% 尾部分位数差异。波段不是独立样本。',
        '- Pearson 和 Spearman 波段相关矩阵均在各半样本内独立计算。矩阵 pattern r 为两个矩阵上三角元素之间的相关，只是结构相似度，不为这些相关边计算独立观测 p 值。另报告矩阵元素 RMSE，避免相似度掩盖相关强度差异。',
        '- source 间比较同样使用 64 对 64：分别比较两 source 的 A/A、B/B，取两个距离均值；分母为同次两 source 内部 A/B 距离均值。比值大于 1 表示 source 间差异超过该次内部波动，不是分类准确率或显著性门。',
        '- 全部 128 对 128 曲线另作联合分布 energy 置换检验，9999 次整条样本重分组。统计量为 2 mean(dXY)−mean(dXX)−mean(dYY)，保留自距离，d 为 Euclidean/sqrt(维数)，不再开平方。80 维为主、2000 维为预定压缩敏感性检查，六对 source × 三种分量 × 两种坐标共 36 项统一 BH。两种坐标使用同一组置换。', '',
        '随机拆分仅衡量当前有限 bank 的内部重复性。下列 5–95% 范围是拆分范围，不是置信区间。置换检验假设行可交换；当前缺少叶片/采集批次分组，不能将其解释为跨批次或跨仪器推广。', '',
        '## 法向分布的 source 内重复性', '',
        'W1/IQR 的分母为该 source 全样本 80 个波段 IQR 的均值，仅用于描述误差大小。该标度未进入 source 间主距离或置换检验。', '',
        '| Source | W1 / 平均 IQR，中位数 [5%,95%] | 中位数差 / IQR | 宽度差 / IQR | 尾部分位差 / IQR | Pearson 结构 r | Spearman 结构 r |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in source_summary[source_summary.component == 'normal'].itertuples():
        lines.append(f'| {r.source} | {r.marginal_w1_over_iqr_median:.3f} [{r.marginal_w1_over_iqr_q05:.3f}, {r.marginal_w1_over_iqr_q95:.3f}] | {r.median_mae_over_iqr_median:.3f} | {r.iqr_mae_over_iqr_median:.3f} | {r.tail_mae_over_iqr_median:.3f} | {r.pearson_pattern_r_median:.3f} | {r.spearman_pattern_r_median:.3f} |')
    lines += ['', '## 法向 source 间差异相对于内部波动', '',
        '| Source pair | 边缘 W1 比值，中位数 [5%,95%] | Pearson 矩阵差比值 | Spearman 矩阵差比值 |',
        '|---|---:|---:|---:|']
    for r in pair_summary[pair_summary.component == 'normal'].itertuples():
        lines.append(f'| {r.pair} | {r.marginal_w1_between_over_within_median:.2f} [{r.marginal_w1_between_over_within_q05:.2f}, {r.marginal_w1_between_over_within_q95:.2f}] | {r.pearson_corr_rmse_between_over_within_median:.2f} | {r.spearman_corr_rmse_between_over_within_median:.2f} |')
    lines += ['', '## 全分量与压缩敏感性', '']
    for component in COMPONENTS:
        w = source_summary[source_summary.component == component]
        e = energy[energy.component == component]
        lines.append(f'- {component}：W1/IQR 中位数跨 source 范围 {w.marginal_w1_over_iqr_median.min():.3f}–{w.marginal_w1_over_iqr_median.max():.3f}；Pearson 结构 r 中位数 {w.pearson_pattern_r_median.min():.3f}–{w.pearson_pattern_r_median.max():.3f}，Spearman 为 {w.spearman_pattern_r_median.min():.3f}–{w.spearman_pattern_r_median.max():.3f}。80 维置换 BH q<.05：{int(((e.representation=="blocks80") & (e.q_bh<.05)).sum())}/6 对；完整 2000 维：{int(((e.representation=="full2000") & (e.q_bh<.05)).sum())}/6 对。')
    lines += ['', '拒绝“两个 source 的联合分布相同”不说明差异由尾部或相关性独立贡献；均值或尺度差异也可能产生结果。相关矩阵比值为单独的描述性证据，没有对边缘匹配后的依赖结构作机制归因。', '',
        '## 可保留与尚未检验的内容', '',
        '本轮用内部误差基线量化 source 分布描述的可重复程度，并判断 source 差异相对于内部波动的大小。中位数、宽度、尾部和相关结构分别报告，不设事后“全部稳定”的合格阈值。', '',
        '仍是单个旧模型和同一有限样本 bank；未检验跨模型 seed、独立采集批次或状态匹配后的分布不变性。25 nm 均值仅保留前轮报告的约 82–94% 法向中心化变动；完整向量 energy 检查只能判断 source 区分是否依赖该压缩，不能恢复丢失的细节。结论不延伸到真实生化误差或预测风险。', '',
        '## 统计定义参考', '',
        '- [SciPy Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)：一维经验分布距离；等样本数时使用排序后绝对差均值。',
        '- [SciPy energy distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html)：本轮保留平方形式，向量使用欧氏距离。',
        '- [SciPy permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)：整行重分组及 (超越次数+1)/(置换次数+1)。', '']
    for name in ('within_source_reproducibility', 'normal_between_within', 'normal_example_split'):
        lines += [f'![{name}]({name}.png)', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(parent=PARENT, prior=PRIOR, output=OUTPUT):
    parent, prior, output = (Path(p).resolve() for p in (parent, prior, output))
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    bank, inputs = load_bank(parent, prior)
    output.mkdir(parents=True)
    scripts = {str(p): sha256(p) for p in (Path(__file__), ROOT/'source_distribution_step1.py',
                                          ROOT/'source_signed_residual_distribution.py')}
    write_json(output/'protocol_manifest.json', {
        'schema': 'historical_signed_residual_reproducibility_v1', 'exploratory': True,
        'input_sha256': inputs, 'code_sha256': scripts,
        'sources': list(SOURCES), 'n_per_source': 128, 'components': list(COMPONENTS),
        'geometry': 'inherited Exp20A single checkpoint, PP8 clip [.03,.97], step .02, cutoff .001',
        'split_repeats': SPLITS, 'split_size': 64,
        'splits': 'SHA256(20260905|signed_split_v1|repeat|source:sample_index), identical across components',
        'coordinates': '80 fixed signed 25 nm means, common historical wavelength scale; no source-wise scaling in primary comparisons',
        'metrics': list(DISTANCES), 'correlation_similarity': 'Pearson correlation of off-diagonal matrix elements; descriptive only',
        'within_normalization': 'mean marginal full-source IQR, descriptive only',
        'between_ratio': 'mean of cross-source A/A and B/B distances divided by mean of two source-internal A/B distances',
        'split_intervals': '5--95 percentiles of 200 repeated partitions, not confidence intervals or independent replicates',
        'energy_permutations': PERMUTATIONS, 'energy_statistic': 'biased empirical squared energy, Euclidean/sqrt(d), whole-row permutations of full groups 128/128',
        'energy_representations': ['blocks80', 'full2000'], 'energy_bh_family': 'all 36 tests (6 pairs x 3 components x 2 representations)',
        'energy_null_scope': 'equal joint distributions under row exchangeability, not correlation-only equality',
        'display_split': 0, 'no_posthoc_threshold': True,
        'mixed_domain_archive_opened': True, 'target_rows_analyzed': False, 'labels_read': False,
        'neural_training': False, 'limits': ['one historical model', 'no acquisition/leaf group registry',
            'finite bank internal reproducibility only', 'no state matching', 'correlation is descriptive, not causal'],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__}
    })
    splits = {s: [half_indices(bank[s]['ids'], r) for r in range(SPLITS)] for s in SOURCES}
    assignments = []
    for source in SOURCES:
        for repeat, halves in enumerate(splits[source]):
            for half, ix in zip(('A', 'B'), halves):
                assignments.extend({'source': source, 'repeat': repeat, 'sample_id': str(bank[source]['ids'][i]), 'half': half} for i in ix)
    pd.DataFrame(assignments).to_csv(output/'split_assignments.csv.gz', index=False)
    within, between = [], []
    for component in COMPONENTS:
        print(f'Repeated split descriptors: {component}', flush=True)
        cache, internal = {}, {}
        for source in SOURCES:
            x = blocks(bank[source][component])
            scale = float(np.mean(np.subtract(*np.quantile(x, [.75, .25], axis=0))))
            if scale <= 1e-12:
                raise ValueError('Degenerate source IQR scale.')
            cache[source], internal[source] = [], []
            for repeat, (ia, ib) in enumerate(splits[source]):
                a, b = describe(x[ia]), describe(x[ib])
                cache[source].append((a, b))
                value = compare(a, b)
                internal[source].append(value)
                within.append({'source': source, 'component': component, 'repeat': repeat,
                    'mean_full_source_iqr': scale, **value,
                    **{m+'_over_iqr': value[m]/scale for m in DISTANCES[:4]}})
        for sa, sb in combinations(SOURCES, 2):
            for repeat in range(SPLITS):
                aa, ab = cache[sa][repeat]
                ba, bb = cache[sb][repeat]
                da, db = compare(aa, ba), compare(ab, bb)
                row = {'pair': f'{sa} / {sb}', 'source_a': sa, 'source_b': sb,
                       'component': component, 'repeat': repeat}
                for metric in DISTANCES:
                    cross = (da[metric]+db[metric])/2
                    base = (internal[sa][repeat][metric]+internal[sb][repeat][metric])/2
                    if base <= 1e-12:
                        raise ValueError('Zero within-source distance needs separate treatment.')
                    row[metric+'_between'] = cross
                    row[metric+'_within'] = base
                    row[metric+'_between_over_within'] = cross/base
                between.append(row)
    within, between = pd.DataFrame(within), pd.DataFrame(between)
    within.to_csv(output/'within_split_metrics.csv.gz', index=False)
    between.to_csv(output/'between_split_metrics.csv.gz', index=False)
    source_summary = summarize(within, ['source', 'component'],
        [*DISTANCES, 'pearson_pattern_r', 'spearman_pattern_r', *[m+'_over_iqr' for m in DISTANCES[:4]]])
    pair_summary = summarize(between, ['pair', 'component'],
        [m+suffix for m in DISTANCES for suffix in ('_between', '_within', '_between_over_within')])
    source_summary.to_csv(output/'source_summary.csv', index=False)
    pair_summary.to_csv(output/'pair_summary.csv', index=False)
    tests, nulls = [], {}
    for sa, sb in combinations(SOURCES, 2):
        print(f'Joint distribution permutations: {sa} / {sb}', flush=True)
        signs = permutation_signs(128, PERMUTATIONS, f'{sa}|{sb}')
        for component in COMPONENTS:
            for representation in ('blocks80', 'full2000'):
                x, y = bank[sa][component], bank[sb][component]
                if representation == 'blocks80':
                    x, y = blocks(x), blocks(y)
                value, null = energy_test(x, y, signs)
                tests.append({'source_a': sa, 'source_b': sb, 'component': component,
                              'representation': representation, **value})
                nulls[f'{sa}__{sb}__{component}__{representation}'] = null
    energy = pd.DataFrame(tests)
    energy['q_bh'] = bh_adjust(energy.p.to_numpy())
    energy.to_csv(output/'joint_energy_permutation.csv', index=False)
    np.savez_compressed(output/'joint_energy_nulls.npz', **nulls)
    figures(output, bank, within, between, source_summary, pair_summary)
    report(output, source_summary, pair_summary, energy)
    for name, expected in {**inputs, **scripts}.items():
        if sha256(name) != expected:
            raise RuntimeError(f'Input or code changed during analysis: {name}')
    write_json(output/'status.json', {
        'status': 'complete_historical_signed_residual_reproducibility', 'source_count': 4, 'samples': 512,
        'split_repeats': SPLITS, 'permutation_repeats': PERMUTATIONS,
        'target_rows_analyzed': False, 'labels_read': False, 'input_hashes_reverified': True,
        'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    })
    print(source_summary[source_summary.component == 'normal'][['source', 'marginal_w1_over_iqr_median',
          'pearson_pattern_r_median', 'spearman_pattern_r_median']].to_string(index=False), flush=True)
    print(pair_summary[pair_summary.component == 'normal'][['pair', 'marginal_w1_between_over_within_median',
          'pearson_corr_rmse_between_over_within_median', 'spearman_corr_rmse_between_over_within_median']].to_string(index=False), flush=True)
    print(energy.groupby(['component', 'representation']).q_bh.agg(['min', 'max']).to_string(), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.parent, args.prior, args.output)
