"""Local exploratory distributions of frozen Exp31E source residual magnitudes.

Reads an explicit source-feature column allowlist, never labels or target files.
Each encoder seed is a repeated measurement of the same source sample registry.
No neural inference, fitting of geometry, or change to parent artifacts occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import special, stats

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "output_exp31e/exp31e_full_reproducibility/exp31e_current_state_jacobian_v1"
DEFAULT_OUTPUT = ROOT / "output_source_distribution_step1/v1"
SEEDS = (11, 22, 33, 44, 55)
DOMAINS = ("CABO", "Lopex", "Angers", "IFGG", "UW")
VARIABLES = {
    "tangent_rms": "current_tangent_rms",
    "normal_rms": "current_normal_rms",
    "tangent_fraction": "current_tangent_fraction",
}
FAMILIES = {
    "tangent_rms": ("lognormal", "gamma", "weibull"),
    "normal_rms": ("lognormal", "gamma", "weibull"),
    "tangent_fraction": ("beta", "logit_normal"),
}
READ_COLUMNS = ["seed", "domain", "sample_id", *VARIABLES.values(),
                "jacobian_rank", "coordinate_reliable"]
FAMILY_LABELS = {"lognormal": "Lognormal", "gamma": "Gamma", "weibull": "Weibull",
                 "beta": "Beta", "logit_normal": "Logit-normal"}
FAMILY_ZH = {"lognormal": "对数正态", "gamma": "Gamma", "weibull": "Weibull",
             "beta": "Beta", "logit_normal": "logit 正态"}
VARIABLE_ZH = {"tangent_rms": "切向 RMS", "normal_rms": "法向 RMS", "tangent_fraction": "切向能量占比"}
COLORS = ("#2874A6", "#D68910", "#148F77")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keyed_rng(*items):
    key = "|".join(map(str, (20260905, *items)))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fit_family(x, family):
    x = np.asarray(x, dtype=float)
    if len(x) < 8 or not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
        raise ValueError("A continuous fit requires at least eight finite, varying observations.")
    if family in ("beta", "logit_normal"):
        if np.any((x <= 0) | (x >= 1)):
            raise ValueError("Boundary mass requires a separate model; do not silently clip it.")
    elif np.any(x <= 0):
        raise ValueError("Positive family received zero/negative mass; do not silently shift it.")
    if family == "lognormal":
        z = np.log(x)
        return (float(z.std(ddof=0)), 0.0, float(np.exp(z.mean())))
    if family == "logit_normal":
        z = special.logit(x)
        return (float(z.mean()), float(z.std(ddof=0)))
    dist = {"gamma": stats.gamma, "weibull": stats.weibull_min, "beta": stats.beta}[family]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        params = dist.fit(x, floc=0, **({"fscale": 1} if family == "beta" else {}))
    return tuple(map(float, params))


def family_dist(family):
    return {"lognormal": stats.lognorm, "gamma": stats.gamma,
            "weibull": stats.weibull_min, "beta": stats.beta}[family]


def cdf(x, family, params):
    if family == "logit_normal":
        return stats.norm.cdf(special.logit(x), loc=params[0], scale=params[1])
    return family_dist(family).cdf(x, *params)


def logpdf(x, family, params):
    if family == "logit_normal":
        # Evaluate every candidate in the SAME original measure, including Jacobian.
        return stats.norm.logpdf(special.logit(x), loc=params[0], scale=params[1]) - np.log(x) - np.log1p(-x)
    return family_dist(family).logpdf(x, *params)


def ppf(q, family, params):
    if family == "logit_normal":
        return special.expit(stats.norm.ppf(q, loc=params[0], scale=params[1]))
    return family_dist(family).ppf(q, *params)


def draw(n, family, params, rng):
    if family == "logit_normal":
        return special.expit(rng.normal(params[0], params[1], n))
    return family_dist(family).rvs(*params, size=n, random_state=rng)


def ks_distance(x, family, params):
    u = np.sort(cdf(x, family, params))
    n = len(u)
    return float(max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n)))


def bootstrap_gof(x, family, params, repeats, rng):
    """Parametric bootstrap: re-estimate parameters in EVERY null replicate."""
    observed = ks_distance(x, family, params)
    values = []
    for _ in range(repeats):
        simulated = draw(len(x), family, params, rng)
        fitted = fit_family(simulated, family)
        values.append(ks_distance(simulated, family, fitted))
    p = (1 + int(np.sum(np.asarray(values) >= observed))) / (repeats + 1)
    return observed, p


def folds_for_ids(ids, repeat):
    # Fold memberships depend on sample IDs, NOT values, family, or encoder seed.
    keys = [hashlib.sha256(f"20260905|{repeat}|{s}".encode()).hexdigest() for s in ids]
    order = np.argsort(keys)
    folds = np.empty(len(ids), dtype=int)
    folds[order] = np.arange(len(ids)) % 5
    return folds


def cross_validate(x, ids, family, repeats=3):
    scores, pit = [], []
    for repeat in range(repeats):
        folds = folds_for_ids(ids, repeat)
        ll, u = np.empty(len(x)), np.empty(len(x))
        for fold in range(5):
            train, test = folds != fold, folds == fold
            params = fit_family(x[train], family)
            ll[test] = logpdf(x[test], family, params)
            u[test] = cdf(x[test], family, params)
        scores.append(ll)
        pit.append(u)
    ll = np.mean(scores, axis=0)
    u = np.asarray(pit)
    return {"cv_mean_logpdf": float(ll.mean()),
            "cv_mean_logpdf_sample_se": float(ll.std(ddof=1) / math.sqrt(len(ll))),
            "cv_central90_coverage": float(np.mean((u >= .05) & (u <= .95))),
            "cv_central50_coverage": float(np.mean((u >= .25) & (u <= .75)))}


def bh_adjust(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    q_sorted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(q_sorted, 1)
    return out


def load_sources(parent):
    manifest = json.loads((parent / "protocol_manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["config"]["seeds"]) != SEEDS:
        raise ValueError("Unexpected parent seed registry.")
    if set(manifest["source_validation"]) != set(DOMAINS):
        raise ValueError("Unexpected source registry.")
    inputs, pieces = {}, []
    for seed in SEEDS:
        folder = parent / "seed_cache" / f"seed_{seed}"
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete_seed":
            raise ValueError("Incomplete parent cache.")
        for name in ("source_validation_features.csv.gz", "reference.npz", "checkpoint.json"):
            path = folder / name
            actual = sha256(path)
            relative = path.relative_to(parent).as_posix()
            if actual != status["artifact_sha256"][relative]:
                raise ValueError(f"Parent hash mismatch: {path}")
            inputs[relative] = actual
        # Label/error columns exist in this CSV but are deliberately not parsed.
        table = pd.read_csv(folder / "source_validation_features.csv.gz", usecols=READ_COLUMNS,
                            dtype={"sample_id": str})
        if table.duplicated(["domain", "sample_id"]).any() or set(table.seed) != {seed}:
            raise ValueError("Duplicate sample or wrong seed.")
        if set(table.domain) != set(DOMAINS):
            raise ValueError("Unexpected domain in source-only table.")
        values = table[list(VARIABLES.values())].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite geometry: audit before fitting.")
        ratio = values[:, 0] ** 2 / (values[:, 0] ** 2 + values[:, 1] ** 2)
        if not np.allclose(ratio, values[:, 2], atol=2e-6, rtol=2e-6):
            raise ValueError("Tangent energy accounting failed.")
        pieces.append(table)
    table = pd.concat(pieces, ignore_index=True)
    registries = [set(zip(p.domain, p.sample_id)) for p in pieces]
    if any(r != registries[0] for r in registries[1:]):
        raise ValueError("Seed sample registries differ; do not pool unmatched samples.")
    inputs["protocol_manifest.json"] = sha256(parent / "protocol_manifest.json")
    return table, inputs


def dependence(table, repeats, bootstrap_repeats):
    t = table.current_tangent_rms.to_numpy()
    n = table.current_normal_rms.to_numpy()
    count = len(t)
    seed, domain = int(table.seed.iloc[0]), str(table.domain.iloc[0])
    rng = keyed_rng("dependence", seed, domain)
    tr, nr = stats.rankdata(t), stats.rankdata(n)
    tr, nr = (tr-tr.mean())/np.linalg.norm(tr-tr.mean()), (nr-nr.mean())/np.linalg.norm(nr-nr.mean())
    rho = float(tr @ nr)
    permuted = np.array([tr @ rng.permutation(nr) for _ in range(repeats)])
    p = (1 + np.sum(np.abs(permuted) >= abs(rho))) / (repeats + 1)
    bootstrap = []
    for _ in range(bootstrap_repeats):
        ix = rng.integers(count, size=count)
        bootstrap.append(float(stats.spearmanr(t[ix], n[ix]).statistic))
    # Joint lognormal is only a fixed descriptive candidate, not presumed true.
    z = np.log(np.column_stack([t, n]))
    joint, independent = [], []
    for repeat in range(3):
        folds = folds_for_ids(table.sample_id.to_numpy(), repeat)
        for fold in range(5):
            train, test = z[folds != fold], z[folds == fold]
            mean = train.mean(axis=0)
            cov = np.cov(train, rowvar=False, ddof=0)
            cov += np.eye(2) * 1e-8 * np.trace(cov) / 2
            joint.extend(stats.multivariate_normal.logpdf(test, mean=mean, cov=cov) - test.sum(axis=1))
            independent.extend(stats.multivariate_normal.logpdf(test, mean=mean, cov=np.diag(np.diag(cov))) - test.sum(axis=1))
    # Correlation is only monotone dependence; failure to reject is not independence.
    return {"seed": seed, "domain": domain, "n": count, "spearman": rho,
            "rho_ci_low": float(np.quantile(bootstrap, .025)),
            "rho_ci_high": float(np.quantile(bootstrap, .975)), "permutation_p": float(p),
            "joint_lognormal_cv_logpdf_gain_over_independent": float(np.mean(joint)-np.mean(independent))}


def split_stability(x, repeats, rng):
    scale = np.subtract(*np.quantile(x, [.75, .25]))
    values = []
    for _ in range(repeats):
        order = rng.permutation(len(x))
        a, b = x[order[:len(x)//2]], x[order[len(x)//2:]]
        values.append(stats.wasserstein_distance(a, b) / scale)
    return {"split_w1_over_iqr_median": float(np.median(values)),
            "split_w1_over_iqr_q95": float(np.quantile(values, .95))}


def make_figures(data, fits, dep, output):
    figures = output / "figures"
    figures.mkdir()
    primary = data[data.seed == 11]
    labels = {"tangent_rms": "Tangent RMS", "normal_rms": "Normal RMS", "tangent_fraction": "Tangent energy fraction"}
    for variable, column in VARIABLES.items():
        fig, axes = plt.subplots(2, 5, figsize=(17, 7))
        for j, domain in enumerate(DOMAINS):
            x = primary.loc[primary.domain == domain, column].to_numpy()
            rows = fits[(fits.seed == 11) & (fits.domain == domain) & (fits.variable == variable)].sort_values("cv_mean_logpdf", ascending=False)
            xx = np.linspace(max(x.min()*.7, 1e-8), x.max()*1.1, 350) if variable != "tangent_fraction" else np.linspace(.001, .999, 350)
            axes[0,j].hist(x, bins="fd", density=True, color="#CDD8DF", edgecolor="white", alpha=.9)
            for k, (_, row) in enumerate(rows.iterrows()):
                params = json.loads(row.parameters)
                axes[0,j].plot(xx, np.exp(logpdf(xx, row.family, params)), color=COLORS[k], lw=1.5,
                               label=f"{FAMILY_LABELS[row.family]}: p={row.gof_p:.3f}")
            row = rows.iloc[0]
            quantile = (np.arange(len(x))+.5)/len(x)
            theoretical = ppf(quantile, row.family, json.loads(row.parameters))
            axes[1,j].scatter(theoretical, np.sort(x), s=11, color=COLORS[0], alpha=.7)
            lo, hi = min(theoretical.min(),x.min()), max(theoretical.max(),x.max())
            axes[1,j].plot([lo,hi],[lo,hi], "--", color="#666666", lw=1)
            axes[0,j].set_title(f"{domain}  n={len(x)}")
            axes[0,j].legend(fontsize=7, frameon=False)
            axes[0,j].set_xlabel(labels[variable])
            axes[1,j].set_xlabel(f"CV-best {FAMILY_LABELS[row.family]} quantiles")
            axes[1,j].set_ylabel("Observed quantiles")
        axes[0,0].set_ylabel("Density")
        fig.suptitle(f"Source distributions: {labels[variable]} | seed 11\nRefitted-bootstrap KS p; non-rejection does not prove a distribution", fontsize=13)
        fig.tight_layout(rect=(0,0,1,.93))
        fig.savefig(figures/f"{variable}_fits.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1,5,figsize=(17,3.8))
    for ax,domain in zip(axes,DOMAINS):
        d=primary[primary.domain==domain]
        row=dep[(dep.seed==11)&(dep.domain==domain)].iloc[0]
        ax.scatter(d.current_tangent_rms,d.current_normal_rms,s=15,alpha=.65,color=COLORS[0])
        ax.set(xscale="log",yscale="log",xlabel="Tangent RMS",ylabel="Normal RMS",
               title=f"{domain}\nrho={row.spearman:.2f}, p={row.permutation_p:.3f}")
        ax.grid(alpha=.15)
    fig.suptitle("Joint magnitude distributions | same samples, orthogonal vectors can have dependent magnitudes",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.9))
    fig.savefig(figures/"joint_magnitudes.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,variable in zip(axes,VARIABLES):
        rows=fits[fits.variable==variable]
        families=FAMILIES[variable]
        matrix=np.array([[np.sum((rows.domain==d)&(rows.family==f)&(rows.gof_p>=.05)) for f in families] for d in DOMAINS])
        ax.imshow(matrix,vmin=0,vmax=5,cmap="YlGnBu",aspect="auto")
        ax.set_xticks(range(len(families)),[FAMILY_LABELS[f] for f in families],rotation=20)
        ax.set_yticks(range(5),DOMAINS)
        ax.set_title(labels[variable])
        for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha="center",va="center",color="white" if value>=4 else "black")
    fig.suptitle("Seeds without rejection at raw bootstrap p >= .05 (out of 5)\nSeeds share samples; counts describe model sensitivity, not five independent replications",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.9))
    fig.savefig(figures/"seed_sensitivity.png",dpi=170)
    plt.close(fig)


def report(output, data, fits, dep, descriptive):
    primary=fits[fits.seed==11]
    best=primary.sort_values("cv_mean_logpdf",ascending=False).groupby(["domain","variable"],sort=False).head(1)
    lines=["# Source 概率分布检查：本地第一步", "",
        "本轮对象是 Exp31E 五个 source 的验证样本及其冻结几何特征。结果属于探索性分布诊断，未训练模型、未访问目标域文件，也未将源域标签/误差列用于拟合。", "",
        "## 检查对象与方法", "",
        "- 切向与法向 RMS：比较零位置的对数正态、Gamma、Weibull；切向能量占比：比较固定 [0,1] 支持的 Beta、logit 正态。",
        "- 每个 source、每个 Encoder seed 单独拟合；seed 11 是事先指定的展示基准，其余四个作模型敏感性检查。样本 ID 在五个 seed 间完全相同，不能当成五倍样本。",
        "- 三次五折交叉验证以留出平均 log density 排序；所有密度均在原变量尺度计算，logit 正态包含变换 Jacobian。比较只在同 source、同变量内进行。",
        "- KS 拟合优度采用参数自举，每次模拟重新估计参数。seed 11 使用 499 次，其余 seed 使用 199 次。表中 p 为未校正值，完整 CSV 另附每 seed 40 个检验的 BH q。",
        "- 不拒绝某个分布只表示当前样本未排除它，不能证明真实分布就是该分布；CV 第一名也可能拟合不足。",
        "- 保留全部冻结样本；另对 coordinate_reliable 子集只重算 CV 排名，作为几何质量敏感性检查。",
        "- 随机拆半 W1/IQR 描述有限样本波动，不是独立采集或跨时间稳定性的证明。数据仅有行 ID，不能核查潜在叶片/采集批次内依赖；p 与区间采用行可交换假设。", "",
        "## Seed 11 的候选分布结果", "",
        "| Source | n | 变量 | 留出表现最佳 | bootstrap p | 该分布五 seed 未拒绝数 |",
        "|---|---:|---|---|---:|---:|"]
    summaries=[]
    for domain in DOMAINS:
        for variable in VARIABLES:
            row=best[(best.domain==domain)&(best.variable==variable)].iloc[0]
            across=fits[(fits.domain==domain)&(fits.variable==variable)&(fits.family==row.family)]
            count=int((across.gof_p>=.05).sum())
            lines.append(f"| {domain} | {row.n} | {VARIABLE_ZH[variable]} | {FAMILY_ZH[row.family]} | {row.gof_p:.3f} | {count}/5 |")
            summaries.append({"domain":domain,"variable":variable,"seed11_cv_best":row.family,"seed11_gof_p":float(row.gof_p),"same_family_nonrejection_seeds":count})
    lines += ["", "## 切向与法向幅度的关系", "",
              "| Source | seed 11 Spearman | 95% bootstrap 区间 | 五 seed rho 范围 | 联合 lognormal 相对独立模型的留出增益（seed 11） |",
              "|---|---:|---|---|---:|"]
    for domain in DOMAINS:
        d=dep[dep.domain==domain]
        r=d[d.seed==11].iloc[0]
        lines.append(f"| {domain} | {r.spearman:.3f} | [{r.rho_ci_low:.3f}, {r.rho_ci_high:.3f}] | [{d.spearman.min():.3f}, {d.spearman.max():.3f}] | {r.joint_lognormal_cv_logpdf_gain_over_independent:.3f} |")
    lines += ["", "联合模型比较只检验保留相关性是否改善该候选模型的留出描述；未证明二维 lognormal 是正确联合分布。Spearman 检验仅覆盖单调关联，未拒绝也不能宣布独立。切向能量占比由两个 RMS 确定，不能把三个变量当作独立维度。", "",
              "## 当前结论边界", "",
              "本步骤检查的是局部几何分解后的**幅度边缘分布与幅度依赖**。它没有拟合完整有符号切向/法向向量的概率密度，也没有控制预测性状状态。分布结论始终相对于当前混合源域 Encoder、Decoder 和每 seed 冻结的 source 光谱尺度。Exp33 资格验证继续 pending。", "",
              "## 图与复现", ""]
    for name in ["tangent_rms_fits", "normal_rms_fits", "tangent_fraction_fits", "joint_magnitudes", "seed_sensitivity"]:
        lines += [f"![{name}](figures/{name}.png)", ""]
    lines += ["运行入口：`python source_distribution_step1.py --output <新的结果目录>`。现有结果目录不覆盖；协议和输入 SHA-256 在 protocol_manifest.json 中。", "",
              "方法参考：[SciPy Monte Carlo goodness-of-fit](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.goodness_of_fit.html)。参数拟合后不使用假定参数已知的普通 KS p 值。", ""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return summaries


def run(parent=PARENT, output=DEFAULT_OUTPUT):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory; results are not overwritten: {output}")
    data,inputs=load_sources(parent)
    output.mkdir(parents=True)
    manifest={"schema":"source_distribution_step1_v1","exploratory":True,
              "parent":str(parent),"input_sha256":inputs,"script_sha256":sha256(__file__),
              "seeds":list(SEEDS),"primary_seed":11,"domains":list(DOMAINS),
              "read_columns":READ_COLUMNS,"families":{k:list(v) for k,v in FAMILIES.items()},
              "source_labels_used":False,"target_files_read":False,"training":False,
              "cv":{"folds":5,"repeats":3,"split_key":"sample_id, fixed repeat salt"},
              "bootstrap":{"primary_seed":499,"other_seeds":199,"refit_every_replicate":True},
              "scope":"source-validation residual magnitudes, not full signed vector density",
              "scale":"parent per-seed source-reference residual scale; common across domains within seed; no recentering",
              "interpretation":"non-rejection is not distribution proof; seed repeats are not independent samples",
              "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__}}
    write_json(output/"protocol_manifest.json",manifest)
    data.to_csv(output/"source_geometry_allowlist.csv.gz",index=False)
    fits,desc,deps,reliable,splits=[],[],[],[],[]
    for seed in SEEDS:
        for domain in DOMAINS:
            part=data[(data.seed==seed)&(data.domain==domain)].sort_values("sample_id")
            print(f"seed={seed} source={domain} n={len(part)}: fitting distributions",flush=True)
            ids=part.sample_id.to_numpy()
            for repeat in range(3):
                for sample_id,fold in zip(ids,folds_for_ids(ids,repeat)):
                    if seed==11:splits.append({"domain":domain,"sample_id":sample_id,"repeat":repeat,"fold":int(fold)})
            for variable,column in VARIABLES.items():
                x=part[column].to_numpy(dtype=float)
                desc.append({"seed":seed,"domain":domain,"variable":variable,"n":len(x),
                             "mean":float(x.mean()),"std":float(x.std(ddof=1)),"median":float(np.median(x)),
                             "q05":float(np.quantile(x,.05)),"q95":float(np.quantile(x,.95)),
                             "skewness":float(stats.skew(x,bias=False)),"excess_kurtosis":float(stats.kurtosis(x,bias=False)),
                             "coordinate_reliable_fraction":float(part.coordinate_reliable.mean()),
                             **split_stability(x,200,keyed_rng("split",seed,domain,variable))})
                for family in FAMILIES[variable]:
                    params=fit_family(x,family)
                    ks,p=bootstrap_gof(x,family,params,499 if seed==11 else 199,keyed_rng("gof",seed,domain,variable,family))
                    fits.append({"seed":seed,"domain":domain,"variable":variable,"family":family,"n":len(x),
                                 "parameters":json.dumps(params),"ks":ks,"gof_p":p,
                                 **cross_validate(x,ids,family)})
                    mask=part.coordinate_reliable.to_numpy(dtype=bool)
                    if mask.sum()>=30:
                        reliable.append({"seed":seed,"domain":domain,"variable":variable,"family":family,"n":int(mask.sum()),
                                         **cross_validate(x[mask],ids[mask],family)})
            deps.append(dependence(part,4999,999 if seed==11 else 199))
    fit=pd.DataFrame(fits)
    fit["gof_q_bh_within_seed"]=fit.groupby("seed").gof_p.transform(lambda p:bh_adjust(p.to_numpy()))
    fit["cv_rank"]=fit.groupby(["seed","domain","variable"]).cv_mean_logpdf.rank(ascending=False,method="min")
    dep=pd.DataFrame(deps)
    dep["permutation_q_bh_within_seed"]=dep.groupby("seed").permutation_p.transform(lambda p:bh_adjust(p.to_numpy()))
    descriptive=pd.DataFrame(desc)
    rel=pd.DataFrame(reliable)
    rel["cv_rank"]=rel.groupby(["seed","domain","variable"]).cv_mean_logpdf.rank(ascending=False,method="min")
    for name,frame in [("distribution_fits",fit),("descriptive_statistics",descriptive),("tangent_normal_dependence",dep),
                       ("reliable_subset_fits",rel),("cv_assignments",pd.DataFrame(splits))]:
        frame.to_csv(output/f"{name}.csv",index=False)
    make_figures(data,fit,dep,output)
    summary=report(output,data,fit,dep,descriptive)
    for relative,expected in inputs.items():
        if sha256(parent/relative)!=expected:raise RuntimeError("Parent artifact changed during analysis.")
    write_json(output/"status.json",{"status":"complete_exploratory_source_magnitude_distribution_audit",
               "source_samples":len(data[data.seed==11]),"source_count":5,"encoder_seeds":5,
               "target_files_read":False,"source_labels_used":False,"parent_hashes_reverified":True,
               "fits":len(fit),"summary":summary,
               "artifact_sha256":{str(p.relative_to(output)):sha256(p) for p in output.rglob("*") if p.is_file()}})
    print(f"Complete: {output / 'report.md'}",flush=True)
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,default=PARENT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    run(args.parent,args.output)
