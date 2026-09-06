"""Source-only empirical signed residual distributions from the historical Exp20A bank.

An explicitly separate pilot: four sources, 128 spectra each, one historical
checkpoint and 400--2399 nm. This is not an Exp31E five-seed confirmation.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from source_distribution_step1 import ROOT, sha256, write_json, bh_adjust, folds_for_ids

PARENT=ROOT/'experiment_v3_20260724/exp20a_jacobian_geometry_v1'
OUTPUT=ROOT/'output_source_signed_residual_distribution/historical_v1'
SOURCES=('Lopex','Angers','IFGG_train','UW_train')
COMPONENTS=('residual','tangent','normal')
COLORS=('#2874A6','#D68910','#148F77','#9B59B6')


def blocks(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2000:raise ValueError('Expected 400--2399 nm, 2000 columns.')
    return x.reshape(len(x),80,25).mean(axis=2)


def normal_ks_batch(rows):
    """Composite-normal KS statistic: fit mean and MLE SD independently per row."""
    rows=np.asarray(rows,dtype=float)
    sd=rows.std(axis=1,ddof=0)
    if np.any(sd<=1e-12):raise ValueError('Degenerate projection requires separate point-mass treatment.')
    z=(rows-rows.mean(axis=1,keepdims=True))/sd[:,None]
    u=stats.norm.cdf(np.sort(z,axis=1))
    n=rows.shape[1]
    return np.maximum((np.arange(1,n+1)/n-u).max(axis=1),(u-np.arange(n)/n).max(axis=1))


def projection_capture(x,b):
    denominator=np.sum(x*x,axis=1)
    captured=25*np.sum(b*b,axis=1)
    xc=x-x.mean(axis=0)
    bc=b-b.mean(axis=0)
    return float(np.median(captured/denominator)),float(25*np.sum(bc*bc)/np.sum(xc*xc))


def joint_cv(b,ids):
    """Fixed 80D candidates, source-internal train-only covariance and mean fitting."""
    independent,joint,coverage=[],[],[]
    assignments=folds_for_ids(ids,0)
    for fold in range(5):
        train,test=b[assignments!=fold],b[assignments==fold]
        mean=train.mean(axis=0)
        var=train.var(axis=0,ddof=0)
        floor=1e-8*max(float(var.mean()),1e-12)
        var=np.maximum(var,floor)
        independent.extend(np.sum(stats.norm.logpdf(test,mean,np.sqrt(var)),axis=1)/80)
        model=LedoitWolf().fit(train)
        cov=model.covariance_+np.eye(80)*floor
        joint.extend(stats.multivariate_normal.logpdf(test,mean=model.location_,cov=cov)/80)
        u=stats.norm.cdf(test,mean,np.sqrt(var))
        coverage.extend(np.mean((u>=.05)&(u<=.95),axis=1))
    return {'independent_gaussian_cv_logpdf_per_coordinate':float(np.mean(independent)),
            'correlated_gaussian_cv_logpdf_per_coordinate':float(np.mean(joint)),
            'joint_minus_independent_cv_logpdf_per_coordinate':float(np.mean(joint)-np.mean(independent)),
            'pointwise_gaussian_cv_90_coverage':float(np.mean(coverage))}


def figures(bank,output,gof):
    wavelength=np.arange(400,2400)
    fig,axes=plt.subplots(3,4,figsize=(18,10))
    for i,component in enumerate(COMPONENTS):
        for j,source in enumerate(SOURCES):
            x=bank[source][component]
            q=np.quantile(x,[.05,.25,.5,.75,.95],axis=0)
            ax=axes[i,j]
            ax.fill_between(wavelength,q[0],q[4],alpha=.15,color=COLORS[j],label='Pointwise 5--95%')
            ax.fill_between(wavelength,q[1],q[3],alpha=.3,color=COLORS[j],label='Pointwise 25--75%')
            ax.plot(wavelength,q[2],lw=1,color=COLORS[j],label='Median')
            ax.axhline(0,lw=.7,color='#777777')
            ax.set_title(f'{source} | {component}')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Signed source-scaled residual')
            if i==0 and j==0:ax.legend(fontsize=7)
    fig.suptitle('Historical Exp20A: signed residual distributions | 128 samples per source\nPointwise population quantiles, not confidence intervals or simultaneous spectral bands',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(output/'signed_spectral_quantiles.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,source in zip(axes,SOURCES):
        b=blocks(bank[source]['normal'])
        corr=np.corrcoef(b,rowvar=False)
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r',origin='lower',extent=(400,2400,400,2400))
        ax.set(title=source,xlabel='Wavelength (nm)',ylabel='Wavelength (nm)')
    fig.colorbar(im,ax=list(axes),shrink=.8,label='Across-sample Pearson correlation')
    fig.suptitle('Normal residual: source-specific joint band structure | 80 fixed signed 25 nm averages')
    fig.savefig(output/'normal_band_correlations.png',dpi=170,bbox_inches='tight')
    plt.close(fig)
    probes=(500,700,900,1200,1450,1650,1950,2150)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    q=(np.arange(128)+.5)/128
    expected=stats.norm.ppf(q)
    for ax,probe in zip(axes.ravel(),probes):
        group=(probe-400)//25
        for source,color in zip(SOURCES,COLORS):
            x=blocks(bank[source]['normal'])[:,group]
            z=(x-x.mean())/x.std(ddof=0)
            ax.plot(expected,np.sort(z),'.',ms=3,color=color,label=source)
        ax.plot([-3,3],[-3,3],'--',color='#777777',lw=1)
        ax.set(title=f'{400+25*group}--{424+25*group} nm',xlabel='Standard normal quantile',ylabel='Standardized signed band average')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Normal residual Gaussian Q--Q diagnostics | 8 display windows fixed before analysis\nEach source/window standardized only for this shape comparison; primary vectors retain common source scale',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(output/'normal_signed_qq.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,component in zip(axes,COMPONENTS):
        x=np.concatenate([blocks(bank[s][component]) for s in SOURCES])
        pca=PCA(n_components=2,svd_solver='full').fit(x)
        z=pca.transform(x)
        for j,source in enumerate(SOURCES):
            ax.scatter(z[j*128:(j+1)*128,0],z[j*128:(j+1)*128,1],s=10,alpha=.55,color=COLORS[j],label=source)
        ax.set(title=f'{component}: PC1+2 = {pca.explained_variance_ratio_.sum():.1%}',xlabel='Shared PC1',ylabel='Shared PC2')
        ax.legend(fontsize=8)
    fig.suptitle('Empirical source distributions in shared coordinates | descriptive source-pooled PCA only',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(output/'shared_empirical_coordinates.png',dpi=170)
    plt.close(fig)


def run(parent=PARENT,output=OUTPUT):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(f'Choose a new output directory: {output}')
    status=json.loads((parent/'exp20a_status.json').read_text(encoding='utf-8'))
    inputs={}
    for name in ('protocol_manifest.json','empirical_normal_bank.npz','source_wavelength_scale.npy'):
        value=sha256(parent/name)
        if value!=status['artifact_hashes'][name]:raise ValueError(f'Historical parent hash mismatch: {name}')
        inputs[name]=value
    bank={}
    with np.load(parent/'empirical_normal_bank.npz',allow_pickle=False) as z:
        domains=z['dataset'].astype(str)
        # Archive contains other domains; none of their rows enter any statistic.
        for source in SOURCES:
            mask=domains==source
            residual=z['residual_whitened'][mask].astype(float)
            normal=z['normal_whitened'][mask].astype(float)
            tangent=residual-normal
            ids=np.array([f'{source}:{i}' for i in z['sample_index'][mask]])
            if len(ids)!=128 or len(set(ids))!=128:raise ValueError('Unexpected source registry.')
            if not np.isfinite(residual).all() or not np.isfinite(normal).all():raise ValueError('Nonfinite source vectors.')
            cosine=np.abs(np.sum(tangent*normal,axis=1))/(np.linalg.norm(tangent,axis=1)*np.linalg.norm(normal,axis=1))
            if cosine.max()>1e-4:raise ValueError('Tangent-normal orthogonality failed.')
            bank[source]={'residual':residual,'tangent':tangent,'normal':normal,'ids':ids}
    output.mkdir(parents=True)
    write_json(output/'protocol_manifest.json',{
        'schema':'historical_source_signed_residual_distribution_v1','exploratory':True,
        'parent':str(parent),'input_sha256':inputs,'script_sha256':sha256(__file__),
        'helper_sha256':sha256(ROOT/'source_distribution_step1.py'),
        'sources':list(SOURCES),'n_per_source':128,'checkpoint_scope':'one historical Exp20A checkpoint, not Exp31E five seeds',
        'geometry':'inherited PP8, latent clip [.03,.97], difference step .02, SVD cutoff .001',
        'support_nm':[400,2399],'components':list(COMPONENTS),
        'projection':'80 nonoverlapping signed averages, 25 nm each, no per-source normalization',
        'normality':'composite normal KS, refit mean/MLE SD in 9999 null draws of n=128; common null justified by location-scale invariance',
        'multiplicity':'BH within each source/component, 80 band tests; exploratory correlated tests',
        'joint_probe':'5-fold source-internal CV, independent Gaussian vs LedoitWolf covariance Gaussian, log density per coordinate',
        'joint_probe_boundary':'relative heldout comparison of two approximations, not proof of Gaussianity or isolated causal dependence',
        'mixed_domain_archive_opened':True,'target_rows_analyzed':False,'labels_read':False,'neural_training':False,
        'limits':['historical mixed-source model','no seed confirmation','source rows may have acquisition/leaf dependence',
                  'quantile bands are pointwise, not simultaneous','Gaussian rejection does not identify a mixture mechanism'],
    })
    rng=np.random.default_rng(20260905)
    null=np.sort(normal_ks_batch(rng.standard_normal((9999,128))))
    summaries,tests,quantiles=[],[],[]
    for source in SOURCES:
        print(f'Signed distribution pilot: {source}',flush=True)
        for component in COMPONENTS:
            x=bank[source][component]
            b=blocks(x)
            ks=normal_ks_batch(b.T)
            p=(1+len(null)-np.searchsorted(null,ks,side='left'))/(len(null)+1)
            q=bh_adjust(p)
            corr=np.corrcoef(b,rowvar=False)
            upper=np.abs(corr[np.triu_indices(80,1)])
            raw_capture,variation_capture=projection_capture(x,b)
            row={'source':source,'component':component,'n':128,
                 'median_projection_energy_capture':raw_capture,'centered_variation_capture':variation_capture,
                 'mean_profile_energy_fraction':float(np.sum(x.mean(axis=0)**2)/np.mean(np.sum(x*x,axis=1))),
                 'gaussian_rejected_bands_raw':int((p<.05).sum()),'gaussian_rejected_bands_bh':int((q<.05).sum()),
                 'median_abs_band_correlation':float(np.median(upper)),
                 **joint_cv(b,bank[source]['ids'])}
            summaries.append(row)
            for j in range(80):
                v=b[:,j]
                tests.append({'source':source,'component':component,'group':j,'start_nm':400+25*j,'end_nm':424+25*j,
                              'mean':float(v.mean()),'std':float(v.std(ddof=0)),'skewness':float(stats.skew(v,bias=False)),
                              'excess_kurtosis':float(stats.kurtosis(v,bias=False)),'ks':float(ks[j]),'p':float(p[j]),'q_bh':float(q[j])})
            pointwise=np.quantile(x,[.05,.25,.5,.75,.95],axis=0)
            for j,wave in enumerate(range(400,2400)):
                quantiles.append({'source':source,'component':component,'wavelength':wave,
                                  **{name:float(pointwise[k,j]) for k,name in enumerate(['q05','q25','median','q75','q95'])}})
    summary=pd.DataFrame(summaries)
    gof=pd.DataFrame(tests)
    summary.to_csv(output/'source_component_summary.csv',index=False)
    gof.to_csv(output/'signed_band_gaussian_checks.csv',index=False)
    pd.DataFrame(quantiles).to_csv(output/'spectral_quantiles.csv.gz',index=False)
    figures(bank,output,gof)
    lines=['# 有符号残差分布：历史缓存试探','',
           '**基座限定：Exp20A 单个历史 checkpoint，四个 source 各 128 样本，400–2399 nm。不能与 Exp31E 五种子的结论合并。**','',
           '本轮直接使用整条有符号标准化残差，以及其切向/法向向量。历史文件包含其他域，仅四个 source 的行进入计算；没有读取性状标签、没有新训练。','',
           '## 如何理解这次的“分布”','',
           '- 每个样本是一条 2000 维残差曲线；同一波长上跨样本取分位数，描述 source 在该位置的有符号偏差分布。波长不作为独立样本。',
           '- 80 个固定 25 nm 有符号均值提供共同线性坐标，报告总能量与中心化变动的保留率，避免只保留平滑平均形状。',
           '- 用参数重新拟合的 Monte Carlo KS 检查每个坐标的 Gaussian 近似，并在每 source/component 的 80 个检验内作 BH 校正。候选被拒绝不说明真实分布必为混合族，未拒绝也不证明 Gaussian。',
           '- 相关 Gaussian 与独立 Gaussian 只作五折留出比较，前者用训练折内 LedoitWolf 协方差。两者维度、均值族及协方差估计方式不同，增益只说明联合描述候选更好，不是等容量机制检验。',
           '- 光谱分位带是逐点经验分布区间，不是均值置信区间或全谱同时覆盖带。','',
           '## 法向有符号分布结果','',
           '| Source | 25 nm 均值的中心化变动保留率 | Gaussian 被拒绝波段数（BH，/80） | 波段间绝对相关中位数 | 联合候选留出增益/坐标 |',
           '|---|---:|---:|---:|---:|']
    for r in summary[summary.component=='normal'].itertuples():
        lines.append(f'| {r.source} | {r.centered_variation_capture:.3f} | {r.gaussian_rejected_bands_bh} | {r.median_abs_band_correlation:.3f} | {r.joint_minus_independent_cv_logpdf_per_coordinate:.3f} |')
    lines += ['', '## 保留的研究边界','',
              '这次得到 source 的经验有符号分布、Gaussian 基线失配位置及跨波段统计结构。没有定义新的风险分数，没有判断真实生化误差，没有控制预测状态，也没有得到完整高维概率密度的资格结论。旧几何的 clipping/conditioning 曾在 Exp20A.1 获得稳健性支持，但本次不冒充其 boundary-aware 向量或最新五种子模型。','']
    for name in ['signed_spectral_quantiles','normal_band_correlations','normal_signed_qq','shared_empirical_coordinates']:
        lines += [f'![{name}]({name}.png)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    for name,digest in inputs.items():
        if sha256(parent/name)!=digest:raise RuntimeError('Historical parent changed during analysis.')
    write_json(output/'status.json',{'status':'complete_historical_signed_residual_distribution_pilot','source_count':4,'samples':512,
        'target_rows_analyzed':False,'labels_read':False,'parent_hashes_reverified':True,
        'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(summary[summary.component=='normal'].to_string(index=False),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    run(args.parent,args.output)
