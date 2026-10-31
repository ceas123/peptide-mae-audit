import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, spearmanr, rankdata


COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#666666']
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'paper_assets'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight'})


def group(source):
    if '_s42_mr' in source:
        return 'New mask-ratio group'
    if source.startswith('ckpt'):
        return 'Old orthogonal group'
    if source.startswith('randinit'):
        return 'Random initialization'
    if source.startswith('esm'):
        return 'ESM-2 upper bound'
    return 'Descriptors / sentinels'


def label(source):
    match = re.search(r'_s42_mr([0-9.]+)_', source)
    if match:
        return 'New MAE mr=' + match.group(1)
    if source.startswith('randinit'):
        return source.replace('randinit_d256_', 'Random ')
    if source.startswith('ckpt'):
        return 'Old ' + re.sub(r'_\d{4}_\d{6}_mr[0-9.]+_d256$', '', source[5:])
    return source.replace('esm_esm2_', 'ESM-2 ')


def save_table(name, frame, note=''):
    frame.to_csv(ASSETS / (name + '.csv'), index=False, encoding='utf-8-sig')
    printable = frame.copy().fillna('NA')
    for column in printable:
        printable[column] = printable[column].map(lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))
    text = '| ' + ' | '.join(printable.columns) + ' |\n'
    text += '| ' + ' | '.join(['---'] * len(printable.columns)) + ' |\n'
    text += '\n'.join('| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |' for row in printable.values)
    (ASSETS / (name + '.md')).write_text(text + '\n\n' + note + '\n', encoding='utf-8')


def save_fig(name, fig):
    fig.savefig(ASSETS / (name + '.pdf'))
    fig.savefig(ASSETS / (name + '.png'), dpi=300)
    plt.close(fig)


def read_probe(task, pool='mean'):
    return pd.read_csv(ROOT / 'paper_run/probe' / f'metrics_eval_{task}_{task}_{pool}.csv')


def subspace_tables():
    frames = {}
    for kind in ['resid', 'dim', 'dim_curve']:
        chunks = []
        for path in (ROOT / 'paper_run/subspace').glob(f'*/{kind}_eval_*.csv'):
            chunks.append(pd.read_csv(path))
        if chunks:
            frame = pd.concat(chunks, ignore_index=True)
            keys = ['set', 'source'] + (['span'] if kind == 'resid' else ['basis', 'k'] if kind == 'dim_curve' else [])
            frame = frame.drop_duplicates(keys, keep='last')
            frame.to_csv(ROOT / 'paper_run/subspace' / (kind + '_all.csv'), index=False, encoding='utf-8-sig')
            frames[kind] = frame
    return frames


def table1():
    rows = []
    corpus = pd.read_csv(ROOT / 'alldataset.csv').Sequence.astype(str)
    clean = pd.read_csv(ROOT / 'decontam_out_sne/sets/corpus_clean.csv').Sequence.astype(str)
    for task, pos in [('amp', 'AMP'), ('afab', 'Antifungal'), ('acp', 'ACP')]:
        path = ROOT / f'decontam_out_sne/sets/eval_{task}.csv'
        d = pd.read_csv(path)
        lengths = d.Sequence.str.len()
        mask = d.Label.eq(pos)
        rows.append(dict(dataset=task.upper(), n=len(d), n_positive=int(mask.sum()), prevalence=mask.mean(), length_KS=ks_2samp(lengths[mask], lengths[~mask]).statistic, exact_overlap_n=int(d.Sequence.isin(set(corpus)).sum()), origin=str(path.relative_to(ROOT))))
    for split in ['two_vs_rest', 'three_vs_rest', 'low_vs_high']:
        d = pd.read_csv(ROOT / f'splits/splits/{split}.csv')
        rows.append(dict(dataset='GB1 ' + split, n=len(d), n_positive=np.nan, prevalence=np.nan, length_KS=np.nan, exact_overlap_n=int(d.sequence.isin(set(corpus)).sum()), origin=f'splits/splits/{split}.csv'))
    rows += [dict(dataset='Pretraining original', n=len(corpus)), dict(dataset='Pretraining clean file', n=len(clean)), dict(dataset='Removed from corpus', n=len(set(corpus) - set(clean)))]
    save_table('Table1_dataset_audit', pd.DataFrame(rows), 'GB1 is regression: prevalence and positive/negative length KS are not applicable. AMP length matching is not exact: the KS statistic is measured from current files. Historical report says 43754, but the current corpus_clean.csv contains 43846 rows. 9-mer containment is a proxy, not sequence identity. New MAE checkpoints were trained on alldataset.csv, not on the clean corpus.')
    audit = pd.DataFrame([dict(item='original_unique', value=len(set(corpus)), evidence='alldataset.csv'), dict(item='sne_exact_overlap', value=2266, evidence='plan_report.txt [2]'), dict(item='sne_9mer_additional_at_0.9', value=384, evidence='plan_report.txt [4]'), dict(item='negative_pool_AMP_like_at_0.9', value=146, evidence='plan_report.txt [4]B; not necessarily final removal'), dict(item='ACP_overlap_reported', value=908, evidence='plan_report.txt [7x]'), dict(item='ACP_near_homolog_reported', value=204, evidence='plan_report.txt [7x]; overlaps not additive'), dict(item='clean_file_rows', value=len(clean), evidence='current file'), dict(item='historical_final_clean_claim', value=43754, evidence='plan_report.txt final; inconsistent with file')])
    save_table('Table1b_decontamination_evidence', audit)


def table2():
    rows = []
    for task in ['afab', 'acp', 'amp']:
        m = read_probe(task)
        c = pd.read_csv(ROOT / f'paper_run/probe/compare_eval_{task}_{task}_mean.csv')
        random = m[m.source.str.startswith('randinit')]
        for _, row in m.iterrows():
            source = row.source
            pair = c[(c.a.eq(source) & c.b.eq('randinit_d256_s0')) | (c.b.eq(source) & c.a.eq('randinit_d256_s0'))]
            extra = {}
            if len(pair):
                p = pair.iloc[0]
                sign = 1 if p.a == source else -1
                extra = dict(delta_AP_vs_randinit_s0=sign*p.dAP, delta_lo=p.dAP_lo if sign == 1 else -p.dAP_hi, delta_hi=p.dAP_hi if sign == 1 else -p.dAP_lo, MDE_80=p.mde_80, verdict=p.verdict)
            rows.append(dict(task=task, group=group(source), source=source, dim=row.dim, AP=row.AP, AP_lo=row.AP_lo, AP_hi=row.AP_hi, AUC=row.AUC, AUC_lo=row.AUC_lo, AUC_hi=row.AUC_hi, **extra))
        rows.append(dict(task=task, group='Random initialization summary', source='randinit mean +/- seed SD (n=3)', AP=random.AP.mean(), AP_seed_sd=random.AP.std(ddof=1), AUC=random.AUC.mean(), AUC_seed_sd=random.AUC.std(ddof=1)))
    save_table('Table2_mean_benchmark', pd.DataFrame(rows), 'ESM-2 based on UR50D is a Memorization-contaminated Upper Bound (potential overlap; no measured UR50D hit rate). ACP is balanced 500/500 (prevalence 0.5); AP and AUC are both reported, but balance does not make them mathematically equivalent. Source-script CIs use class-stratified individual bootstrap, not cluster bootstrap. Delta and MDE in this table use random seed 0; seed mean +/- SD is a separate summary. Specification curves use the mean random prediction baseline and cluster bootstrap. Unresolved differences do not establish equivalence; MDE is not an equivalence margin.')


def table3(frames):
    if 'resid' not in frames or 'dim' not in frames:
        return
    d = frames['resid'].query("span == 'comp_quad'").copy()
    d['span_minus_ctrl'] = d.AP_span - d.AP_ctrl
    d['resid_minus_ctrl'] = d.AP_resid - d.AP_ctrl
    columns = ['set', 'source', 'AP_full', 'AP_span', 'AP_resid', 'AP_ctrl', 'span_minus_ctrl', 'resid_minus_ctrl', 'r2_span', 'dim_span', 'rank_span']
    d = d[columns].merge(frames['dim'][['set', 'source', 'PR', 'erank', 'k_var95', 'k95_pca', 'k95_pls']], on=['set', 'source'], how='left')
    d.insert(2, 'group', d.source.map(group))
    save_table('Table3_subspace_mechanism', d, 'comp_quad in the source script includes length (211 columns), whereas aac_off_nolen has 210 columns. R2 is an in-sample descriptive fit, not cross-validated explanatory variance. Subspace performance uses 2 CV repeats; the main benchmark uses 5. Subspace CIs use individual stratified bootstrap. Dimensional recovery is evaluated on the first fold assignment only. k95 defaults to full dimension if the grid does not reach the target; consult reached flags in raw output.')


def table4():
    chunks = []
    for split in ['two_vs_rest', 'three_vs_rest', 'low_vs_high']:
        p = ROOT / f'paper_run/flip/{split}.csv'
        if p.exists():
            d = pd.read_csv(p)
            d['split'] = split
            d['group'] = d.source.map(group)
            chunks.append(d)
    if chunks:
        save_table('Table4_GB1_regression', pd.concat(chunks, ignore_index=True), 'All sources use fixed ridge alpha=1.0, mean pooling and the first 100 residues. Original sequences are 265 residues; all variable sites are retained, but the constant tail is omitted. Fit and validation are combined by the existing source script. Individual-model 95% CIs are test-sample bootstrap; paired difference CIs are reported separately. ESM-2 is a potentially contaminated upper bound. Zero has undefined Spearman rho.')


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    n = len(pd.read_csv(ROOT / 'alldataset.csv'))
    clean = len(pd.read_csv(ROOT / 'decontam_out_sne/sets/corpus_clean.csv'))
    axes[0].bar(['Original', 'Removed', 'Clean file'], [n, n-clean, clean], color=[COLORS[0], COLORS[1], COLORS[2]])
    for i, v in enumerate([n, n-clean, clean]):
        axes[0].text(i, v+800, f'{v:,}', ha='center')
    axes[0].set(ylabel='Sequences', title='a  Audited corpus flow', ylim=(0, n*1.15))
    a = pd.read_csv(ROOT / 'decontam_out_sne/sets/eval_amp.csv')
    original = pd.read_csv(ROOT / 'alldataset.csv').Sequence
    known = set(pd.read_csv(ROOT / 'alldataset_old.csv').Sequence)
    before = original[~original.isin(known)].str.len().to_numpy()
    pos = a[a.Label.eq('AMP')].Sequence.str.len().to_numpy()
    neg = a[a.Label.ne('AMP')].Sequence.str.len().to_numpy()
    for vals, name, color, style in [(pos, 'AMP positive', COLORS[0], '-'), (before, 'Negative pool before matching', COLORS[1], '--'), (neg, 'Matched negative', COLORS[2], '-')]:
        x = np.sort(vals)
        axes[1].step(x, np.arange(1, len(x)+1)/len(x), where='post', label=name, color=color, linestyle=style)
    axes[1].set(xlabel='Sequence length (aa)', ylabel='ECDF', title=f'b  Length matching: KS={ks_2samp(pos,neg).statistic:.4f}', xlim=(0, 150))
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_fig('Fig1_audit', fig)


def fig2():
    p = ROOT / 'paper_run/spec/spec_rows.csv'
    if not p.exists():
        return
    d = pd.read_csv(p).sort_values('delta').reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), gridspec_kw={'height_ratios': [3, 1.5]}, sharex=True)
    x = np.arange(len(d))
    colors = np.where(d.lo.gt(0), COLORS[0], np.where(d.hi.lt(0), COLORS[1], '#999999'))
    for i, row in d.iterrows():
        axes[0].plot([i,i],[row.lo,row.hi],color=colors[i],alpha=0.25,lw=0.5)
    axes[0].scatter(x,d.delta,c=colors,s=4)
    axes[0].axhline(0,color='black',lw=0.7)
    axes[0].set(ylabel='Paired metric difference',title='Specification curve: task, pool, baseline, metric and checkpoint')
    matrix = np.array([pd.Categorical(d[c], categories=sorted(d[c].unique())).codes for c in ['task','pool','baseline','metric']])
    axes[1].imshow(matrix,aspect='auto',interpolation='nearest',cmap='tab10',extent=(-0.5,len(d)-0.5,3.5,-0.5),vmin=0,vmax=9)
    axes[1].set(yticks=range(4),yticklabels=['Task','Pool','Baseline','Metric'],xlabel=f'{len(d)} specifications sorted by effect')
    fig.text(0.01,0.005,'Task: acp / afab / amp; pool: cls / mean; baseline: aac_off_nolen / prevalence / randinit; metric: AP / AUC. Colors in each barcode row encode categories in this order.',fontsize=6)
    fig.tight_layout(rect=(0,0.03,1,1))
    save_fig('Fig2_specification_curve',fig)


def fig3():
    fig, axes = plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    ladder=['length','aac','physchem_nolen','class_quad','aac_off_nolen']
    for ax,task in zip(axes,['afab','acp','amp']):
        m=read_probe(task).set_index('source'); cls=read_probe(task,'cls').set_index('source')
        vals=m.loc[ladder]
        ax.errorbar(range(5),vals.AP,yerr=[np.maximum(0,vals.AP-vals.AP_lo),np.maximum(0,vals.AP_hi-vals.AP)],color='black',marker='o',capsize=2,label='Descriptor AP')
        for frame,mask,name,color in [(m,m.index.str.startswith('randinit'),'Random mean',COLORS[0]),(m,m.index.str.contains('_s42_'),'New MAE mean',COLORS[1]),(m,m.index.str.startswith('ckpt') & ~m.index.str.contains('_s42_'),'Old MAE mean',COLORS[2]),(cls,cls.index.str.startswith('ckpt'),'MAE CLS',COLORS[3])]:
            v=frame.loc[mask,'AP']; ax.axhspan(v.min(),v.max(),color=color,alpha=0.12); ax.axhline(v.mean(),color=color,lw=1,label=name)
        ax.set(title=task.upper(),xticks=range(5),xticklabels=['Length\n1d','AAC\n20d','PhysChem\n26d','ClassQuad\n41d','AAQuad\n210d'],ylim=(0.15,1.0))
    axes[0].set_ylabel('Average precision')
    axes[-1].legend(fontsize=6,loc='lower right')
    fig.tight_layout(); save_fig('Fig3_descriptor_ladder',fig)


def selected_sources(d):
    names=d.source.unique().tolist()
    want=['randinit_d256_s0']+[s for s in names if '_s42_' in s]
    old=[s for s in names if s.startswith('ckpt') and '_s42_' not in s]
    return [s for s in want+(old[:1]) if s in names]


def fig4(frames):
    if not all(k in frames for k in ['resid','dim','dim_curve']):
        return
    d=frames['resid'].query("set == 'eval_amp' and span == 'comp_quad'")
    names=selected_sources(d)
    if not names:
        return
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for j,col in enumerate(['AP_full','AP_span','AP_ctrl','AP_resid']):
        axes[0].bar(np.arange(len(names))+(j-1.5)*0.19,d.set_index('source').loc[names,col],width=0.19,label=col.replace('AP_',''),color=COLORS[j])
    axes[0].set(xticks=range(len(names)),xticklabels=[label(n) for n in names],ylabel='AMP average precision',title='a  Composition subspace and PCA controls')
    axes[0].tick_params(axis='x',labelrotation=35);axes[0].legend(fontsize=7)
    dims=frames['dim'].query("set == 'eval_amp'").set_index('source')
    for i,n in enumerate(names):
        curve=frames['dim_curve'].query("set == 'eval_amp' and basis == 'pca'")
        curve=curve[curve.source.eq(n)].sort_values('k')
        if len(curve):
            er=dims.loc[n,'erank']; kv=dims.loc[n,'k_var95']
            axes[1].plot(curve.k,curve.AP,marker='.',color=COLORS[i%len(COLORS)],label=f'{label(n)}; erank={er:.1f}, k95={kv:.0f}')
    axes[1].set(xscale='log',xlabel='Retained PCA dimensions k',ylabel='AMP average precision',title='b  Performance recovery (first CV assignment)')
    axes[1].legend(fontsize=6)
    fig.tight_layout();save_fig('Fig4_subspace_dimensions',fig)


def fig5():
    folder=ROOT/'paper_run/axis'; paths=[folder/'afab_acp_aac_weights.csv',folder/'afab_amp_aac_weights.csv']
    if not all(p.exists() for p in paths):
        return
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    import re as regex
    aa=regex.search(r'AA_STD\s*=\s*[\"\']([^\"\']+)',(ROOT/'probe_suite.py').read_text(encoding='utf-8')).group(1)
    for ax,p,title in zip(axes[:2],paths,['AFAB vs ACP (shared AB controls)','AFAB vs AMP']):
        w=pd.read_csv(p); ax.scatter(w.weight_a,w.weight_b,s=12,color=COLORS[0])
        for i,row in w.iterrows():ax.annotate(aa[i],(row.weight_a,row.weight_b),fontsize=7,xytext=(2,2),textcoords='offset points')
        ax.axhline(0,color='grey',lw=0.5);ax.axvline(0,color='grey',lw=0.5);ax.set(xlabel='Normalized AFAB weight',ylabel='Normalized other-task weight',title=title)
    matrix=np.full((3,3),np.nan)
    for pair,a,b in [('afab_amp',0,2),('acp_amp',1,2),('afab_acp',0,1)]:
        p=folder/(pair+'_aac.csv')
        if p.exists():
            d=pd.read_csv(p).set_index('metric');matrix[a,a]=d.loc['self_a','mean'];matrix[b,b]=d.loc['self_b','mean'];matrix[a,b]=d.loc['a_to_b','mean'];matrix[b,a]=d.loc['b_to_a','mean']
    image=axes[2].imshow(matrix,vmin=0.4,vmax=1,cmap='viridis')
    for a in range(3):
        for b in range(3):axes[2].text(b,a,'NA' if np.isnan(matrix[a,b]) else f'{matrix[a,b]:.3f}',ha='center',va='center',color='white',fontsize=8)
    axes[2].set(xticks=range(3),yticks=range(3),xticklabels=['AFAB','ACP','AMP'],yticklabels=['AFAB','ACP','AMP'],xlabel='Target task',ylabel='Source axis',title='AAC OOB transfer AUC')
    fig.colorbar(image,ax=axes[2],shrink=0.8)
    fig.text(0.01,0.01,'AFAB-ACP cells use shared-source AB controls; other pairs use benchmark sets. Diagonal values vary with pair-specific scaling: see raw CSVs.',fontsize=6)
    fig.tight_layout(rect=(0,0.05,1,1));save_fig('Fig5_axis_transfer',fig)


def paired_flip():
    rows=[]
    for split in ['two_vs_rest','three_vs_rest','low_vs_high']:
        p=ROOT/f'paper_run/flip/{split}_predictions.csv'
        if not p.exists():continue
        d=pd.read_csv(p); y=d.target.to_numpy(); random=[c for c in d if c.startswith('randinit')]
        if len(random)!=3:continue
        base=d[random].mean(axis=1).to_numpy(); br=spearmanr(y,base).statistic
        names=[c for c in d if c.startswith('ckpt')]
        arrays=np.column_stack([base]+[d[n].to_numpy() for n in names]); point=np.array([spearmanr(y,arrays[:,i]).statistic for i in range(arrays.shape[1])])
        rng=np.random.RandomState(42);boot=np.zeros((1000,len(names)))
        for b in range(1000):
            ix=rng.randint(0,len(y),len(y)); yr=rankdata(y[ix]); xr=rankdata(arrays[ix],axis=0); yr=yr-yr.mean();xr=xr-xr.mean(axis=0)
            rho=(yr[:,None]*xr).sum(axis=0)/np.sqrt((yr**2).sum()*(xr**2).sum(axis=0));boot[b]=rho[1:]-rho[0]
        for j,name in enumerate(names):
            lo,hi=np.percentile(boot[:,j],[2.5,97.5]);rows.append(dict(split=split,source=name,baseline='mean_prediction_randinit_s0_s1_s2',rho=point[j+1],base_rho=br,delta=point[j+1]-br,lo=lo,hi=hi,n_boot=1000))
    if rows:save_table('Table4b_GB1_paired_differences',pd.DataFrame(rows),'Paired test-sample bootstrap, 1000 draws, seed 42. Baseline is the mean prediction across three random seeds, not the mean of their individual rho values. Conditional on fitted models; training variability is not included.')


def fig6():
    p=ROOT/'paper_run/spec/spec_rows.csv'; f=ASSETS/'Table4b_GB1_paired_differences.csv'
    if not p.exists() or not f.exists():return
    d=pd.read_csv(p);d=d.query("pool == 'mean' and baseline == 'randinit' and metric == 'AP'"); d=d[d.ckpt.str.contains('_s42_')].rename(columns={'ckpt':'source','task':'dataset'})
    b=pd.read_csv(f);b=b[b.source.str.contains('_s42_')].rename(columns={'split':'dataset'})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,data,title,xlabel in [(axes[0],d,'Composition-associated classification','Delta AP vs random prediction ensemble'),(axes[1],b,'GB1 extrapolation regression','Delta Spearman rho vs random prediction ensemble')]:
        data=data.sort_values(['dataset','source']).reset_index(drop=True)
        for i,row in data.iterrows():
            color=COLORS[0] if row.lo>0 else COLORS[1] if row.hi<0 else '#777777'
            ax.plot([row.lo,row.hi],[i,i],color=color,lw=1.5);ax.plot(row.delta,i,'o',color=color,ms=4)
        ax.axvline(0,color='black',lw=0.7);ax.set(yticks=range(len(data)),yticklabels=[row.dataset+' | '+label(row.source) for _,row in data.iterrows()],xlabel=xlabel,title=title);ax.invert_yaxis()
    fig.text(0.01,0.01,'Classification uses cluster bootstrap; GB1 uses paired test-sample bootstrap. Different metrics have separate axes. Grey intervals cross zero; no universal sign is imposed.',fontsize=6)
    fig.tight_layout(rect=(0,0.05,1,1));save_fig('Fig6_regime_effects',fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paired_flip',action='store_true');args=ap.parse_args();ASSETS.mkdir(exist_ok=True)
    if args.paired_flip:paired_flip()
    table1();table2();frames=subspace_tables();table3(frames);table4();fig1();fig2();fig3();fig4(frames);fig5();fig6()
    files=[dict(file=p.name,bytes=p.stat().st_size) for p in ASSETS.iterdir() if p.is_file()]
    pd.DataFrame(files).to_csv(ASSETS/'asset_inventory.csv',index=False,encoding='utf-8-sig')
    print('Assets:',len(files),'subspace rows:',{k:len(v) for k,v in frames.items()},flush=True)


if __name__=='__main__':main()
