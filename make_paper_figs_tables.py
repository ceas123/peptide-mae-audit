from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
R=Path('D:/创'); O=R/'paper_figures_redesigned'; O.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('base',R/'make_paper_figs_tables.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.ASSETS=O
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.titlepad':10,'savefig.bbox':None})
C=m.COLORS

def fig2():
 d=pd.read_csv(R/'paper_run/spec/spec_rows.csv').sort_values('delta').reset_index(drop=True)
 f,axes=plt.subplots(2,1,figsize=(7.1,6.8),sharex=True,gridspec_kw={'height_ratios':[3,1]}); f.subplots_adjust(left=.13,right=.98,top=.92,bottom=.31,hspace=.17)
 colors=np.where(d.lo>0,C[0],np.where(d.hi<0,C[1],'#888888'))
 axes[0].vlines(np.arange(len(d)),d.lo,d.hi,color=colors,alpha=.35,lw=.6);axes[0].scatter(np.arange(len(d)),d.delta,c=colors,s=5)
 axes[0].axhline(0,c='black',ls='--',lw=.8);m.add_ygrid(axes[0]);axes[0].set(ylabel='Paired metric difference',title=f'{len(d)} evaluation specifications')
 handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for c,l in [(C[1],'95% CI below zero'),('#888888','95% CI includes zero'),(C[0],'95% CI above zero')]]
 axes[0].legend(handles=handles,loc='upper left',frameon=False,fontsize=8)
 cats=[sorted(d[k].unique()) for k in ['task','pool','baseline','metric']]
 mat=np.array([pd.Categorical(d[k],categories=v).codes for k,v in zip(['task','pool','baseline','metric'],cats)])
 axes[1].imshow(mat,aspect='auto',interpolation='nearest',cmap=plt.get_cmap('tab10'),vmin=0,vmax=9,extent=(-.5,len(d)-.5,3.5,-.5))
 axes[1].set(yticks=range(4),yticklabels=['Task','Pooling','Baseline','Metric'],xlabel='Specifications ordered by paired difference')
 palette=plt.get_cmap('tab10')
 for j,(title,values) in enumerate(zip(['Task','Pooling','Baseline','Metric'],cats)):
  names=[str(v).upper() if title in ['Task','Metric'] else str(v).replace('aac_off_nolen','AAC quadratic').replace('randinit','Random ensemble').replace('prevalence','Prevalence') for v in values]
  hs=[Line2D([],[],marker='s',ls='',color=palette(i),label=n) for i,n in enumerate(names)]
  f.legend(handles=hs,title=title,loc='center',bbox_to_anchor=(.55,.225-j*.057),ncol=3,frameon=False,fontsize=8,title_fontsize=8)
 save('Fig2_specification_curve',f)

def fig3():
 f,axes=plt.subplots(1,3,figsize=(7.1,4.2),sharey=True);f.subplots_adjust(left=.09,right=.99,bottom=.25,top=.75,wspace=.16)
 ladder=['length','aac','physchem_nolen','class_quad','aac_off_nolen']
 for ax,task in zip(axes,['afab','acp','amp']):
  m.add_ygrid(ax);d=m.read_probe(task).set_index('source');cl=m.read_probe(task,'cls').set_index('source');v=d.loc[ladder]
  ax.errorbar(range(5),v.AP,yerr=[v.AP-v.AP_lo,v.AP_hi-v.AP],color='black',marker='o',ms=3,lw=1.3,capsize=2,label='Descriptors')
  for df,mask,name,c in [(d,d.index.str.startswith('randinit'),'Random, mean',C[0]),(d,d.index.str.contains('_s42_'),'New MAE, mean',C[1]),(d,d.index.str.startswith('ckpt')&~d.index.str.contains('_s42_'),'Old MAE, mean',C[2]),(cl,cl.index.str.startswith('ckpt'),'MAE, CLS',C[3])]:
   v=df.loc[mask,'AP'];ax.axhspan(v.min(),v.max(),color=c,alpha=.14);ax.axhline(v.mean(),color=c,lw=1.2,label=name)
  ax.set(title=task.upper(),xticks=range(5),xticklabels=['Length','AAC','Phys.','Class Q.','AAC Q.'],ylim=(.15,.96));ax.tick_params(axis='x',rotation=45)
 for ax in axes:
  for tick in ax.get_xticklabels(): tick.set_ha('right'); tick.set_rotation_mode('anchor')
 axes[0].set_ylabel('Average precision');h,l=axes[0].get_legend_handles_labels();f.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.53,.99))
 save('Fig3_descriptor_ladder',f)

def frames():
 return {k:pd.read_csv(R/f'paper_run/subspace/{k}_all.csv') for k in ['resid','dim','dim_curve']}
def fig6():
 d=pd.read_csv(R/'paper_run/spec/spec_rows.csv').query("pool == 'mean' and baseline == 'randinit' and metric == 'AP'");d=d[d.ckpt.str.contains('_s42_')].rename(columns={'ckpt':'source','task':'dataset'})
 b=pd.read_csv(R/'paper_assets/Table4b_GB1_paired_differences.csv');b=b[b.source.str.contains('_s42_')].rename(columns={'split':'dataset'})
 f,axes=plt.subplots(2,1,figsize=(7.1,7.6));f.subplots_adjust(left=.33,right=.97,bottom=.08,top=.94,hspace=.48)
 for ax,data,title,xlab in [(axes[0],d,'a  Peptide classification','ΔAP versus random ensemble'),(axes[1],b,'b  GB1 extrapolation','ΔSpearman ρ versus random ensemble')]:
  data=data.sort_values(['dataset','source']).reset_index(drop=True)
  for i,row in data.iterrows():
   c=C[0] if row.lo>0 else C[1] if row.hi<0 else '#777777';ax.plot([row.lo,row.hi],[i,i],color=c,lw=1.7);ax.plot(row.delta,i,'o',color=c,ms=4)
  labs=[row.dataset.replace('_',' ').upper()+' | '+m.label(row.source).replace('New MAE ','') for _,row in data.iterrows()]
  ax.set(yticks=range(len(data)),yticklabels=labs,title=title,xlabel=xlab);ax.tick_params(axis='y',labelsize=8);ax.invert_yaxis();ax.axvline(0,color='black',ls='--',lw=.9);ax.xaxis.grid(True,ls='--',alpha=.3)
 save('Fig6_regime_effects',f)


from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle
O=R/'paper_figures_redesigned'; O.mkdir(exist_ok=True)

def save(name,fig):
 fig.savefig(O/(name+'.pdf'))
 fig.savefig(O/(name+'.svg'))
 fig.savefig(O/(name+'.png'),dpi=600)
 fig.savefig(O/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(O/(name+'_preview.png'),dpi=150)
 plt.close(fig)

def fig1():
 orig=pd.read_csv(R/'alldataset.csv').Sequence.astype(str);clean=pd.read_csv(R/'decontam_out_sne/sets/corpus_clean.csv').Sequence.astype(str)
 a=pd.read_csv(R/'decontam_out_sne/sets/eval_amp.csv');known=set(pd.read_csv(R/'alldataset_old.csv').Sequence)
 f,axes=plt.subplots(1,2,figsize=(7.1,4.4));f.subplots_adjust(left=.04,right=.98,top=.82,bottom=.32,wspace=.37)
 ax=axes[0];ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');ax.set_title('a  Corpus audit')
 def box(x,y,w,h,text,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',fc=color,ec='#526474',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
 box(.18,.73,.64,.25,f'Original corpus\n{len(set(orig)):,}', '#E5EFF8')
 box(.03,.20,.43,.28,f'Removed\n{len(set(orig)-set(clean)):,}', '#FCE9DF')
 box(.55,.20,.43,.28,f'Retained\n{len(set(clean)):,}', '#E1F2EB')
 for x in [.245,.765]:ax.add_patch(FancyArrowPatch((.5,.72),(x,.50),arrowstyle='-|>',mutation_scale=13,color='#526474',lw=1.2))
 ax.text(.5,.04,'Unique-sequence set counts',ha='center',fontsize=9,color='#526474')
 pos=a[a.Label.eq('AMP')].Sequence.str.len();neg=a[a.Label.ne('AMP')].Sequence.str.len();before=orig[~orig.isin(known)].str.len()
 for v,n,c,ls in [(pos,'AMP positive',C[0],'-'),(before,'Unmatched pool',C[1],'--'),(neg,'Matched negative',C[2],'-')]:
  x=np.sort(v);axes[1].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=n,color=c,ls=ls,lw=1.6)
 axes[1].set(xlabel='Length (amino acids)',ylabel='Cumulative proportion',title=f'b  Length matching\nKS = {m.ks_2samp(pos,neg).statistic:.4f}',xlim=(0,105));m.add_ygrid(axes[1]);axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False,fontsize=8)
 save('Fig1_audit',f)

def fig4():
 fr=frames();d=fr['resid'].query("set == 'eval_amp' and span == 'comp_quad'").set_index('source')
 names=['randinit_d256_s0']+sorted(n for n in d.index if '_s42_' in n)+['ckpt_ce_alltok_ls0.1_1007_162139_mr0.75_d256']
 labs=['Random s0','New mr=.20','New mr=.50','New mr=.75','Old CE/all/LS']
 f,axes=plt.subplots(2,1,figsize=(7.1,7.6));f.subplots_adjust(left=.12,right=.98,top=.94,bottom=.17,hspace=.52)
 for j,(col,name) in enumerate(zip(['AP_full','AP_span','AP_ctrl','AP_resid'],['Full','Projection','PCA control','Residual'])):
  axes[0].bar(np.arange(5)+(j-1.5)*.19,d.loc[names,col],width=.18,color=C[j],label=name,zorder=3)
 axes[0].set(xticks=range(5),xticklabels=labs,ylabel='AMP average precision',title='a  Subspace attribution',ylim=(0,.95));m.add_ygrid(axes[0]);axes[0].legend(loc='upper center',ncol=4,frameon=False,fontsize=8)
 for i,(n,l) in enumerate(zip(names,labs)):
  v=fr['dim_curve'].query("set == 'eval_amp' and basis == 'pca'");v=v[v.source.eq(n)].sort_values('k');s=fr['dim'].query("set == 'eval_amp'").set_index('source').loc[n]
  axes[1].plot(v.k,v.AP,marker='o',ms=3,lw=1.4,color=C[i],label=f'{l}: erank {s.erank:.1f}, variance k95 {s.k_var95:.0f}')
 axes[1].set(xscale='log',xlabel='Retained PCA dimensions',ylabel='AMP average precision',title='b  Performance versus retained dimensions');m.add_ygrid(axes[1]);axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,fontsize=8,frameon=False)
 save('Fig4_subspace_dimensions',f)

def fig5():
 f,axes=plt.subplots(3,1,figsize=(7.1,6.6));f.subplots_adjust(left=.23,right=.85,top=.91,bottom=.10,hspace=.82)
 titles=['a  AFAB / ACP: shared AB controls','b  AFAB / AMP: main evaluation sets','c  ACP / AMP: main evaluation sets']
 for ax,pair,title in zip(axes,['afab_acp','afab_amp','acp_amp'],titles):
  d=pd.read_csv(R/f'paper_run/axis/{pair}_aac.csv').set_index('metric');keys=[['self_a','a_to_b'],['b_to_a','self_b']];mat=np.array([[d.loc[k,'mean'] for k in row] for row in keys]);a,b=pair.upper().split('_')
  image=ax.imshow(mat,cmap='RdBu',vmin=.15,vmax=.85,aspect='auto')
  for i in range(2):
   for j in range(2):
    r=d.loc[keys[i][j]];ax.text(j,i,f'{r["mean"]:.3f}\n[{r["lo"]:.3f}, {r["hi"]:.3f}]',ha='center',va='center',fontsize=10,color='white' if abs(r['mean']-.5)>.24 else '#182636')
  ax.set(xticks=[0,1],xticklabels=[a,b],yticks=[0,1],yticklabels=[a,b],ylabel='Source',title=title);ax.tick_params(length=0)
  for sp in ax.spines.values():sp.set_visible(False)
 axes[-1].set_xlabel('Target task');cax=f.add_axes([.89,.28,.025,.43]);cb=f.colorbar(image,cax=cax);cb.set_label('Out-of-bag AUC');cb.set_ticks([.2,.35,.5,.65,.8])
 save('Fig5_axis_transfer',f)

def workflow():
 f=plt.figure(figsize=(7.1,7.8));ax=f.add_axes([0,0,1,1]);ax.set(xlim=(0,100),ylim=(0,110));ax.axis('off')
 ink='#253849';blue='#0072B2';green='#008565';orange='#BD641B'
 def txt(x,y,s,size=9,weight='normal',color=ink,ha='center'):
  ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va='center',linespacing=1.45)
 def box(x,y,w,h,fc='#F2F6FA',ec='#B5C3D0'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.5,rounding_size=1.2',fc=fc,ec=ec,lw=.8))
 def arrow(x1,y1,x2,y2,dash=False):
  ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,color='#647786',lw=1.2,linestyle='--' if dash else '-'))
 def database(x,y,color):
  ax.add_patch(Rectangle((x-4,y-3),8,6,fc=color,ec=ink,lw=.7));ax.add_patch(Ellipse((x,y-3),8,2.3,fc=color,ec=ink,lw=.7));ax.add_patch(Ellipse((x,y+3),8,2.3,fc=color,ec=ink,lw=.7))
  for yy in [y-1,y+1]:ax.plot([x-3.8,x+3.8],[yy,yy],color='white',lw=.8)
 def network(x,y):
  for yy in [-2,2]:
   for zz in [-3,0,3]:ax.plot([x-4,x],[y+yy,y+zz],color='#8C9DAB',lw=.7)
  for yy in [-3,0,3]:
   for zz in [-2,2]:ax.plot([x,x+4],[y+yy,y+zz],color='#8C9DAB',lw=.7)
  for xx,ys in [(x-4,[-2,2]),(x,[-3,0,3]),(x+4,[-2,2])]:
   for yy in ys:ax.add_patch(Circle((xx,y+yy),.8,fc='#66B5D4',ec=ink,lw=.6))
 def chart(x,y):
  ax.plot([x-4,x-4,x+5],[y+4,y-3,y-3],color=ink,lw=.8)
  for i,h in enumerate([3,5,4,6]):ax.add_patch(Rectangle((x-3+i*2,y-3),1.2,h,fc=[blue,orange,green,'#A77CB5'][i],ec='none'))
 txt(50,106,'Audit of masked pretraining for peptide representations',12,'bold')
 # Three-column data row
 txt(4,99,'1  Data and provenance',10,'bold',ha='left')
 box(3,77,29,18);database(8.5,89,'#B9D7EB');txt(22,89,'Pretraining\ncorpus',9,'bold');txt(17.5,81.5,'53,167 sequences\nAll audited MAEs trained here',8)
 box(35.5,77,29,18);database(41,89,'#BDE0D0');txt(54,89,'Peptide tasks',9,'bold');txt(50,81.5,'AMP 8,650 · AFAB 2,266\nACP 1,000',8.5)
 box(68,77,29,18);database(73.5,89,'#E8D7B7');txt(86,89,'GB1\nbenchmark',9,'bold');txt(82.5,81.5,'Three extrapolation splits\n8,733 variants per split',8.5)
 box(3,66.5,94,7,fc='#FFF8EE',ec='#D6B58C');txt(50,70,'Overlap audit: 9,321 removed → 43,846 retained\nRetained corpus is an audit output, not the MAE training corpus',8.5)
 arrow(17.5,77,17.5,74,True)
 # Representation bank
 txt(4,61,'2  Frozen representation bank',10,'bold',ha='left')
 box(3,40,94,17)
 for x,title,detail in [(16,'Descriptors','Composition, physicochemical\nand local-order features'),(40,'Random encoders','Same architecture\n3 initialization seeds'),(64,'MAE checkpoints','8 objective variants\n+ 3 masking ratios'),(87,'ESM-2','8M / 150M\nPotential corpus overlap')]:
  if title=='Descriptors':
   for j,h in enumerate([3,5,4,6]):ax.add_patch(Rectangle((x-4+j*2,49),1.3,h,fc=[blue,orange,green,'#A77CB5'][j],ec='none'))
  else: network(x,52)
  txt(x,46,title,9,'bold');txt(x,41.8,detail,7.6)
 # Audit is a side annotation, not a training-data processing stage.
 ax.plot([1.2,1.2],[84,49],color='#647786',lw=1)
 arrow(3,84,1.2,84);arrow(1.2,49,3,49)
 ax.plot([98.8,98.8],[96.5,30.5],color='#647786',lw=1)
 ax.plot([50,50,98.8],[95,96.5,96.5],color='#647786',lw=1)
 arrow(97,84,98.8,84);arrow(98.8,30.5,87.5,30.5)
 txt(50,58.8,'Fixed feature sources evaluated on downstream tasks',8,color='#647786')
 # readout
 box(13,26,74,9,fc='#EAF4F0',ec='#80B59F');txt(50,32,'Frozen features → linear readouts',10,'bold');txt(50,28.5,'Peptides: grouped cross-validation  |  GB1: fixed-split ridge regression',8)
 arrow(50,39,50,36)
 # analysis
 txt(4,22,'3  Complementary analyses',10,'bold',ha='left')
 for x,title,detail in [(3,'Benchmark & robustness','Descriptor ladder; 396 specs\nPaired bootstrap contrasts'),(35.5,'Subspace & spectrum','Projection / residual / PCA\nEffective rank and recovery'),(68,'Cross-task\ntransfer','Pair-specific task controls\nComposition-axis alignment')]:
  box(x,4,29,14);chart(x+5,12);txt(x+18,13,title.replace(' & ',' &\n'),8.5,'bold');txt(x+14.5,6.5,detail,7.8)
 for x in [17.5,50,82.5]:
  ax.plot([x,x],[24.5,19.7],color='#647786',lw=1)
  arrow(x,19.7,x,18.5)
 ax.plot([17.5,82.5],[24.5,24.5],color='#647786',lw=1);ax.plot([50,50],[26,24.5],color='#647786',lw=1)
 save('Fig0_study_workflow',f)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Render audited paper figures without rerunning experiments.')
 parser.add_argument('--only',nargs='+',choices=['workflow','fig1','fig2','fig3','fig4','fig5','fig6'],default=['workflow','fig1','fig2','fig3','fig4','fig5','fig6'])
 args=parser.parse_args()
 for name in args.only:
  print(name,flush=True);globals()[name]()
 print(O)
