# When Does Masked Pretraining Help Peptide Representation?

Official source code, curated benchmark datasets, and pretrained checkpoints for the study: **"When Does Masked Pretraining Help Peptide Representation? Protocol Sensitivity, Subspace Dilution, and Task-Regime Reversal"**.

## Overview

Self-supervised masked pretraining is widely adopted in biological sequence modeling, yet apparent downstream gains are frequently confounded by homologous data leakage, sequence-length disparities, and evaluation protocol choices. Using a lightweight one-dimensional Masked Autoencoder (**1D-MAE**, 7.96M parameters, $D = 256$) with a fully audited training corpus, this repository provides a systematic benchmarking and mechanistic audit framework across three bioactive peptide classification tasks (**AMP**, **AFAB**, **ACP**) and three protein mutational fitness extrapolation splits (**FLIP GB1**).

## Pretrained Checkpoints

All 11 pretrained 1D-MAE checkpoints ($D = 256$) evaluated in our study are publicly available in [**Releases (v1.0.0)**](https://github.com/ceas123/peptide-mae-audit/releases/tag/v1.0.0):
- **Orthogonal $2 \times 2 \times 2$ Objective Cohort ($r = 0.75$, 8 models)**: `{ce, focal}` $\times$ `{maskonly, alltok}` $\times$ `{ls0.1, nols}` (`ckpt_*_mr0.75_d256.pt`).
- **Controlled Mask-Ratio Gradient Cohort ($\text{seed} = 42$, 3 models)**: $r \in \{0.20, 0.50, 0.75\}$ (`ckpt_ce_maskonly_ls0.1_s42_mr{0.2,0.5,0.75}_d256.pt`).

*(Download the `.pt` files from the Release page and place them in your working directory or `runs/` folder before running checkpoint probes.)*

## Repository Contents

### Curated Peptide Benchmarks
- `eval_amp.csv` — Length-stratified antimicrobial vs. non-antimicrobial peptide benchmark ($n = 8,650$; 5-aa bin matched, $\text{KS} = 0.0245$).
- `eval_afab.csv` — Mutually exclusive antifungal vs. antibacterial peptide benchmark ($n = 2,266$).
- `eval_acp.csv` — Length-matched anticancer vs. antibacterial peptide benchmark ($n = 1,000$).

### Corpus Construction & Decontamination Audit
- `cleanSNE.py` — Extracts mutually exclusive antifungal and antibacterial peptide records from DRAMP.
- `make_acp_task.py` — Constructs the length-matched ACP vs. non-ACP benchmark.
- `build_corpus.py` — Performs 9-mer sliding-window homology auditing ($C_9 \ge 0.9$), builds length-matched negative sets, and generates the decontaminated corpus.

### 1D-MAE Architecture & Pretraining
- `dataset.py` — Amino acid tokenizer (`vocab_size = 24`: `PAD=0`, `CLS=1`, `MASK=2`, `UNK=3`, plus 20 standard amino acids) and fixed-length sequence dataset ($L_{\max} = 100$).
- `model.py` — 1D-MAE backbone (6-layer encoder, 4-layer decoder, 8 attention heads, $D = 256$), supporting deterministic full-sequence unmasked feature extraction (`encode_full`).
- `pretrainold.py` — Pretraining script for the $2 \times 2 \times 2$ orthogonal objective cohort ($r = 0.75$).
- `pretrainnew.py` — Controlled pretraining script across mask ratios $r \in \{0.20, 0.50, 0.75\}$.

### Representation Auditing & Mechanistic Probes
- `probe_suite.py` — Unified frozen linear probe suite (`C = 1.0`, 9-mer cluster-grouped $5 \times 5$ cross-validation), 17-tier physicochemical descriptor ladder, and paired bootstrap Minimum Detectable Effect (MDE) evaluation.
- `spec_curve.py` — Multiverse specification curve meta-analysis across 396 evaluation configurations using 1,000 cluster-bootstrap resamplings.
- `subspace_probe.py` — Orthogonal subspace attribution (`Z_span` vs. `Z_resid`), rank-matched PCA controls, participation ratio, effective rank (`erank`), and PCA/PLS dimensionality recovery curves.
- `axis_align.py` — Cross-task discriminative axis cosine alignment and out-of-bag (OOB) transfer ROC-AUC evaluation.
- `flip_probe.py` — Cross-regime Ridge regression probe ($\alpha = 1.0$) on FLIP GB1 mutational fitness extrapolation splits (`two_vs_rest`, `three_vs_rest`, `low_vs_high`).
- `make_paper_figs_tables.py` — Reproduces all manuscript figures and statistical summary tables from probe outputs.

## Requirements

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib openpyxl
```
*(Optional: install `fair-esm` to reproduce ESM-2 upper-bound baselines.)*

## Usage Examples

```bash
# 1. Frozen linear probe & 17-tier physicochemical descriptor ladder
python probe_suite.py --set eval_amp.csv --pos_label AMP --pool mean --feat_reps 1 --folds 5 --repeats 5 --out_dir results/probe_amp

# 2. Orthogonal subspace attribution & spectral effective rank analysis
python subspace_probe.py --npz results/probe_amp/features_eval_amp_amp_mean.npz --set eval_amp.csv --pos_label AMP --spans length,physchem,comp_quad,kmer2 --ctrl 1 --out_dir results/subspace_amp

# 3. Multiverse specification curve (396 configurations)
python spec_curve.py --specs "results/oof_amp_mean.csv|AMP|amp|mean,results/oof_amp_cls.csv|AMP|amp|cls" --out_dir results/spec --n_boot 1000

# 4. Cross-regime FLIP GB1 fitness extrapolation probe
python flip_probe.py --set splits/low_vs_high.csv --pool mean --alphas 1.0 --n_boot 1000 --out_csv results/flip_low_vs_high.csv
```

## License

This project is released under the [MIT License](LICENSE).
