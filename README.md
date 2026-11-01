# When Does Masked Pretraining Help Peptide Representation?
Official source code, curated benchmark datasets, and pretrained checkpoints for the study: "When Does Masked Pretraining Help Peptide Representation? Protocol Sensitivity, Subspace Dilution, and Task-Regime Reversal".

## Overview
Self-supervised masked pretraining is widely adopted in biological sequence modeling, yet apparent downstream gains are frequently confounded by homologous data leakage, sequence-length disparities, and evaluation protocol choices. Using a lightweight one-dimensional Masked Autoencoder (1D-MAE, 7.96M parameters, D=256) with a fully audited training corpus, this repository provides a systematic benchmarking and mechanistic audit framework across three bioactive peptide classification tasks (AMP, AFAB, ACP) and three protein mutational fitness extrapolation splits (FLIP GB1).

## Pretrained Checkpoints
All 11 pretrained 1D-MAE checkpoints (D=256) evaluated in our study are publicly available in Releases (v1.0.0):

*   **Orthogonal 2×2×2 Objective Cohort (r=0.75, 8 models):** {ce, focal} × {maskonly, alltok} × {ls0.1, nols} (ckpt_*_mr0.75_d256.pt).
*   **Controlled Mask-Ratio Gradient Cohort (seed=42, 3 models):** r∈{0.20, 0.50, 0.75} (ckpt_ce_maskonly_ls0.1_s42_mr{0.2,0.5,0.75}_d256.pt).
*(Download the .pt files from the Release page and place them in your working directory or runs/ folder before running checkpoint probes.)*

## Repository Contents

### Curated Peptide Benchmarks
*   **eval_amp.csv** — Length-stratified antimicrobial vs. non-antimicrobial peptide benchmark (n=8,650; 5-aa bin matched, KS=0.0245).
*   **eval_afab.csv** — Mutually exclusive antifungal vs. antibacterial peptide benchmark (n=2,266).
*   **eval_acp.csv** — Length-matched anticancer vs. antibacterial peptide benchmark (n=1,000).

### Corpus Construction & Decontamination Audit
*   **cleanSNE.py** — Extracts mutually exclusive antifungal and antibacterial peptide records from DRAMP.
*   **make_acp_task.py** — Constructs the length-matched ACP vs. non-ACP benchmark.
*   **build_corpus.py** — Performs 9-mer sliding-window homology auditing (C9 ≥ 0.9), builds length-matched negative sets, and generates the decontaminated corpus.

### 1D-MAE Architecture & Pretraining
*   **dataset.py** — Amino acid tokenizer and fixed-length sequence dataset.
*   **model.py** — 1D-MAE backbone (6-layer encoder, 4-layer decoder, 8 attention heads, D=256), supporting deterministic unmasked feature extraction.
*   **pretrainold.py** / **pretrainnew.py** — Controlled pretraining scripts for cohort specifications.

### Representation Auditing & Mechanistic Probes
*   **probe_suite.py** — Unified frozen linear probe suite, 17-tier physicochemical descriptor ladder, and bootstrap MDE evaluation.
*   **spec_curve.py** — Multiverse specification curve meta-analysis.
*   **subspace_probe.py** — Orthogonal subspace attribution (Z_span vs. Z_resid) and spectral effective rank analysis.
*   **gb1_subspace_all.py** — Orthogonal subspace decomposition on FLIP GB1 mutational fitness extrapolation tasks (quadratic composition projection vs. residual).
*   **axis_align.py** — Cross-task discriminative axis cosine alignment and OOB transfer evaluation.
*   **flip_probe.py** — Cross-regime Ridge regression probe on FLIP GB1 extrapolation splits.
*   **make_paper_figs_tables.py** — Reproduces manuscript figures and statistical tables.

## Usage Examples

# 1. Frozen linear probe & 17-tier physicochemical descriptor ladder
python probe_suite.py --set eval_amp.csv --pos_label AMP --pool mean --out_dir results/probe_amp

# 2. Orthogonal subspace attribution & spectral effective rank analysis
python subspace_probe.py --npz results/probe_amp/features_eval_amp_amp_mean.npz --set eval_amp.csv --pos_label AMP --spans comp_quad --out_dir results/subspace_amp

# 3. GB1 orthogonal subspace decomposition (Composition vs. Residual)
python gb1_subspace_all.py --split low_vs_high --batch 32 --n_boot 10000

# 4. Cross-regime FLIP GB1 fitness extrapolation probe
python flip_probe.py --set splits/low_vs_high.csv --pool mean --out_csv results/flip_low_vs_high.csv


## License

This project is released under the [MIT License](LICENSE).
