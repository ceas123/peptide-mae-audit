import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
import probe_suite as ps
from flip_probe import extract
from gb1_subspace_check import predict, rho


def intervals(y, predictions, n_boot):
    names = list(predictions)
    points = np.array([rho(y, predictions[k]) for k in names])
    boot = np.empty((n_boot, len(names)))
    rng = np.random.default_rng(42)
    for j in range(n_boot):
        ix = rng.integers(len(y), size=len(y))
        yy = rankdata(y[ix])
        yy -= yy.mean()
        for k, name in enumerate(names):
            pp = rankdata(predictions[name][ix])
            pp -= pp.mean()
            denom = np.linalg.norm(yy) * np.linalg.norm(pp)
            boot[j, k] = yy @ pp / denom if denom else np.nan
    result = {f'rho_{name}': points[k] for k, name in enumerate(names)}
    for first, second in [('full', 'span'), ('full', 'composition'), ('residual', 'composition')]:
        i, j = names.index(first), names.index(second)
        key = f'{first}_minus_{second}'
        delta = boot[:, i] - boot[:, j]
        result[key] = points[i] - points[j]
        result[key + '_lo'], result[key + '_hi'] = np.nanpercentile(delta, [2.5, 97.5])
        if first == 'full':
            tail = 2.5 / 66
            result[key + '_sim_lo'], result[key + '_sim_hi'] = np.nanpercentile(delta, [tail, 100 - tail])
    return result


def decompose(z, f, tr, te, y):
    sf = StandardScaler().fit(f[tr])
    a = np.column_stack([np.ones(tr.sum()), sf.transform(f[tr])])
    b = np.column_stack([np.ones(te.sum()), sf.transform(f[te])])
    sz = StandardScaler().fit(z[tr])
    ztr, zte = sz.transform(z[tr]), sz.transform(z[te])
    coef, _, rank, _ = np.linalg.lstsq(a, ztr, rcond=None)
    st, se = a @ coef, b @ coef
    rt, re = ztr - st, zte - se
    orth = np.linalg.norm(a.T @ rt) / max(np.linalg.norm(a) * np.linalg.norm(ztr), 1e-12)
    assert orth < 1e-9
    diagnostic = dict(design_rank=rank, orth_error=orth,
                      train_residual_fraction=np.sum(rt**2)/np.sum(ztr**2),
                      test_residual_fraction=np.sum(re**2)/np.sum(zte**2))
    predictions = {'full': predict(z[tr], z[te], y[tr]),
                   'span': predict(st, se, y[tr]),
                   'residual': predict(rt, re, y[tr]),
                   'composition': predict(f[tr], f[te], y[tr])}
    return predictions, diagnostic


def evaluate(source, seqs, f, y, tr, te, batch, n_boot):
    paths = [Path(source + '.pt'), Path('runs') / (source + '.pt')]
    hits = [p for p in paths if p.is_file()]
    assert len(hits) == 1, (source, hits)
    aa_map = {c: i + 1 for i, c in enumerate(ps.AA_STD)}
    aa_map.update({'X': 23, 'B': 22, 'U': 21})
    z = extract('ckpt', str(hits[0]), {}, seqs, 'cuda', batch, 'mean', 1,
                aa_map, lambda s: print(s, flush=True))
    z = np.asarray(z, dtype=np.float64)
    assert z.shape == (len(y), 256) and np.isfinite(z).all()
    predictions, diagnostic = decompose(z, f, tr, te, y)
    result = intervals(y[te], predictions, n_boot)
    result.update(diagnostic)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['low_vs_high', 'two_vs_rest', 'three_vs_rest'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--n_boot', type=int, default=10000)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    frame = ps.read_csv_any(f'splits/splits/{args.split}.csv')
    cols = {str(c).lower().strip(): c for c in frame.columns}
    seqs = np.array([''.join(str(s).split()).upper()[:100] for s in frame[cols['sequence']]])
    y = frame[cols['target']].to_numpy(dtype=float)
    split = frame[cols['set']].astype(str).str.lower().str.strip().to_numpy()
    tr, te = split == 'train', split == 'test'
    assert np.all(tr | te) and tr.any() and te.any() and np.isfinite(y).all()
    assert not set(seqs[tr]) & set(seqs[te])
    f = np.asarray(ps.aac_quad_matrix(list(seqs), False, diagonal=False), dtype=np.float64)
    assert np.isfinite(f).all()
    old = pd.read_csv(f'paper_run/flip/{args.split}.csv')
    models = old[old.source.str.startswith('ckpt_') & old.source.str.endswith('_d256')]
    assert len(models) == 11 and models.source.is_unique
    output = Path('paper_run/flip') / f'{args.split}_subspace_all.csv'
    rows = []
    print(f'[DATA] split={args.split} train={tr.sum()} test={te.sum()} models=11 bootstrap={args.n_boot}', flush=True)
    for _, model in models.iterrows():
        result = evaluate(model.source, seqs, f, y, tr, te, args.batch, args.n_boot)
        error = abs(result['rho_full'] - model.rho)
        assert error < 1e-5, (model.source, error)
        result.update(source=model.source, split=args.split, reproduction_error=error,
                      n_boot=args.n_boot, n_train=int(tr.sum()), n_test=int(te.sum()))
        rows.append(result)
        pd.DataFrame(rows).to_csv(output, index=False, encoding='utf-8-sig')
        print(f'[DONE] {len(rows)}/11 source={model.source} full={result["rho_full"]:.6f} span={result["rho_span"]:.6f} composition={result["rho_composition"]:.6f} residual={result["rho_residual"]:.6f} reproduction_error={error:.2e}', flush=True)
    print(f'[COMPLETE] output={output}', flush=True)


if __name__ == '__main__':
    main()
