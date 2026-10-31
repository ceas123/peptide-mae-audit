
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import probe_suite as ps

BOOT_CACHE = {}
METRIC_CACHE = {}


def load_oof(path, pos):
    d = ps.read_csv_any(path)
    cm = {str(c).strip().lower(): c for c in d.columns}
    lab = d[cm["label"]].astype(str).values
    y = (lab == pos).astype(np.float64)
    if y.sum() == 0:
        raise RuntimeError(f"{path} 中无正类 {pos}，标签={sorted(set(lab))}")
    cl = d[cm["cluster"]].values if "cluster" in cm else np.arange(len(d))
    meta = {cm.get("sequence", "Sequence"), cm["label"],
            cm.get("cluster", "cluster")}
    srcs = {c: d[c].astype(float).values for c in d.columns if c not in meta}
    return y, cl, srcs


def baseline_score(srcs, kind, n):
    if kind == "prevalence":
        return np.zeros(n)
    if kind == "randinit":
        cols = [v for k, v in srcs.items() if k.startswith("randinit")]
        if not cols:
            return None
        return np.mean(np.column_stack(cols), axis=1)
    return srcs.get(kind)


def metric(y, s, which):
    if which == "AUC":
        return ps.rank_auc(s[y == 1], s[y == 0])
    return ps.average_precision(y, s)


def cluster_boot(cl, n_boot, seed):
    key = (cl.tobytes(), n_boot, seed)
    if key in BOOT_CACHE:
        yield from BOOT_CACHE[key]
        return
    uniq = np.unique(cl)
    members = [np.where(cl == g)[0] for g in uniq]
    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(members), len(members))
        indices.append(np.concatenate([members[i] for i in pick]))
    BOOT_CACHE[key] = indices
    yield from indices


def delta_ci(y, sa, sb, cl, which, n_boot, seed):
    dm = metric(y, sa, which) - metric(y, sb, which)
    arrays = []
    for scores in (sa, sb):
        key = (scores.tobytes(), which, n_boot, seed)
        if key not in METRIC_CACHE:
            values = []
            for idx in cluster_boot(cl, n_boot, seed):
                yy = y[idx]
                if yy.sum() == 0 or yy.sum() == len(yy):
                    continue
                values.append(metric(yy, scores[idx], which))
            METRIC_CACHE[key] = np.asarray(values)
        arrays.append(METRIC_CACHE[key])
    ds = arrays[0] - arrays[1]
    lo, hi = np.percentile(ds, [2.5, 97.5]) if len(ds) else (np.nan, np.nan)
    verdict = ("正显著" if lo > 0 else "负显著" if hi < 0 else "不显著")
    return float(dm), float(lo), float(hi), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--baselines", default="randinit,aac_off_nolen,prevalence")
    ap.add_argument("--metrics", default="AUC,AP")
    ap.add_argument("--probe_cols", default="ckpt")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="spec_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bl = [x.strip() for x in args.baselines.split(",") if x.strip()]
    mt = [x.strip() for x in args.metrics.split(",") if x.strip()]
    pc = tuple(x.strip() for x in args.probe_cols.split(",") if x.strip())

    specs = []
    for item in args.specs.split(","):
        p, pos, task, pool = item.split("|")
        specs.append((p.strip(), pos.strip(), task.strip(), pool.strip()))

    rows = []
    for path, pos, task, pool in specs:
        BOOT_CACHE.clear()
        METRIC_CACHE.clear()
        y, cl, srcs = load_oof(path, pos)
        n = len(y)
        prev = float(y.mean())
        ckpts = [c for c in srcs if c.startswith(pc)]
        for base in bl:
            sb = baseline_score(srcs, base, n)
            if sb is None:
                continue
            for which in mt:
                base_m = metric(y, sb, which)
                for ck in ckpts:
                    dm, lo, hi, vd = delta_ci(y, srcs[ck], sb, cl, which,
                                              args.n_boot, args.seed)
                    print(f"Completed {task} {pool} {base} {which} {ck}", flush=True)
                    rows.append({
                        "task": task, "pool": pool, "baseline": base,
                        "metric": which, "ckpt": ck, "prev": prev,
                        "ckpt_val": metric(y, srcs[ck], which),
                        "base_val": base_m, "delta": dm,
                        "lo": lo, "hi": hi, "verdict": vd})

        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "spec_rows.partial.csv"), index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "spec_rows.csv"),
              index=False, encoding="utf-8-sig")

    print("=" * 92)
    print("规格曲线：预训练增益(ckpt − 基线)随协议翻转")
    print("=" * 92)
    print(f"规格数 {len(df)}  = 任务×池化×基线×指标×ckpt   自助 n_boot={args.n_boot}")
    g = (df.groupby(["pool", "baseline", "metric"])
         .agg(n=("delta", "size"),
              pos_sig=("verdict", lambda s: int((s == "正显著").sum())),
              neg_sig=("verdict", lambda s: int((s == "负显著").sum())),
              ns=("verdict", lambda s: int((s == "不显著").sum())),
              mean_delta=("delta", "mean")).reset_index())
    for _, r in g.iterrows():
        print(f"  pool={r['pool']:<5s} base={r['baseline']:<14s} "
              f"{r['metric']:<3s}  均值Δ={r['mean_delta']:+.4f}  "
              f"正显著 {r['pos_sig']:2d} / 负显著 {r['neg_sig']:2d} / "
              f"不显著 {r['ns']:2d}  (共{r['n']})")
    print("")
    print("  读法：同一组 ckpt，换 pool(cls↔mean) 或换 base(randinit↔prevalence)，")
    print("        正显著↔负显著计数翻转 → 增益非稳定可测，是协议选择的联合产物")
    print(f"\n落盘: {args.out_dir}/spec_rows.csv")


if __name__ == "__main__":
    main()
