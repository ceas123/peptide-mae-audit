
from __future__ import annotations

import argparse
import os
import pandas as pd

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import probe_suite as ps


def load_set(path, pos_label):
    d = ps.read_csv_any(path)
    cm = {str(c).strip().lower(): c for c in d.columns}
    seqs = d[cm["sequence"]].astype(str).values
    lab = d[cm["label"]].astype(str).values
    lv = sorted(set(lab))
    p = pos_label if pos_label in lv else lv[1]
    return seqs, (lab == p).astype(np.float64), p


def build(kind, seqs):
    if kind == "aac":
        return ps.aac_freq(list(seqs))
    if kind == "class_quad":
        return ps.class_quad_matrix(list(seqs), False)
    if kind == "aac_off_nolen":
        return ps.aac_quad_matrix(list(seqs), False, diagonal=False)
    raise ValueError(kind)


def direction(Xs, y, C):
    clf = LogisticRegression(C=C / len(y), max_iter=5000, solver="lbfgs")
    clf.fit(Xs, y)
    w = clf.coef_.ravel()
    return w / max(np.linalg.norm(w), 1e-12)


def auc_dir(Z, w, y, flip):
    s = -(Z @ w) if flip else (Z @ w)
    return ps.rank_auc(s[y == 1], s[y == 0])


def oob(n, idx):
    m = np.ones(n, dtype=bool)
    m[idx] = False
    return np.where(m)[0]


def two_class(y, idx):
    return len(idx) > 0 and len(set(y[idx])) == 2


def ci(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_a", required=True)
    ap.add_argument("--pos_a", default="")
    ap.add_argument("--set_b", required=True)
    ap.add_argument("--pos_b", default="")
    ap.add_argument("--feature", default="aac")
    ap.add_argument("--probe_C", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    sa, ya, pa = load_set(args.set_a, args.pos_a)
    sb, yb, pb = load_set(args.set_b, args.pos_b)
    Xa, Xb = build(args.feature, sa), build(args.feature, sb)

    sc = StandardScaler().fit(np.concatenate([Xa, Xb], axis=0))
    Za, Zb = sc.transform(Xa), sc.transform(Xb)
    na, nb = len(ya), len(yb)

    wa = direction(Za, ya, args.probe_C)
    wb = direction(Zb, yb, args.probe_C)
    cos_full = float(wa @ wb)

    rng = np.random.RandomState(args.seed)
    cos_b, sa_b, sb_b, ab_b, ba_b = [], [], [], [], []
    for _ in range(args.n_boot):
        ia = rng.randint(0, na, na)
        ib = rng.randint(0, nb, nb)
        if not (two_class(ya, ia) and two_class(yb, ib)):
            continue
        oa, ob = oob(na, ia), oob(nb, ib)
        if not (two_class(ya, oa) and two_class(yb, ob)):
            continue
        va = direction(Za[ia], ya[ia], args.probe_C)
        vb = direction(Zb[ib], yb[ib], args.probe_C)
        cos_b.append(float(va @ vb))
        fa = auc_dir(Za[oa], va, ya[oa], False) < 0.5
        fb = auc_dir(Zb[ob], vb, yb[ob], False) < 0.5
        sa_b.append(auc_dir(Za[oa], va, ya[oa], fa))
        sb_b.append(auc_dir(Zb[ob], vb, yb[ob], fb))
        ab_b.append(auc_dir(Zb[ob], va, yb[ob], fa))
        ba_b.append(auc_dir(Za[oa], vb, ya[oa], fb))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        records = []
        for name, values in [("cos_signed", cos_b), ("cos_absolute", np.abs(cos_b)), ("self_a", sa_b), ("self_b", sb_b), ("a_to_b", ab_b), ("b_to_a", ba_b)]:
            mean, lo, hi = ci(values)
            records.append(dict(feature=args.feature, set_a=args.set_a, set_b=args.set_b, pos_a=pa, pos_b=pb, metric=name, mean=mean, lo=lo, hi=hi, n_boot=len(values), seed=args.seed))
        pd.DataFrame(records).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        stem = os.path.splitext(args.out_csv)[0]
        pd.DataFrame(dict(coordinate=np.arange(len(wa)), weight_a=wa, weight_b=wb)).to_csv(stem + "_weights.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(dict(cos_signed=cos_b, self_a=sa_b, self_b=sb_b, a_to_b=ab_b, b_to_a=ba_b)).to_csv(stem + "_bootstrap.csv", index=False, encoding="utf-8-sig")
    cos_m, cos_lo, cos_hi = ci(cos_b)
    sa_m, sa_lo, sa_hi = ci(sa_b)
    sb_m, sb_lo, sb_hi = ci(sb_b)
    ab_m, ab_lo, ab_hi = ci(ab_b)
    ba_m, ba_lo, ba_hi = ci(ba_b)

    print("=" * 80)
    print("跨任务探针轴对齐")
    print("=" * 80)
    print(f"  特征基        {args.feature}  dim={Xa.shape[1]}  "
          f"C_eff=C/n (C={args.probe_C})")
    print(f"  A  {args.set_a}  正类={pa}  n={na}  n_pos={int(ya.sum())}")
    print(f"  B  {args.set_b}  正类={pb}  n={nb}  n_pos={int(yb.sum())}")
    print(f"  有效 bootstrap: {len(cos_b)}/{args.n_boot}")
    print("")
    print(f"  轴夹角 |cos|  全样本 {abs(cos_full):.4f}  有符号 {cos_full:+.4f}")
    print(f"               OOB 均值 {abs(cos_m):.4f}  "
          f"95%CI [{cos_lo:+.4f}, {cos_hi:+.4f}]")
    print("")
    print("  AUC 均为 OOB 样本外估计，自集与迁移同口径可比：")
    print(f"  自集  A  {sa_m:.4f}  95%CI [{sa_lo:.4f}, {sa_hi:.4f}]")
    print(f"        B  {sb_m:.4f}  95%CI [{sb_lo:.4f}, {sb_hi:.4f}]")
    print(f"  迁移  A轴→B  {ab_m:.4f}  95%CI [{ab_lo:.4f}, {ab_hi:.4f}]")
    print(f"        B轴→A  {ba_m:.4f}  95%CI [{ba_lo:.4f}, {ba_hi:.4f}]")
    print("")
    print("  判据: |cos| 的 CI 不跨 0 且 > 0.7 → 两任务共用同一条组成轴，")
    print("        天花板相同更可能是同一根轴在饱和，而非各自的可学结构")
    print("        迁移 AUC 的 CI 明显低于自集 CI → 轴不通用，天花板相同是巧合")


if __name__ == "__main__":
    main()
