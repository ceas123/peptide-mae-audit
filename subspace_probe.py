
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import probe_suite as ps

SPANS = ("length", "physchem", "comp_quad", "comp_quad_diag", "kmer2")
IDENT_R2 = 0.999
DUST_STD = 1e-6
PCA_CACHE = {}


def load_set(path, pos_label):
    d = ps.read_csv_any(path)
    cm = {str(c).strip().lower(): c for c in d.columns}
    if "sequence" not in cm or "label" not in cm:
        raise RuntimeError(f"{path} 需要 Sequence 与 Label 列: {list(d.columns)}")
    seqs = np.array(["".join(str(s).split()).upper()
                     for s in d[cm["sequence"]].values])
    lab = d[cm["label"]].astype(str).values
    lv = sorted(set(lab))
    if len(lv) != 2:
        raise RuntimeError(f"需要二分类标签，实际为 {lv}")
    up = {str(v).strip().upper(): v for v in lv}
    if pos_label in lv:
        p, note = pos_label, "显式指定"
    else:
        hit = [up[k] for k in ("AMP", "POS", "POSITIVE", "TRUE", "1") if k in up]
        if not hit:
            raise RuntimeError(f"标签为 {lv}，无法确定正类，请传 --pos_label")
        p, note = hit[0], "按名称自动取，AUC 普遍<0.5 说明方向反了"
    y = (lab == p).astype(np.float64)
    return seqs, y, p, note


def build_span(name, seqs):
    if name == "length":
        return np.array([[float(len(s))] for s in seqs], dtype=np.float64)
    if name == "physchem":
        return ps.physchem_matrix(seqs, True)
    if name == "comp_quad":
        return ps.aac_quad_matrix(seqs, True, diagonal=False)
    if name == "comp_quad_diag":
        return ps.aac_quad_matrix(seqs, True, diagonal=True)
    if name == "kmer2":
        return ps.kmer_matrix([str(s) for s in seqs], 2, 5)
    raise ValueError(f"未知 span {name}，可选 {SPANS}")


def design(Ftr, Fte):
    mu = Ftr.mean(0)
    sd = Ftr.std(0)
    sd[sd == 0.0] = 1.0
    return (np.column_stack([np.ones(len(Ftr)), (Ftr - mu) / sd]),
            np.column_stack([np.ones(len(Fte)), (Fte - mu) / sd]))


def span_rank(F):
    return int(np.linalg.matrix_rank(np.column_stack([np.ones(len(F)), F])))


def oof_variant(Z, F, y, folds, n_folds, C, variant):
    acc = np.zeros(len(y), dtype=np.float64)
    hit = np.zeros(len(y), dtype=np.float64)
    kept = []
    for f in range(n_folds):
        te = folds == f
        tr = ~te
        if tr.sum() < 20 or te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        if variant == "raw":
            Xtr, Xte = Z[tr], Z[te]
        else:
            A, B = design(F[tr], F[te])
            mu_z = Z[tr].mean(0)
            sd_z = Z[tr].std(0)
            sd_z[sd_z == 0.0] = 1.0
            Ztr = (Z[tr] - mu_z) / sd_z
            Zte = (Z[te] - mu_z) / sd_z
            coef = np.linalg.lstsq(A, Ztr, rcond=None)[0]
            if variant == "span":
                Xtr, Xte = A @ coef, B @ coef
            elif variant == "resid":
                Rtr, Rte = Ztr - A @ coef, Zte - B @ coef
                keep = Rtr.std(0) > DUST_STD
                if keep.any():
                    Xtr, Xte = Rtr[:, keep], Rte[:, keep]
                    kept.append(int(keep.sum()))
                else:
                    Xtr = np.zeros((len(Ztr), 1))
                    Xte = np.zeros((len(Zte), 1))
                    kept.append(0)
            else:
                raise ValueError(variant)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        clf.fit(sc.transform(Xtr), y[tr])
        acc[te] += clf.predict_proba(sc.transform(Xte))[:, 1]
        hit[te] += 1.0
    n_keep = int(round(np.mean(kept))) if kept else int(Z.shape[1])
    return acc / np.maximum(hit, 1.0), n_keep


def oof_reduced(Z, y, folds, n_folds, C, kind, k):
    acc = np.zeros(len(y), dtype=np.float64)
    hit = np.zeros(len(y), dtype=np.float64)
    for f in range(n_folds):
        te = folds == f
        tr = ~te
        if tr.sum() < 20 or te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(Z[tr])
        A, B = sc.transform(Z[tr]), sc.transform(Z[te])
        kk = int(min(k, A.shape[1], A.shape[0] - 1))
        if kk < 1:
            continue
        if kind == "pca":
            mu = A.mean(0)
            cache_key = (id(Z), folds.tobytes(), f)
            if cache_key not in PCA_CACHE:
                PCA_CACHE[cache_key] = np.linalg.svd(A - mu, full_matrices=False)[2]
            V = PCA_CACHE[cache_key][:kk].T
            At, Bt = (A - mu) @ V, (B - mu) @ V
        else:
            pls = PLSRegression(n_components=kk, scale=False).fit(A, y[tr])
            W = pls.x_weights_
            W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
            At, Bt = A @ W, B @ W
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        clf.fit(At, y[tr])
        acc[te] += clf.predict_proba(Bt)[:, 1]
        hit[te] += 1.0
    return acc / np.maximum(hit, 1.0)


def ap_auc(y, s):
    return (ps.average_precision(y, s),
            ps.rank_auc(s[y == 1], s[y == 0]))


def boot_pair(y, sa, sb, ipos, ineg, n_boot, seed):
    rng = np.random.RandomState(seed)
    out = np.empty((n_boot, 2), dtype=np.float64)
    for b in range(n_boot):
        pi = ipos[rng.randint(0, len(ipos), len(ipos))]
        ni = ineg[rng.randint(0, len(ineg), len(ineg))]
        idx = np.concatenate([pi, ni])
        yy = y[idx]
        out[b, 0] = (ps.average_precision(yy, sa[idx])
                     - ps.average_precision(yy, sb[idx]))
        out[b, 1] = (ps.rank_auc(sa[idx][yy == 1], sa[idx][yy == 0])
                     - ps.rank_auc(sb[idx][yy == 1], sb[idx][yy == 0]))
    return out


def judge(d, lo, hi, mde):
    if lo <= 0.0 <= hi:
        return "不可分辨"
    ratio = abs(d) / mde if mde > 1e-12 else float("inf")
    return "弱" if ratio < 1.5 else "可分辨"


def pair_stats(d_boot, dm, zz):
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    mde = float(zz * d_boot.std(ddof=1))
    return float(dm), float(lo), float(hi), mde, judge(dm, lo, hi, mde)


def effective_dim(Z):
    A = Z - Z.mean(0)
    sd = A.std(0)
    keep = sd > 0
    A = A[:, keep] / sd[keep]
    s = np.linalg.svd(A, compute_uv=False)
    lam = (s ** 2) / max(len(A) - 1, 1)
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    erank = float(np.exp(-(p * np.log(p)).sum()))
    k95 = int(np.searchsorted(np.cumsum(p), 0.95) + 1)
    return pr, erank, k95, int(keep.sum())


def multi_r2(Z, F):
    A, _ = design(F, F[:1])
    mu_z = Z.mean(0)
    sd_z = Z.std(0)
    sd_z[sd_z == 0.0] = 1.0
    Zs = (Z - mu_z) / sd_z
    coef = np.linalg.lstsq(A, Zs, rcond=None)[0]
    res = Zs - A @ coef
    ss_res = (res ** 2).sum(0)
    ss_tot = ((Zs - Zs.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(np.median(r2)), float(np.mean(r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--pos_label", default="")
    ap.add_argument("--spans", default="physchem,comp_quad")
    ap.add_argument("--sources", default="")
    ap.add_argument("--out_dir", default="subspace_out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--probe_C", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--k_grid",
                    default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256")
    ap.add_argument("--ctrl", type=int, default=1)
    ap.add_argument("--cluster_thr", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--min_dim", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zz = ps.norm_ppf(1 - args.alpha / 2) + ps.norm_ppf(args.power)
    seqs, y, pos_label, note = load_set(args.set, args.pos_label)
    n = len(y)
    sname = os.path.splitext(os.path.basename(args.set))[0]
    tag = f"_{args.tag}" if args.tag else ""

    data = np.load(args.npz)
    names = list(data.files)
    if args.sources:
        want = [x.strip() for x in args.sources.split(",") if x.strip()]
        names = [k for k in names if k in want]
    names = [k for k in names
             if data[k].ndim == 2 and data[k].shape[1] >= args.min_dim]
    bad = [k for k in names if data[k].shape[0] != n]
    if bad:
        raise RuntimeError(
            f"行数不匹配: set={n}，{bad[0]}={data[bad[0]].shape[0]}；"
            f"npz 必须与 eval CSV 同序、同长度")
    if not names:
        raise RuntimeError("没有可用来源")

    span_names = [x.strip() for x in args.spans.split(",") if x.strip()]
    span_F = {sp: build_span(sp, seqs) for sp in span_names}
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip().isdigit()]

    L = []
    def say(message):
        L.append(message)
        print(message, flush=True)
    say("=" * 108)
    say("子空间归属与有效维数")
    say("=" * 108)
    say(f"评测集 {sname}  n={n}  正类={pos_label} n_pos={int(y.sum())}  "
        f"n_neg={int(n - y.sum())}  流行率={y.mean():.4f}  [{note}]")
    say(f"npz={os.path.basename(args.npz)}  来源 {len(names)} 个  "
        f"折={args.folds}  重复={args.repeats}  probe_C={args.probe_C}  "
        f"n_boot={args.n_boot}  z={zz:.4f}  同维对照={'开' if args.ctrl else '关'}")
    say("span=" + "  ".join(
        f"{sp}(d={span_F[sp].shape[1]},rank={span_rank(span_F[sp])})"
        for sp in span_names))

    groups = ps.greedy_clusters(list(seqs), k=9, thr=args.cluster_thr)
    cl = set(groups.tolist())
    say(f"聚簇(k=9, thr={args.cluster_thr}): {len(cl)} 簇，"
        f"最大簇 {max(int((groups == g).sum()) for g in cl)} 条")
    folds_list = [ps.grouped_folds(groups, args.folds, args.seed + 1000 * r)
                  for r in range(args.repeats)]
    f0 = folds_list[0]
    ipos, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    prev = float(y.mean())

    arows, brows, crows = [], [], []
    for nm in names:
        PCA_CACHE.clear()
        Z = np.asarray(data[nm], dtype=np.float64)
        raw = np.zeros(n)
        for fd in folds_list:
            raw += oof_variant(Z, None, y, fd, args.folds, args.probe_C, "raw")[0]
        raw /= len(folds_list)
        ap_full, auc_full = ap_auc(y, raw)

        say("")
        say(f"[{nm}] dim={Z.shape[1]}  AP={ap_full:.4f}  AUC={auc_full:.4f}  "
            f"可学习量(AP−流行率)={ap_full - prev:+.4f}")

        for sp in span_names:
            F = span_F[sp]
            kc = min(span_rank(F), Z.shape[1])
            spa = np.zeros(n)
            res = np.zeros(n)
            n_res_dim = 0
            for fd in folds_list:
                s1, _ = oof_variant(Z, F, y, fd, args.folds, args.probe_C, "span")
                s2, nd = oof_variant(Z, F, y, fd, args.folds, args.probe_C, "resid")
                spa += s1
                res += s2
                n_res_dim = nd
            spa /= len(folds_list)
            res /= len(folds_list)
            ap_s, auc_s = ap_auc(y, spa)
            ap_r, auc_r = ap_auc(y, res)
            bs = boot_pair(y, spa, raw, ipos, ineg, args.n_boot, args.seed + 11)
            br = boot_pair(y, res, raw, ipos, ineg, args.n_boot, args.seed + 23)
            ds = pair_stats(bs[:, 0], ap_s - ap_full, zz)
            dr = pair_stats(br[:, 0], ap_r - ap_full, zz)
            r2_med, _ = multi_r2(Z, F)
            ident = (n_res_dim == 0)
            if ident:
                dr = (dr[0], dr[1], dr[2], dr[3], "恒等残差")
            share = (ap_s - prev) / max(ap_full - prev, 1e-12)

            ap_c = float("nan")
            if args.ctrl:
                sc_c = np.zeros(n)
                for fd in folds_list:
                    sc_c += oof_reduced(Z, y, fd, args.folds,
                                        args.probe_C, "pca", kc)
                sc_c /= len(folds_list)
                ap_c = ap_auc(y, sc_c)[0]

            say(f"  {sp:<14s} span  AP={ap_s:.4f}  ΔAP={ds[0]:+.4f} "
                f"[{ds[1]:+.4f},{ds[2]:+.4f}]  MDE={ds[3]:.4f} {ds[4]:>9s}  "
                f"占比={share:6.3f}  R²={r2_med:6.3f}")
            say(f"  {'':<14s} resid AP={ap_r:.4f}  ΔAP={dr[0]:+.4f} "
                f"[{dr[1]:+.4f},{dr[2]:+.4f}]  MDE={dr[3]:.4f} {dr[4]:>9s}  "
                f"dim={n_res_dim}  ctrl AP={ap_c:.4f}  "
                f"span−ctrl={ap_s - ap_c:+.4f}  resid−ctrl={ap_r - ap_c:+.4f}")

            arows.append({
                "set": sname, "source": nm, "span": sp,
                "dim": Z.shape[1], "dim_span": F.shape[1],
                "rank_span": span_rank(F), "dim_resid": n_res_dim,
                "r2_span": r2_med, "is_identity": bool(ident),
                "AP_full": ap_full, "AUC_full": auc_full,
                "AP_span": ap_s, "AUC_span": auc_s,
                "AP_resid": ap_r, "AUC_resid": auc_r,
                "AP_ctrl": ap_c, "k_ctrl": kc,
                "dAP_span": ds[0], "dAP_span_lo": ds[1], "dAP_span_hi": ds[2],
                "mde_span": ds[3], "verdict_span": ds[4],
                "dAP_resid": dr[0], "dAP_resid_lo": dr[1],
                "dAP_resid_hi": dr[2], "mde_resid": dr[3],
                "verdict_resid": dr[4], "span_share": share,
                "dAUC_span": float(np.mean(bs[:, 1])),
                "dAUC_resid": float(np.mean(br[:, 1])),
                "dAUC_resid_lo": float(np.percentile(br[:, 1], 2.5)),
                "dAUC_resid_hi": float(np.percentile(br[:, 1], 97.5)),
            })

        pr, er, k95_var, n_cl = effective_dim(Z)
        raw0, _ = oof_variant(Z, None, y, f0, args.folds, args.probe_C, "raw")
        ap0 = ap_auc(y, raw0)[0]
        target = 0.95 * ap0
        curves = {}
        for kind in ("pca", "pls"):
            for k in k_grid:
                if k >= Z.shape[1]:
                    continue
                apk = ap_auc(y, oof_reduced(Z, y, f0, args.folds,
                                            args.probe_C, kind, k))[0]
                curves[(kind, k)] = apk
                crows.append({"set": sname, "source": nm, "basis": kind,
                              "k": k, "AP": apk})
        k95, reached = {}, {}
        for kind in ("pca", "pls"):
            hits = [k for k in k_grid
                    if k < Z.shape[1] and curves.get((kind, k), -1) >= target]
            k95[kind] = min(hits) if hits else Z.shape[1]
            reached[kind] = bool(hits)
        say(f"  有效维数  PR={pr:8.2f}  erank={er:8.2f}  "
            f"PCA95%方差维数={k95_var}  非零列={n_cl}/{Z.shape[1]}")
        say(f"  AP(f0)={ap0:.4f}  追平 95%={target:.4f} 需  "
            f"k_PCA={k95['pca']}{'' if reached['pca'] else '(未达)'}  "
            f"k_PLS={k95['pls']}{'' if reached['pls'] else '(未达)'}  "
            f"(全维 {Z.shape[1]})")
        brows.append({"set": sname, "source": nm, "dim": Z.shape[1],
                      "AP_full": ap_full, "AUC_full": auc_full,
                      "AP_f0": ap0, "AP_target95": target,
                      "PR": pr, "erank": er, "k_var95": k95_var,
                      "k95_pca": k95["pca"], "k95_pls": k95["pls"],
                      "k95_pca_reached": reached["pca"],
                      "k95_pls_reached": reached["pls"],
                      "n_active_cols": n_cl})

        pd.DataFrame(arows).to_csv(os.path.join(args.out_dir, f"resid_{sname}{tag}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(brows).to_csv(os.path.join(args.out_dir, f"dim_{sname}{tag}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(crows).to_csv(os.path.join(args.out_dir, f"dim_curve_{sname}{tag}.csv"), index=False, encoding="utf-8-sig")
    adf = pd.DataFrame(arows)
    adf.to_csv(os.path.join(args.out_dir, f"resid_{sname}{tag}.csv"),
               index=False, encoding="utf-8-sig")
    pd.DataFrame(brows).to_csv(
        os.path.join(args.out_dir, f"dim_{sname}{tag}.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(crows).to_csv(
        os.path.join(args.out_dir, f"dim_curve_{sname}{tag}.csv"),
        index=False, encoding="utf-8-sig")

    say("")
    say("=" * 108)
    say("[判读门]")
    say("=" * 108)
    for sp in span_names:
        sub = adf[adf["span"] == sp]
        if sub.empty:
            continue
        ident = sub["is_identity"]
        eff = sub[~ident]
        n_span = int((eff["verdict_span"] == "不可分辨").sum()) if len(eff) else 0
        n_res = int((eff["verdict_resid"] != "不可分辨").sum()) if len(eff) else 0
        say(f"  span={sp}: 恒等残差 {int(ident.sum())} 格（R²>{IDENT_R2}，"
            f"不可作证据）；有效 {len(eff)} 格中，"
            f"{n_span} 个的判别力落在此 span 内，{n_res} 个残差仍可分辨")
        if args.ctrl and len(eff):
            better = int((eff["AP_span"] > eff["AP_ctrl"]).sum())
            say(f"            同维 PCA 对照：{better}/{len(eff)} 格 span > ctrl"
                f"（span 有信息），其余为降维去噪")
    say("  若 AP_span ≈ AP_full 且 AP_resid 不可分辨 → 判别信号可由该 span"
        "线性读出")
    say("  若 AP_resid 可分辨且 span−ctrl > 0 → span 之外有表征内容，"
        "按 ΔAP_resid 排序决定是否进入解码器反演")
    say("  注意：只检验线性可读性；非线性读出未测，"
        "结论不得写成『表征只含组成』")

    txt = "\n".join(L)
    with open(os.path.join(args.out_dir, f"subspace_report_{sname}{tag}.txt"),
              "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n落盘: {args.out_dir}")


if __name__ == "__main__":
    main()
