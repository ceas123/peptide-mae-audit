import argparse
import csv
import os
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import probe_suite as ps
import warnings
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)


ALPHAS = [0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
DEFAULT_SOURCES = ("esm:D:/创/esm2_t6_8M_UR50D,esm:D:/创/esm2_t30_150M_UR50D,"
                   "randinit:dim=256;seed=0,aac_off_nolen,zero")


def sp(a, b):
    return float(spearmanr(a, b)[0])


def extract(kind, path, kv, seqs, device, batch, pool, feat_reps, aa_map, say):
    if kind == "esm":
        X = ps.esm_features(list(seqs), path, device, batch)
        say(f"[{ps.source_tag(kind, path, kv)}] ESM 冻结特征 dim={X.shape[1]} "
            f"⚠️ UR50D 含记忆上界")
        return X
    if kind == "aac_off_nolen":
        X = ps.aac_quad_matrix(list(seqs), False, diagonal=False)
        say(f"[aac_off_nolen] 组成+非对角乘积 dim={X.shape[1]} "
            f"(GB1 突变体组成近恒定，预计信号极低)")
        return X
    if kind == "zero":
        say("[zero] 全零哨兵 dim=1 (ρ 应为 nan/≈0)")
        return np.zeros((len(seqs), 1))
    if kind in ("ckpt", "randinit"):
        if kind == "ckpt":
            ck = ps.torch_load_any(path)
            if isinstance(ck, dict) and "state_dict" in ck:
                state, cfg = ck["state_dict"], dict(ck.get("args", {}))
            elif isinstance(ck, dict) and "model_state_dict" in ck:
                state, cfg = ck["model_state_dict"], dict(ck.get("args", {}))
            else:
                state, cfg = ck, {}
            arch = ps.infer_arch(state, cfg)
            arch["encoder_head"] = arch.get("encoder_head") or 8
            arch["decoder_head"] = arch.get("decoder_head") or 8
            m = ps.build_mae(arch, device)
            m.load_state_dict(state, strict=True)
            tg = ps.source_tag(kind, path, kv)
        else:
            dim = int(kv.get("dim", 256))
            arch = {"dim": dim, "max_len": int(kv.get("max_len", 100)),
                    "vocab_size": int(kv.get("vocab_size", 24)),
                    "encoder_layer": int(kv.get("encoder_layer", 6)),
                    "decoder_layer": int(kv.get("decoder_layer", 4)),
                    "encoder_head": 8, "decoder_head": 8,
                    "mask_ratio": float(kv.get("mask_ratio", 0.75))}
            torch.manual_seed(int(kv.get("seed", 0)))
            m = ps.build_mae(arch, device)
            tg = ps.source_tag(kind, path, kv)
        tok = ps.encode_tokens(list(seqs), arch["max_len"], aa_map)
        snaps = ps.mae_features_snapshots(m, tok, device, batch,
                                          {feat_reps}, pool)
        X = snaps[feat_reps]
        say(f"[{tg}] MAE 编码 dim={X.shape[1]} pool={pool} max_len={arch['max_len']}")
        return X
    raise ValueError(kind)


def ridge_eval(X, y, fit_m, val_m, test_m, n_boot, rng):
    from sklearn.model_selection import KFold
    tr = fit_m | val_m
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    Xtr, ytr = Xs[tr], y[tr]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best_a, best_v = ALPHAS[0], -2.0
    for a in ALPHAS:
        vs = []
        for ti, vi in kf.split(Xtr):
            r = Ridge(alpha=a).fit(Xtr[ti], ytr[ti])
            vs.append(sp(r.predict(Xtr[vi]), ytr[vi]))
        vs = [v for v in vs if np.isfinite(v)]
        mv = float(np.mean(vs)) if vs else float("-inf")
        if mv > best_v:
            best_v, best_a = mv, a
    r = Ridge(alpha=best_a).fit(Xtr, ytr)
    pred = r.predict(Xs[test_m])
    yt = y[test_m]
    rho = sp(pred, yt)
    n = len(yt)
    boot = [sp(pred[i], yt[i]) for i in (rng.integers(0, n, n)
                                         for _ in range(n_boot))]
    boot = [b for b in boot if np.isfinite(b)]
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2)
    return rho, lo, hi, best_a, X.shape[1], pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--pool", default="mean", choices=["cls", "mean"])
    ap.add_argument("--feat_reps", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trunc", type=int, default=100)
    ap.add_argument("--alphas", default=None,
                    help="逗号分隔，固定 ridge α 网格；给单值即对齐所有来源")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()
    if args.alphas:
        global ALPHAS
        ALPHAS = [float(x) for x in args.alphas.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    aa_map = {c: i + 1 for i, c in enumerate(ps.AA_STD)}
    aa_map["X"], aa_map["B"], aa_map["U"] = 23, 22, 21

    df = ps.read_csv_any(args.set)
    cm = {str(c).strip().lower(): c for c in df.columns}
    seqs = np.array(["".join(str(s).split()).upper()
                     for s in df[cm["sequence"]].values])
    if args.trunc > 0:
        seqs = np.array([s[:args.trunc] for s in seqs])
    y = df[cm["target"]].astype(float).values
    setcol = df[cm["set"]].astype(str).str.strip().str.lower().values
    valcol = df[cm["validation"]].astype(str).str.strip().str.lower().values
    train_all = setcol == "train"
    test_m = setcol == "test"
    val_m = train_all & np.isin(valcol, ["true", "1", "1.0"])
    fit_m = train_all & ~val_m
    if val_m.sum() == 0:
        print("[WARN] validation 列无 TRUE，α 用 fit 集自选（略泄漏）")
        val_m, fit_m = train_all, train_all

    L = []
    def say(message):
        L.append(message)
        print(message, flush=True)
    say("=" * 84)
    say(f"FLIP 回归探针  set={args.set}")
    say(f"n={len(seqs)}  fit={fit_m.sum()} val={val_m.sum()} test={test_m.sum()}"
        f"  长度 {int(np.min([len(s) for s in seqs]))}-"
        f"{int(np.max([len(s) for s in seqs]))}  device={device}")
    say("=" * 84)

    rows = []
    predictions = {"target": y[test_m]}
    for spec in ps.split_sources(args.sources):
        kind, path, kv = ps.parse_source(spec)
        X = extract(kind, path, kv, seqs, device, args.batch,
                    args.pool, args.feat_reps, aa_map, say)
        rho, lo, hi, a, dim, pred = ridge_eval(X, y, fit_m, val_m, test_m,
                                         args.n_boot, rng)
        rows.append((ps.source_tag(kind, path, kv), dim, rho, lo, hi, a))
        predictions[ps.source_tag(kind, path, kv)] = pred
        say(f"Completed {rows[-1][0]} rho={rho:.4f} [{lo:.4f}, {hi:.4f}]")
        if args.out_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
            import pandas as pd
            pd.DataFrame(rows, columns=["source", "dim", "rho", "rho_lo", "rho_hi", "alpha"]).assign(set=args.set, pool=args.pool, seed=args.seed, n_boot=args.n_boot, n_fit=int(fit_m.sum()), n_val=int(val_m.sum()), n_test=int(test_m.sum())).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(predictions).to_csv(os.path.splitext(args.out_csv)[0] + "_predictions.csv", index=False, encoding="utf-8-sig")

    say("")
    say(f"{'来源':<34}{'dim':>6}{'test ρ':>10}{'95%CI':>22}{'α':>8}")
    for tg, dim, rho, lo, hi, a in rows:
        say(f"{tg:<34}{dim:>6}{rho:>10.4f}  [{lo:+.4f}, {hi:+.4f}]{a:>8.1f}")
    print("\n".join(str(x) for x in L))
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as out:
            writer = csv.DictWriter(out, fieldnames=["set", "source", "dim", "rho", "rho_lo", "rho_hi", "alpha", "pool", "seed", "n_boot", "n_fit", "n_val", "n_test"])
            writer.writeheader()
            for tg, dim, rho, lo, hi, a in rows:
                writer.writerow(dict(set=args.set, source=tg, dim=dim, rho=rho, rho_lo=lo, rho_hi=hi, alpha=a, pool=args.pool, seed=args.seed, n_boot=args.n_boot, n_fit=int(fit_m.sum()), n_val=int(val_m.sum()), n_test=int(test_m.sum())))
        print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
