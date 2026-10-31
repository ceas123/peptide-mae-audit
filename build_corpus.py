
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

AA_STD = "ACDEFGHIKLMNPQRSTVWY"
K_MER = 9
BASE = 26
_POW = (BASE ** np.arange(K_MER - 1, -1, -1)).astype(np.int64)

_WS = re.compile(r"\s+")
_NON_STD = re.compile(f"[^{AA_STD}]")
_TERM = re.compile(r"^(?:AC-|ACE-|ACETYL-|FOR-|NH2-|H-|D-|BETA-)"
                   r"|(?:-NH2|-OH|-COOH|-CONH2|-AMIDE|-NH3|-AMINE)$")

AA_ORD = np.full(256, 20, dtype=np.int64)
for _i, _c in enumerate(AA_STD):
    AA_ORD[ord(_c)] = _i

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}

EISENBERG = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
             "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
             "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
             "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}

LEN_BINS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 101]


def norm_exact(s):
    """去空白 + 转大写，不碰字母表。"""
    return _WS.sub("", str(s)).upper()


def norm_canon(s):
    """再剥末端修饰标注与非标准残基，只留 20 种标准氨基酸。"""
    return _NON_STD.sub("", _TERM.sub("", norm_exact(s)))


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, dtype=str)
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError as exc:
            last = exc
        except Exception as exc:
            last = exc
            break
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str,
                               engine="python", on_bad_lines="skip")
        except Exception as exc:
            last = exc
    raise RuntimeError(f"无法解析文件 {path}: {last}")


def find_seq_col(df):
    for c in df.columns:
        if str(c).strip().lower() == "sequence":
            return c
    for c in df.columns:
        if str(c).strip().lower() in ("seq", "sequences", "peptide"):
            return c
    return df.columns[0] if len(df.columns) else None


def load_frame(path):
    df = read_table(path)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    sc = find_seq_col(df)
    out = pd.DataFrame(index=range(len(df)))
    out["raw"] = df[sc].astype(str).values if sc is not None else ""
    out["exact"] = out["raw"].map(norm_exact)
    out["canon"] = out["raw"].map(norm_canon)
    out = out[out["exact"] != ""].reset_index(drop=True)
    out["length"] = out["canon"].str.len()
    return {"df": out, "src": df, "seq_col": sc, "path": path,
            "n_rows": len(df), "n_used": len(out)}


def net_charge(s):
    q = 0.0
    for c in s:
        if c in "KR":
            q += 1.0
        elif c in "DE":
            q -= 1.0
        elif c == "H":
            q += 0.1
    return q


def gravy(s):
    return float(np.mean([KD[c] for c in s])) if s else float("nan")


def hydro_moment(s, delta_deg=100.0):
    if not s:
        return float("nan")
    h = np.array([EISENBERG[c] for c in s], dtype=np.float64)
    ang = np.deg2rad(delta_deg) * np.arange(len(h))
    return float(np.hypot((h * np.sin(ang)).sum(),
                          (h * np.cos(ang)).sum()) / len(h))


def build_meta(seqs):
    rows = [(len(s), net_charge(s), gravy(s), hydro_moment(s)) for s in seqs]
    return pd.DataFrame(rows, columns=["length", "charge", "gravy", "muH"],
                        index=list(seqs))


def kmer_codes(seq):
    if len(seq) < K_MER:
        return None
    idx = AA_ORD[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    win = np.lib.stride_tricks.sliding_window_view(idx, K_MER)
    return (win * _POW).sum(axis=1)


def make_pool(seqs):
    parts = [c for c in (kmer_codes(s) for s in seqs) if c is not None]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


def cont_one(seq, pool):
    code = kmer_codes(seq)
    if code is None or pool.size == 0:
        return float("nan")
    pos = np.clip(np.searchsorted(pool, code), 0, pool.size - 1)
    return float((pool[pos] == code).mean())


def cont_many(seqs, pool, rng=None):
    out = np.full(len(seqs), np.nan, dtype=np.float64)
    if pool.size == 0:
        return out
    for i, s in enumerate(seqs):
        if rng is not None:
            a = list(s)
            rng.shuffle(a)
            s = "".join(a)
        out[i] = cont_one(s, pool)
    return out


def dist_row(series, name):
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"group": name, "n": 0}
    return {"group": name, "n": int(x.size), "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x)), "q10": float(np.percentile(x, 10)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "q90": float(np.percentile(x, 90)), "min": float(x.min()),
            "max": float(x.max())}


def option_row(name, remove_set, A, old_in_A, nonamp_in_A, meta, args, note=""):
    rem = remove_set & A
    kept = A - rem
    amp_left = old_in_A - rem
    sub = meta.loc[sorted(amp_left)] if amp_left else meta.iloc[0:0]
    chem = int(((sub["length"] <= args.amp_max_len)
                & (sub["charge"] >= args.amp_min_charge)
                & (sub["muH"] >= args.amp_min_muh)).sum()) if len(sub) else 0
    return {"option": name, "remove_total": len(rem),
            "corpus_left": len(kept), "corpus_pct": len(kept) / max(len(A), 1),
            "amp_left": len(amp_left),
            "amp_left_pct": len(amp_left) / max(len(old_in_A), 1),
            "amp_chemspace_left": chem,
            "nonamp_left": len(nonamp_in_A - rem),
            "loss_h": 0.0, "note": note}


def stratified_negatives(nonamp, pos_lens, n_want, seed):
    rng = np.random.RandomState(seed)
    nonamp = sorted(nonamp)
    nl = pd.Series([len(s) for s in nonamp], index=nonamp)
    nbin = pd.cut(nl, bins=LEN_BINS, right=False)
    pbin = pd.cut(pd.Series(pos_lens), bins=LEN_BINS, right=False)
    share = pbin.value_counts(normalize=True)
    picked, shortfalls = [], []
    for b, frac in share.items():
        want = int(round(frac * n_want))
        pool = list(nbin.index[nbin == b])
        if want <= len(pool):
            picked += list(rng.choice(pool, size=want, replace=False))
        else:
            picked += pool
            shortfalls.append((str(b), int(want), len(pool)))
    picked = list(dict.fromkeys(picked))
    rest = [s for s in nonamp if s not in set(picked)]
    need = n_want - len(picked)
    if need > 0 and rest:
        picked += list(rng.choice(rest, size=min(need, len(rest)), replace=False))
    return picked, shortfalls


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="alldataset.csv")
    ap.add_argument("--amp", type=str, default="alldataset_old.csv")
    ap.add_argument("--eval", type=str, default="snedataset.csv")
    ap.add_argument("--extra_eval", nargs="*", default=[],
                    help="额外带标签评测集 csv(Sequence+Label), 其正负类及近同源一并移出语料")
    ap.add_argument("--sources", nargs="*",
                    default=["111.csv", "222.xlsx", "333.csv"])
    ap.add_argument("--out_root", type=str, default="decontam_out")
    ap.add_argument("--homolog_thr", type=float, default=0.90)
    ap.add_argument("--amp_max_len", type=int, default=50)
    ap.add_argument("--amp_min_charge", type=float, default=2.0)
    ap.add_argument("--amp_min_muh", type=float, default=0.40)
    ap.add_argument("--n_neg", type=int, default=0)
    ap.add_argument("--pos_source", type=str, default="sne_h",
                    choices=["sne", "sne_h", "old"])
    ap.add_argument("--emit", type=int, default=0, choices=[0, 1])
    ap.add_argument("--emit_corpus", type=int, default=0, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    args.corpus, args.amp, args.eval = rp(args.corpus), rp(args.amp), rp(args.eval)
    args.sources = [rp(p) for p in args.sources]
    args.extra_eval = [rp(p) for p in args.extra_eval]
    out = rp(args.out_root)
    os.makedirs(os.path.join(out, "sets"), exist_ok=True)
    rng = np.random.RandomState(args.seed)

    L = []
    say = lambda s="": L.append(s)

    say("=" * 86)
    say("预训练语料去污染方案核查")
    say("=" * 86)

    A = load_frame(args.corpus)
    old = load_frame(args.amp)
    eve = load_frame(args.eval)
    srcs = {os.path.basename(p): load_frame(p)
            for p in args.sources if os.path.exists(p)}

    say("")
    say("[0] 输入文件")
    for tag, info in [("corpus", A), ("amp", old), ("eval", eve)] + \
            [(k, v) for k, v in srcs.items()]:
        say(f"  {tag:>10s}  {os.path.basename(info['path']):>22s}  "
            f"rows={info['n_rows']:6d}  used={info['n_used']:6d}  "
            f"seq_col='{info['seq_col']}'")
        say(f"{'':>14s}列名: {[str(c) for c in info['src'].columns][:12]}")

    A_set = set(A["df"]["canon"])
    old_set = set(old["df"]["canon"])
    eve_set = set(eve["df"]["canon"])
    nonamp_set = A_set - old_set
    amp_in_A = A_set & old_set
    if "Label" in eve["src"].columns:
        say("")
        say(f"  eval 的 Label 分布: "
            f"{eve['src']['Label'].value_counts().to_dict()}")
    else:
        say("")
        say(f"  警告: {os.path.basename(args.eval)} 未找到 Label 列，"
            f"实际列: {[str(c) for c in eve['src'].columns]}")

    say("")
    say("=" * 86)
    say("[1] 语料构成")
    say("=" * 86)
    say(f"  alldataset 去重后 canon: {len(A_set)}")
    say(f"  old 去重后 canon        : {len(old_set)}")
    say(f"  old ∩ alldataset        : {len(amp_in_A)}  "
        f"({len(amp_in_A) / max(len(old_set), 1):.4%} of old)")
    say(f"  alldataset − old        : {len(nonamp_set)}")
    say(f"  校验: {len(amp_in_A)} + {len(nonamp_set)} = "
        f"{len(amp_in_A) + len(nonamp_set)}  vs  {len(A_set)}")
    if len(old_set - A_set):
        say(f"  注意: old 中有 {len(old_set - A_set)} 条不在 alldataset 内")
    say("  → alldataset 无标签列，AMP 注释唯一来源是 old；"
        "两者构成二分，不存在 old 之外的 AMP。")

    say("")
    say("[2] 与评测集的关系")
    say(f"  sne ⊆ alldataset : {eve_set <= A_set}")
    say(f"  sne ⊆ old        : {eve_set <= old_set}")
    say(f"  sne ∩ old        : {len(eve_set & old_set)}")
    say(f"  old − sne        : {len(old_set - eve_set)}   ← 仅删 sne 时语料中保留的 AMP")
    for k, v in srcs.items():
        s = set(v["df"]["canon"])
        say(f"  sne ∩ {k:>12s}: {len(eve_set & s):5d}   "
            f"({len(eve_set & s) / max(len(eve_set), 1):.2%} of sne)   "
            f"|{k}|={len(s)}  {k}⊆A={s <= A_set}")

    say("")
    say("[3] 源文件对语料的贡献（验证 cleannew 的产物构成）")
    union = set()
    for k, v in srcs.items():
        s = set(v["df"]["canon"])
        union |= s
        say(f"  {k:>12s}  rows={v['n_rows']:6d}  canon={len(s):6d}  "
            f"∩A={len(s & A_set):6d} ({len(s & A_set) / max(len(s), 1):.2%})  "
            f"∩A−old={len(s & nonamp_set):6d}  ∩old={len(s & old_set):6d}")
    say(f"  sources 并集         canon={len(union):6d}  ∩A={len(union & A_set):6d}  "
        f"∩(A−old)={len(union & nonamp_set):6d}")
    orphan = nonamp_set - union
    say(f"  A − old 中未被任何源文件解释的条目: {len(orphan)}")
    if orphan:
        say("     → 这 44436 条非 AMP 的来源不在 111/222/333 里，"
            "应为更早版本的 alldataset（cleannew 的旧库分支）。")
        say("       请确认那份文件的筛选标准，否则负类难度不可控。")
    else:
        say("     → 非 AMP 全部可由源文件解释，provenance 闭合。")

    say("")
    say("=" * 86)
    say("[4] 近同源扫描（{} -mer 包含度，非同源判定）".format(K_MER))
    say("=" * 86)
    pool_sne = make_pool(sorted(eve_set))
    pool_old = make_pool(sorted(old_set))
    say(f"  sne 的 {K_MER}-mer 基数: {pool_sne.size}")
    say(f"  old 的 {K_MER}-mer 基数: {pool_old.size}")

    old_minus_sne = sorted(old_set - eve_set)
    c1 = cont_many(old_minus_sne, pool_sne, rng=None)
    c1n = cont_many(old_minus_sne, pool_sne, rng=rng)
    d1 = dist_row(c1, "old−sne vs sne")
    say("")
    say(f"  A) old−sne({len(old_minus_sne)}) 对 sne 池:")
    say(f"     实测 mean={d1['mean']:.4f} median={d1['median']:.4f}   "
        f"打乱零假设 mean={np.nanmean(c1n):.4f}")
    for t in (0.5, 0.8, 0.9, 0.99):
        say(f"     ≥{t:.2f}: {int((c1 >= t).sum()):6d}  "
            f"({np.nanmean(c1 >= t):.2%})")
    n_h = int((c1 >= args.homolog_thr).sum())
    say(f"     → 阈值 {args.homolog_thr} 下需一并从语料移除的近同源: {n_h} 条")

    na_list = sorted(nonamp_set)
    c2 = cont_many(na_list, pool_old, rng=None)
    c2n = cont_many(na_list, pool_old, rng=rng)
    d2 = dist_row(c2, "A−old vs old")
    say("")
    say(f"  B) A−old({len(na_list)}) 对 old 池（负类标签噪声估计）:")
    say(f"     实测 mean={d2['mean']:.4f} median={d2['median']:.4f}   "
        f"打乱零假设 mean={np.nanmean(c2n):.4f}")
    for t in (0.5, 0.8, 0.9, 0.99):
        say(f"     ≥{t:.2f}: {int((c2 >= t).sum()):6d}  "
            f"({np.nanmean(c2 >= t):.2%})")
    say(f"     → 这些是疑似 AMP 却标为 non-AMP 的条目，"
        f"构成正类的假阴性。比例高则 AMP 探针必须改用硬负样本臂。")

    h2 = (pd.DataFrame({"canon": old_minus_sne, "cont_sne": c1,
                        "cont_sne_shuf": c1n})
          .sort_values("cont_sne", ascending=False))
    h2.to_csv(os.path.join(out, "homolog_scan.csv"),
              index=False, encoding="utf-8-sig")

    say("")
    say("=" * 86)
    say("[5] 理化画像")
    say("=" * 86)
    meta = build_meta(sorted(A_set & old_set | A_set))
    lab = eve["src"]["Label"].astype(str).values if "Label" in eve["src"].columns else None
    eve_canon = eve["df"]["canon"].values
    groups = {
        "sne_ALL": eve_set,
        "old−sne": old_set - eve_set,
        "A−old(负类)": nonamp_set,
    }
    if lab is not None:
        for v in pd.unique(lab):
            groups[f"sne_{v}"] = set(eve_canon[lab == v])
    pc_rows = []
    say(f"  {'group':>16s} {'n':>7s} {'len':>7s} {'q25':>6s} {'q75':>6s} "
        f"{'charge':>8s} {'gravy':>7s} {'muH':>7s}")
    for g, s in groups.items():
        if not s:
            continue
        m = meta.loc[sorted(s)]
        row = {"group": g, "n": len(s),
               "len_median": float(m["length"].median()),
               "charge_mean": float(m["charge"].mean()),
               "gravy_mean": float(m["gravy"].mean()),
               "muH_mean": float(m["muH"].mean())}
        pc_rows.append(row)
        say(f"  {g:>16s} {len(s):7d} {m['length'].median():7.1f} "
            f"{m['length'].quantile(.25):6.1f} {m['length'].quantile(.75):6.1f} "
            f"{m['charge'].mean():8.2f} {m['gravy'].mean():7.3f} "
            f"{m['muH'].mean():7.3f}")
    pd.DataFrame(pc_rows).to_csv(os.path.join(out, "physchem_by_group.csv"),
                                 index=False, encoding="utf-8-sig")

    say("")
    say("=" * 86)
    say("[6] 去污染方案对照")
    say("=" * 86)
    sne_h = eve_set | {s for s, v in zip(old_minus_sne, c1)
                       if np.isfinite(v) and v >= args.homolog_thr}
    cands = [
        ("R1 = sne", eve_set, "最小移除；AMP 正类上限 2266"),
        (f"R2 = sne ∪ 近同源({args.homolog_thr})", sne_h,
         "推荐主臂：切断同源通道，保留 ~5400 AMP"),
        ("R3 = old ∩ A", amp_in_A, "移除全部已知 AMP；需无 AMP 对照臂"),
    ]
    rows = [option_row(n, s, A_set, amp_in_A, nonamp_set, meta, args, note)
            for n, s, note in cands]
    say(f"  {'option':>28s} {'移除':>7s} {'语料':>7s} {'占比':>7s} "
        f"{'AMP留':>7s} {'保留率':>7s} {'化学空间':>8s}")
    for r in rows:
        say(f"  {r['option']:>28s} {r['remove_total']:7d} {r['corpus_left']:7d} "
            f"{r['corpus_pct']:7.1%} {r['amp_left']:7d} {r['amp_left_pct']:7.1%} "
            f"{r['amp_chemspace_left']:8d}")
    say("")
    say(f"  化学空间判据: len<={args.amp_max_len} 且 charge>={args.amp_min_charge} "
        f"且 muH>={args.amp_min_muh}")

    P = {"sne": eve_set, "sne_h": sne_h, "old": amp_in_A}[args.pos_source]
    say("")
    say(f"  AMP 探针正类来源 pos_source={args.pos_source}: P={len(P)}")
    say(f"  {'N':>7s} {'流行率':>8s} {'总移除':>8s} {'干净语料':>9s} {'占比':>7s}")
    for nb in (1, 2, 3, 5, 8):
        n = nb * len(P)
        if n > len(nonamp_set):
            continue
        tot = len(P | sne_h) + n if args.pos_source != "old" else len(amp_in_A) + n
        say(f"  {n:7d} {len(P) / (len(P) + n):8.3f} {tot:8d} "
            f"{len(A_set) - tot:9d} {(len(A_set) - tot) / len(A_set):7.1%}")
    say("  注: 负类池一次留够后，可在同一 ckpt 上按不同流行率子采样评估，"
        "无需重训。")

    removal = set()
    if args.emit:
        removal = set(sne_h) | set(P)
        neg = []
        if args.n_neg > 0:
            pos_list = sorted(P)
            pos_lens = [len(s) for s in pos_list]
            neg, sf = stratified_negatives(nonamp_set, pos_lens, args.n_neg,
                                           args.seed)
            removal |= set(neg)
            say("")
            say(f"[7] 负类分层抽样 N={len(neg)}（请求 {args.n_neg}）")
            if sf:
                say(f"  长度分箱不足: {sf}")
            pl = np.array(pos_lens)
            nl = np.array([len(s) for s in neg])
            say(f"  正类 len: mean={pl.mean():.1f} median={np.median(pl):.1f}")
            say(f"  负类 len: mean={nl.mean():.1f} median={np.median(nl):.1f}")
            pb = pd.cut(pd.Series(pl), bins=LEN_BINS, right=False).value_counts(normalize=True)
            nb2 = pd.cut(pd.Series(nl), bins=LEN_BINS, right=False).value_counts(normalize=True)
            maxdiff = float((pb - nb2).abs().max())
            say(f"  分箱占比最大偏差: {maxdiff:.4f}")
        for ep in args.extra_eval:
            if not os.path.exists(ep):
                say(f"  警告: extra_eval 不存在 {ep}")
                continue
            ef = load_frame(ep)
            ecanon = set(ef["df"]["canon"]) - {""}
            name = os.path.splitext(os.path.basename(ep))[0]
            n_in = len(ecanon & A_set)
            pool_e = make_pool(sorted(ecanon))
            outside = sorted(A_set - ecanon)
            cont_e = cont_many(outside, pool_e, rng=None)
            homo = {s for s, v in zip(outside, cont_e)
                    if np.isfinite(v) and v >= args.homolog_thr}
            removal |= ecanon | homo
            amp_hit = len((ecanon | homo) & amp_in_A)
            say("")
            say(f"[7x] 额外评测集 {name}: 序列 {len(ecanon)}  "
                f"在语料内 {n_in}  近同源 {len(homo)}  占用已知 AMP {amp_hit}")
            ee = read_table(ep)
            ee.to_csv(os.path.join(out, "sets", f"eval_{name}.csv"),
                      index=False, encoding="utf-8-sig")
            say(f"  写出 sets/eval_{name}.csv ({len(ee)} 条)")
        ordered = sorted(removal)
        with open(os.path.join(out, "sets", "remove_final.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(ordered) + "\n")

        if args.n_neg > 0:
            ev = pd.DataFrame(
                [{"Sequence": s, "Label": "AMP"} for s in sorted(P)]
                + [{"Sequence": s, "Label": "nonAMP"} for s in sorted(neg)])
            ev.to_csv(os.path.join(out, "sets", "eval_amp.csv"),
                      index=False, encoding="utf-8-sig")
            say(f"  写出 sets/eval_amp.csv  "
                f"(AMP={len(P)}, nonAMP={len(neg)})")

        fa = eve["df"][["raw", "canon"]].copy()
        fa.columns = ["Sequence", "canon"]
        if lab is not None:
            fa["Label"] = lab
        fa = fa[["Sequence", "Label"]] if lab is not None else fa[["Sequence"]]
        fa.to_csv(os.path.join(out, "sets", "eval_afab.csv"),
                  index=False, encoding="utf-8-sig")
        say(f"  写出 sets/eval_afab.csv  ({len(fa)} 条)")

        if args.emit_corpus:
            kept = sorted(A_set - removal)
            pd.DataFrame({"Sequence": kept}).to_csv(
                os.path.join(out, "sets", "corpus_clean.csv"),
                index=False, encoding="utf-8-sig")
            say(f"  写出 sets/corpus_clean.csv  ({len(kept)} 条)")

        say("")
        say(f"  最终: 移除 {len(removal)}  干净语料 {len(A_set) - len(removal)} "
            f"({(len(A_set) - len(removal)) / len(A_set):.1%})  "
            f"预计单次训练 "
            f"{9.0 * (len(A_set) - len(removal)) / len(A_set):.1f}h")

    say("")
    say("=" * 86)
    say("[8] 判读要点")
    say("=" * 86)
    say(f"  1. 删除集起点是 sne（含有标签的评测集），不是 old。"
        f"old−sne 的 {len(old_set - eve_set)} 条应保留在语料中。")
    say(f"  2. 阈值 {args.homolog_thr} 下 sne 的近同源 {n_h} 条也须移除，"
        f"否则 af/ab 探针仍有同源泄漏。")
    say(f"  3. AMP 探针正类上限为移除出的 old 子集 P={len(P)}，"
        f"负类 N 由 alldataset−old 留出。两者都不可留在语料内。")
    say(f"  4. A−old 中疑似 AMP 的条目见 [4]B；若比例高，"
        f"AMP 探针必须加硬负样本臂。")
    say("  5. MMseqs2 仍需对最终删除集跑一次验证（"
        "--min-seq-id 0.3 -c 0.5），本次 {} -mer 扫描只是代理。".format(K_MER))

    txt = "\n".join(L)
    with open(os.path.join(out, "plan_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    pd.DataFrame(rows).to_csv(os.path.join(out, "options.csv"),
                              index=False, encoding="utf-8-sig")
    rel = {
        "corpus_n": len(A_set), "old_n": len(old_set),
        "old_in_corpus": len(amp_in_A), "nonamp_n": len(nonamp_set),
        "sne_n": len(eve_set), "sne_subset_corpus": eve_set <= A_set,
        "sne_subset_old": eve_set <= old_set,
        "old_minus_sne": len(old_set - eve_set),
        "sne_homolog_hits": n_h, "homolog_thr": args.homolog_thr,
        "pos_source": args.pos_source, "P": len(P),
        "orphan_nonamp": len(orphan),
        "n_neg_requested": args.n_neg,
        "remove_final": len(removal) if args.emit else None,
    }
    with open(os.path.join(out, "set_relations.json"), "w",
              encoding="utf-8") as f:
        json.dump(rel, f, indent=2, ensure_ascii=False)
    print(f"\n落盘: {out}")


if __name__ == "__main__":
    main()
