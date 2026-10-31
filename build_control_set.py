
import argparse
import os

import numpy as np
import pandas as pd

from build_corpus import norm_canon, AA_STD, stratified_negatives

STD = set(AA_STD)


def clean_canon(s):
    c = norm_canon(s)
    if not c or any(ch not in STD for ch in c):
        return ""
    return c


def pick(pos_canon, neg_pool, cap, seed):
    rng = np.random.RandomState(seed)
    pos = list(pos_canon)
    if cap > 0 and len(pos) > cap:
        pos = list(rng.choice(pos, cap, replace=False))
    neg, sf = stratified_negatives(neg_pool, [len(s) for s in pos],
                                   len(pos), seed)
    return sorted(pos), sorted(neg), sf


def write_task(path, pos, neg, pos_lab, neg_lab):
    rows = ([{"Sequence": s, "Label": pos_lab} for s in pos]
            + [{"Sequence": s, "Label": neg_lab} for s in neg])
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def med(xs):
    return float(np.median([len(s) for s in xs])) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="222.xlsx")
    ap.add_argument("--out_dir", default="control_out/sets")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_excel(args.src, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    act = df["Activity"].fillna("").str.lower()
    is_fun = act.str.contains("antifungal")
    is_can = act.str.contains("anticancer")
    is_bac = act.str.contains("antibacterial")

    canon = df["Sequence"].map(clean_canon)
    ok = (canon != "") & canon.str.len().between(args.min_len, args.max_len)

    fun = set(canon[ok & is_fun & ~is_bac])
    can = set(canon[ok & is_can & ~is_bac])
    bac = set(canon[ok & is_bac & ~is_can & ~is_fun])
    ambig = fun & can
    fun -= ambig
    can -= ambig

    af_pos, af_neg, af_sf = pick(fun, bac, args.cap, args.seed)
    ac_pos, ac_neg, ac_sf = pick(can, bac, args.cap, args.seed + 1)

    write_task(os.path.join(args.out_dir, "eval_afab.csv"),
               af_pos, af_neg, "AF", "AB")
    write_task(os.path.join(args.out_dir, "eval_acp.csv"),
               ac_pos, ac_neg, "ACP", "AB")

    used = set(af_neg) | set(ac_neg)
    print(f"源 {args.src}  抗真菌纯 {len(fun)}  抗癌纯 {len(can)}  "
          f"抗菌纯(负类池) {len(bac)}  两活性冲突剔除 {len(ambig)}")
    print(f"afab  正类 AF {len(af_pos)}  负类 AB {len(af_neg)}  "
          f"total {len(af_pos) + len(af_neg)}")
    print(f"acp   正类 ACP {len(ac_pos)}  负类 AB {len(ac_neg)}  "
          f"total {len(ac_pos) + len(ac_neg)}")
    print(f"负类池占用 {len(used)}/{len(bac)} = {len(used)/max(len(bac),1):.3f}"
          f"  两任务负类交集 {len(set(af_neg) & set(ac_neg))}")
    print(f"len median  AF={med(af_pos):.0f}  ACP={med(ac_pos):.0f}  "
          f"AB_afab={med(af_neg):.0f}  AB_acp={med(ac_neg):.0f}")
    if af_sf:
        print(f"afab 负类分箱不足: {af_sf}")
    if ac_sf:
        print(f"acp 负类分箱不足: {ac_sf}")
    print(f"写出 {args.out_dir}/eval_afab.csv 与 eval_acp.csv")


if __name__ == "__main__":
    main()
