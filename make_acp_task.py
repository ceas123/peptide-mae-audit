
import argparse

import numpy as np
import pandas as pd

from build_corpus import norm_canon, AA_STD, stratified_negatives

STD = set(AA_STD)


def clean_canon(s):
    c = norm_canon(s)
    if not c or any(ch not in STD for ch in c):
        return ""
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="222.xlsx")
    ap.add_argument("--out", default="acp.csv")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_excel(args.src, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    act = df["Activity"].fillna("").str.lower()
    is_acp = act.str.contains("anticancer")
    is_abac = act.str.contains("antibacterial")
    role = np.where(is_acp & ~is_abac, "ACP",
                    np.where(is_abac & ~is_acp, "nonACP", ""))

    rec = pd.DataFrame({"canon": df["Sequence"].map(clean_canon), "Label": role})
    rec = rec[(rec["Label"] != "") & (rec["canon"] != "")]
    rec = rec[rec["canon"].str.len().between(args.min_len, args.max_len)]

    nlab = rec.groupby("canon")["Label"].nunique()
    conflict = set(nlab[nlab > 1].index)
    rec = rec[~rec["canon"].isin(conflict)].drop_duplicates("canon")

    pos = rec.loc[rec["Label"] == "ACP", "canon"].tolist()
    neg_all = set(rec.loc[rec["Label"] == "nonACP", "canon"])

    rng = np.random.RandomState(args.seed)
    if args.cap > 0 and len(pos) > args.cap:
        pos = list(rng.choice(pos, args.cap, replace=False))
    pos_lens = [len(s) for s in pos]
    neg, sf = stratified_negatives(neg_all, pos_lens, len(pos), args.seed)

    out = pd.DataFrame(
        [{"Sequence": s, "Label": "ACP"} for s in sorted(pos)]
        + [{"Sequence": s, "Label": "nonACP"} for s in sorted(neg)])
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    pl, nl = np.array(pos_lens), np.array([len(s) for s in neg])
    print(f"冲突剔除(两类同序列): {len(conflict)}")
    print(f"ACP 正类: {len(pos)}  nonACP 负类: {len(neg)}  总 {len(out)}")
    print(f"正类 len median={np.median(pl):.0f}  负类 len median={np.median(nl):.0f}")
    if sf:
        print(f"负类长度分箱不足: {sf}")
    print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
