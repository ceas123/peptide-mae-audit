
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)


AA_STD = "ACDEFGHIKLMNPQRSTVWY"
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
EIS = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
       "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
       "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
       "Y": 0.26, "V": 1.08}
BOMAN = {"A": -0.5, "R": -1.8, "N": -4.8, "D": -9.0, "C": -0.8, "Q": -4.1,
         "E": -8.7, "G": 0.0, "H": -1.4, "I": 2.8, "L": 2.6, "K": -5.1,
         "M": 2.1, "F": 2.7, "P": -0.2, "S": -2.8, "T": -1.4, "W": -1.3,
         "Y": -0.4, "V": 3.2}
PKA = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00,
       "K": 10.53, "R": 12.48}
PKA_N, PKA_C = 8.0, 3.1

CLASS_SETS = {
    "cat": "KRH",
    "ani": "DE",
    "pol": "STNQ",
    "spe": "GPC",
    "ali": "AVLIM",
    "aro": "FWY",
}

SIMPLE = ("length", "physchem", "physchem_nolen",
          "aac", "aac_len",
          "aac_off_nolen", "aac_off",
          "aac_quad_nolen", "aac_quad",
          "aac_simpson", "aac_simpson_len", "aac_simpson_freq",
          "class_quad", "amphi_hc",
          "kmer2", "kmer3", "zero")

KIND_LABEL = {
    "length": "单变量长度",
    "aac": "组成 aac",
    "aac_len": "组成+长度",
    "aac_off_nolen": "组成+非对角乘积",
    "aac_off": "组成+非对角乘积+长度",
    "aac_quad_nolen": "组成+全部乘积",
    "aac_quad": "组成+全部乘积+长度",
    "aac_simpson": "辛普森集中度",
    "aac_simpson_len": "辛普森+长度",
    "aac_simpson_freq": "组成+辛普森",
    "class_quad": "类级二次",
    "amphi_hc": "组成+两亲性项",
    "kmer2": "2-mer 频率",
    "kmer3": "3-mer 频率",
    "physchem": "理化含长度",
    "physchem_nolen": "理化去长度",
}

CONSTANT_AP_CONVENTION = "流行率"

CI_RAISE = 0.02
CI_WARN = 0.005


def norm_ppf(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
            ) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4])
                    * r + 1)


def read_csv_any(path):
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"无法解析 {path}: {last}")


def torch_load_any(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def net_charge(seq, ph=7.0):
    q = 1.0 / (1.0 + 10 ** (ph - PKA_N)) - 1.0 / (1.0 + 10 ** (PKA_C - ph))
    for c in seq:
        if c in "KRH":
            q += 1.0 / (1.0 + 10 ** (ph - PKA[c]))
        elif c in "DECY":
            q -= 1.0 / (1.0 + 10 ** (PKA[c] - ph))
    return q


def iso_point(seq):
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hydro_moment(seq, window=11, deg=100.0):
    if len(seq) == 0:
        return 0.0
    if len(seq) < window:
        window = len(seq)
    ang = np.deg2rad(deg * np.arange(window))
    cs, sn = np.cos(ang), np.sin(ang)
    best = 0.0
    for i in range(len(seq) - window + 1):
        h = np.array([EIS.get(c, 0.0) for c in seq[i:i + window]])
        best = max(best, float(np.hypot(h @ cs, h @ sn) / window))
    return best


def physchem_row(seq, with_len):
    n = max(len(seq), 1)
    freq = [seq.count(c) / n for c in AA_STD]
    row = [net_charge(seq), sum(KD.get(c, 0.0) for c in seq) / n,
           iso_point(seq), sum(seq.count(c) for c in "FWY") / n,
           sum(BOMAN.get(c, 0.0) for c in seq) / n, hydro_moment(seq)]
    row = freq + row
    if with_len:
        row = [float(len(seq))] + row
    return row


def physchem_matrix(seqs, with_len):
    return np.array([physchem_row(s, with_len) for s in seqs],
                    dtype=np.float64)


def aac_freq(seqs):
    rows = []
    for s in seqs:
        n = max(len(s), 1)
        rows.append([s.count(c) / n for c in AA_STD])
    return np.array(rows, dtype=np.float64)


def with_length(seqs, X):
    L = np.array([[float(len(s))] for s in seqs], dtype=np.float64)
    return np.concatenate([L, X], axis=1)


def aac_matrix(seqs, with_len):
    X = aac_freq(seqs)
    return with_length(seqs, X) if with_len else X


def aac_products(seqs, diagonal=True):
    f = aac_freq(seqs)
    n = f.shape[0]
    cols = []
    for i in range(20):
        for j in range(i if diagonal else i + 1, 20):
            cols.append((f[:, i] * f[:, j]).reshape(n, 1))
    return np.concatenate(cols, axis=1) if cols else np.zeros((n, 0))


def aac_quad_matrix(seqs, with_len, diagonal=True):
    base = aac_freq(seqs)
    P = aac_products(seqs, diagonal=diagonal)
    X = np.concatenate([base, P], axis=1)
    return with_length(seqs, X) if with_len else X


def aac_simpson_matrix(seqs, with_len):
    f = aac_freq(seqs)
    S = (f ** 2).sum(1, keepdims=True)
    return with_length(seqs, S) if with_len else S


def aac_simpson_freq_matrix(seqs, with_len):
    f = aac_freq(seqs)
    S = (f ** 2).sum(1, keepdims=True)
    X = np.concatenate([f, S], axis=1)
    return with_length(seqs, X) if with_len else X


def aac_class_totals(seqs):
    f = aac_freq(seqs)
    idx = {c: i for i, c in enumerate(AA_STD)}
    T = np.zeros((f.shape[0], len(CLASS_SETS)), dtype=np.float64)
    for k, letters in enumerate(CLASS_SETS.values()):
        T[:, k] = f[:, [idx[c] for c in letters]].sum(1)
    return T


def class_quad_matrix(seqs, with_len):
    f = aac_freq(seqs)
    T = aac_class_totals(seqs)
    cols = [f]
    for i in range(T.shape[1]):
        for j in range(i, T.shape[1]):
            cols.append((T[:, i] * T[:, j]).reshape(-1, 1))
    X = np.concatenate(cols, axis=1)
    return with_length(seqs, X) if with_len else X


def amphi_matrix(seqs, with_len):
    f = aac_freq(seqs)
    idx = {c: i for i, c in enumerate(AA_STD)}
    cat = f[:, [idx[c] for c in CLASS_SETS["cat"]]].sum(1)
    hyd = f[:, [idx[c] for c in CLASS_SETS["ali"]]].sum(1) \
        + f[:, [idx[c] for c in CLASS_SETS["aro"]]].sum(1)
    X = np.concatenate([f, (cat * hyd).reshape(-1, 1)], axis=1)
    return with_length(seqs, X) if with_len else X


def kmer_matrix(seqs, k, min_df=5):
    grid = ["".join(p) for p in itertools.product(AA_STD, repeat=k)]
    idx = {km: i for i, km in enumerate(grid)}
    X = np.zeros((len(seqs), len(grid)), dtype=np.float64)
    for r, s in enumerate(seqs):
        n = max(len(s) - k + 1, 1)
        for j in range(len(s) - k + 1):
            X[r, idx[s[j:j + k]]] += 1.0
        X[r] /= n
    keep = X.sum(0) >= min_df
    return X[:, keep]


def greedy_clusters(seqs, k=9, thr=0.5):
    inv, labels, n_cl = {}, np.full(len(seqs), -1, dtype=np.int64), 0
    for i, s in enumerate(seqs):
        if len(s) < k:
            labels[i] = -2
            continue
        km = {s[j:j + k] for j in range(len(s) - k + 1)}
        cnt = {}
        for kmer in km:
            for cid in inv.get(kmer, ()):
                cnt[cid] = cnt.get(cid, 0) + 1
        best, bid = 0, -1
        for cid, c in cnt.items():
            if c > best:
                best, bid = c, cid
        if bid >= 0 and best / len(km) >= thr:
            labels[i] = bid
            for kmer in km:
                inv.setdefault(kmer, set()).add(bid)
        else:
            labels[i] = n_cl
            for kmer in km:
                inv.setdefault(kmer, set()).add(n_cl)
            n_cl += 1
    return labels


def grouped_folds(groups, n_folds, seed):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    sizes = {int(g): int((groups == g).sum()) for g in uniq}
    order = [int(g) for g in uniq]
    order = [order[i] for i in rng.permutation(len(order))]
    order.sort(key=lambda g: -sizes[g])
    load = [0] * n_folds
    assign = {}
    for g in order:
        f = int(np.argmin(load))
        load[f] += sizes[g]
        assign[g] = f
    return np.array([assign[int(g)] for g in groups], dtype=np.int64)


def rank_auc(pos, neg):
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sv[1:] != sv[:-1]) + 1]
    ends = np.r_[starts[1:], len(sv)]
    ranks[order] = np.repeat((starts + ends - 1) / 2.0 + 1.0, ends - starts)
    r = ranks[:n_pos].sum()
    return float((r - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def is_constant(v):
    v = np.asarray(v, dtype=np.float64)
    return bool(np.nanstd(v) == 0.0)


def average_precision(y, s):
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    if is_constant(s):
        return float(y.mean())
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float(prec[yy == 1].sum() / n_pos)


def oof_probe_given(X, y, folds_vec, n_folds, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    acc = np.zeros(len(y), dtype=np.float64)
    hit = np.zeros(len(y), dtype=np.float64)
    for f in range(n_folds):
        te = folds_vec == f
        tr = ~te
        if tr.sum() < 10 or te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        clf.fit(sc.transform(X[tr]), y[tr])
        acc[te] += clf.predict_proba(sc.transform(X[te]))[:, 1]
        hit[te] += 1.0
    return acc / np.maximum(hit, 1.0)


def oof_probe(X, y, groups, n_folds, C, n_repeats, seed):
    acc = np.zeros(len(y), dtype=np.float64)
    for r in range(n_repeats):
        fv = grouped_folds(groups, n_folds, seed + 1000 * r)
        acc += oof_probe_given(X, y, fv, n_folds, C)
    return acc / max(n_repeats, 1)


def infer_arch(state, cfg):
    d = {}
    for k in ("dim", "max_len", "vocab_size", "encoder_layer", "decoder_layer",
              "encoder_head", "decoder_head"):
        v = cfg.get(k, 0)
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = 0
        d[k] = v or None
    try:
        d["mask_ratio"] = float(cfg.get("mask_ratio", 0.75))
    except (TypeError, ValueError):
        d["mask_ratio"] = 0.75

    for k, v in state.items():
        if k.endswith("encoder.pos_embedding"):
            d["max_len"] = d["max_len"] or int(v.shape[1] - 1)
            d["dim"] = d["dim"] or int(v.shape[2])
        if k.endswith("encoder.token_emb.weight"):
            d["vocab_size"] = d["vocab_size"] or int(v.shape[0])
            d["dim"] = d["dim"] or int(v.shape[1])
    for tag, key in (("encoder_layer", "encoder.transformer."),
                     ("decoder_layer", "decoder.transformer.")):
        idx = set()
        for k in state:
            if key in k:
                head = k.split(key)[1].split(".")[0]
                if head.isdigit():
                    idx.add(int(head))
        if idx:
            d[tag] = d[tag] or (max(idx) + 1)
    return d


def build_mae(arch, device):
    from model import MAE_Peptide
    m = MAE_Peptide(
        vocab_size=arch["vocab_size"], max_len=arch["max_len"],
        dim=arch["dim"], encoder_layer=arch["encoder_layer"],
        encoder_head=arch["encoder_head"], decoder_layer=arch["decoder_layer"],
        decoder_head=arch["decoder_head"], mask_ratio=arch["mask_ratio"])
    return m.to(device)


def encode_tokens(seqs, max_len, aa_map):
    out = np.zeros((len(seqs), max_len + 1), dtype=np.int64)
    default = aa_map.get("X", 23)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:max_len]):
            out[i, j + 1] = aa_map.get(c, default)
    return torch.from_numpy(out)


@torch.no_grad()
def mae_features_snapshots(model, tokens, device, batch, snapshots, pool="cls"):
    enc = model.encoder.eval()
    dim = enc.pos_embedding.shape[-1]
    want = sorted({int(x) for x in snapshots if int(x) >= 1})
    if not want:
        return {}
    feat = np.zeros((len(tokens), dim), dtype=np.float64)
    for i in range(0, len(tokens), batch):
        t = tokens[i:i + batch].to(device)
        x = enc.encode_full(t)
        if pool == "mean":
            m = (t != 0).float().unsqueeze(-1)
            m[:, 0, :] = 0.0
            feat[i:i + batch] = ((x * m).sum(1)
                                 / m.sum(1).clamp(min=1.0)).float().cpu().numpy()
        else:
            feat[i:i + batch] = x[:, 0, :].float().cpu().numpy()
    return {r: feat for r in want}




@torch.no_grad()
def esm_features(seqs, model_path, device, batch):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_path)
    mod = AutoModel.from_pretrained(model_path).to(device).eval()
    out = []
    for i in range(0, len(seqs), batch):
        chunk = list(seqs[i:i + batch])
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        r = mod(**enc, output_hidden_states=True)
        h = r.hidden_states[-1]
        mask = enc["attention_mask"].bool().clone()
        last = enc["attention_mask"].sum(1) - 1
        for b in range(mask.shape[0]):
            mask[b, 0] = False
            mask[b, int(last[b])] = False
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, 0)


def split_sources(s):
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    out = []
    for p in parts:
        if out and ("=" in p) and (":" not in p) \
                and out[-1].startswith("randinit:"):
            out[-1] = out[-1] + ";" + p
        else:
            out.append(p)
    return out


def parse_source(spec):
    if spec.startswith("ckpt:"):
        return "ckpt", spec[5:], {}
    if spec.startswith("randinit:"):
        kv = {}
        for x in spec[9:].replace(",", ";").split(";"):
            if "=" in x:
                k, v = x.split("=", 1)
                kv[k.strip()] = v.strip()
        return "randinit", None, kv
    if spec.startswith("esm:"):
        return "esm", spec[4:], {}
    if spec in SIMPLE:
        return spec, None, {}
    raise ValueError(f"无法识别的 source: {spec}")


def source_tag(kind, path, kv):
    if kind == "ckpt":
        return os.path.splitext(os.path.basename(path))[0]
    if kind == "randinit":
        return f"randinit_d{kv.get('dim', '?')}_s{kv.get('seed', 0)}"
    if kind == "esm":
        return "esm_" + os.path.basename(path.rstrip("/\\"))
    return kind


def kind_label(kind, nm, dim):
    base = KIND_LABEL.get(kind, kind)
    return f"{base} ({dim})" if dim else base


def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan")
    return float(np.mean(v)), float(np.std(v, ddof=1))


def build_features(kind, seqs, kv, args, tag, say):
    if kind == "zero":
        return (np.zeros((len(seqs), 1)),
                "全零特征 dim=1（泄漏哨兵，AUC 必须为 0.5000）")
    if kind == "length":
        return (np.array([[len(s)] for s in seqs], dtype=np.float64),
                "序列长度 dim=1")
    if kind == "physchem":
        X = physchem_matrix(list(seqs), True)
        return X, f"理化描述符 dim={X.shape[1]}  含长度=True"
    if kind == "physchem_nolen":
        X = physchem_matrix(list(seqs), False)
        return X, f"理化描述符 dim={X.shape[1]}  含长度=False"
    if kind in ("aac", "aac_len"):
        wl = kind == "aac_len"
        X = aac_matrix(list(seqs), wl)
        return X, f"氨基酸组成 dim={X.shape[1]}  含长度={wl}"
    if kind in ("aac_off_nolen", "aac_off"):
        wl = kind == "aac_off"
        X = aac_quad_matrix(list(seqs), wl, diagonal=False)
        return X, (f"组成 + 非对角乘积 dim={X.shape[1]}  含长度={wl}  "
                   f"(190 个 f_i*f_j, i<j)")
    if kind in ("aac_quad_nolen", "aac_quad"):
        wl = kind == "aac_quad"
        X = aac_quad_matrix(list(seqs), wl, diagonal=True)
        return X, (f"组成 + 全部乘积 dim={X.shape[1]}  含长度={wl}  "
                   f"(210 个 f_i*f_j, i<=j)")
    if kind in ("aac_simpson", "aac_simpson_len"):
        wl = kind == "aac_simpson_len"
        X = aac_simpson_matrix(list(seqs), wl)
        return X, f"辛普森集中度 dim={X.shape[1]}  含长度={wl}"
    if kind == "aac_simpson_freq":
        X = aac_simpson_freq_matrix(list(seqs), False)
        return X, (f"组成 + 辛普森 dim={X.shape[1]}  含长度=False  "
                   f"(判据: 190 乘积能否被单变量辛普森替代)")
    if kind == "class_quad":
        X = class_quad_matrix(list(seqs), False)
        return X, (f"组成 + 类级乘积 dim={X.shape[1]}  含长度=False  "
                   f"({len(CLASS_SETS)} 类 → "
                   f"{len(CLASS_SETS) * (len(CLASS_SETS) + 1) // 2} 个类乘积)")
    if kind == "amphi_hc":
        X = amphi_matrix(list(seqs), False)
        return X, (f"组成 + 单一两亲性交叉项 dim={X.shape[1]}  "
                   f"含长度=False  (f_cat * f_hyd)")
    if kind in ("kmer2", "kmer3"):
        k = 2 if kind == "kmer2" else 3
        X = kmer_matrix(list(seqs), k, args.kmer_min_df)
        return X, (f"{k}-mer 频率 dim={X.shape[1]}  min_df={args.kmer_min_df} "
                   f"(从 {20 ** k} 中筛出)")
    raise ValueError(kind)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--extra_scores", default="")
    ap.add_argument("--out_root", default="probe_out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pos_label", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--probe_C", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--cluster", type=int, default=1, choices=[0, 1])
    ap.add_argument("--cluster_thr", type=float, default=0.5)
    ap.add_argument("--feat_reps", type=int, default=8)
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--diag_reps", default="1,2,4,8,16,32")
    ap.add_argument("--diag_repeats", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--kmer_min_df", type=int, default=5)
    ap.add_argument("--aa_map", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    out = rp(args.out_root)
    os.makedirs(out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zz = norm_ppf(1 - args.alpha / 2) + norm_ppf(args.power)
    diag_repeats = args.diag_repeats if args.diag_repeats > 0 else args.repeats

    if args.aa_map:
        with open(rp(args.aa_map), encoding="utf-8") as f:
            aa_map = {k: int(v) for k, v in json.load(f).items()}
    else:
        aa_map = {c: i + 1 for i, c in enumerate(AA_STD)}
        aa_map["X"] = 23
        aa_map["B"] = 22
        aa_map["U"] = 21

    diag_reps = sorted({int(x) for x in str(args.diag_reps).split(",")
                        if x.strip().isdigit() and int(x) >= 1})

    L = []
    say = lambda s="": L.append(s)
    say("=" * 96)
    say("统一探针台")
    say("=" * 96)

    sdf = read_csv_any(rp(args.set))
    cols = {str(c).strip().lower(): c for c in sdf.columns}
    if "sequence" not in cols or "label" not in cols:
        raise RuntimeError(f"{args.set} 需要 Sequence 与 Label 列: "
                           f"{list(sdf.columns)}")
    seqs = np.array(["".join(str(s).split()).upper()
                     for s in sdf[cols["sequence"]].values])
    bad_chars = sorted(set("".join(seqs.tolist())) - set(AA_STD))
    if bad_chars:
        bad_mask = np.array([bool(set(s) - set(AA_STD)) for s in seqs])
        n_bad = int(bad_mask.sum())
        say(f"[WARN] {n_bad} 条序列含非标准字符 {bad_chars}，"
            f"示例 {[s[:40] for s in seqs[bad_mask][:3]]}")
    lab = sdf[cols["label"]].astype(str).values
    sname = os.path.splitext(os.path.basename(args.set))[0]

    lv = sorted(set(lab))
    if len(lv) != 2:
        raise RuntimeError(f"需要二分类标签，实际为 {lv}")
    up = {str(v).strip().upper(): v for v in lv}
    if args.pos_label in lv:
        pos_label, pos_note = args.pos_label, "显式指定"
    else:
        hit = [up[k] for k in ("AMP", "POS", "POSITIVE", "TRUE", "1", "正类")
               if k in up]
        if not hit:
            raise RuntimeError(f"标签为 {lv}，无法确定正类，"
                               f"请显式传 --pos_label {' / '.join(lv)}")
        pos_label = hit[0]
        pos_note = "未显式指定，按名称自动取，AUC 普遍<0.5 说明方向反了"
    neg_label = [x for x in lv if x != pos_label][0]
    y = (lab == pos_label).astype(np.float64)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())

    say("")
    say(f"评测集 {sname}  n={len(y)}  正类={pos_label} n_pos={n_pos}  "
        f"负类={neg_label} n_neg={n_neg}  流行率={n_pos / len(y):.4f}  "
        f"[{pos_note}]")

    say(f"device={device}  folds={args.folds} repeats={args.repeats}  "
        f"probe_C={args.probe_C}  feat_reps={args.feat_reps}  "
        f"diag_repeats={diag_repeats}")
    say(f"z = norm_ppf(1-{args.alpha}/2) + norm_ppf({args.power}) = {zz:.4f}")

    if args.cluster:
        groups = greedy_clusters(list(seqs), k=9, thr=args.cluster_thr)
        cl = sorted(set(groups.tolist()))
        say(f"聚簇(k=9, thr={args.cluster_thr}): {len(cl)} 簇，"
            f"最大簇 {max((groups == g).sum() for g in cl)} 条，"
            f"长度<9 单独成簇 {int((groups == -2).sum())} 条")
        pd.DataFrame({"Sequence": seqs, "cluster": groups}).to_csv(
            os.path.join(out, f"cluster_{sname}{tag}.csv"), index=False,
            encoding="utf-8-sig")
    else:
        groups = np.arange(len(y), dtype=np.int64)
        say("聚簇已关闭 → 每条序列自成一簇，等效随机分折，"
            "同源序列可能跨折，CV 偏乐观")

    base_folds = grouped_folds(groups, args.folds, args.seed)
    sizes = [int((base_folds == f).sum()) for f in range(args.folds)]
    say(f"基准分折 (seed={args.seed}) 各折大小: {sizes}")

    feats, fixed_scores, names = {}, {}, []
    src_kind, src_dim, src_meta = {}, {}, {}
    diag_rows = []

    for spec in split_sources(args.sources):
        kind, path, kv = parse_source(spec)
        tg = source_tag(kind, path, kv)
        if tg in feats or tg in fixed_scores:
            tg = f"{tg}_{len(names)}"
        names.append(tg)
        src_kind[tg] = kind
        src_meta[tg] = dict(kv)

        if kind in SIMPLE:
            X, msg = build_features(kind, seqs, kv, args, tg, say)
            feats[tg] = X
            src_dim[tg] = X.shape[1]
            say(f"\n[{tg}] {msg}")
            if kind == "zero":
                say("  全零 → 常向量，AP 按流行率定义，AUC 恒 0.5000")
            continue

        if kind == "esm":
            say(f"\n[{tg}] ESM-2 冻结特征  路径={path}")
            feats[tg] = esm_features(list(seqs), rp(path), device, args.batch)
            src_dim[tg] = feats[tg].shape[1]
            say(f"  特征 dim={src_dim[tg]}")
            say("  ⚠️ ESM-2 预训练语料为 UR50D，与 UniProt 来源的抗菌肽库存在")
            say("     结构性重叠。本行是『含记忆上界』，不是同协议基线。")
            continue

        snaps = {}
        if kind == "ckpt":
            ck = torch_load_any(rp(path))
            if isinstance(ck, dict) and "state_dict" in ck:
                state, cfg = ck["state_dict"], dict(ck.get("args", {}))
            elif isinstance(ck, dict) and "model_state_dict" in ck:
                state, cfg = ck["model_state_dict"], dict(ck.get("args", {}))
            else:
                state, cfg = ck, {}
            arch = infer_arch(state, cfg)
            miss = [k for k in ("dim", "max_len", "vocab_size",
                                "encoder_layer", "decoder_layer")
                    if not arch.get(k)]
            if miss:
                raise RuntimeError(f"{tg} 无法确定架构字段 {miss}，"
                                   f"请提供含 args 的 ckpt")
            arch["encoder_head"] = arch.get("encoder_head") or 8
            arch["decoder_head"] = arch.get("decoder_head") or 8
            say(f"\n[{tg}] ckpt  dim={arch['dim']} "
                f"enc={arch['encoder_layer']}L/{arch['encoder_head']}H "
                f"dec={arch['decoder_layer']}L/{arch['decoder_head']}H  "
                f"max_len={arch['max_len']}  vocab={arch['vocab_size']}  "
                f"mask_ratio={arch['mask_ratio']}")
            if not cfg:
                say("  ⚠️ ckpt 未内嵌 args，架构由参数形状推断；"
                    "head 数用了默认 8")
            m = build_mae(arch, device)
            m.load_state_dict(state, strict=True)
            say(f"  参数量 {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
            tok = encode_tokens(list(seqs), arch["max_len"], aa_map)
            src_dim[tg] = arch["dim"]
            snaps = mae_features_snapshots(
                m, tok, device, args.batch,
                sorted(set(diag_reps) | {args.feat_reps}), args.pool)
            say(f"  特征 dim={src_dim[tg]}  掩码重复 {args.feat_reps} 次取均值")
        elif kind == "randinit":
            dim = int(kv.get("dim", 256))
            arch = {"dim": dim,
                    "max_len": int(kv.get("max_len", 100)),
                    "vocab_size": int(kv.get("vocab_size", 24)),
                    "encoder_layer": int(kv.get("encoder_layer", 6)),
                    "decoder_layer": int(kv.get("decoder_layer", 4)),
                    "encoder_head": int(kv.get("encoder_head", 8)),
                    "decoder_head": int(kv.get("decoder_head", 8)),
                    "mask_ratio": float(kv.get("mask_ratio", 0.75))}
            torch.manual_seed(int(kv.get("seed", 0)))
            m = build_mae(arch, device)
            say(f"\n[{tg}] 随机初始化  dim={dim} seed={kv.get('seed', 0)} "
                f"enc={arch['encoder_layer']}L/{arch['encoder_head']}H "
                f"dec={arch['decoder_layer']}L/{arch['decoder_head']}H  "
                f"mask_ratio={arch['mask_ratio']}")
            say(f"  参数量 {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
            tok = encode_tokens(list(seqs), arch["max_len"], aa_map)
            src_dim[tg] = dim
            snaps = mae_features_snapshots(
                m, tok, device, args.batch,
                sorted(set(diag_reps) | {args.feat_reps}), args.pool)
            say(f"  特征 dim={dim}  掩码重复 {args.feat_reps} 次取均值")
        else:
            raise ValueError(kind)

        feats[tg] = snaps.get(args.feat_reps, snaps[max(snaps)])

        if diag_reps:
            for r in sorted(snaps):
                s = oof_probe(snaps[r], y, groups, args.folds, args.probe_C,
                              diag_repeats, args.seed)
                diag_rows.append({
                    "set": sname, "source": tg, "kind": kind, "reps": r,
                    "dim": src_dim[tg],
                    "AUC": rank_auc(s[y == 1], s[y == 0]),
                    "AP": average_precision(y, s),
                    "folds": args.folds, "cv_repeats": diag_repeats})

    if args.extra_scores:
        for f in sorted(glob.glob(os.path.join(rp(args.extra_scores),
                                               "scores_*.csv"))):
            nm = "ext_" + os.path.basename(f)[7:-4]
            d = read_csv_any(f)
            cm = {str(c).strip().lower(): c for c in d.columns}
            sc = cm.get("score") or cm.get("logit")
            if cm.get("sequence") is None or sc is None:
                say(f"\n跳过 {f}: 缺 Sequence 或 score 列")
                continue
            mp = dict(zip(d[cm["sequence"]].astype(str),
                          pd.to_numeric(d[sc], errors="coerce").values))
            v = np.array([mp.get(s, np.nan) for s in seqs], dtype=np.float64)
            if not np.isfinite(v).all():
                say(f"\n跳过 {f}: 有 {int((~np.isfinite(v)).sum())} 条缺失")
                continue
            if nm in fixed_scores:
                nm = f"{nm}_{len(names)}"
            fixed_scores[nm] = v
            names.append(nm)
            src_kind[nm] = "ext"
            src_dim[nm] = None
            src_meta[nm] = {}
            say(f"\n[{nm}] 外部分数（不重训探针，直接算指标）")

    if not names:
        raise RuntimeError("没有任何可用来源")

    np.savez_compressed(os.path.join(out, f"features_{sname}{tag}.npz"),
                        **{k: v for k, v in feats.items()})

    say("")
    say("=" * 96)
    say("[探针训练] 簇分组 CV + 重复，输出 OOF 分数")
    say("=" * 96)
    say(f"  {'来源':<34s} {'kind':<16s} {'dim':>7s} {'重复':>5s} {'状态':>10s}")
    oof = {}
    for nm in names:
        if nm in fixed_scores:
            oof[nm] = fixed_scores[nm]
            say(f"  {nm:<34s} {src_kind[nm]:<16s} {'-':>7s} {'-':>5s} "
                f"{'直接使用':>10s}")
            continue
        X = feats[nm]
        if is_constant(X):
            oof[nm] = np.zeros(len(y))
            say(f"  {nm:<34s} {src_kind[nm]:<16s} {X.shape[1]:7d} "
                f"{args.repeats:5d} {'零方差':>10s}")
            continue
        oof[nm] = oof_probe(X, y, groups, args.folds, args.probe_C,
                            args.repeats, args.seed)
        say(f"  {nm:<34s} {src_kind[nm]:<16s} {X.shape[1]:7d} "
            f"{args.repeats:5d} {'OK':>10s}")

    oof_df = pd.DataFrame({nm: oof[nm] for nm in names})
    oof_df.insert(0, "Sequence", seqs)
    oof_df.insert(1, "Label", lab)
    oof_df.insert(2, "cluster", groups)
    oof_df.to_csv(os.path.join(out, f"oof_{sname}{tag}.csv"), index=False,
                  encoding="utf-8-sig")
    for nm in names:
        pd.DataFrame({"Sequence": seqs, "score": oof[nm]}).to_csv(
            os.path.join(out, f"scores_{nm}_{sname}{tag}.csv"), index=False,
            encoding="utf-8-sig")

    CONST = {nm for nm in names if is_constant(oof[nm])}
    if "zero" in names and "zero" in oof:
        zauc = rank_auc(oof["zero"][y == 1], oof["zero"][y == 0])
        if not np.isfinite(zauc) or abs(zauc - 0.5) > 1e-9:
            raise RuntimeError(f"哨兵失败：zero 的 AUC = {zauc}，应为 0.5")

    ipos, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    point = {}
    for nm in names:
        s = oof[nm]
        point[nm] = (average_precision(y, s), rank_auc(s[ipos], s[ineg]))

    rng = np.random.RandomState(args.seed + 7)
    boot = np.empty((args.n_boot, len(names), 2), dtype=np.float64)
    for b in range(args.n_boot):
        pi = ipos[rng.randint(0, len(ipos), len(ipos))]
        ni = ineg[rng.randint(0, len(ineg), len(ineg))]
        idx = np.concatenate([pi, ni])
        yy = y[idx]
        for k, nm in enumerate(names):
            s = oof[nm][idx]
            boot[b, k, 0] = average_precision(yy, s)
            boot[b, k, 1] = rank_auc(s[:len(pi)], s[len(pi):])

    say("")
    say("=" * 96)
    say("[1] 单来源指标")
    say("=" * 96)
    say(f"  {'来源':<30s} {'dim':>5s} {'AP':>8s} {'AP 95%CI':>20s} "
        f"{'SE_b':>7s} {'SE_a':>7s} {'比':>5s} {'AUC':>8s} {'AUC 95%CI':>20s} "
        f"{'注':>4s}")
    mrows = []
    for k, nm in enumerate(names):
        a, u = boot[:, k, 0], boot[:, k, 1]
        a_lo, a_hi = np.percentile(a, [2.5, 97.5])
        u_lo, u_hi = np.percentile(u, [2.5, 97.5])
        se_b = float(a.std(ddof=1))
        se_a = float(sqrt(point[nm][0] * (1 - point[nm][0]) / max(n_pos, 1)))
        flag = "常" if nm in CONST else ""
        say(f"  {nm:<30s} {str(src_dim.get(nm) or '-'):>5s} "
            f"{point[nm][0]:8.4f} [{a_lo:.4f}, {a_hi:.4f}] "
            f"{se_b:7.4f} {se_a:7.4f} {se_b / max(se_a, 1e-9):5.2f} "
            f"{point[nm][1]:8.4f} [{u_lo:.4f}, {u_hi:.4f}] {flag:>4s}")
        mrows.append({"set": sname, "source": nm, "kind": src_kind.get(nm),
                      "dim": src_dim.get(nm), "is_constant": nm in CONST,
                      "AP": point[nm][0], "AP_lo": float(a_lo),
                      "AP_hi": float(a_hi), "AP_se_boot": se_b,
                      "AP_se_analytic": se_a,
                      "AP_se_ratio": se_b / max(se_a, 1e-9),
                      "AUC": point[nm][1], "AUC_lo": float(u_lo),
                      "AUC_hi": float(u_hi),
                      "AUC_se_boot": float(u.std(ddof=1))})
    pd.DataFrame(mrows).to_csv(os.path.join(out, f"metrics_{sname}{tag}.csv"),
                               index=False, encoding="utf-8-sig")
    say(f"  常 = 常向量来源，正类流行率 = {n_pos / len(y):.4f}，"
        f"AP 按该值定义，AUC 恒 0.5000")

    ref = args.ref if args.ref in names else names[0]
    say("")
    say("=" * 96)
    say(f"[2] 两两配对 Δ 与判读（参考来源 {ref}）")
    say("=" * 96)
    say(f"  {'对比':<44s} {'ΔAP':>8s} {'SE':>7s} {'ΔAP 95%CI':>20s} "
        f"{'MDE':>7s} {'Δ/MDE':>6s} {'AP判':>9s} {'AUC判':>9s}")
    crows = []
    pairs = list(itertools.combinations(range(len(names)), 2))
    pairs.sort(key=lambda p: (ref not in (names[p[0]], names[p[1]]),
                              names[p[0]], names[p[1]]))
    for i, j in pairs:
        a, b = names[i], names[j]
        dap = boot[:, i, 0] - boot[:, j, 0]
        dau = boot[:, i, 1] - boot[:, j, 1]
        lo, hi = np.percentile(dap, [2.5, 97.5])
        ulo, uhi = np.percentile(dau, [2.5, 97.5])
        dm = point[a][0] - point[b][0]
        du = point[a][1] - point[b][1]
        dev = abs(dm - float(np.clip(dm, lo, hi))) if np.isfinite(dm) else 0.0
        if dev > CI_RAISE:
            raise RuntimeError(
                f"一致性检查失败: {a} − {b} 点估计 {dm:.4f} 不在 "
                f"bootstrap CI [{lo:.4f}, {hi:.4f}] 内，偏差 {dev:.4f} "
                f"> {CI_RAISE}")
        se = float(dap.std(ddof=1))
        se_u = float(dau.std(ddof=1))
        mde = zz * se
        mde_u = zz * se_u
        ratio = abs(dm) / mde if mde > 0 else float("inf")
        cross = bool(lo <= 0 <= hi)
        if cross:
            verdict = "不可分辨"
        elif ratio < 1.5:
            verdict = "弱"
        else:
            verdict = "可分辨"
        ucross = bool(ulo <= 0 <= uhi)
        if ucross or mde_u <= 0:
            verdict_auc = "不可分辨"
        elif abs(du) / mde_u < 1.5:
            verdict_auc = "弱"
        else:
            verdict_auc = "可分辨"
        note = ""
        if a in CONST or b in CONST:
            note = " [常量基准]"
        elif dev > CI_WARN:
            note = f" [CI偏差{dev:.3f}]"
        say(f"  {a + ' − ' + b:<44s} {dm:8.4f} {se:7.4f} "
            f"[{lo:+.4f}, {hi:+.4f}] {mde:7.4f} {ratio:6.2f} "
            f"{verdict:>9s} {verdict_auc:>9s}{note}")
        crows.append({"set": sname, "a": a, "b": b,
                      "kind_a": src_kind.get(a), "kind_b": src_kind.get(b),
                      "dim_a": src_dim.get(a), "dim_b": src_dim.get(b),
                      "dAP": dm, "dAP_se": se, "dAP_lo": float(lo),
                      "dAP_hi": float(hi), "dAP_crosses_zero": cross,
                      "mde_80": mde, "delta_over_mde": ratio,
                      "verdict": verdict, "verdict_auc": verdict_auc,
                      "ci_dev": dev,
                      "touches_constant": (a in CONST or b in CONST),
                      "dAUC": du, "dAUC_se": se_u,
                      "dAUC_lo": float(ulo), "dAUC_hi": float(uhi),
                      "dAUC_crosses_zero": ucross,
                      "dAUC_over_mde": (abs(du) / mde_u if mde_u > 0
                                        else float("inf"))})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(os.path.join(out, f"compare_{sname}{tag}.csv"), index=False,
               encoding="utf-8-sig")

    def find_cmp(a, b):
        for r in crows:
            if (r["a"], r["b"]) == (a, b):
                return r
            if (r["b"], r["a"]) == (a, b):
                r2 = dict(r)
                r2["a"], r2["b"] = a, b
                r2["dAP"] = -r["dAP"]
                r2["dAP_lo"], r2["dAP_hi"] = -r["dAP_hi"], -r["dAP_lo"]
                r2["dAUC"] = -r["dAUC"]
                r2["dAUC_lo"], r2["dAUC_hi"] = -r["dAUC_hi"], -r["dAUC_lo"]
                return r2
        return None

    def have(*kinds):
        return all(k in names for k in kinds)

    say("")
    say("=" * 96)
    say("[3] 关键对照")
    say("=" * 96)

    def show(a, b):
        r = find_cmp(a, b)
        if r is None:
            return
        say(f"  {a + ' − ' + b:<44s} {r['dAP']:8.4f} "
            f"[{r['dAP_lo']:+.4f}, {r['dAP_hi']:+.4f}] {r['mde_80']:7.4f} "
            f"{r['delta_over_mde']:6.2f} {r['verdict']:>9s}"
            + (" [常量基准]" if r.get("touches_constant") else ""))
        say(f"    ΔAUC = {r['dAUC']:+.4f} "
            f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]  "
            f"AUC判 {r.get('verdict_auc', '-')}")

    shown = set()

    def show_pair(a, b):
        if a in names and b in names and (a, b) not in shown:
            shown.add((a, b))
            show(a, b)

    for nm in names:
        if src_kind.get(nm) in ("ckpt", "randinit", "esm"):
            for other in ("randinit_d%s_s0" % src_dim.get(nm), "aac_quad",
                          "physchem", "aac"):
                show_pair(nm, other)

    for pair in (("ckpt", "randinit"), ("ckpt", "esm"), ("ckpt", "aac_quad"),
                 ("ckpt", "physchem"), ("ckpt", "aac"), ("ckpt", "length"),
                 ("randinit", "aac"), ("randinit", "aac_quad"),
                 ("randinit", "length"),
                 ("physchem", "physchem_nolen"),
                 ("physchem", "aac"), ("physchem", "aac_quad"),
                 ("physchem_nolen", "aac"),
                 ("aac_len", "aac"),
                 ("aac_quad_nolen", "aac_off_nolen"),
                 ("aac_off_nolen", "aac"),
                 ("aac_off_nolen", "aac_simpson_freq"),
                 ("aac_off_nolen", "class_quad"),
                 ("aac_off_nolen", "amphi_hc"),
                 ("class_quad", "aac"),
                 ("amphi_hc", "aac"),
                 ("aac_quad", "aac_quad_nolen"),
                 ("aac_quad", "aac"), ("aac_off", "aac_off_nolen"),
                 ("aac_simpson", "length"), ("aac_simpson_len", "length")):
        ca = [n for n in names if src_kind.get(n) == pair[0]]
        cb = [n for n in names if src_kind.get(n) == pair[1]]
        if ca and cb:
            show_pair(ca[0], cb[0])

    say("")
    say("=" * 96)
    say("[4] 判读要点")
    say("=" * 96)

    def aucs_of(kind):
        return [point[nm][1] for nm in names if src_kind.get(nm) == kind]

    for kind in SIMPLE:
        if kind == "zero":
            continue
        v = aucs_of(kind)
        if not v:
            continue
        m, s = mean_sd(v)
        dim = src_dim.get([n for n in names
                           if src_kind.get(n) == kind][0])
        say(f"  {kind_label(kind, kind, dim):<28s} AUC = {m:.4f}"
            + (f" ± {s:.4f}" if np.isfinite(s) else "") + f"  (n={len(v)})")
        if len(v) > 1:
            say(f"      各实例: " + ", ".join(f"{x:.4f}" for x in v))

    for kind, label in (("randinit", "随机初始化"), ("ckpt", "训练 ckpt"),
                        ("esm", "ESM（含记忆）"), ("ext", "外部分数")):
        v = aucs_of(kind)
        if v:
            m, s = mean_sd(v)
            say(f"  {label:<28s} AUC = {m:.4f}"
                + (f" ± {s:.4f}" if np.isfinite(s) else "")
                + f"  (n={len(v)})")

    say("")
    say("=" * 96)
    say("[5] 维度对照（同 kind，按 dim 分组）")
    say("=" * 96)
    by_dim = {}
    for nm in names:
        if src_kind.get(nm) in ("ckpt", "randinit") and src_dim.get(nm):
            by_dim.setdefault((src_kind[nm], src_dim[nm]), []).append(
                point[nm][1])
    if by_dim:
        say(f"  {'kind':<12s} {'dim':>6s} {'n':>4s} {'AUC 均值':>10s} "
            f"{'sd':>8s} {'AP 均值':>10s}")
        for (kd, dm), _ in sorted(by_dim.items()):
            subs = [nm for nm in names
                    if src_kind.get(nm) == kd and src_dim.get(nm) == dm]
            av = [point[nm][1] for nm in subs]
            ap = [point[nm][0] for nm in subs]
            m, s = mean_sd(av)
            say(f"  {kd:<12s} {dm:6d} {len(subs):4d} {m:10.4f} {s:8.4f} "
                f"{np.mean(ap):10.4f}")
        rd = {dm: mean_sd([point[nm][1] for nm in names
                           if src_kind.get(nm) == "randinit"
                           and src_dim.get(nm) == dm])[0]
              for (kd, dm) in by_dim if kd == "randinit"}
        rw = {dm: mean_sd([point[nm][1] for nm in names
                           if src_kind.get(nm) == "randinit"
                           and src_dim.get(nm) == dm])[1]
              for (kd, dm) in by_dim if kd == "randinit"}
        if len(rd) >= 2:
            ds = sorted(rd)
            diff = rd[ds[-1]] - rd[ds[0]]
            fins = [v for v in rw.values() if np.isfinite(v)]
            sd_lo = min(fins) if fins else float("nan")
            say(f"  randinit 跨维度差 = {diff:+.4f} AUC "
                f"({ds[0]} → {ds[-1]})")
            if np.isfinite(sd_lo):
                say(f"  同 dim 内最小种子 sd = {sd_lo:.4f}")
                say(f"  判据: |跨维度差| {'<' if abs(diff) < sd_lo else '>='} "
                    f"种子 sd → "
                    + ("维度可忽略，不同 dim 的 ckpt 可并表"
                       if abs(diff) < sd_lo
                       else "维度不可忽略，不同 dim 的 ckpt 必须分节"))
    else:
        say("  本次无 ckpt / randinit，跳过")

    say("")
    say("=" * 96)
    say("[6] 等效性界（训练 ckpt vs 同 dim 随机初始化）")
    say("=" * 96)
    eq_rows = []
    for r in crows:
        if {r["kind_a"], r["kind_b"]} != {"ckpt", "randinit"}:
            continue
        a, b = r["a"], r["b"]
        ck = a if r["kind_a"] == "ckpt" else b
        ri = b if r["kind_a"] == "ckpt" else a
        sgn = 1.0 if r["kind_a"] == "ckpt" else -1.0
        d = sgn * r["dAP"]
        lo = min(sgn * r["dAP_lo"], sgn * r["dAP_hi"])
        hi = max(sgn * r["dAP_lo"], sgn * r["dAP_hi"])
        da, db = src_dim.get(a), src_dim.get(b)
        if da != db:
            say(f"  {ck} − {ri}  跳过（dim {da} vs {db} 不匹配，"
                f"需同 dim 才能归因）")
            continue
        hw = max(abs(lo), abs(hi))
        say(f"  {ck} − {ri}")
        say(f"    ΔAP = {d:+.4f}   95%CI = [{lo:+.4f}, {hi:+.4f}]   "
            f"MDE(80%) = {r['mde_80']:.4f}")
        say(f"    等效性界 = ±{hw:.4f} AP（CI 半宽）")
        for target in (0.002, 0.005, 0.01):
            need = int(np.ceil(n_pos * (r["mde_80"] / target) ** 2))
            say(f"    要把 |ΔAP| = {target:.3f} 做到 80% 功效，"
                f"约需 {need:,} 正类")
        eq_rows.append({"set": sname, "ckpt": ck, "randinit": ri, "dim": da,
                        "dAP": d, "lo": lo, "hi": hi, "half_width": hw,
                        "mde_80": r["mde_80"], "n_pos": n_pos})
    if not eq_rows:
        say("  本次没有同 dim 的 ckpt-vs-randinit 配对，无法给出等效性界")

    say("")
    say("=" * 96)
    say("[7] 掩码噪声收敛（MAE 编码器的随机 shuffle）")
    say("=" * 96)
    if diag_rows:
        ddf = pd.DataFrame(diag_rows)
        ddf.to_csv(os.path.join(out, f"diag_masknoise_{sname}{tag}.csv"),
                   index=False, encoding="utf-8-sig")
        say(f"  与主表同折同重复（folds={args.folds}, "
            f"cv_repeats={diag_repeats}），逐 reps 评估")
        say(f"  {'来源':<30s} "
            + " ".join(f"{'r' + str(r):>9s}" for r in diag_reps)
            + f" {'r1−rmax':>9s} {'AP漂移':>9s}")
        for nm in names:
            sub = ddf[ddf["source"] == nm].sort_values("reps")
            if sub.empty:
                continue
            av = {int(r): float(v) for r, v in zip(sub["reps"], sub["AUC"])}
            ap = {int(r): float(v) for r, v in zip(sub["reps"], sub["AP"])}
            row = " ".join(f"{av.get(r, float('nan')):9.4f}"
                           for r in diag_reps)
            dr_a = av.get(diag_reps[0], np.nan) - av.get(diag_reps[-1], np.nan)
            dr_p = ap.get(diag_reps[0], np.nan) - ap.get(diag_reps[-1], np.nan)
            say(f"  {nm:<30s} {row} {dr_a:9.4f} {dr_p:9.4f}")
        say("  判据: |AUC(r1) − AUC(rmax)| > 0.005 或 |AP 漂移| > 0.01 "
            "→ feat_reps 不足")
        bad = []
        for nm in names:
            sub = ddf[ddf["source"] == nm]
            if sub.empty:
                continue
            av = {int(r): float(v) for r, v in zip(sub["reps"], sub["AUC"])}
            ap = {int(r): float(v) for r, v in zip(sub["reps"], sub["AP"])}
            if diag_reps[0] not in av or diag_reps[-1] not in av:
                continue
            if abs(av[diag_reps[0]] - av[diag_reps[-1]]) > 0.005 \
                    or abs(ap[diag_reps[0]] - ap[diag_reps[-1]]) > 0.01:
                bad.append(nm)
        if bad:
            say(f"  ⚠️ 漂移超标: {', '.join(bad)} → "
                f"请把 --feat_reps 提到 {diag_reps[-1]} 后重跑")
        else:
            say(f"  ✅ 全部来源在 reps={diag_reps[-1]} 内收敛，"
                f"--feat_reps={args.feat_reps} 可用")
    else:
        say("  已关闭（--diag_reps 为空）")

    say("")
    say("=" * 96)
    say("[8] 组成模型分解（长度 / 平方项 / 交互项）")
    say("=" * 96)
    decomp = [
        ("aac_len", "aac", "显式长度列（线性模型内）"),
        ("aac_off_nolen", "aac", "非对角交互项 190 个（不含长度）"),
        ("aac_off_nolen", "aac_simpson_freq",
         "190 乘积 vs 单变量辛普森（同 20 维频率底）"),
        ("aac_off_nolen", "class_quad", "190 乘积 vs 21 个类级乘积"),
        ("class_quad", "aac", "类级二次 vs 线性"),
        ("amphi_hc", "aac", "单一两亲性项 vs 线性"),
        ("aac_quad_nolen", "aac_off_nolen", "对角平方项 20 个"),
        ("aac_quad", "aac_quad_nolen", "二次模型内长度列的边际"),
        ("aac_off", "aac_off_nolen", "长度列（非对角模型内）"),
        ("aac_simpson_len", "aac_simpson", "长度列（仅辛普森作底）"),
    ]
    got = False
    for a, b, desc in decomp:
        r = find_cmp(a, b)
        if r is None:
            continue
        got = True
        say(f"  {desc}")
        say(f"    {a} − {b}:  ΔAP = {r['dAP']:+.4f} "
            f"[{r['dAP_lo']:+.4f}, {r['dAP_hi']:+.4f}]   "
            f"ΔAUC = {r['dAUC']:+.4f} "
            f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]   "
            f"AP判 {r['verdict']} / AUC判 {r.get('verdict_auc', '-')}")

    if not got:
        say("  缺少对比所需来源，至少需要 aac, aac_len, aac_off_nolen,"
            " aac_simpson_freq, class_quad, amphi_hc, aac_quad_nolen, aac_quad")

    if got:
        say("")
        say("  恒等式: sum_{i<j} f_i f_j = (1 - Simpson) / 2")
        say("  故 190 个乘积的『全一方向』本身就是辛普森，")
        say("  aac_off_nolen 相对 aac 的增益必须与 aac_simpson_freq 对照后归因")
        if have("aac_off_nolen") and have("aac_simpson_freq"):
            r = find_cmp("aac_off_nolen", "aac_simpson_freq")
            if r and r["dAUC_crosses_zero"]:
                say("  ⚠️ 190 乘积 vs 单变量辛普森: ΔAUC CI 跨 0 →")
                say("     非线性增益可被单变量辛普森解释，"
                    "不存在可分辨的成对结构")
            elif r:
                say(f"  ✅ 190 乘积 − 单变量辛普森  ΔAUC = {r['dAUC']:+.4f} "
                    f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]  "
                    f"AUC判 {r.get('verdict_auc')}")
                say("     超出辛普森 → 存在可分辨的成对组成结构")
        if have("aac_simpson") and have("length"):
            ds = abs(point["aac_simpson"][1] - 0.5)
            dl = abs(point["length"][1] - 0.5)
            say(f"  |AUC-0.5|: 辛普森 {ds:.4f}  vs  长度 {dl:.4f}"
                f"  (长度 AUC = {point['length'][1]:.4f}，"
                f"方向可能 < 0.5)")
            r = find_cmp("aac_simpson", "length")
            if r is None:
                say("  缺少 aac_simpson 与 length 的配对，无法判定代理关系")
            elif r["dAUC_crosses_zero"]:
                say(f"  ΔAUC(aac_simpson − length) = {r['dAUC']:+.4f} "
                    f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]  CI 跨 0 → "
                    f"二者不可分辨，不得称辛普森为长度的强代理")
            elif r["dAUC"] > 0:
                say(f"  ΔAUC(aac_simpson − length) = {r['dAUC']:+.4f} "
                    f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]  "
                    f"辛普森强于长度 → 平方/对角通道不能全部归给『真实交互』")
            else:
                say(f"  ΔAUC(aac_simpson − length) = {r['dAUC']:+.4f} "
                    f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]  "
                    f"辛普森弱于长度，非其强代理")
        if have("class_quad") and have("aac_off_nolen"):
            r = find_cmp("class_quad", "aac_off_nolen")
            if r and r["dAUC_crosses_zero"]:
                say("  ✅ 21 个类级乘积追平 190 个氨基酸级乘积 → "
                    "可把 210 维模型降为 41 维")
            elif r:
                say(f"  类级乘积 − 氨基酸级乘积  ΔAUC = {r['dAUC']:+.4f} "
                    f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}] → "
                    f"类级不足以替代")

    txt = "\n".join(L)
    with open(os.path.join(out, f"probe_report_{sname}{tag}.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    summary = {
        "set": sname, "tag": args.tag, "n": int(len(y)),
        "n_pos": n_pos, "n_neg": n_neg,
        "pos_label": pos_label, "neg_label": neg_label, "ref": ref,
        "folds": args.folds, "repeats": args.repeats,
        "diag_repeats": diag_repeats,
        "probe_C": args.probe_C, "feat_reps": args.feat_reps,
        "diag_reps": diag_reps, "z": zz,
        "cluster": bool(args.cluster), "cluster_thr": args.cluster_thr,
        "n_clusters": int(len(set(groups.tolist()))),
        "constant_sources": sorted(CONST),
        "constant_ap_convention": CONSTANT_AP_CONVENTION,
        "sources": names,
        "metrics": {nm: {"kind": src_kind.get(nm), "dim": src_dim.get(nm),
                         "AP": point[nm][0], "AUC": point[nm][1],
                         "is_constant": nm in CONST} for nm in names},
        "compare": crows,
        "equivalence": eq_rows,
        "masknoise": diag_rows,
    }
    with open(os.path.join(out, f"probe_summary_{sname}{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(txt)
    print(f"\n落盘: {out}")


if __name__ == "__main__":
    main()
