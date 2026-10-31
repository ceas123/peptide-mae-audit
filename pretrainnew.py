
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import MAE_Peptide
from dataset import PeptideDataset
from utils import setup_seed


torch.backends.cudnn.benchmark = True

FOCAL_ALPHA = None
IGNORE_INDEX = 0


def build_run_name(args):
    ls_tag = f"ls{args.label_smoothing}" if args.label_smoothing > 0 else "nols"
    mask_tag = "maskonly" if args.mask_only_loss else "alltok"
    name = (f"{args.loss_type}_{mask_tag}_{ls_tag}_s{args.seed}"
            f"_mr{args.mask_ratio}_d{args.dim}")
    if args.tag:
        name = f"{name}_{args.tag}"
    return name


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0,
                 ignore_index=0, loss_type="ce"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.loss_type = loss_type

    def forward(self, logits, targets, loss_mask=None):
        V = logits.shape[-1]

        valid = (targets != self.ignore_index)
        if loss_mask is not None:
            valid = valid & loss_mask
        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing > 0:
            true_dist = torch.full_like(log_probs,
                                        self.label_smoothing / max(V - 1, 1))
            true_dist.scatter_(1, targets.unsqueeze(1),
                               1.0 - self.label_smoothing)
        else:
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0)

        ce_per_sample = -(true_dist * log_probs).sum(dim=-1)

        if self.loss_type == "ce":
            return ce_per_sample.mean()

        pt = (true_dist * probs).sum(dim=-1)
        focal_factor = (1.0 - pt).clamp(min=1e-6) ** self.gamma
        loss = focal_factor * ce_per_sample
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--data_path", type=str, default="alldataset.csv")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    p.add_argument("--model_path", type=str, default="")

    p.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--mask_only_loss", type=int, default=1, choices=[0, 1])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument("--max_len", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--encoder_head", type=int, default=8)
    p.add_argument("--decoder_head", type=int, default=8)
    p.add_argument("--encoder_layer", type=int, default=6)
    p.add_argument("--decoder_layer", type=int, default=4)
    p.add_argument("--mask_ratio", type=float, default=0.75)

    p.add_argument("--base_learning_rate", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--total_epoch", type=int, default=600)
    p.add_argument("--warmup_epoch", type=int, default=60)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.99995)

    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="logs")
    return p


def main():
    args = build_parser().parse_args()

    assert args.dim % args.encoder_head == 0, "dim 必须能被 encoder_head 整除"
    assert args.dim % args.decoder_head == 0, "dim 必须能被 decoder_head 整除"
    assert 0.0 < args.mask_ratio < 1.0, "mask_ratio 必须在 (0,1) 开区间"
    assert args.val_every >= 1

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = build_run_name(args)
    if not args.model_path:
        args.model_path = os.path.join(args.out_dir, f"ckpt_{run_name}.pt")

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp) and device == "cuda"
    pin = device == "cuda"

    print("=" * 78)
    print(f"run_name      : {run_name}")
    print(f"model_path    : {os.path.abspath(args.model_path)}")
    print(f"device        : {device}   amp={use_amp}")
    print(f"loss          : {args.loss_type}  gamma={args.focal_gamma}  "
          f"alpha={FOCAL_ALPHA}")
    print(f"mask_only     : {bool(args.mask_only_loss)}   "
          f"label_smoothing={args.label_smoothing}")
    print(f"arch          : dim={args.dim} enc={args.encoder_layer}L/"
          f"{args.encoder_head}H dec={args.decoder_layer}L/{args.decoder_head}H")
    print(f"mask_ratio    : {args.mask_ratio}   max_len={args.max_len}")
    print(f"seeds         : init/shuffle={args.seed}   split={args.split_seed}")
    print("=" * 78)

    if args.loss_type == "focal" and args.label_smoothing > 0:
        print("[note] focal x label_smoothing: 调制因子取自平滑目标下的期望后验 "
              "sum_c q_c p_c，lambda=0 时退化为标准 focal。该组合按此定义解释，"
              "不要再按未平滑的 p_true 复算。")
    if not args.mask_only_loss:
        print("[note] mask_only_loss=0 (alltok): 未被掩码的位置对编码器可见，"
              "解码器可从同位置直接复制 token 嵌入，这批位置的损失趋于平凡值，"
              "等效于放大整体梯度尺度。请按'有效学习率修正'而非'监督范围'解释该格。")

    full_dataset = PeptideDataset(args.data_path, max_len=args.max_len)
    n_total = len(full_dataset)
    n_val = int(round(args.val_ratio * n_total))
    n_train = n_total - n_val
    if n_val < 2 or n_train < 2:
        raise ValueError(f"数据集过小: n_total={n_total} n_train={n_train} n_val={n_val}")

    gen = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=gen)
    print(f"dataset       : n_total={n_total} n_train={n_train} n_val={n_val} "
          f"vocab={full_dataset.vocab_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=pin,
        drop_last=False, persistent_workers=min(args.num_workers, 2) > 0)

    model = MAE_Peptide(
        vocab_size=full_dataset.vocab_size,
        max_len=args.max_len,
        dim=args.dim,
        encoder_head=args.encoder_head,
        decoder_head=args.decoder_head,
        encoder_layer=args.encoder_layer,
        decoder_layer=args.decoder_layer,
        mask_ratio=args.mask_ratio,
    ).to(device)

    ema_model = copy.deepcopy(model).eval()
    for pm in ema_model.parameters():
        pm.requires_grad_(False)

    n_param = sum(x.numel() for x in model.parameters()) / 1e6
    print(f"params        : {n_param:.2f}M")

    scaler = make_grad_scaler(use_amp)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate,
                              weight_decay=args.weight_decay)

    def lr_func(epoch):
        if epoch < args.warmup_epoch:
            return (epoch + 1) / args.warmup_epoch
        return 0.5 * (1.0 + math.cos(
            math.pi * (epoch - args.warmup_epoch)
            / max(args.total_epoch - args.warmup_epoch, 1)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    criterion = MaskedFocalLoss(
        gamma=args.focal_gamma,
        alpha=FOCAL_ALPHA,
        label_smoothing=args.label_smoothing,
        ignore_index=IGNORE_INDEX,
        loss_type=args.loss_type,
    )

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
    with open(os.path.join(args.log_dir, run_name, "config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"args": vars(args), "run_name": run_name,
                   "n_total": n_total, "n_train": n_train, "n_val": n_val,
                   "vocab_size": int(full_dataset.vocab_size),
                   "n_param_M": round(n_param, 4)},
                  f, indent=2, ensure_ascii=False)

    best_val_loss = float("inf")
    no_improve_count = 0
    t0 = time.time()
    stopped_epoch = args.total_epoch - 1

    for e in range(args.total_epoch):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {e}")
        for tokens in pbar:
            tokens = tokens.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits, loss_mask = model(tokens)
                targets = tokens[:, 1:]
                lm = (loss_mask.reshape(-1)
                      if (args.mask_only_loss and loss_mask is not None) else None)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1), lm)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optim)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}"})

        lr_scheduler.step()
        cur_lr = lr_scheduler.get_last_lr()[0]

        if e % args.val_every != 0 and e != args.total_epoch - 1:
            continue

        ema_model.eval()
        val_losses = []
        with torch.no_grad():
            for tokens in val_loader:
                tokens = tokens.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=use_amp):
                    logits, loss_mask = ema_model(tokens)
                    targets = tokens[:, 1:]
                    lm = (loss_mask.reshape(-1)
                          if (args.mask_only_loss and loss_mask is not None) else None)
                    loss = criterion(logits.reshape(-1, logits.size(-1)),
                                     targets.reshape(-1), lm)
                val_losses.append(loss.item())

        if not val_losses:
            raise RuntimeError("验证集为空，检查 --val_ratio 与 --batch_size")

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {e} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} "
              f"| lr {cur_lr:.6f} | best {best_val_loss:.4f} "
              f"| {time.time() - t0:.0f}s")
        writer.add_scalar("Loss/Train", avg_train_loss, e)
        writer.add_scalar("Loss/Validation", avg_val_loss, e)
        writer.add_scalar("LearningRate", cur_lr, e)
        writer.flush()

        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save({
                "state_dict": ema_model.state_dict(),
                "args": vars(args),
                "run_name": run_name,
                "epoch": e,
                "val_loss": best_val_loss,
            }, args.model_path)
            print(f"--> saved {args.model_path} (val {best_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                stopped_epoch = e
                print(f"\n[early stop] no improvement for {args.patience} epochs.")
                print(f"best val loss: {best_val_loss:.4f}")
                break

    writer.add_hparams(
        {k: v for k, v in vars(args).items()
         if isinstance(v, (int, float, str, bool))},
        {"hparam/best_val_loss": best_val_loss})
    writer.close()

    print("=" * 78)
    print(f"finished: {run_name}")
    print(f"epochs   : {stopped_epoch + 1}   best_val_loss={best_val_loss:.4f}")
    print(f"ckpt     : {os.path.abspath(args.model_path)}")
    print(f"elapsed  : {time.time() - t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
