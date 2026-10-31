import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import MAE_Peptide
from dataset import PeptideDataset
from utils import setup_seed


torch.backends.cudnn.benchmark = True


# ============================================================================
LOSS_TYPE = 'focal'          # TODO 实验开关: 'ce' / 'focal'
MASK_ONLY_LOSS = True        # TODO 实验开关: True=仅掩码 / False=全部token
LABEL_SMOOTHING = 0.1        # TODO 实验开关: 0.0 关闭 / 0.1 开启
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None
IGNORE_INDEX = 0
# ============================================================================
def build_run_name():
    ls_tag = f"ls{LABEL_SMOOTHING}" if LABEL_SMOOTHING > 0 else "nols"
    mask_tag = "maskonly" if MASK_ONLY_LOSS else "alltok"
    stamp = time.strftime('%m%d_%H%M%S')
    return f"{LOSS_TYPE}_{mask_tag}_{ls_tag}_{stamp}"


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0,
                 ignore_index=0, loss_type='focal'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.loss_type = loss_type

    def forward(self, logits, targets, loss_mask=None):
        N, V = logits.shape

        valid = (targets != self.ignore_index)
        if loss_mask is not None:
            valid = valid & loss_mask
        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]

        log_probs = F.log_softmax(logits, dim=-1)

        if self.label_smoothing > 0:
            true_dist = torch.full_like(log_probs, self.label_smoothing / (V - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0)

        ce_per_sample = -(true_dist * log_probs).sum(dim=-1)

        if self.loss_type == 'ce':
            loss = ce_per_sample
        else:
            pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
            focal_factor = (1.0 - pt).clamp(min=1e-6) ** self.gamma
            loss = focal_factor * ce_per_sample
            if self.alpha is not None:
                loss = self.alpha * loss

        return loss.mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='alldataset.csv')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--encoder_head', type=int, default=8)
    parser.add_argument('--decoder_head', type=int, default=8)
    parser.add_argument('--encoder_layer', type=int, default=6)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--base_learning_rate', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=400)
    parser.add_argument('--warmup_epoch', type=int, default=40)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--model_path', type=str, default='peptide_mae_best.pt')
    args = parser.parse_args()


    RUN_NAME = build_run_name() + f"_mr{args.mask_ratio}_d256"
    args.model_path = f"ckpt_{RUN_NAME}.pt"

    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    full_dataset = PeptideDataset(args.data_path, max_len=args.max_len)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MAE_Peptide(
        vocab_size=full_dataset.vocab_size,
        max_len=args.max_len,
        dim=256,
        encoder_head=args.encoder_head,
        decoder_head=args.decoder_head,
        encoder_layer=args.encoder_layer,
        decoder_layer=args.decoder_layer,
        mask_ratio=args.mask_ratio
    ).to(device)

    ema_model = torch.optim.swa_utils.AveragedModel(model)

    scaler = GradScaler()
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate, weight_decay=args.weight_decay)

    def lr_func(epoch):
        if epoch < args.warmup_epoch:
            return (epoch + 1) / args.warmup_epoch
        else:
            return 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epoch) / (args.total_epoch - args.warmup_epoch)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    criterion = MaskedFocalLoss(
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA,
        label_smoothing=LABEL_SMOOTHING,
        ignore_index=IGNORE_INDEX,
        loss_type=LOSS_TYPE,
    )

    best_val_loss = float('inf')
    no_improve_count = 0
    writer = SummaryWriter(log_dir=f'logs/{RUN_NAME}')

    for e in range(args.total_epoch):
        model.train()
        train_losses = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {e}')
        for tokens in pbar:
            tokens = tokens.to(device, non_blocking=True)

            optim.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss_mask = model(tokens)
                targets = tokens[:, 1:]
                lm = loss_mask.reshape(-1) if (MASK_ONLY_LOSS and loss_mask is not None) else None
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), lm)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optim)
            scaler.update()
            update_ema(ema_model, model)

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{lr_scheduler.get_last_lr()[0]:.6f}"})

        lr_scheduler.step()
        ema_model.eval()
        val_losses = []
        with torch.no_grad():
            for tokens in val_dataloader:
                tokens = tokens.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss_mask = ema_model(tokens)
                    targets = tokens[:, 1:]
                    lm = loss_mask.reshape(-1) if (MASK_ONLY_LOSS and loss_mask is not None) else None
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), lm)
                val_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Epoch {e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        writer.add_scalar('Loss/Train', avg_train_loss, e)
        writer.add_scalar('Loss/Validation', avg_val_loss, e)
        writer.add_scalar('LearningRate', lr_scheduler.get_last_lr()[0], e)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(ema_model.module.state_dict(), args.model_path)
            print(f"--> Saved best model to {args.model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"\n[Early Stopping] Val loss no improvement for {args.patience} epochs. Stop training.")
                print(f"Best val loss: {best_val_loss:.4f}")
                break
# D:\ana\envs\pytorch2.3.1\Scripts\tensorboard.exe --logdir=logs
