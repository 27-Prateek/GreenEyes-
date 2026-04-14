# """
# training/train.py
# =================
# Complete training loop for GreenEyes++.
# """

# import sys
# import time
# import math
# import argparse
# import json
# import csv
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import joblib

# sys.path.append(str(Path(__file__).parent.parent))
# from configs.config import *
# from preprocessing.preprocess import AQIDataset
# from models.model import (
#     GreenEyesPlus, LSTMBaseline, GRUBaseline,
#     TransformerBaseline, WaveNetOnlyBaseline
# )

# set_seed(RANDOM_SEED)


# # ══════════════════════════════════════════════════════════════════════════════
# # DATASET
# # ══════════════════════════════════════════════════════════════════════════════

# class AQISequenceDataset(AQIDataset):
#     """
#     Thin wrapper around AQIDataset that adds city_idx and returns
#     a 4-tuple (X, y_reg, y_cls, city_idx) expected by the training loop.
#     """
#     def __init__(self, pt_path):
#         # Accept paths like DATA_PROCESSED / 'train.pt' for backward compat
#         path     = Path(str(pt_path).replace('.pt', ''))
#         split    = path.stem          # 'train', 'val', or 'test'
#         data_dir = path.parent
#         super().__init__(data_dir, split)
#         # _split is already stored by AQIDataset.__init__
#         self.city_idx = torch.tensor(
#             [m['city_idx'] for m in self.meta], dtype=torch.long
#         )

#     def __getitem__(self, idx):
#         X, y_reg, y_cls = super().__getitem__(idx)
#         return X, y_reg, y_cls, self.city_idx[idx]


# # ══════════════════════════════════════════════════════════════════════════════
# # LOSS
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_class_weights(dataset: AQISequenceDataset, n_classes: int = 6) -> torch.Tensor:
#     """Inverse-frequency class weights — reads directly from the memmap npy file."""
#     y_cls_np = np.load(
#         str(dataset.data_dir / f'{dataset._split}_y_cls.npy'), mmap_mode='r'
#     )
#     y_cls   = torch.from_numpy(np.array(y_cls_np)).long()
#     counts  = torch.bincount(y_cls.clamp(0, n_classes - 1), minlength=n_classes).float()
#     counts  = counts + 1                            # Laplace smoothing
#     weights = 1.0 / counts
#     weights = weights / weights.sum() * n_classes
#     return weights


# class JointLoss(nn.Module):
#     def __init__(self, alpha: float, beta: float, class_weights: torch.Tensor = None):
#         super().__init__()
#         self.alpha = alpha
#         self.beta  = beta
#         self.mse   = nn.MSELoss(reduction='mean')
#         self.ce    = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

#     def forward(self, reg_preds, cat_logits, y_reg, y_cls):
#         reg_loss = torch.stack([
#             self.mse(pred.squeeze(), y_reg[:, i])
#             for i, pred in enumerate(reg_preds)
#         ]).mean()
#         valid = y_cls >= 0
#         if valid.any():
#             cls_loss = self.ce(cat_logits[valid], y_cls[valid])
#         else:
#             cls_loss = torch.tensor(0.0, device=reg_loss.device)
#         total = self.alpha * reg_loss + self.beta * cls_loss
#         return total, float(reg_loss), float(cls_loss)


# # ══════════════════════════════════════════════════════════════════════════════
# # METRICS
# # ══════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def compute_metrics(reg_preds, cat_logits, y_reg, y_cls, horizons):
#     metrics = {}
#     all_mae, all_rmse, all_r2 = [], [], []
#     for i, (pred, h) in enumerate(zip(reg_preds, horizons)):
#         p = pred.squeeze().cpu().numpy()
#         t = y_reg[:, i].cpu().numpy()
#         mae    = float(np.abs(p - t).mean())
#         rmse   = float(np.sqrt(((p - t)**2).mean()))
#         ss_res = ((t - p)**2).sum()
#         ss_tot = ((t - t.mean())**2).sum()
#         r2     = float(1 - ss_res / (ss_tot + 1e-8))
#         metrics[f'MAE_t{h}h']  = round(mae,  6)
#         metrics[f'RMSE_t{h}h'] = round(rmse, 6)
#         metrics[f'R2_t{h}h']   = round(r2,   6)
#         all_mae.append(mae); all_rmse.append(rmse); all_r2.append(r2)
#     metrics['MAE_mean']  = round(float(np.mean(all_mae)),  6)
#     metrics['RMSE_mean'] = round(float(np.mean(all_rmse)), 6)
#     metrics['R2_mean']   = round(float(np.mean(all_r2)),   6)
#     valid = y_cls >= 0
#     if valid.any():
#         pred_cls = cat_logits[valid].argmax(dim=1).cpu()
#         acc = float((pred_cls == y_cls[valid].cpu()).float().mean())
#         metrics['cat_acc'] = round(acc, 6)
#     return metrics


# # ══════════════════════════════════════════════════════════════════════════════
# # LR SCHEDULE
# # ══════════════════════════════════════════════════════════════════════════════

# def make_scheduler(optimizer, n_epochs: int, warmup: int = WARMUP_EPOCHS):
#     def lr_lambda(epoch):
#         if epoch < warmup:
#             return (epoch + 1) / max(1, warmup)
#         p = (epoch - warmup) / max(1, n_epochs - warmup)
#         return 0.5 * (1 + math.cos(math.pi * p))
#     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# # ══════════════════════════════════════════════════════════════════════════════
# # ONE EPOCH
# # ══════════════════════════════════════════════════════════════════════════════

# def run_epoch(model, loader, loss_fn, device, adj, optimizer=None, horizons=None):
#     is_train = optimizer is not None
#     model.train() if is_train else model.eval()
#     total_loss = total_reg = total_cls = 0.0
#     all_reg_preds = [[] for _ in horizons]
#     all_y_reg, all_cat_logits, all_y_cls = [], [], []
#     ctx = torch.enable_grad() if is_train else torch.no_grad()
#     with ctx:
#         for X, y_reg, y_cls, city_idx in loader:
#             X        = X.to(device)
#             y_reg    = y_reg.to(device)
#             y_cls    = y_cls.to(device)
#             city_idx = city_idx.to(device)
#             reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)
#             loss, reg_l, cls_l = loss_fn(reg_preds, cat_logits, y_reg, y_cls)
#             if is_train:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
#                 optimizer.step()
#             total_loss += float(loss)
#             total_reg  += reg_l
#             total_cls  += cls_l
#             for i, p in enumerate(reg_preds):
#                 all_reg_preds[i].append(p.detach().cpu())
#             all_y_reg.append(y_reg.detach().cpu())
#             all_cat_logits.append(cat_logits.detach().cpu())
#             all_y_cls.append(y_cls.detach().cpu())
#     n       = len(loader)
#     reg_cat = [torch.cat(all_reg_preds[i]) for i in range(len(all_reg_preds))]
#     y_reg_c = torch.cat(all_y_reg)
#     cat_c   = torch.cat(all_cat_logits)
#     y_cls_c = torch.cat(all_y_cls)
#     metrics = compute_metrics(reg_cat, cat_c, y_reg_c, y_cls_c, horizons)
#     metrics['loss']     = round(total_loss / n, 6)
#     metrics['reg_loss'] = round(total_reg  / n, 6)
#     metrics['cls_loss'] = round(total_cls  / n, 6)
#     return metrics


# # ══════════════════════════════════════════════════════════════════════════════
# # TRAINING MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# MODEL_REGISTRY = {
#     'greeneyes':   GreenEyesPlus,
#     'lstm':        LSTMBaseline,
#     'gru':         GRUBaseline,
#     'transformer': TransformerBaseline,
#     'wavenet':     WaveNetOnlyBaseline,
# }


# def train(model_name: str = 'greeneyes'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\nDevice: {device}")

#     meta        = joblib.load(DATA_PROCESSED / 'meta.joblib')
#     n_feat      = meta['n_features']
#     horizons    = meta['forecast_hours']
#     adj_np      = meta['adj']
#     city_to_idx = meta['city_to_idx']
#     adj         = torch.tensor(adj_np, dtype=torch.float32).to(device)

#     train_ds = AQISequenceDataset(DATA_PROCESSED / 'train.pt')
#     val_ds   = AQISequenceDataset(DATA_PROCESSED / 'val.pt')

#     # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
#     #                           num_workers=2, pin_memory=(device.type=='cuda'),
#     #                           drop_last=True)
#     # val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
#     #                           num_workers=2)
    
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
#                               num_workers=0,drop_last=True)
#     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
#                               num_workers=0)
    
#     print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Features: {n_feat}")

#     ModelClass = MODEL_REGISTRY.get(model_name, GreenEyesPlus)
#     if model_name == 'greeneyes':
#         model = ModelClass(
#             n_features    = n_feat,
#             n_cities      = N_CITIES,
#             hidden        = HIDDEN_DIM,
#             wavenet_layers= WAVENET_LAYERS,
#             kernel_size   = KERNEL_SIZE,
#             lstm_layers   = LSTM_LAYERS,
#             lstm_dropout  = LSTM_DROPOUT,
#             gnn_heads     = GNN_HEADS,
#             n_categories  = N_CATEGORIES,
#             horizons      = horizons,
#             dropout       = DROPOUT,
#         )
#     else:
#         model = ModelClass(
#             n_features   = n_feat,
#             horizons     = horizons,
#             n_categories = N_CATEGORIES,
#         )
#     model = model.to(device)
#     print(f"Model: {model_name} | Parameters: {model.count_parameters():,}")

#     class_weights = compute_class_weights(train_ds, N_CATEGORIES).to(device)
#     loss_fn   = JointLoss(LOSS_ALPHA, LOSS_BETA, class_weights)
#     optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

#     ckpt_dir = CHECKPOINTS_DIR / model_name
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     log_path   = ckpt_dir / 'training_log.csv'
#     log_fields = ['epoch', 'phase', 'loss', 'reg_loss', 'cls_loss',
#                   'MAE_mean', 'RMSE_mean', 'R2_mean', 'cat_acc', 'lr', 'elapsed_s']
#     with open(log_path, 'w', newline='') as f:
#         csv.DictWriter(f, fieldnames=log_fields).writeheader()

#     best_val_loss = float('inf')
#     patience_cnt  = 0
#     history       = []

#     print(f"\nTraining {model_name} for up to {EPOCHS} epochs...\n")
#     print(f"{'Ep':>4} | {'Train Loss':>10} | {'Val Loss':>9} | "
#           f"{'Val MAE':>8} | {'Val R²':>7} | {'Val Acc':>7} | {'LR':>9}")
#     print("-" * 75)

#     for epoch in range(1, EPOCHS + 1):
#         t0      = time.time()
#         train_m = run_epoch(model, train_loader, loss_fn, device, adj,
#                             optimizer=optimizer, horizons=horizons)
#         val_m   = run_epoch(model, val_loader,   loss_fn, device, adj,
#                             horizons=horizons)
#         scheduler.step()
#         elapsed = time.time() - t0
#         lr_now  = optimizer.param_groups[0]['lr']

#         print(f"{epoch:>4} | {train_m['loss']:>10.4f} | {val_m['loss']:>9.4f} | "
#               f"{val_m['MAE_mean']:>8.4f} | {val_m['R2_mean']:>7.4f} | "
#               f"{val_m.get('cat_acc',0):>7.4f} | {lr_now:>9.2e}")

#         for phase, m in [('train', train_m), ('val', val_m)]:
#             row = {k: m.get(k, '') for k in log_fields}
#             row['epoch'] = epoch; row['phase'] = phase
#             row['lr'] = lr_now; row['elapsed_s'] = round(elapsed, 2)
#             with open(log_path, 'a', newline='') as f:
#                 csv.DictWriter(f, fieldnames=log_fields).writerow(row)

#         history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

#         if val_m['loss'] < best_val_loss:
#             best_val_loss = val_m['loss']
#             patience_cnt  = 0
#             torch.save({
#                 'epoch':       epoch,
#                 'model_state': model.state_dict(),
#                 'val_metrics': val_m,
#                 'model_name':  model_name,
#                 'n_features':  n_feat,
#                 'horizons':    horizons,
#             }, ckpt_dir / 'best_model.pt')
#             print(f"       ↑ saved best (val_loss={best_val_loss:.4f})")
#         else:
#             patience_cnt += 1
#             if patience_cnt >= PATIENCE:
#                 print(f"\nEarly stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
#                 break

#     with open(ckpt_dir / 'history.json', 'w') as f:
#         json.dump(history, f, indent=2)

#     print(f"\nBest val loss: {best_val_loss:.6f}")
#     print(f"Checkpoint: {ckpt_dir / 'best_model.pt'}")
#     return str(ckpt_dir / 'best_model.pt')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default='greeneyes',
#                         choices=list(MODEL_REGISTRY.keys()))
#     args = parser.parse_args()
#     train(args.model)





"""
training/train.py
=================
Complete training loop for GreenEyes++.
Updated with:
  - Automatic mixed precision (AMP) — halves VRAM, ~1.5x faster on RTX 3050
  - Gradient accumulation — effective batch size = BATCH_SIZE × GRAD_ACCUM_STEPS
  - Explicit CUDA device print on startup
  - num_workers=0 for Windows compatibility (multiprocessing fork issues on Windows)
"""

import sys
import time
import math
import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import joblib

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *
from preprocessing.preprocess import AQIDataset
from models.model import (
    GreenEyesPlus, LSTMBaseline, GRUBaseline,
    TransformerBaseline, WaveNetOnlyBaseline
)

set_seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class AQISequenceDataset(AQIDataset):
    """
    Thin wrapper: adds city_idx and returns 4-tuple
    (X, y_reg, y_cls, city_idx) expected by the training loop.
    Accepts paths like DATA_PROCESSED / 'train.pt' for convenience
    even though the backend is now .npy memmaps.
    """
    def __init__(self, pt_path):
        path     = Path(str(pt_path).replace('.pt', ''))
        split    = path.stem        # 'train', 'val', or 'test'
        data_dir = path.parent
        super().__init__(data_dir, split)
        self.city_idx = torch.tensor(
            [m['city_idx'] for m in self.meta], dtype=torch.long
        )

    def __getitem__(self, idx):
        X, y_reg, y_cls = super().__getitem__(idx)
        return X, y_reg, y_cls, self.city_idx[idx]


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(dataset: AQISequenceDataset, n_classes: int = 6) -> torch.Tensor:
    """Reads y_cls directly from the memmap .npy file — no RAM spike."""
    y_cls_np = np.load(
        str(dataset.data_dir / f'{dataset._split}_y_cls.npy'), mmap_mode='r'
    )
    y_cls   = torch.from_numpy(np.array(y_cls_np)).long()
    counts  = torch.bincount(y_cls.clamp(0, n_classes - 1), minlength=n_classes).float()
    counts  = counts + 1                             # Laplace smoothing
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


class JointLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.mse   = nn.MSELoss(reduction='mean')
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(self, reg_preds, cat_logits, y_reg, y_cls):
        reg_loss = torch.stack([
            self.mse(pred.squeeze(), y_reg[:, i])
            for i, pred in enumerate(reg_preds)
        ]).mean()
        valid = y_cls >= 0
        cls_loss = (self.ce(cat_logits[valid], y_cls[valid])
                    if valid.any()
                    else torch.tensor(0.0, device=reg_loss.device))
        total = self.alpha * reg_loss + self.beta * cls_loss
        return total, float(reg_loss), float(cls_loss)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(reg_preds, cat_logits, y_reg, y_cls, horizons):
    metrics = {}
    all_mae, all_rmse, all_r2 = [], [], []
    for i, (pred, h) in enumerate(zip(reg_preds, horizons)):
        p      = pred.squeeze().cpu().numpy()
        t      = y_reg[:, i].cpu().numpy()
        mae    = float(np.abs(p - t).mean())
        rmse   = float(np.sqrt(((p - t) ** 2).mean()))
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2     = float(1 - ss_res / (ss_tot + 1e-8))
        metrics[f'MAE_t{h}h']  = round(mae,  6)
        metrics[f'RMSE_t{h}h'] = round(rmse, 6)
        metrics[f'R2_t{h}h']   = round(r2,   6)
        all_mae.append(mae); all_rmse.append(rmse); all_r2.append(r2)
    metrics['MAE_mean']  = round(float(np.mean(all_mae)),  6)
    metrics['RMSE_mean'] = round(float(np.mean(all_rmse)), 6)
    metrics['R2_mean']   = round(float(np.mean(all_r2)),   6)
    valid = y_cls >= 0
    if valid.any():
        pred_cls       = cat_logits[valid].argmax(dim=1).cpu()
        metrics['cat_acc'] = round(float((pred_cls == y_cls[valid].cpu()).float().mean()), 6)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def make_scheduler(optimizer, n_epochs: int, warmup: int = WARMUP_EPOCHS):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        p = (epoch - warmup) / max(1, n_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# ONE EPOCH  (with AMP + gradient accumulation)
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, loss_fn, device, adj,
              optimizer=None, horizons=None,
              scaler: GradScaler = None,
              accum_steps: int = 1):
    """
    optimizer=None  → eval mode (no gradients, no scaler)
    scaler=None     → FP32 training (AMP disabled)
    accum_steps > 1 → gradient accumulation
    """
    is_train = optimizer is not None
    use_amp  = (scaler is not None) and is_train and (device.type == 'cuda')
    model.train() if is_train else model.eval()

    total_loss = total_reg = total_cls = 0.0
    all_reg_preds = [[] for _ in horizons]
    all_y_reg, all_cat_logits, all_y_cls = [], [], []

    if is_train:
        optimizer.zero_grad()
    loop = tqdm(loader, leave=False)
    # for batch_idx, (X, y_reg, y_cls, city_idx) in enumerate(loader):
    for batch_idx, (X, y_reg, y_cls, city_idx) in enumerate(loop):
        X        = X.to(device, non_blocking=True)
        y_reg    = y_reg.to(device, non_blocking=True)
        y_cls    = y_cls.to(device, non_blocking=True)
        city_idx = city_idx.to(device, non_blocking=True)

        if is_train:
            with autocast(enabled=use_amp):
                reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)
                loss, reg_l, cls_l   = loss_fn(reg_preds, cat_logits, y_reg, y_cls)
                loss = loss / accum_steps          # scale for accumulation

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step only on accumulation boundary or last batch
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad()

        else:
            with torch.no_grad():
                with autocast(enabled=(device.type == 'cuda' and USE_AMP)):
                    reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)
                    loss, reg_l, cls_l   = loss_fn(reg_preds, cat_logits, y_reg, y_cls)
                    

        total_loss += float(loss) * (accum_steps if is_train else 1)
        total_reg  += reg_l
        total_cls  += cls_l
        if batch_idx % 100 == 0:
            loop.set_postfix({
            "loss": f"{float(loss):.4f}"
            }
        )
        for i, p in enumerate(reg_preds):
            all_reg_preds[i].append(p.detach().cpu())
        all_y_reg.append(y_reg.detach().cpu())
        all_cat_logits.append(cat_logits.detach().cpu())
        all_y_cls.append(y_cls.detach().cpu())

    n       = len(loader)
    reg_cat = [torch.cat(all_reg_preds[i]) for i in range(len(all_reg_preds))]
    y_reg_c = torch.cat(all_y_reg)
    cat_c   = torch.cat(all_cat_logits)
    y_cls_c = torch.cat(all_y_cls)

    metrics = compute_metrics(reg_cat, cat_c, y_reg_c, y_cls_c, horizons)
    metrics['loss']     = round(total_loss / n, 6)
    metrics['reg_loss'] = round(total_reg  / n, 6)
    metrics['cls_loss'] = round(total_cls  / n, 6)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING MAIN
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    'greeneyes':   GreenEyesPlus,
    'lstm':        LSTMBaseline,
    'gru':         GRUBaseline,
    'transformer': TransformerBaseline,
    'wavenet':     WaveNetOnlyBaseline,
}


def train(model_name: str = 'greeneyes'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {total_vram:.2f} GB")
        print(f"AMP    : {'enabled' if USE_AMP else 'disabled'}")
    print(f"Batch  : {BATCH_SIZE} × {GRAD_ACCUM_STEPS} accum = "
          f"effective {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    meta        = joblib.load(DATA_PROCESSED / 'meta.joblib')
    n_feat      = meta['n_features']
    horizons    = meta['forecast_hours']
    adj_np      = meta['adj']
    adj         = torch.tensor(adj_np, dtype=torch.float32).to(device)

    train_ds = AQISequenceDataset(DATA_PROCESSED / 'train.pt')
    val_ds   = AQISequenceDataset(DATA_PROCESSED / 'val.pt')

    # num_workers=0 is required on Windows to avoid multiprocessing issues
    # pin_memory=True speeds up CPU→GPU transfers
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )

    print(f"Train  : {len(train_ds):,} | Val: {len(val_ds):,} | Features: {n_feat}")

    # Build model
    ModelClass = MODEL_REGISTRY.get(model_name, GreenEyesPlus)
    if model_name == 'greeneyes':
        model = ModelClass(
            n_features    = n_feat,
            n_cities      = N_CITIES,
            hidden        = HIDDEN_DIM,
            wavenet_layers= WAVENET_LAYERS,
            kernel_size   = KERNEL_SIZE,
            lstm_layers   = LSTM_LAYERS,
            lstm_dropout  = LSTM_DROPOUT,
            gnn_heads     = GNN_HEADS,
            n_categories  = N_CATEGORIES,
            horizons      = horizons,
            dropout       = DROPOUT,
        )
    else:
        model = ModelClass(
            n_features   = n_feat,
            horizons     = horizons,
            n_categories = N_CATEGORIES,
        )
    model = model.to(device)
    print(f"Model  : {model_name} | Parameters: {model.count_parameters():,}")

    # VRAM estimate
    if device.type == 'cuda':
        dummy = torch.randn(BATCH_SIZE, WINDOW_SIZE, n_feat, device=device)
        with autocast(enabled=USE_AMP):
            _ = model(dummy, adj=adj)
        del dummy
        used = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM used after forward pass: {used:.2f} GB")
        torch.cuda.empty_cache()

    class_weights = compute_class_weights(train_ds, N_CATEGORIES).to(device)
    loss_fn   = JointLoss(LOSS_ALPHA, LOSS_BETA, class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    # AMP scaler — only used when USE_AMP=True and GPU present
    scaler = GradScaler() if (USE_AMP and device.type == 'cuda') else None

    ckpt_dir = CHECKPOINTS_DIR / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path   = ckpt_dir / 'training_log.csv'
    log_fields = ['epoch', 'phase', 'loss', 'reg_loss', 'cls_loss',
                  'MAE_mean', 'RMSE_mean', 'R2_mean', 'cat_acc', 'lr', 'elapsed_s']
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_loss = float('inf')
    patience_cnt  = 0
    history       = []

    print(f"\nTraining {model_name} for up to {EPOCHS} epochs...\n")
    print(f"{'Ep':>4} | {'Train Loss':>10} | {'Val Loss':>9} | "
          f"{'Val MAE':>8} | {'Val R²':>7} | {'Val Acc':>7} | {'LR':>9}")
    print("-" * 75)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_m = run_epoch(
            model, train_loader, loss_fn, device, adj,
            optimizer=optimizer, horizons=horizons,
            scaler=scaler, accum_steps=GRAD_ACCUM_STEPS
        )
        val_m = run_epoch(
            model, val_loader, loss_fn, device, adj,
            horizons=horizons
        )
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f"{epoch:>4} | {train_m['loss']:>10.4f} | {val_m['loss']:>9.4f} | "
              f"{val_m['MAE_mean']:>8.4f} | {val_m['R2_mean']:>7.4f} | "
              f"{val_m.get('cat_acc', 0):>7.4f} | {lr_now:>9.2e}")

        for phase, m in [('train', train_m), ('val', val_m)]:
            row = {k: m.get(k, '') for k in log_fields}
            row.update({'epoch': epoch, 'phase': phase,
                        'lr': lr_now, 'elapsed_s': round(elapsed, 2)})
            with open(log_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            patience_cnt  = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_m,
                'model_name':  model_name,
                'n_features':  n_feat,
                'horizons':    horizons,
            }, ckpt_dir / 'best_model.pt')
            print(f"       ↑ saved best (val_loss={best_val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    with open(ckpt_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss : {best_val_loss:.6f}")
    print(f"Checkpoint    : {ckpt_dir / 'best_model.pt'}")
    return str(ckpt_dir / 'best_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='greeneyes',
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()
    train(args.model)