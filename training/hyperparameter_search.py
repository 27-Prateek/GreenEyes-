# """
# training/hyperparameter_search.py
# ==================================
# Optuna hyperparameter search for GreenEyes++.
# """

# import sys
# import argparse
# import json
# from pathlib import Path

# import torch
# import joblib
# import optuna
# from torch.utils.data import DataLoader, Subset
# import torch.optim as optim

# sys.path.append(str(Path(__file__).parent.parent))
# from configs.config import *
# from models.model import GreenEyesPlus
# from training.train import AQISequenceDataset, JointLoss, compute_class_weights, run_epoch, make_scheduler

# set_seed(RANDOM_SEED)
# optuna.logging.set_verbosity(optuna.logging.WARNING)


# def objective(trial, meta, train_sub, val_ds, device, adj):
#     hidden       = trial.suggest_categorical('hidden', [64, 128, 256])
#     lr           = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
#     alpha        = trial.suggest_float('alpha', 0.60, 0.85)
#     dropout      = trial.suggest_float('dropout', 0.05, 0.25)
#     lstm_layers  = trial.suggest_int('lstm_layers', 1, 3)
#     wn1          = trial.suggest_int('wn_block1', 4, 10)
#     wn2          = trial.suggest_int('wn_block2', 2, 6)
#     wn3          = trial.suggest_int('wn_block3', 1, 4)
#     batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])
#     weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

#     train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
#                               num_workers=0, drop_last=True)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
#                               num_workers=0)

#     model = GreenEyesPlus(
#         n_features    = meta['n_features'],
#         n_cities      = N_CITIES,
#         hidden        = hidden,
#         wavenet_layers= [wn1, wn2, wn3],
#         lstm_layers   = lstm_layers,
#         dropout       = dropout,
#         horizons      = meta['forecast_hours'],
#     ).to(device)

#     class_weights = compute_class_weights(train_sub.dataset, N_CATEGORIES).to(device)
#     loss_fn   = JointLoss(alpha, 1-alpha, class_weights)
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = make_scheduler(optimizer, n_epochs=25, warmup=3)

#     best_val = float('inf')
#     for epoch in range(25):
#         run_epoch(model, train_loader, loss_fn, device, adj,
#                   optimizer=optimizer, horizons=meta['forecast_hours'])
#         val_m    = run_epoch(model, val_loader, loss_fn, device, adj,
#                              horizons=meta['forecast_hours'])
#         scheduler.step()
#         val_loss = val_m['loss']
#         if val_loss < best_val:
#             best_val = val_loss
#         trial.report(val_loss, epoch)
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
#     return best_val


# def run_search(n_trials: int, timeout: int = None):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Device: {device}")

#     meta = joblib.load(DATA_PROCESSED / 'meta.joblib')
#     adj  = torch.tensor(meta['adj'], dtype=torch.float32).to(device)

#     train_ds = AQISequenceDataset(DATA_PROCESSED / 'train.pt')
#     val_ds   = AQISequenceDataset(DATA_PROCESSED / 'val.pt')

#     n_sub     = min(len(train_ds), 15000)
#     idx       = torch.randperm(len(train_ds))[:n_sub]
#     train_sub = Subset(train_ds, idx.tolist())

#     study = optuna.create_study(
#         direction    = 'minimize',
#         sampler      = optuna.samplers.TPESampler(seed=RANDOM_SEED),
#         pruner       = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
#         study_name   = 'greeneyes_plus',
#         storage      = f'sqlite:///{DATA_PROCESSED}/optuna.db',
#         load_if_exists=True,
#     )
#     study.optimize(
#         lambda trial: objective(trial, meta, train_sub, val_ds, device, adj),
#         n_trials=n_trials, timeout=timeout, show_progress_bar=True,
#     )

#     best = study.best_trial
#     print(f"\n{'='*50}")
#     print(f"Best trial #{best.number}  val_loss={best.value:.6f}")
#     print("Best hyperparameters:")
#     for k, v in best.params.items():
#         print(f"  {k}: {v}")

#     out      = {'value': best.value, 'params': best.params}
#     out_path = DATA_PROCESSED / 'best_hparams.json'
#     with open(out_path, 'w') as f:
#         json.dump(out, f, indent=2)
#     print(f"\nSaved to {out_path}")

#     p = best.params
#     print("\nSuggested config updates:")
#     print(f"  HIDDEN_DIM    = {p.get('hidden', HIDDEN_DIM)}")
#     print(f"  WAVENET_LAYERS= [{p.get('wn_block1',8)}, {p.get('wn_block2',5)}, {p.get('wn_block3',3)}]")
#     print(f"  LSTM_LAYERS   = {p.get('lstm_layers', LSTM_LAYERS)}")
#     print(f"  DROPOUT       = {p.get('dropout', DROPOUT):.3f}")
#     print(f"  LEARNING_RATE = {p.get('lr', LEARNING_RATE):.2e}")
#     print(f"  BATCH_SIZE    = {p.get('batch_size', BATCH_SIZE)}")
#     print(f"  LOSS_ALPHA    = {p.get('alpha', LOSS_ALPHA):.3f}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_trials', type=int, default=50)
#     parser.add_argument('--timeout',  type=int, default=None)
#     args = parser.parse_args()
#     run_search(args.n_trials, args.timeout)





"""
training/hyperparameter_search.py
==================================
Optuna hyperparameter search for GreenEyes++.
Fixed to use the memmap-backed AQIDataset from preprocess.py
instead of the old .pt-based AQISequenceDataset.
"""

import sys
import argparse
import json
from pathlib import Path

import torch
import joblib
import optuna
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *
from models.model import GreenEyesPlus
from training.train import AQISequenceDataset, JointLoss, compute_class_weights, run_epoch, make_scheduler

set_seed(RANDOM_SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, meta, train_sub, val_ds, device, adj):
    hidden       = trial.suggest_categorical('hidden', [64, 128, 256])
    lr           = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    alpha        = trial.suggest_float('alpha', 0.60, 0.85)
    dropout      = trial.suggest_float('dropout', 0.05, 0.25)
    lstm_layers  = trial.suggest_int('lstm_layers', 1, 3)
    wn1          = trial.suggest_int('wn_block1', 4, 10)
    wn2          = trial.suggest_int('wn_block2', 2, 6)
    wn3          = trial.suggest_int('wn_block3', 1, 4)
    batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    model = GreenEyesPlus(
        n_features    = meta['n_features'],
        n_cities      = N_CITIES,
        hidden        = hidden,
        wavenet_layers= [wn1, wn2, wn3],
        lstm_layers   = lstm_layers,
        dropout       = dropout,
        horizons      = meta['forecast_hours'],
    ).to(device)

    # compute_class_weights now reads from the memmap npy directly
    class_weights = compute_class_weights(train_sub.dataset, N_CATEGORIES).to(device)
    loss_fn   = JointLoss(alpha, 1 - alpha, class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, n_epochs=25, warmup=3)

    best_val = float('inf')
    for epoch in range(25):
        run_epoch(model, train_loader, loss_fn, device, adj,
                  optimizer=optimizer, horizons=meta['forecast_hours'])
        val_m    = run_epoch(model, val_loader, loss_fn, device, adj,
                             horizons=meta['forecast_hours'])
        scheduler.step()
        val_loss = val_m['loss']
        if val_loss < best_val:
            best_val = val_loss
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_val


def run_search(n_trials: int, timeout: int = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    meta = joblib.load(DATA_PROCESSED / 'meta.joblib')
    adj  = torch.tensor(meta['adj'], dtype=torch.float32).to(device)

    # Use the memmap-backed dataset via AQISequenceDataset wrapper
    train_ds = AQISequenceDataset(DATA_PROCESSED / 'train.pt')
    val_ds   = AQISequenceDataset(DATA_PROCESSED / 'val.pt')

    n_sub     = min(len(train_ds), 15000)
    idx       = torch.randperm(len(train_ds))[:n_sub]
    train_sub = Subset(train_ds, idx.tolist())

    study = optuna.create_study(
        direction     = 'minimize',
        sampler       = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner        = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
        study_name    = 'greeneyes_plus',
        storage       = f'sqlite:///{DATA_PROCESSED}/optuna.db',
        load_if_exists= True,
    )
    study.optimize(
        lambda trial: objective(trial, meta, train_sub, val_ds, device, adj),
        n_trials=n_trials, timeout=timeout, show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n{'='*50}")
    print(f"Best trial #{best.number}  val_loss={best.value:.6f}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out      = {'value': best.value, 'params': best.params}
    out_path = DATA_PROCESSED / 'best_hparams.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    p = best.params
    print("\nSuggested config updates (edit configs/config.py):")
    print(f"  HIDDEN_DIM     = {p.get('hidden',      HIDDEN_DIM)}")
    print(f"  WAVENET_LAYERS = [{p.get('wn_block1', 8)}, "
          f"{p.get('wn_block2', 5)}, {p.get('wn_block3', 3)}]")
    print(f"  LSTM_LAYERS    = {p.get('lstm_layers',  LSTM_LAYERS)}")
    print(f"  DROPOUT        = {p.get('dropout',      DROPOUT):.3f}")
    print(f"  LEARNING_RATE  = {p.get('lr',           LEARNING_RATE):.2e}")
    print(f"  BATCH_SIZE     = {p.get('batch_size',   BATCH_SIZE)}")
    print(f"  LOSS_ALPHA     = {p.get('alpha',        LOSS_ALPHA):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--timeout',  type=int, default=None,
                        help='Max wall-clock seconds for the search')
    args = parser.parse_args()
    run_search(args.n_trials, args.timeout)