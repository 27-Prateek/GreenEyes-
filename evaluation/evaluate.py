# # """
# # evaluation/evaluate.py
# # =======================
# # Comprehensive research evaluation.
# # """

# # import sys
# # import argparse
# # import json
# # import warnings
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import torch
# # import joblib
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from torch.utils.data import DataLoader
# # from sklearn.metrics import (
# #     classification_report, confusion_matrix,
# #     mean_absolute_error, mean_squared_error, r2_score, f1_score
# # )

# # warnings.filterwarnings('ignore')
# # sys.path.append(str(Path(__file__).parent.parent))
# # from configs.config import *
# # from models.model import (
# #     GreenEyesPlus, LSTMBaseline, GRUBaseline,
# #     TransformerBaseline, WaveNetOnlyBaseline, ConformalPredictor
# # )
# # from training.train import AQISequenceDataset, JointLoss, compute_class_weights

# # set_seed(RANDOM_SEED)

# # plt.rcParams.update({
# #     'figure.dpi': 150, 'font.size': 11,
# #     'axes.spines.top': False, 'axes.spines.right': False,
# # })

# # AQI_CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
# # HORIZON_LABELS = {1: 't+1h', 24: 't+24h', 72: 't+72h'}


# # # ══════════════════════════════════════════════════════════════════════════════
# # # MODEL LOADING
# # # ══════════════════════════════════════════════════════════════════════════════

# # def load_model(model_name, n_features, horizons, device):
# #     ckpt_path = CHECKPOINTS_DIR / model_name / 'best_model.pt'
# #     if not ckpt_path.exists():
# #         print(f"  [SKIP] No checkpoint: {ckpt_path}")
# #         return None
# #     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
# #     constructors = {
# #         'greeneyes':   lambda: GreenEyesPlus(
# #             n_features=n_features, n_cities=N_CITIES, hidden=HIDDEN_DIM,
# #             wavenet_layers=WAVENET_LAYERS, lstm_layers=LSTM_LAYERS,
# #             horizons=horizons, dropout=DROPOUT,
# #         ),
# #         'lstm':        lambda: LSTMBaseline(n_features, horizons=horizons),
# #         'gru':         lambda: GRUBaseline(n_features, horizons=horizons),
# #         'transformer': lambda: TransformerBaseline(n_features, horizons=horizons),
# #         'wavenet':     lambda: WaveNetOnlyBaseline(n_features, horizons=horizons),
# #     }
# #     model = constructors[model_name]()
# #     model.load_state_dict(ckpt['model_state'])
# #     return model.to(device).eval()


# # # ══════════════════════════════════════════════════════════════════════════════
# # # FULL PREDICTION PASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @torch.no_grad()
# # def predict_all(model, loader, device, adj, scalers, meta_info):
# #     all_preds  = [[] for _ in model.horizons]
# #     all_y_reg, all_cat_l, all_y_cls = [], [], []
# #     all_city, all_dt = [], []

# #     for (X, y_reg, y_cls, city_idx), batch_meta in zip(loader, _batch_meta(meta_info, loader.batch_size)):
# #         X        = X.to(device)
# #         y_reg    = y_reg.to(device)
# #         city_idx = city_idx.to(device)
# #         reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)
# #         for i, p in enumerate(reg_preds):
# #             all_preds[i].append(p.squeeze().cpu())
# #         all_y_reg.append(y_reg.cpu())
# #         all_cat_l.append(cat_logits.cpu())
# #         all_y_cls.append(y_cls)
# #         all_city.extend([m['city']    for m in batch_meta])
# #         all_dt.extend(  [m['datetime'] for m in batch_meta])

# #     preds   = [torch.cat(all_preds[i]).numpy() for i in range(len(all_preds))]
# #     y_reg   = torch.cat(all_y_reg).numpy()
# #     cat_log = torch.cat(all_cat_l)
# #     y_cls   = torch.cat(all_y_cls).numpy()

# #     if 'AQI_poly' in scalers:
# #         sc    = scalers['AQI_poly']
# #         preds = [sc.inverse_transform(p.reshape(-1,1)).flatten() for p in preds]
# #         y_reg = np.column_stack([
# #             sc.inverse_transform(y_reg[:, i].reshape(-1,1)).flatten()
# #             for i in range(y_reg.shape[1])
# #         ])
# #     return preds, y_reg, cat_log, y_cls, all_city, all_dt


# # def _batch_meta(meta_list, batch_size):
# #     for i in range(0, len(meta_list), batch_size):
# #         yield meta_list[i:i+batch_size]


# # # ══════════════════════════════════════════════════════════════════════════════
# # # METRICS
# # # ══════════════════════════════════════════════════════════════════════════════

# # def compute_all_metrics(preds, y_reg, cat_logits, y_cls, horizons, model_name=''):
# #     rows = []
# #     for i, h in enumerate(horizons):
# #         p, t     = preds[i], y_reg[:, i]
# #         mae      = mean_absolute_error(t, p)
# #         rmse     = np.sqrt(mean_squared_error(t, p))
# #         r2       = r2_score(t, p)
# #         within10 = np.mean(np.abs(p - t) / (np.abs(t) + 1e-6) <= 0.10) * 100
# #         rows.append({
# #             'model': model_name,
# #             'horizon': HORIZON_LABELS.get(h, f't+{h}h'),
# #             'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
# #             'R2': round(r2, 4), 'Within_10pct': round(within10, 2),
# #         })
# #     y_pred_cls = cat_logits.argmax(dim=1).numpy()
# #     valid = y_cls >= 0
# #     if valid.any():
# #         acc = (y_pred_cls[valid] == y_cls[valid]).mean() * 100
# #         f1  = f1_score(y_cls[valid], y_pred_cls[valid], average='weighted', zero_division=0) * 100
# #         rows.append({
# #             'model': model_name, 'horizon': 'classification',
# #             'MAE': '-', 'RMSE': '-',
# #             'R2': round(acc/100, 4), 'Within_10pct': round(f1, 2),
# #         })
# #     return rows


# # def per_city_metrics(preds, y_reg, horizons, cities):
# #     result = []
# #     for h_idx, h in enumerate(horizons):
# #         df_h = pd.DataFrame({'city': cities, 'pred': preds[h_idx], 'true': y_reg[:, h_idx]})
# #         city_metrics = df_h.groupby('city').apply(
# #             lambda g: pd.Series({
# #                 'MAE':  round(mean_absolute_error(g['true'], g['pred']), 3),
# #                 'RMSE': round(np.sqrt(mean_squared_error(g['true'], g['pred'])), 3),
# #                 'R2':   round(r2_score(g['true'], g['pred']), 3),
# #                 'n':    len(g),
# #             })
# #         ).reset_index()
# #         city_metrics['horizon'] = HORIZON_LABELS.get(h, f't+{h}h')
# #         result.append(city_metrics)
# #     return pd.concat(result, ignore_index=True)


# # # ══════════════════════════════════════════════════════════════════════════════
# # # FIGURES
# # # ══════════════════════════════════════════════════════════════════════════════

# # def plot_training_curves(model_name, out_dir):
# #     hist_path = CHECKPOINTS_DIR / model_name / 'history.json'
# #     if not hist_path.exists():
# #         return
# #     with open(hist_path) as f:
# #         history = json.load(f)
# #     epochs     = [h['epoch'] for h in history]
# #     train_loss = [h['train']['loss'] for h in history]
# #     val_loss   = [h['val']['loss']   for h in history]
# #     val_mae    = [h['val']['MAE_mean'] for h in history]
# #     val_r2     = [h['val']['R2_mean']  for h in history]
# #     val_acc    = [h['val'].get('cat_acc', 0) for h in history]

# #     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# #     fig.suptitle(f'{model_name} — Training Curves', fontsize=13, fontweight='bold')
# #     axes[0].plot(epochs, train_loss, label='Train', color='#2196F3', linewidth=1.5)
# #     axes[0].plot(epochs, val_loss,   label='Val',   color='#F44336', linewidth=1.5)
# #     axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
# #     axes[0].set_title('Joint Loss'); axes[0].legend(); axes[0].set_yscale('log')
# #     axes[1].plot(epochs, val_mae, color='#4CAF50', linewidth=1.5)
# #     axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
# #     axes[1].set_title('Validation MAE')
# #     axes[2].plot(epochs, val_r2,  label='R²',      color='#9C27B0', linewidth=1.5)
# #     axes[2].plot(epochs, val_acc, label='Cat. Acc', color='#FF9800', linewidth=1.5)
# #     axes[2].set_xlabel('Epoch'); axes[2].legend()
# #     axes[2].set_title('R² and Category Accuracy')
# #     plt.tight_layout()
# #     out_path = out_dir / f'{model_name}_training_curves.png'
# #     plt.savefig(out_path, bbox_inches='tight'); plt.close()
# #     print(f"  Saved: {out_path.name}")


# # def plot_prediction_vs_truth(preds, y_reg, horizons, scalers, out_dir, model_name, n_samples=500):
# #     n    = min(n_samples, len(preds[0]))
# #     fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5))
# #     if len(horizons) == 1: axes = [axes]
# #     fig.suptitle(f'{model_name} — Predicted vs. True AQI', fontsize=13, fontweight='bold')
# #     for i, (h, ax) in enumerate(zip(horizons, axes)):
# #         p   = preds[i][:n]; t = y_reg[:n, i]
# #         lim = (min(p.min(), t.min())-5, max(p.max(), t.max())+5)
# #         ax.scatter(t, p, alpha=0.3, s=8, color='#1976D2', rasterized=True)
# #         ax.plot(lim, lim, 'r--', linewidth=1.2, label='Perfect')
# #         ax.set_xlabel('True AQI'); ax.set_ylabel('Predicted AQI')
# #         ax.set_title(f'{HORIZON_LABELS.get(h)}  R²={r2_score(t,p):.3f}  MAE={mean_absolute_error(t,p):.2f}')
# #         ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(fontsize=9)
# #     plt.tight_layout()
# #     out_path = out_dir / f'{model_name}_pred_vs_true.png'
# #     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
# #     print(f"  Saved: {out_path.name}")


# # def plot_sample_forecast(preds, y_reg, datetimes, horizons, out_dir, model_name):
# #     fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4*len(horizons)), sharex=False)
# #     if len(horizons) == 1: axes = [axes]
# #     fig.suptitle(f'{model_name} — Sample Forecast vs Ground Truth', fontsize=13, fontweight='bold')
# #     n_show = min(500, len(preds[0]))
# #     for i, (h, ax) in enumerate(zip(horizons, axes)):
# #         ax.plot(range(n_show), y_reg[:n_show, i], label='Ground truth',
# #                 color='#333', linewidth=1.2, alpha=0.8)
# #         ax.plot(range(n_show), preds[i][:n_show], label='Predicted',
# #                 color='#E53935', linewidth=1.2, alpha=0.8, linestyle='--')
# #         ax.set_ylabel('AQI'); ax.set_title(HORIZON_LABELS.get(h, f't+{h}h'))
# #         ax.legend(fontsize=9, loc='upper right')
# #     axes[-1].set_xlabel('Sample index')
# #     plt.tight_layout()
# #     out_path = out_dir / f'{model_name}_sample_forecast.png'
# #     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
# #     print(f"  Saved: {out_path.name}")


# # def plot_city_heatmap(city_df, out_dir, model_name):
# #     pivot = city_df.pivot_table(index='city', columns='horizon', values='MAE')
# #     if pivot.empty: return
# #     fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2), max(6, len(pivot)*0.4)))
# #     sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
# #                 linewidths=0.5, cbar_kws={'label': 'MAE (AQI units)'})
# #     ax.set_title(f'{model_name} — Per-City MAE by Horizon', fontsize=13, fontweight='bold')
# #     plt.tight_layout()
# #     out_path = out_dir / f'{model_name}_city_heatmap.png'
# #     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
# #     print(f"  Saved: {out_path.name}")


# # def plot_confusion_matrix(y_true, y_pred, out_dir, model_name):
# #     valid  = y_true >= 0
# #     cm     = confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(len(AQI_CATEGORIES))))
# #     cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
# #                 xticklabels=AQI_CATEGORIES, yticklabels=AQI_CATEGORIES,
# #                 cbar_kws={'label': '%'}, linewidths=0.5)
# #     ax.set_xlabel('Predicted'); ax.set_ylabel('True')
# #     ax.set_title(f'{model_name} — AQI Category Confusion Matrix (%)', fontsize=13)
# #     plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
# #     out_path = out_dir / f'{model_name}_confusion_matrix.png'
# #     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
# #     np.save(out_dir / f'{model_name}_confusion_matrix.npy', cm)
# #     print(f"  Saved: {out_path.name}")


# # def plot_ablation_bar(ablation_df, out_dir):
# #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# #     fig.suptitle('Ablation Study — Model Comparison', fontsize=13, fontweight='bold')
# #     reg_df  = ablation_df[ablation_df['horizon'] != 'classification']
# #     mean_df = reg_df.groupby('model')[['MAE', 'RMSE', 'R2']].mean().reset_index()
# #     colors  = plt.cm.Set2(np.linspace(0, 1, len(mean_df)))
# #     axes[0].barh(mean_df['model'], mean_df['MAE'], color=colors)
# #     axes[0].set_xlabel('Mean MAE'); axes[0].set_title('Mean MAE (lower = better)')
# #     axes[0].invert_xaxis()
# #     axes[1].barh(mean_df['model'], mean_df['R2'], color=colors)
# #     axes[1].set_xlabel('Mean R²'); axes[1].set_title('Mean R² (higher = better)')
# #     axes[1].set_xlim(0, 1)
# #     plt.tight_layout()
# #     out_path = out_dir / 'ablation_comparison.png'
# #     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
# #     print(f"  Saved: {out_path.name}")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # FEATURE IMPORTANCE (Integrated Gradients)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def compute_feature_importance_ig(model, test_ds, device, adj, feature_cols, n_samples=200):
# #     """Uses __getitem__ to load samples — works with memmap-backed AQIDataset."""
# #     model.eval()
# #     n_samples   = min(n_samples, len(test_ds))
# #     samples     = [test_ds[i] for i in range(n_samples)]
# #     X_sample    = torch.stack([s[0] for s in samples]).to(device)
# #     city_idx    = torch.stack([s[3] for s in samples]).to(device)
# #     baseline    = torch.zeros_like(X_sample)
# #     importances = torch.zeros(len(feature_cols))
# #     n_steps     = 30

# #     for alpha in np.linspace(0, 1, n_steps):
# #         x_interp = (baseline + alpha * (X_sample - baseline)).requires_grad_(True)
# #         reg_preds, _ = model(x_interp, adj=adj, city_idx=city_idx)
# #         loss = reg_preds[0].sum()
# #         loss.backward()
# #         with torch.no_grad():
# #             grad = x_interp.grad.abs().mean(dim=(0, 1)).cpu()
# #         importances += grad / n_steps
# #         x_interp.grad = None

# #     return importances.numpy()


# # def plot_feature_importance(importances, feature_cols, out_dir, model_name, top_n=25):
# #     ranked = np.argsort(importances)[::-1][:top_n]
# #     names  = [feature_cols[i] for i in ranked]
# #     vals   = importances[ranked] / (importances[ranked].max() + 1e-8)
# #     fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
# #     colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
# #     ax.barh(names[::-1], vals[::-1], color=colors[::-1])
# #     ax.set_xlabel('Normalised Importance (Integrated Gradients)')
# #     ax.set_title(f'{model_name} — Top-{top_n} Feature Importances', fontsize=13, fontweight='bold')
# #     ax.set_xlim(0, 1.05); plt.tight_layout()
# #     out_path = out_dir / f'{model_name}_feature_importance.png'
# #     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
# #     print(f"  Saved: {out_path.name}")
# #     imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
# #     imp_df.sort_values('importance', ascending=False).to_csv(
# #         out_dir / f'{model_name}_feature_importance.csv', index=False)


# # # ══════════════════════════════════════════════════════════════════════════════
# # # CONFORMAL PREDICTION
# # # ══════════════════════════════════════════════════════════════════════════════

# # def evaluate_conformal(model, val_loader, test_loader, device, adj, horizons, out_dir, model_name):
# #     cp = ConformalPredictor(model, alpha=CONFORMAL_ALPHA)
# #     print(f"\n  Calibrating conformal predictor (α={CONFORMAL_ALPHA})...")
# #     cp.calibrate(val_loader, device, adj)

# #     covered = {i: 0 for i in range(len(horizons))}
# #     widths  = {i: [] for i in range(len(horizons))}
# #     total   = 0

# #     model.eval()
# #     with torch.no_grad():
# #         for X, y_reg, y_cls, city_idx in test_loader:
# #             X, y_reg = X.to(device), y_reg.to(device)
# #             city_idx = city_idx.to(device)
# #             intervals, _ = cp.predict_with_intervals(X, adj=adj, city_idx=city_idx)
# #             for i, iv in enumerate(intervals):
# #                 lo = iv['lower'].cpu().numpy()
# #                 hi = iv['upper'].cpu().numpy()
# #                 t  = y_reg[:, i].cpu().numpy()
# #                 covered[i] += ((t >= lo) & (t <= hi)).sum()
# #                 widths[i].extend((hi - lo).tolist())
# #             total += len(X)

# #     rows = []
# #     for i, h in enumerate(horizons):
# #         cov   = covered[i] / total
# #         w_med = float(np.median(widths[i]))
# #         rows.append({
# #             'model': model_name,
# #             'horizon': HORIZON_LABELS.get(h, f't+{h}h'),
# #             'target_coverage': f'{int((1-CONFORMAL_ALPHA)*100)}%',
# #             'empirical_coverage': f'{cov*100:.2f}%',
# #             'median_width': round(w_med, 3),
# #         })
# #         print(f"    {HORIZON_LABELS.get(h)}: coverage={cov*100:.2f}%  "
# #               f"(target={(1-CONFORMAL_ALPHA)*100:.0f}%)  width={w_med:.3f}")

# #     cov_df = pd.DataFrame(rows)
# #     cov_df.to_csv(out_dir / f'{model_name}_conformal_coverage.csv', index=False)
# #     return cov_df


# # # ══════════════════════════════════════════════════════════════════════════════
# # # MAIN EVALUATION
# # # ══════════════════════════════════════════════════════════════════════════════

# # def evaluate_model(model_name, device, meta, test_ds, val_ds, adj, scalers, out_dir):
# #     print(f"\n{'='*55}")
# #     print(f"  Evaluating: {model_name}")
# #     print(f"{'='*55}")
# #     n_feat    = meta['n_features']
# #     horizons  = meta['forecast_hours']
# #     feat_cols = meta['feature_cols']

# #     model = load_model(model_name, n_feat, horizons, device)
# #     if model is None:
# #         return None

# #     # test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
# #     # val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=2)
# #     test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
# #     val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)

# #     preds, y_reg, cat_logits, y_cls, cities, datetimes = predict_all(
# #         model, test_loader, device, adj, scalers, test_ds.meta
# #     )

# #     metric_rows = compute_all_metrics(preds, y_reg, cat_logits, y_cls, horizons, model_name)
# #     print(f"\n  Regression metrics:")
# #     for r in metric_rows:
# #         if r['horizon'] != 'classification':
# #             print(f"    {r['horizon']:>7s}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
# #                   f"R²={r['R2']:.4f}  Within10%={r['Within_10pct']:.1f}%")
# #     cls_row = next((r for r in metric_rows if r['horizon'] == 'classification'), None)
# #     if cls_row:
# #         print(f"  Classification: Acc={float(cls_row['R2'])*100:.2f}%  "
# #               f"WtdF1={cls_row['Within_10pct']:.2f}%")

# #     plot_training_curves(model_name, out_dir)
# #     plot_prediction_vs_truth(preds, y_reg, horizons, scalers, out_dir, model_name)
# #     plot_sample_forecast(preds, y_reg, datetimes, horizons, out_dir, model_name)

# #     city_df = per_city_metrics(preds, y_reg, horizons, cities)
# #     city_df.to_csv(out_dir / f'{model_name}_per_city_metrics.csv', index=False)
# #     plot_city_heatmap(city_df, out_dir, model_name)

# #     y_pred_cls = cat_logits.argmax(dim=1).numpy()
# #     plot_confusion_matrix(y_cls, y_pred_cls, out_dir, model_name)

# #     valid = y_cls >= 0
# #     if valid.any():
# #         present = sorted(set(y_cls[valid]))
# #         report  = classification_report(
# #             y_cls[valid], y_pred_cls[valid],
# #             labels=present,
# #             target_names=[AQI_CATEGORIES[i] for i in present],
# #             zero_division=0,
# #         )
# #         print(f"\n  Full classification report:\n{report}")
# #         with open(out_dir / f'{model_name}_classification_report.txt', 'w') as f:
# #             f.write(report)

# #     if model_name == 'greeneyes' and hasattr(model, 'wn_blocks'):
# #         print("\n  Computing feature importance (Integrated Gradients)...")
# #         imp = compute_feature_importance_ig(model, test_ds, device, adj, feat_cols)
# #         plot_feature_importance(imp, feat_cols, out_dir, model_name)

# #     evaluate_conformal(model, val_loader, test_loader, device, adj, horizons, out_dir, model_name)
# #     return metric_rows


# # def main(args):
# #     device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     out_dir = RESULTS_DIR
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #     print(f"Device: {device} | Results → {out_dir}")

# #     meta    = joblib.load(DATA_PROCESSED / 'meta.joblib')
# #     scalers = joblib.load(DATA_PROCESSED / 'scalers.joblib')
# #     adj     = torch.tensor(meta['adj'], dtype=torch.float32).to(device)

# #     test_ds = AQISequenceDataset(DATA_PROCESSED / 'test.pt')
# #     val_ds  = AQISequenceDataset(DATA_PROCESSED / 'val.pt')
# #     print(f"Test sequences: {len(test_ds):,}")

# #     models_to_eval = (
# #         ['greeneyes', 'lstm', 'gru', 'transformer', 'wavenet']
# #         if args.ablation else [args.model]
# #     )

# #     all_rows = []
# #     for m in models_to_eval:
# #         rows = evaluate_model(m, device, meta, test_ds, val_ds, adj, scalers, out_dir)
# #         if rows:
# #             all_rows.extend(rows)

# #     if all_rows:
# #         df = pd.DataFrame(all_rows)
# #         df.to_csv(out_dir / 'overall_metrics.csv', index=False)
# #         reg     = df[df['horizon'] != 'classification']
# #         summary = reg.groupby('model')[['MAE', 'RMSE', 'R2', 'Within_10pct']].mean()
# #         print("\n\n" + "="*60)
# #         print("  SUMMARY TABLE (mean over all horizons)")
# #         print("="*60)
# #         print(summary.to_string())
# #         if args.ablation:
# #             plot_ablation_bar(df, out_dir)
# #             df.to_csv(out_dir / 'ablation_table.csv', index=False)

# #     print(f"\nAll results saved to: {out_dir.resolve()}")


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--model',    default='greeneyes',
# #                         choices=['greeneyes','lstm','gru','transformer','wavenet'])
# #     parser.add_argument('--ablation', action='store_true')
# #     args = parser.parse_args()
# #     main(args)  

# """
# evaluation/evaluate.py
# =======================
# Comprehensive research evaluation.
# """

# import sys
# import argparse
# import json
# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import joblib
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader
# from sklearn.metrics import (
#     classification_report, confusion_matrix,
#     mean_absolute_error, mean_squared_error, r2_score, f1_score
# )
# from tqdm import tqdm

# warnings.filterwarnings('ignore')
# sys.path.append(str(Path(__file__).parent.parent))
# from configs.config import *
# from models.model import (
#     GreenEyesPlus, LSTMBaseline, GRUBaseline,
#     TransformerBaseline, WaveNetOnlyBaseline, ConformalPredictor
# )
# from training.train import AQISequenceDataset, JointLoss, compute_class_weights

# set_seed(RANDOM_SEED)

# plt.rcParams.update({
#     'figure.dpi': 150, 'font.size': 11,
#     'axes.spines.top': False, 'axes.spines.right': False,
# })

# AQI_CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
# HORIZON_LABELS = {1: 't+1h', 24: 't+24h', 72: 't+72h'}


# # ══════════════════════════════════════════════════════════════════════════════
# # MODEL LOADING
# # ══════════════════════════════════════════════════════════════════════════════

# def load_model(model_name, n_features, horizons, device):
#     ckpt_path = CHECKPOINTS_DIR / model_name / 'best_model.pt'
#     if not ckpt_path.exists():
#         print(f"  [SKIP] No checkpoint: {ckpt_path}")
#         return None
#     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
#     constructors = {
#         'greeneyes':   lambda: GreenEyesPlus(
#             n_features=n_features, n_cities=N_CITIES, hidden=HIDDEN_DIM,
#             wavenet_layers=WAVENET_LAYERS, lstm_layers=LSTM_LAYERS,
#             horizons=horizons, dropout=DROPOUT,
#         ),
#         'lstm':        lambda: LSTMBaseline(n_features, horizons=horizons),
#         'gru':         lambda: GRUBaseline(n_features, horizons=horizons),
#         'transformer': lambda: TransformerBaseline(n_features, horizons=horizons),
#         'wavenet':     lambda: WaveNetOnlyBaseline(n_features, horizons=horizons),
#     }
#     model = constructors[model_name]()
#     model.load_state_dict(ckpt['model_state'])
#     return model.to(device).eval()


# # ══════════════════════════════════════════════════════════════════════════════
# # FULL PREDICTION PASS
# # ══════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def predict_all(model, loader, device, adj, scalers, meta_info):
#     all_preds  = [[] for _ in model.horizons]
#     all_y_reg, all_cat_l, all_y_cls = [], [], []
#     all_city, all_dt = [], []

#     pbar = tqdm(
#         zip(loader, _batch_meta(meta_info, loader.batch_size)),
#         total=len(loader),
#         desc="  Predicting batches",
#         unit="batch",
#         ncols=90,
#         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
#     )
#     for (X, y_reg, y_cls, city_idx), batch_meta in pbar:
#         X        = X.to(device)
#         y_reg    = y_reg.to(device)
#         city_idx = city_idx.to(device)
#         reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)
#         for i, p in enumerate(reg_preds):
#             all_preds[i].append(p.squeeze().cpu())
#         all_y_reg.append(y_reg.cpu())
#         all_cat_l.append(cat_logits.cpu())
#         all_y_cls.append(y_cls)
#         all_city.extend([m['city']    for m in batch_meta])
#         all_dt.extend(  [m['datetime'] for m in batch_meta])

#     preds   = [torch.cat(all_preds[i]).numpy() for i in range(len(all_preds))]
#     y_reg   = torch.cat(all_y_reg).numpy()
#     cat_log = torch.cat(all_cat_l)
#     y_cls   = torch.cat(all_y_cls).numpy()

#     if 'AQI_poly' in scalers:
#         sc    = scalers['AQI_poly']
#         preds = [sc.inverse_transform(p.reshape(-1,1)).flatten() for p in preds]
#         y_reg = np.column_stack([
#             sc.inverse_transform(y_reg[:, i].reshape(-1,1)).flatten()
#             for i in range(y_reg.shape[1])
#         ])
#     return preds, y_reg, cat_log, y_cls, all_city, all_dt


# def _batch_meta(meta_list, batch_size):
#     for i in range(0, len(meta_list), batch_size):
#         yield meta_list[i:i+batch_size]


# # ══════════════════════════════════════════════════════════════════════════════
# # METRICS
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_all_metrics(preds, y_reg, cat_logits, y_cls, horizons, model_name=''):
#     rows = []
#     for i, h in enumerate(tqdm(horizons, desc="  Computing regression metrics", ncols=90, leave=False)):
#         p, t     = preds[i], y_reg[:, i]
#         mae      = mean_absolute_error(t, p)
#         rmse     = np.sqrt(mean_squared_error(t, p))
#         r2       = r2_score(t, p)
#         within10 = np.mean(np.abs(p - t) / (np.abs(t) + 1e-6) <= 0.10) * 100
#         rows.append({
#             'model': model_name,
#             'horizon': HORIZON_LABELS.get(h, f't+{h}h'),
#             'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
#             'R2': round(r2, 4), 'Within_10pct': round(within10, 2),
#         })
#     y_pred_cls = cat_logits.argmax(dim=1).numpy()
#     valid = y_cls >= 0
#     if valid.any():
#         acc = (y_pred_cls[valid] == y_cls[valid]).mean() * 100
#         f1  = f1_score(y_cls[valid], y_pred_cls[valid], average='weighted', zero_division=0) * 100
#         rows.append({
#             'model': model_name, 'horizon': 'classification',
#             'MAE': '-', 'RMSE': '-',
#             'R2': round(acc/100, 4), 'Within_10pct': round(f1, 2),
#         })
#     return rows


# def per_city_metrics(preds, y_reg, horizons, cities):
#     result = []
#     for h_idx, h in enumerate(tqdm(horizons, desc="  Per-city metrics", ncols=90, leave=False)):
#         df_h = pd.DataFrame({'city': cities, 'pred': preds[h_idx], 'true': y_reg[:, h_idx]})
#         city_metrics = df_h.groupby('city').apply(
#             lambda g: pd.Series({
#                 'MAE':  round(mean_absolute_error(g['true'], g['pred']), 3),
#                 'RMSE': round(np.sqrt(mean_squared_error(g['true'], g['pred'])), 3),
#                 'R2':   round(r2_score(g['true'], g['pred']), 3),
#                 'n':    len(g),
#             })
#         ).reset_index()
#         city_metrics['horizon'] = HORIZON_LABELS.get(h, f't+{h}h')
#         result.append(city_metrics)
#     return pd.concat(result, ignore_index=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # FIGURES
# # ══════════════════════════════════════════════════════════════════════════════

# def plot_training_curves(model_name, out_dir):
#     hist_path = CHECKPOINTS_DIR / model_name / 'history.json'
#     if not hist_path.exists():
#         return
#     with open(hist_path) as f:
#         history = json.load(f)
#     epochs     = [h['epoch'] for h in history]
#     train_loss = [h['train']['loss'] for h in history]
#     val_loss   = [h['val']['loss']   for h in history]
#     val_mae    = [h['val']['MAE_mean'] for h in history]
#     val_r2     = [h['val']['R2_mean']  for h in history]
#     val_acc    = [h['val'].get('cat_acc', 0) for h in history]

#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     fig.suptitle(f'{model_name} — Training Curves', fontsize=13, fontweight='bold')
#     axes[0].plot(epochs, train_loss, label='Train', color='#2196F3', linewidth=1.5)
#     axes[0].plot(epochs, val_loss,   label='Val',   color='#F44336', linewidth=1.5)
#     axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
#     axes[0].set_title('Joint Loss'); axes[0].legend(); axes[0].set_yscale('log')
#     axes[1].plot(epochs, val_mae, color='#4CAF50', linewidth=1.5)
#     axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
#     axes[1].set_title('Validation MAE')
#     axes[2].plot(epochs, val_r2,  label='R²',      color='#9C27B0', linewidth=1.5)
#     axes[2].plot(epochs, val_acc, label='Cat. Acc', color='#FF9800', linewidth=1.5)
#     axes[2].set_xlabel('Epoch'); axes[2].legend()
#     axes[2].set_title('R² and Category Accuracy')
#     plt.tight_layout()
#     out_path = out_dir / f'{model_name}_training_curves.png'
#     plt.savefig(out_path, bbox_inches='tight'); plt.close()
#     print(f"  Saved: {out_path.name}")


# def plot_prediction_vs_truth(preds, y_reg, horizons, scalers, out_dir, model_name, n_samples=500):
#     n    = min(n_samples, len(preds[0]))
#     fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5))
#     if len(horizons) == 1: axes = [axes]
#     fig.suptitle(f'{model_name} — Predicted vs. True AQI', fontsize=13, fontweight='bold')
#     for i, (h, ax) in enumerate(zip(horizons, axes)):
#         p   = preds[i][:n]; t = y_reg[:n, i]
#         lim = (min(p.min(), t.min())-5, max(p.max(), t.max())+5)
#         ax.scatter(t, p, alpha=0.3, s=8, color='#1976D2', rasterized=True)
#         ax.plot(lim, lim, 'r--', linewidth=1.2, label='Perfect')
#         ax.set_xlabel('True AQI'); ax.set_ylabel('Predicted AQI')
#         ax.set_title(f'{HORIZON_LABELS.get(h)}  R²={r2_score(t,p):.3f}  MAE={mean_absolute_error(t,p):.2f}')
#         ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(fontsize=9)
#     plt.tight_layout()
#     out_path = out_dir / f'{model_name}_pred_vs_true.png'
#     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
#     print(f"  Saved: {out_path.name}")


# def plot_sample_forecast(preds, y_reg, datetimes, horizons, out_dir, model_name):
#     fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4*len(horizons)), sharex=False)
#     if len(horizons) == 1: axes = [axes]
#     fig.suptitle(f'{model_name} — Sample Forecast vs Ground Truth', fontsize=13, fontweight='bold')
#     n_show = min(500, len(preds[0]))
#     for i, (h, ax) in enumerate(zip(horizons, axes)):
#         ax.plot(range(n_show), y_reg[:n_show, i], label='Ground truth',
#                 color='#333', linewidth=1.2, alpha=0.8)
#         ax.plot(range(n_show), preds[i][:n_show], label='Predicted',
#                 color='#E53935', linewidth=1.2, alpha=0.8, linestyle='--')
#         ax.set_ylabel('AQI'); ax.set_title(HORIZON_LABELS.get(h, f't+{h}h'))
#         ax.legend(fontsize=9, loc='upper right')
#     axes[-1].set_xlabel('Sample index')
#     plt.tight_layout()
#     out_path = out_dir / f'{model_name}_sample_forecast.png'
#     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
#     print(f"  Saved: {out_path.name}")


# def plot_city_heatmap(city_df, out_dir, model_name):
#     pivot = city_df.pivot_table(index='city', columns='horizon', values='MAE')
#     if pivot.empty: return
#     fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2), max(6, len(pivot)*0.4)))
#     sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
#                 linewidths=0.5, cbar_kws={'label': 'MAE (AQI units)'})
#     ax.set_title(f'{model_name} — Per-City MAE by Horizon', fontsize=13, fontweight='bold')
#     plt.tight_layout()
#     out_path = out_dir / f'{model_name}_city_heatmap.png'
#     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
#     print(f"  Saved: {out_path.name}")


# def plot_confusion_matrix(y_true, y_pred, out_dir, model_name):
#     valid  = y_true >= 0
#     cm     = confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(len(AQI_CATEGORIES))))
#     cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
#                 xticklabels=AQI_CATEGORIES, yticklabels=AQI_CATEGORIES,
#                 cbar_kws={'label': '%'}, linewidths=0.5)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('True')
#     ax.set_title(f'{model_name} — AQI Category Confusion Matrix (%)', fontsize=13)
#     plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
#     out_path = out_dir / f'{model_name}_confusion_matrix.png'
#     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
#     np.save(out_dir / f'{model_name}_confusion_matrix.npy', cm)
#     print(f"  Saved: {out_path.name}")


# def plot_ablation_bar(ablation_df, out_dir):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle('Ablation Study — Model Comparison', fontsize=13, fontweight='bold')
#     reg_df  = ablation_df[ablation_df['horizon'] != 'classification']
#     mean_df = reg_df.groupby('model')[['MAE', 'RMSE', 'R2']].mean().reset_index()
#     colors  = plt.cm.Set2(np.linspace(0, 1, len(mean_df)))
#     axes[0].barh(mean_df['model'], mean_df['MAE'], color=colors)
#     axes[0].set_xlabel('Mean MAE'); axes[0].set_title('Mean MAE (lower = better)')
#     axes[0].invert_xaxis()
#     axes[1].barh(mean_df['model'], mean_df['R2'], color=colors)
#     axes[1].set_xlabel('Mean R²'); axes[1].set_title('Mean R² (higher = better)')
#     axes[1].set_xlim(0, 1)
#     plt.tight_layout()
#     out_path = out_dir / 'ablation_comparison.png'
#     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
#     print(f"  Saved: {out_path.name}")


# # ══════════════════════════════════════════════════════════════════════════════
# # FEATURE IMPORTANCE (Integrated Gradients)
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_feature_importance_ig(model, test_ds, device, adj, feature_cols, n_samples=200):
#     """Uses __getitem__ to load samples — works with memmap-backed AQIDataset."""
#     model.eval()
#     n_samples   = min(n_samples, len(test_ds))
#     samples     = [test_ds[i] for i in range(n_samples)]
#     X_sample    = torch.stack([s[0] for s in samples]).to(device)
#     city_idx    = torch.stack([s[3] for s in samples]).to(device)
#     baseline    = torch.zeros_like(X_sample)
#     importances = torch.zeros(len(feature_cols))
#     n_steps     = 30

#     for alpha in tqdm(
#         np.linspace(0, 1, n_steps),
#         desc="  Integrated Gradients steps",
#         ncols=90,
#         unit="step",
#     ):
#         x_interp = (baseline + alpha * (X_sample - baseline)).requires_grad_(True)
#         reg_preds, _ = model(x_interp, adj=adj, city_idx=city_idx)
#         loss = reg_preds[0].sum()
#         loss.backward()
#         with torch.no_grad():
#             grad = x_interp.grad.abs().mean(dim=(0, 1)).cpu()
#         importances += grad / n_steps
#         x_interp.grad = None

#     return importances.numpy()


# def plot_feature_importance(importances, feature_cols, out_dir, model_name, top_n=25):
#     ranked = np.argsort(importances)[::-1][:top_n]
#     names  = [feature_cols[i] for i in ranked]
#     vals   = importances[ranked] / (importances[ranked].max() + 1e-8)
#     fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
#     colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
#     ax.barh(names[::-1], vals[::-1], color=colors[::-1])
#     ax.set_xlabel('Normalised Importance (Integrated Gradients)')
#     ax.set_title(f'{model_name} — Top-{top_n} Feature Importances', fontsize=13, fontweight='bold')
#     ax.set_xlim(0, 1.05); plt.tight_layout()
#     out_path = out_dir / f'{model_name}_feature_importance.png'
#     plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
#     print(f"  Saved: {out_path.name}")
#     imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
#     imp_df.sort_values('importance', ascending=False).to_csv(
#         out_dir / f'{model_name}_feature_importance.csv', index=False)


# # ══════════════════════════════════════════════════════════════════════════════
# # CONFORMAL PREDICTION
# # ══════════════════════════════════════════════════════════════════════════════

# def evaluate_conformal(model, val_loader, test_loader, device, adj, horizons, out_dir, model_name):
#     cp = ConformalPredictor(model, alpha=CONFORMAL_ALPHA)
#     print(f"\n  Calibrating conformal predictor (α={CONFORMAL_ALPHA})...")
#     cp.calibrate(val_loader, device, adj)

#     covered = {i: 0 for i in range(len(horizons))}
#     widths  = {i: [] for i in range(len(horizons))}
#     total   = 0

#     model.eval()
#     with torch.no_grad():
#         for X, y_reg, y_cls, city_idx in tqdm(
#             test_loader,
#             desc="  Conformal coverage eval",
#             ncols=90,
#             unit="batch",
#         ):
#             X, y_reg = X.to(device), y_reg.to(device)
#             city_idx = city_idx.to(device)
#             intervals, _ = cp.predict_with_intervals(X, adj=adj, city_idx=city_idx)
#             for i, iv in enumerate(intervals):
#                 lo = iv['lower'].cpu().numpy()
#                 hi = iv['upper'].cpu().numpy()
#                 t  = y_reg[:, i].cpu().numpy()
#                 covered[i] += ((t >= lo) & (t <= hi)).sum()
#                 widths[i].extend((hi - lo).tolist())
#             total += len(X)

#     rows = []
#     for i, h in enumerate(horizons):
#         cov   = covered[i] / total
#         w_med = float(np.median(widths[i]))
#         rows.append({
#             'model': model_name,
#             'horizon': HORIZON_LABELS.get(h, f't+{h}h'),
#             'target_coverage': f'{int((1-CONFORMAL_ALPHA)*100)}%',
#             'empirical_coverage': f'{cov*100:.2f}%',
#             'median_width': round(w_med, 3),
#         })
#         print(f"    {HORIZON_LABELS.get(h)}: coverage={cov*100:.2f}%  "
#               f"(target={(1-CONFORMAL_ALPHA)*100:.0f}%)  width={w_med:.3f}")

#     cov_df = pd.DataFrame(rows)
#     cov_df.to_csv(out_dir / f'{model_name}_conformal_coverage.csv', index=False)
#     return cov_df


# # ══════════════════════════════════════════════════════════════════════════════
# # PER-MODEL EVALUATION STEPS
# # ══════════════════════════════════════════════════════════════════════════════

# # Steps shown in the per-model progress bar
# _EVAL_STEPS = [
#     "load model",
#     "predict",
#     "regression metrics",
#     "training curves",
#     "pred vs truth plot",
#     "sample forecast plot",
#     "per-city metrics",
#     "confusion matrix",
#     "classification report",
#     "feature importance",
#     "conformal prediction",
# ]


# def evaluate_model(model_name, device, meta, test_ds, val_ds, adj, scalers, out_dir):
#     print(f"\n{'='*55}")
#     print(f"  Evaluating: {model_name}")
#     print(f"{'='*55}")
#     n_feat    = meta['n_features']
#     horizons  = meta['forecast_hours']
#     feat_cols = meta['feature_cols']

#     step_pbar = tqdm(
#         total=len(_EVAL_STEPS),
#         desc=f"  [{model_name}]",
#         ncols=90,
#         unit="step",
#         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps  {postfix}",
#     )

#     def advance(label):
#         step_pbar.set_postfix_str(label, refresh=True)
#         step_pbar.update(1)

#     # ── 1. Load model ──────────────────────────────────────────────────────
#     advance("loading checkpoint")
#     model = load_model(model_name, n_feat, horizons, device)
#     if model is None:
#         step_pbar.close()
#         return None

#     test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
#     val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)

#     # ── 2. Predict ─────────────────────────────────────────────────────────
#     advance("running inference")
#     preds, y_reg, cat_logits, y_cls, cities, datetimes = predict_all(
#         model, test_loader, device, adj, scalers, test_ds.meta
#     )

#     # ── 3. Regression metrics ──────────────────────────────────────────────
#     advance("computing metrics")
#     metric_rows = compute_all_metrics(preds, y_reg, cat_logits, y_cls, horizons, model_name)
#     print(f"\n  Regression metrics:")
#     for r in metric_rows:
#         if r['horizon'] != 'classification':
#             print(f"    {r['horizon']:>7s}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
#                   f"R²={r['R2']:.4f}  Within10%={r['Within_10pct']:.1f}%")
#     cls_row = next((r for r in metric_rows if r['horizon'] == 'classification'), None)
#     if cls_row:
#         print(f"  Classification: Acc={float(cls_row['R2'])*100:.2f}%  "
#               f"WtdF1={cls_row['Within_10pct']:.2f}%")

#     # ── 4. Training curves ─────────────────────────────────────────────────
#     advance("plotting training curves")
#     plot_training_curves(model_name, out_dir)

#     # ── 5. Pred vs truth ───────────────────────────────────────────────────
#     advance("plotting pred vs truth")
#     plot_prediction_vs_truth(preds, y_reg, horizons, scalers, out_dir, model_name)

#     # ── 6. Sample forecast ─────────────────────────────────────────────────
#     advance("plotting sample forecast")
#     plot_sample_forecast(preds, y_reg, datetimes, horizons, out_dir, model_name)

#     # ── 7. Per-city metrics ────────────────────────────────────────────────
#     advance("per-city metrics")
#     city_df = per_city_metrics(preds, y_reg, horizons, cities)
#     city_df.to_csv(out_dir / f'{model_name}_per_city_metrics.csv', index=False)
#     plot_city_heatmap(city_df, out_dir, model_name)

#     # ── 8. Confusion matrix ────────────────────────────────────────────────
#     advance("confusion matrix")
#     y_pred_cls = cat_logits.argmax(dim=1).numpy()
#     plot_confusion_matrix(y_cls, y_pred_cls, out_dir, model_name)

#     # ── 9. Classification report ───────────────────────────────────────────
#     advance("classification report")
#     valid = y_cls >= 0
#     if valid.any():
#         present = sorted(set(y_cls[valid]))
#         report  = classification_report(
#             y_cls[valid], y_pred_cls[valid],
#             labels=present,
#             target_names=[AQI_CATEGORIES[i] for i in present],
#             zero_division=0,
#         )
#         print(f"\n  Full classification report:\n{report}")
#         with open(out_dir / f'{model_name}_classification_report.txt', 'w') as f:
#             f.write(report)

#     # ── 10. Feature importance ─────────────────────────────────────────────
#     advance("feature importance (IG)")
#     if model_name == 'greeneyes' and hasattr(model, 'wn_blocks'):
#         print("\n  Computing feature importance (Integrated Gradients)...")
#         imp = compute_feature_importance_ig(model, test_ds, device, adj, feat_cols)
#         plot_feature_importance(imp, feat_cols, out_dir, model_name)

#     # ── 11. Conformal prediction ───────────────────────────────────────────
#     advance("conformal prediction")
#     evaluate_conformal(model, val_loader, test_loader, device, adj, horizons, out_dir, model_name)

#     step_pbar.set_postfix_str("done ✓", refresh=True)
#     step_pbar.close()
#     return metric_rows


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN EVALUATION
# # ══════════════════════════════════════════════════════════════════════════════

# def main(args):
#     device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     out_dir = RESULTS_DIR
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Device: {device} | Results → {out_dir}")

#     meta    = joblib.load(DATA_PROCESSED / 'meta.joblib')
#     scalers = joblib.load(DATA_PROCESSED / 'scalers.joblib')
#     adj     = torch.tensor(meta['adj'], dtype=torch.float32).to(device)

#     test_ds = AQISequenceDataset(DATA_PROCESSED / 'test.pt')
#     val_ds  = AQISequenceDataset(DATA_PROCESSED / 'val.pt')
#     print(f"Test sequences: {len(test_ds):,}")

#     models_to_eval = (
#         ['greeneyes', 'lstm', 'gru', 'transformer', 'wavenet']
#         if args.ablation else [args.model]
#     )

#     # ── Overall progress bar across all models ─────────────────────────────
#     all_rows = []
#     overall_pbar = tqdm(
#         models_to_eval,
#         desc="Overall progress",
#         ncols=90,
#         unit="model",
#         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} models  [{elapsed}<{remaining}]",
#     )
#     for m in overall_pbar:
#         overall_pbar.set_postfix_str(m, refresh=True)
#         rows = evaluate_model(m, device, meta, test_ds, val_ds, adj, scalers, out_dir)
#         if rows:
#             all_rows.extend(rows)
#     overall_pbar.set_postfix_str("complete ✓", refresh=True)
#     overall_pbar.close()

#     if all_rows:
#         df = pd.DataFrame(all_rows)
#         df.to_csv(out_dir / 'overall_metrics.csv', index=False)
#         reg     = df[df['horizon'] != 'classification']
#         summary = reg.groupby('model')[['MAE', 'RMSE', 'R2', 'Within_10pct']].mean()
#         print("\n\n" + "="*60)
#         print("  SUMMARY TABLE (mean over all horizons)")
#         print("="*60)
#         print(summary.to_string())
#         if args.ablation:
#             plot_ablation_bar(df, out_dir)
#             df.to_csv(out_dir / 'ablation_table.csv', index=False)

#     print(f"\nAll results saved to: {out_dir.resolve()}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model',    default='greeneyes',
#                         choices=['greeneyes','lstm','gru','transformer','wavenet'])
#     parser.add_argument('--ablation', action='store_true')
#     args = parser.parse_args()
#     main(args)


"""
evaluation/evaluate.py
=======================
Comprehensive research evaluation.
"""

import sys
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, f1_score
)
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *
from models.model import (
    GreenEyesPlus, LSTMBaseline, GRUBaseline,
    TransformerBaseline, WaveNetOnlyBaseline, ConformalPredictor
)
from training.train import AQISequenceDataset, JointLoss, compute_class_weights

set_seed(RANDOM_SEED)

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

AQI_CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
HORIZON_LABELS = {1: 't+1h', 24: 't+24h', 72: 't+72h'}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_name, n_features, horizons, device):
    ckpt_path = CHECKPOINTS_DIR / model_name / 'best_model.pt'
    if not ckpt_path.exists():
        print(f"  [SKIP] No checkpoint: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    constructors = {
        'greeneyes':   lambda: GreenEyesPlus(
            n_features=n_features, n_cities=N_CITIES, hidden=HIDDEN_DIM,
            wavenet_layers=WAVENET_LAYERS, lstm_layers=LSTM_LAYERS,
            horizons=horizons, dropout=DROPOUT,
        ),
        'lstm':        lambda: LSTMBaseline(n_features, horizons=horizons),
        'gru':         lambda: GRUBaseline(n_features, horizons=horizons),
        'transformer': lambda: TransformerBaseline(n_features, horizons=horizons),
        'wavenet':     lambda: WaveNetOnlyBaseline(n_features, horizons=horizons),
    }
    model = constructors[model_name]()
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════════════════════
# FULL PREDICTION PASS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_all(model, loader, device, adj, scalers, meta_info):
    all_preds  = [[] for _ in model.horizons]
    all_y_reg, all_cat_l, all_y_cls = [], [], []
    all_city, all_dt = [], []

    pbar = tqdm(
        zip(loader, _batch_meta(meta_info, loader.batch_size)),
        total=len(loader),
        desc="  Predicting batches",
        unit="batch",
        ncols=90,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for (X, y_reg, y_cls, city_idx), batch_meta in pbar:
        X        = X.to(device)
        y_reg    = y_reg.to(device)
        city_idx = city_idx.to(device)
        reg_preds, cat_logits = model(X, adj=adj, city_idx=city_idx)
        for i, p in enumerate(reg_preds):
            all_preds[i].append(p.squeeze().cpu())
        all_y_reg.append(y_reg.cpu())
        all_cat_l.append(cat_logits.cpu())
        all_y_cls.append(y_cls)
        all_city.extend([m['city']    for m in batch_meta])
        all_dt.extend(  [m['datetime'] for m in batch_meta])

    preds   = [torch.cat(all_preds[i]).numpy() for i in range(len(all_preds))]
    y_reg   = torch.cat(all_y_reg).numpy()
    cat_log = torch.cat(all_cat_l)
    y_cls   = torch.cat(all_y_cls).numpy()

    if 'AQI_poly' in scalers:
        sc    = scalers['AQI_poly']
        preds = [sc.inverse_transform(p.reshape(-1,1)).flatten() for p in preds]
        y_reg = np.column_stack([
            sc.inverse_transform(y_reg[:, i].reshape(-1,1)).flatten()
            for i in range(y_reg.shape[1])
        ])
    return preds, y_reg, cat_log, y_cls, all_city, all_dt


def _batch_meta(meta_list, batch_size):
    for i in range(0, len(meta_list), batch_size):
        yield meta_list[i:i+batch_size]


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(preds, y_reg, cat_logits, y_cls, horizons, model_name=''):
    rows = []
    for i, h in enumerate(tqdm(horizons, desc="  Computing regression metrics", ncols=90, leave=False)):
        p, t     = preds[i], y_reg[:, i]
        mae      = mean_absolute_error(t, p)
        rmse     = np.sqrt(mean_squared_error(t, p))
        r2       = r2_score(t, p)
        within10 = np.mean(np.abs(p - t) / (np.abs(t) + 1e-6) <= 0.10) * 100
        rows.append({
            'model': model_name,
            'horizon': HORIZON_LABELS.get(h, f't+{h}h'),
            'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'R2': round(r2, 4), 'Within_10pct': round(within10, 2),
        })
    y_pred_cls = cat_logits.argmax(dim=1).numpy()
    valid = y_cls >= 0
    if valid.any():
        acc = (y_pred_cls[valid] == y_cls[valid]).mean() * 100
        f1  = f1_score(y_cls[valid], y_pred_cls[valid], average='weighted', zero_division=0) * 100
        rows.append({
            'model': model_name, 'horizon': 'classification',
            'MAE': '-', 'RMSE': '-',
            'R2': round(acc/100, 4), 'Within_10pct': round(f1, 2),
        })
    return rows


def per_city_metrics(preds, y_reg, horizons, cities):
    result = []
    for h_idx, h in enumerate(tqdm(horizons, desc="  Per-city metrics", ncols=90, leave=False)):
        df_h = pd.DataFrame({'city': cities, 'pred': preds[h_idx], 'true': y_reg[:, h_idx]})
        city_metrics = df_h.groupby('city').apply(
            lambda g: pd.Series({
                'MAE':  round(mean_absolute_error(g['true'], g['pred']), 3),
                'RMSE': round(np.sqrt(mean_squared_error(g['true'], g['pred'])), 3),
                'R2':   round(r2_score(g['true'], g['pred']), 3),
                'n':    len(g),
            })
        ).reset_index()
        city_metrics['horizon'] = HORIZON_LABELS.get(h, f't+{h}h')
        result.append(city_metrics)
    return pd.concat(result, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(model_name, out_dir):
    hist_path = CHECKPOINTS_DIR / model_name / 'history.json'
    if not hist_path.exists():
        return
    with open(hist_path) as f:
        history = json.load(f)
    epochs     = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss   = [h['val']['loss']   for h in history]
    val_mae    = [h['val']['MAE_mean'] for h in history]
    val_r2     = [h['val']['R2_mean']  for h in history]
    val_acc    = [h['val'].get('cat_acc', 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{model_name} — Training Curves', fontsize=13, fontweight='bold')
    axes[0].plot(epochs, train_loss, label='Train', color='#2196F3', linewidth=1.5)
    axes[0].plot(epochs, val_loss,   label='Val',   color='#F44336', linewidth=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Joint Loss'); axes[0].legend(); axes[0].set_yscale('log')
    axes[1].plot(epochs, val_mae, color='#4CAF50', linewidth=1.5)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation MAE')
    axes[2].plot(epochs, val_r2,  label='R²',      color='#9C27B0', linewidth=1.5)
    axes[2].plot(epochs, val_acc, label='Cat. Acc', color='#FF9800', linewidth=1.5)
    axes[2].set_xlabel('Epoch'); axes[2].legend()
    axes[2].set_title('R² and Category Accuracy')
    plt.tight_layout()
    out_path = out_dir / f'{model_name}_training_curves.png'
    plt.savefig(out_path, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out_path.name}")


def plot_prediction_vs_truth(preds, y_reg, horizons, scalers, out_dir, model_name, n_samples=500):
    n    = min(n_samples, len(preds[0]))
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle(f'{model_name} — Predicted vs. True AQI', fontsize=13, fontweight='bold')
    for i, (h, ax) in enumerate(zip(horizons, axes)):
        p   = preds[i][:n]; t = y_reg[:n, i]
        lim = (min(p.min(), t.min())-5, max(p.max(), t.max())+5)
        ax.scatter(t, p, alpha=0.3, s=8, color='#1976D2', rasterized=True)
        ax.plot(lim, lim, 'r--', linewidth=1.2, label='Perfect')
        ax.set_xlabel('True AQI'); ax.set_ylabel('Predicted AQI')
        ax.set_title(f'{HORIZON_LABELS.get(h)}  R²={r2_score(t,p):.3f}  MAE={mean_absolute_error(t,p):.2f}')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = out_dir / f'{model_name}_pred_vs_true.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Saved: {out_path.name}")


def plot_sample_forecast(preds, y_reg, datetimes, horizons, out_dir, model_name):
    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4*len(horizons)), sharex=False)
    if len(horizons) == 1: axes = [axes]
    fig.suptitle(f'{model_name} — Sample Forecast vs Ground Truth', fontsize=13, fontweight='bold')
    n_show = min(500, len(preds[0]))
    for i, (h, ax) in enumerate(zip(horizons, axes)):
        ax.plot(range(n_show), y_reg[:n_show, i], label='Ground truth',
                color='#333', linewidth=1.2, alpha=0.8)
        ax.plot(range(n_show), preds[i][:n_show], label='Predicted',
                color='#E53935', linewidth=1.2, alpha=0.8, linestyle='--')
        ax.set_ylabel('AQI'); ax.set_title(HORIZON_LABELS.get(h, f't+{h}h'))
        ax.legend(fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Sample index')
    plt.tight_layout()
    out_path = out_dir / f'{model_name}_sample_forecast.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Saved: {out_path.name}")


def plot_city_heatmap(city_df, out_dir, model_name):
    pivot = city_df.pivot_table(index='city', columns='horizon', values='MAE')
    if pivot.empty: return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2), max(6, len(pivot)*0.4)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'MAE (AQI units)'})
    ax.set_title(f'{model_name} — Per-City MAE by Horizon', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / f'{model_name}_city_heatmap.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Saved: {out_path.name}")


def plot_confusion_matrix(y_true, y_pred, out_dir, model_name):
    valid  = y_true >= 0
    cm     = confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(len(AQI_CATEGORIES))))
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=AQI_CATEGORIES, yticklabels=AQI_CATEGORIES,
                cbar_kws={'label': '%'}, linewidths=0.5)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{model_name} — AQI Category Confusion Matrix (%)', fontsize=13)
    plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    out_path = out_dir / f'{model_name}_confusion_matrix.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
    np.save(out_dir / f'{model_name}_confusion_matrix.npy', cm)
    print(f"  Saved: {out_path.name}")


def plot_ablation_bar(ablation_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ablation Study — Model Comparison', fontsize=13, fontweight='bold')
    reg_df  = ablation_df[ablation_df['horizon'] != 'classification']
    mean_df = reg_df.groupby('model')[['MAE', 'RMSE', 'R2']].mean().reset_index()
    colors  = plt.cm.Set2(np.linspace(0, 1, len(mean_df)))
    axes[0].barh(mean_df['model'], mean_df['MAE'], color=colors)
    axes[0].set_xlabel('Mean MAE'); axes[0].set_title('Mean MAE (lower = better)')
    axes[0].invert_xaxis()
    axes[1].barh(mean_df['model'], mean_df['R2'], color=colors)
    axes[1].set_xlabel('Mean R²'); axes[1].set_title('Mean R² (higher = better)')
    axes[1].set_xlim(0, 1)
    plt.tight_layout()
    out_path = out_dir / 'ablation_comparison.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (Integrated Gradients)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance_ig(model, test_ds, device, adj, feature_cols, n_samples=200):
    """Uses __getitem__ to load samples — works with memmap-backed AQIDataset.

    cuDNN RNN (LSTM/GRU) backward pass requires the model to be in *training*
    mode. We switch to train() here and restore eval() when done. Dropout
    noise is averaged out across the 30 interpolation steps so importances
    remain stable despite stochastic activations.
    """
    # cuDNN RNN backward requires training mode — switch, then restore
    model.train()
    n_samples   = min(n_samples, len(test_ds))
    samples     = [test_ds[i] for i in range(n_samples)]
    X_sample    = torch.stack([s[0] for s in samples]).to(device)
    city_idx    = torch.stack([s[3] for s in samples]).to(device)
    baseline    = torch.zeros_like(X_sample)
    importances = torch.zeros(len(feature_cols))
    n_steps     = 30

    for alpha in tqdm(
        np.linspace(0, 1, n_steps),
        desc="  Integrated Gradients steps",
        ncols=90,
        unit="step",
    ):
        x_interp = (baseline + alpha * (X_sample - baseline)).requires_grad_(True)
        reg_preds, _ = model(x_interp, adj=adj, city_idx=city_idx)
        loss = reg_preds[0].sum()
        loss.backward()
        with torch.no_grad():
            grad = x_interp.grad.abs().mean(dim=(0, 1)).cpu()
        importances += grad / n_steps
        x_interp.grad = None

    model.eval()  # restore eval mode for any subsequent steps
    return importances.numpy()


def plot_feature_importance(importances, feature_cols, out_dir, model_name, top_n=25):
    ranked = np.argsort(importances)[::-1][:top_n]
    names  = [feature_cols[i] for i in ranked]
    vals   = importances[ranked] / (importances[ranked].max() + 1e-8)
    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
    colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel('Normalised Importance (Integrated Gradients)')
    ax.set_title(f'{model_name} — Top-{top_n} Feature Importances', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05); plt.tight_layout()
    out_path = out_dir / f'{model_name}_feature_importance.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150); plt.close()
    print(f"  Saved: {out_path.name}")
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    imp_df.sort_values('importance', ascending=False).to_csv(
        out_dir / f'{model_name}_feature_importance.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_conformal(model, val_loader, test_loader, device, adj, horizons, out_dir, model_name):
    cp = ConformalPredictor(model, alpha=CONFORMAL_ALPHA)
    print(f"\n  Calibrating conformal predictor (α={CONFORMAL_ALPHA})...")
    cp.calibrate(val_loader, device, adj)

    covered = {i: 0 for i in range(len(horizons))}
    widths  = {i: [] for i in range(len(horizons))}
    total   = 0

    model.eval()
    with torch.no_grad():
        for X, y_reg, y_cls, city_idx in tqdm(
            test_loader,
            desc="  Conformal coverage eval",
            ncols=90,
            unit="batch",
        ):
            X, y_reg = X.to(device), y_reg.to(device)
            city_idx = city_idx.to(device)
            intervals, _ = cp.predict_with_intervals(X, adj=adj, city_idx=city_idx)
            for i, iv in enumerate(intervals):
                lo = iv['lower'].cpu().numpy()
                hi = iv['upper'].cpu().numpy()
                t  = y_reg[:, i].cpu().numpy()
                covered[i] += ((t >= lo) & (t <= hi)).sum()
                widths[i].extend((hi - lo).tolist())
            total += len(X)

    rows = []
    for i, h in enumerate(horizons):
        cov   = covered[i] / total
        w_med = float(np.median(widths[i]))
        rows.append({
            'model': model_name,
            'horizon': HORIZON_LABELS.get(h, f't+{h}h'),
            'target_coverage': f'{int((1-CONFORMAL_ALPHA)*100)}%',
            'empirical_coverage': f'{cov*100:.2f}%',
            'median_width': round(w_med, 3),
        })
        print(f"    {HORIZON_LABELS.get(h)}: coverage={cov*100:.2f}%  "
              f"(target={(1-CONFORMAL_ALPHA)*100:.0f}%)  width={w_med:.3f}")

    cov_df = pd.DataFrame(rows)
    cov_df.to_csv(out_dir / f'{model_name}_conformal_coverage.csv', index=False)
    return cov_df


# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL EVALUATION STEPS
# ══════════════════════════════════════════════════════════════════════════════

# Steps shown in the per-model progress bar
_EVAL_STEPS = [
    "load model",
    "predict",
    "regression metrics",
    "training curves",
    "pred vs truth plot",
    "sample forecast plot",
    "per-city metrics",
    "confusion matrix",
    "classification report",
    "feature importance",
    "conformal prediction",
]


def evaluate_model(model_name, device, meta, test_ds, val_ds, adj, scalers, out_dir):
    print(f"\n{'='*55}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*55}")
    n_feat    = meta['n_features']
    horizons  = meta['forecast_hours']
    feat_cols = meta['feature_cols']

    step_pbar = tqdm(
        total=len(_EVAL_STEPS),
        desc=f"  [{model_name}]",
        ncols=90,
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps  {postfix}",
    )

    def advance(label):
        step_pbar.set_postfix_str(label, refresh=True)
        step_pbar.update(1)

    # ── 1. Load model ──────────────────────────────────────────────────────
    advance("loading checkpoint")
    model = load_model(model_name, n_feat, horizons, device)
    if model is None:
        step_pbar.close()
        return None

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)

    # ── 2. Predict ─────────────────────────────────────────────────────────
    advance("running inference")
    preds, y_reg, cat_logits, y_cls, cities, datetimes = predict_all(
        model, test_loader, device, adj, scalers, test_ds.meta
    )

    # ── 3. Regression metrics ──────────────────────────────────────────────
    advance("computing metrics")
    metric_rows = compute_all_metrics(preds, y_reg, cat_logits, y_cls, horizons, model_name)
    print(f"\n  Regression metrics:")
    for r in metric_rows:
        if r['horizon'] != 'classification':
            print(f"    {r['horizon']:>7s}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                  f"R²={r['R2']:.4f}  Within10%={r['Within_10pct']:.1f}%")
    cls_row = next((r for r in metric_rows if r['horizon'] == 'classification'), None)
    if cls_row:
        print(f"  Classification: Acc={float(cls_row['R2'])*100:.2f}%  "
              f"WtdF1={cls_row['Within_10pct']:.2f}%")

    # ── 4. Training curves ─────────────────────────────────────────────────
    advance("plotting training curves")
    plot_training_curves(model_name, out_dir)

    # ── 5. Pred vs truth ───────────────────────────────────────────────────
    advance("plotting pred vs truth")
    plot_prediction_vs_truth(preds, y_reg, horizons, scalers, out_dir, model_name)

    # ── 6. Sample forecast ─────────────────────────────────────────────────
    advance("plotting sample forecast")
    plot_sample_forecast(preds, y_reg, datetimes, horizons, out_dir, model_name)

    # ── 7. Per-city metrics ────────────────────────────────────────────────
    advance("per-city metrics")
    city_df = per_city_metrics(preds, y_reg, horizons, cities)
    city_df.to_csv(out_dir / f'{model_name}_per_city_metrics.csv', index=False)
    plot_city_heatmap(city_df, out_dir, model_name)

    # ── 8. Confusion matrix ────────────────────────────────────────────────
    advance("confusion matrix")
    y_pred_cls = cat_logits.argmax(dim=1).numpy()
    plot_confusion_matrix(y_cls, y_pred_cls, out_dir, model_name)

    # ── 9. Classification report ───────────────────────────────────────────
    advance("classification report")
    valid = y_cls >= 0
    if valid.any():
        present = sorted(set(y_cls[valid]))
        report  = classification_report(
            y_cls[valid], y_pred_cls[valid],
            labels=present,
            target_names=[AQI_CATEGORIES[i] for i in present],
            zero_division=0,
        )
        print(f"\n  Full classification report:\n{report}")
        with open(out_dir / f'{model_name}_classification_report.txt', 'w') as f:
            f.write(report)

    # ── 10. Feature importance ─────────────────────────────────────────────
    advance("feature importance (IG)")
    if model_name == 'greeneyes' and hasattr(model, 'wn_blocks'):
        print("\n  Computing feature importance (Integrated Gradients)...")
        imp = compute_feature_importance_ig(model, test_ds, device, adj, feat_cols)
        plot_feature_importance(imp, feat_cols, out_dir, model_name)

    # ── 11. Conformal prediction ───────────────────────────────────────────
    advance("conformal prediction")
    evaluate_conformal(model, val_loader, test_loader, device, adj, horizons, out_dir, model_name)

    step_pbar.set_postfix_str("done ✓", refresh=True)
    step_pbar.close()
    return metric_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Results → {out_dir}")

    meta    = joblib.load(DATA_PROCESSED / 'meta.joblib')
    scalers = joblib.load(DATA_PROCESSED / 'scalers.joblib')
    adj     = torch.tensor(meta['adj'], dtype=torch.float32).to(device)

    test_ds = AQISequenceDataset(DATA_PROCESSED / 'test.pt')
    val_ds  = AQISequenceDataset(DATA_PROCESSED / 'val.pt')
    print(f"Test sequences: {len(test_ds):,}")

    models_to_eval = (
        ['greeneyes', 'lstm', 'gru', 'transformer', 'wavenet']
        if args.ablation else [args.model]
    )

    # ── Overall progress bar across all models ─────────────────────────────
    all_rows = []
    overall_pbar = tqdm(
        models_to_eval,
        desc="Overall progress",
        ncols=90,
        unit="model",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} models  [{elapsed}<{remaining}]",
    )
    for m in overall_pbar:
        overall_pbar.set_postfix_str(m, refresh=True)
        rows = evaluate_model(m, device, meta, test_ds, val_ds, adj, scalers, out_dir)
        if rows:
            all_rows.extend(rows)
    overall_pbar.set_postfix_str("complete ✓", refresh=True)
    overall_pbar.close()

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(out_dir / 'overall_metrics.csv', index=False)
        reg     = df[df['horizon'] != 'classification']
        summary = reg.groupby('model')[['MAE', 'RMSE', 'R2', 'Within_10pct']].mean()
        print("\n\n" + "="*60)
        print("  SUMMARY TABLE (mean over all horizons)")
        print("="*60)
        print(summary.to_string())
        if args.ablation:
            plot_ablation_bar(df, out_dir)
            df.to_csv(out_dir / 'ablation_table.csv', index=False)

    print(f"\nAll results saved to: {out_dir.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default='greeneyes',
                        choices=['greeneyes','lstm','gru','transformer','wavenet'])
    parser.add_argument('--ablation', action='store_true')
    args = parser.parse_args()
    main(args)