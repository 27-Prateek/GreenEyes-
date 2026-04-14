"""
preprocessing/sequence_fix.py
==============================
Drop-in replacement for Section 10 (build_sequences) in preprocess.py.

Changes vs original:
  - Never materialises all sequences in RAM (was 28.1 GiB → OOM)
  - Writes to numpy memmap in 500 MB chunks instead of one allocation
  - Two-pass approach: count valid sequences first, then fill memmaps
  - AQIDataset wraps memmaps for zero-copy lazy loading in DataLoader
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# How many sequences to accumulate in RAM before flushing to the memmap.
# 500 MB budget:
#   each sequence = window_size x n_features x 4 bytes = 168 x 64 x 4 = 43 008 B
#   500 MB / 43 008 B = ~12 000 sequences per chunk  (we use 11 000 for headroom)
#
# If you want a different budget, change CHUNK_MB — everything else is derived.
CHUNK_MB = 500


# ══════════════════════════════════════════════════════════════════════════════
# LAZY DATASET
# ══════════════════════════════════════════════════════════════════════════════

class AQIDataset(Dataset):
    """
    Reads sequences lazily from memory-mapped .npy files.
    RAM footprint at any time = one batch, not the whole dataset.

    Files on disk (all inside DATA_PROCESSED):
        {split}_X.npy        float32  (N, window_size, n_features)
        {split}_y_reg.npy    float32  (N, n_horizons)
        {split}_y_cls.npy    int64    (N,)
        {split}_info.joblib  metadata dict
    """
    def __init__(self, data_dir: Path, split: str):
        self.data_dir = Path(data_dir)
        self.split    = split
        info          = joblib.load(self.data_dir / f'{split}_info.joblib')
        self.n        = info['n']
        self.shape_X  = info['shape_X']
        self.shape_yr = info['shape_yr']
        self.meta     = info['meta']

        # mmap_mode='r' -> OS pages sequences in on demand, no full load
        self._X     = np.load(str(self.data_dir / f'{split}_X.npy'),     mmap_mode='r')
        self._y_reg = np.load(str(self.data_dir / f'{split}_y_reg.npy'), mmap_mode='r')
        self._y_cls = np.load(str(self.data_dir / f'{split}_y_cls.npy'), mmap_mode='r')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.tensor(np.array(self._X[idx]),     dtype=torch.float32),
            torch.tensor(np.array(self._y_reg[idx]), dtype=torch.float32),
            torch.tensor(int(self._y_cls[idx]),       dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_size(window_size: int, n_feat: int, budget_mb: int) -> int:
    """How many sequences fit in `budget_mb` MB of RAM."""
    bytes_per_seq = window_size * n_feat * 4   # float32
    chunk = max(1, (budget_mb * 1024 * 1024) // bytes_per_seq)
    return int(chunk)


def _write_empty_split(out_dir: Path, split_name: str,
                       window_size: int, n_feat: int, n_horizons: int):
    """Write zero-length arrays so downstream code never sees missing files."""
    np.save(str(out_dir / f'{split_name}_X.npy'),
            np.empty((0, window_size, n_feat), dtype=np.float32))
    np.save(str(out_dir / f'{split_name}_y_reg.npy'),
            np.empty((0, n_horizons), dtype=np.float32))
    np.save(str(out_dir / f'{split_name}_y_cls.npy'),
            np.empty((0,), dtype=np.int64))
    joblib.dump(
        dict(n=0, shape_X=(0, window_size, n_feat),
             shape_yr=(0, n_horizons), meta=[]),
        out_dir / f'{split_name}_info.joblib'
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(
    df,
    feature_cols:      list,
    out_dir:           Path,
    split_name:        str,
    target_col:        str   = 'AQI_poly',
    label_col:         str   = 'AQI_label',
    window_size:       int   = 168,
    forecast_horizons: list  = None,
    city_to_idx:       dict  = None,
    nan_threshold:     float = 0.30,
    chunk_mb:          int   = CHUNK_MB,
) -> AQIDataset:
    """
    Stream sliding-window sequences to disk in `chunk_mb` MB chunks.

    Peak RAM = chunk_mb MB (500 MB), regardless of dataset size.
    Returns an AQIDataset ready to pass directly to torch.utils.data.DataLoader.
    """
    if forecast_horizons is None:
        forecast_horizons = [1, 24, 72]

    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_h         = max(forecast_horizons)
    n_feat        = len(feature_cols)
    n_horizons    = len(forecast_horizons)
    n_mask_cols   = sum(1 for c in feature_cols if c.endswith('_mask'))
    n_signal_cols = n_feat - n_mask_cols
    chunk         = _chunk_size(window_size, n_feat, chunk_mb)

    print(f"  [{split_name}] chunk_mb={chunk_mb} -> {chunk:,} seqs/flush")

    # ── Pass 1: count valid sequences (no data copied to RAM) ────────────────
    print(f"  [{split_name}] Pass 1/2 — counting valid sequences...")
    n_valid       = 0
    skipped_nan   = 0
    skipped_short = 0

    for city, grp in df.groupby('City'):
        grp     = grp.sort_values('datetime').reset_index(drop=True)
        feats   = grp[feature_cols].values.astype(np.float32)
        targets = grp[target_col].values.astype(np.float32)
        n       = len(grp)

        if n < window_size + max_h:
            skipped_short += 1
            continue

        for i in range(window_size, n - max_h + 1):
            window = feats[i - window_size : i, :n_signal_cols]
            if np.isnan(window).mean() > nan_threshold:
                skipped_nan += 1
                continue
            if all(
                (i + h - 1 < n) and not np.isnan(targets[i + h - 1])
                for h in forecast_horizons
            ):
                n_valid += 1

    print(f"  [{split_name}] Valid={n_valid:,} | "
          f"skipped_nan={skipped_nan:,} | skipped_short={skipped_short}")

    if n_valid == 0:
        _write_empty_split(out_dir, split_name, window_size, n_feat, n_horizons)
        print(f"  [{split_name}] Empty split — skipping.")
        return AQIDataset(out_dir, split_name)

    # ── Allocate memmaps on disk ──────────────────────────────────────────────
    mmap_X     = np.lib.format.open_memmap(
        str(out_dir / f'{split_name}_X.npy'),
        mode='w+', dtype=np.float32, shape=(n_valid, window_size, n_feat))
    mmap_y_reg = np.lib.format.open_memmap(
        str(out_dir / f'{split_name}_y_reg.npy'),
        mode='w+', dtype=np.float32, shape=(n_valid, n_horizons))
    mmap_y_cls = np.lib.format.open_memmap(
        str(out_dir / f'{split_name}_y_cls.npy'),
        mode='w+', dtype=np.int64, shape=(n_valid,))

    # ── Pass 2: fill memmaps in 500 MB chunks ────────────────────────────────
    print(f"  [{split_name}] Pass 2/2 — writing in {chunk_mb} MB chunks...")

    # Staging buffers — only these live in RAM
    buf_X     = np.empty((chunk, window_size, n_feat), dtype=np.float32)
    buf_y_reg = np.empty((chunk, n_horizons),          dtype=np.float32)
    buf_y_cls = np.empty((chunk,),                     dtype=np.int64)

    idx_out = 0   # position in the memmap
    buf_pos = 0   # position in the staging buffer
    meta    = []

    def _flush(buf_pos, idx_out):
        if buf_pos == 0:
            return idx_out
        mmap_X    [idx_out : idx_out + buf_pos] = buf_X    [:buf_pos]
        mmap_y_reg[idx_out : idx_out + buf_pos] = buf_y_reg[:buf_pos]
        mmap_y_cls[idx_out : idx_out + buf_pos] = buf_y_cls[:buf_pos]
        return idx_out + buf_pos

    for city, grp in df.groupby('City'):
        grp     = grp.sort_values('datetime').reset_index(drop=True)
        feats   = grp[feature_cols].values.astype(np.float32)
        targets = grp[target_col].values.astype(np.float32)
        labels  = grp[label_col].values.astype(np.int64)
        n       = len(grp)

        if n < window_size + max_h:
            continue

        for i in range(window_size, n - max_h + 1):
            window = feats[i - window_size : i]
            if np.isnan(window[:, :n_signal_cols]).mean() > nan_threshold:
                continue

            y_horizons = []
            valid = True
            for h in forecast_horizons:
                tidx = i + h - 1
                if tidx >= n or np.isnan(targets[tidx]):
                    valid = False
                    break
                y_horizons.append(targets[tidx])
            if not valid:
                continue

            buf_X    [buf_pos] = np.nan_to_num(window, nan=0.0)
            buf_y_reg[buf_pos] = y_horizons
            buf_y_cls[buf_pos] = int(labels[i])
            meta.append({
                'city':     city,
                'datetime': grp['datetime'].iloc[i],
                'city_idx': city_to_idx.get(city, 0) if city_to_idx else 0,
            })
            buf_pos += 1

            if buf_pos == chunk:
                idx_out = _flush(buf_pos, idx_out)
                buf_pos = 0
                print(f"    [{split_name}] {idx_out:,}/{n_valid:,} written")

    # Flush final partial chunk
    idx_out = _flush(buf_pos, idx_out)

    # Release memmaps — forces OS to flush pages to disk
    del mmap_X, mmap_y_reg, mmap_y_cls

    info = dict(
        n        = n_valid,
        shape_X  = (n_valid, window_size, n_feat),
        shape_yr = (n_valid, n_horizons),
        meta     = meta,
    )
    joblib.dump(info, out_dir / f'{split_name}_info.joblib')
    print(f"  [{split_name}] Done — {n_valid:,} sequences saved to {out_dir}")
    return AQIDataset(out_dir, split_name)


# ══════════════════════════════════════════════════════════════════════════════
# LOADER HELPER  (use this in your training script)
# ══════════════════════════════════════════════════════════════════════════════

def load_split(data_processed_dir: Path, split: str) -> AQIDataset:
    """
    Load a preprocessed split as a lazy AQIDataset.

    Usage in train.py:
        from preprocessing.sequence_fix import load_split
        train_ds = load_split(DATA_PROCESSED, 'train')
        val_ds   = load_split(DATA_PROCESSED, 'val')
        loader   = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    """
    return AQIDataset(Path(data_processed_dir), split)


# ══════════════════════════════════════════════════════════════════════════════
# PASTE THIS BLOCK INTO run_pipeline() in preprocess.py
# Replace everything from "[10] Building..." to the end of the function body.
# ══════════════════════════════════════════════════════════════════════════════
"""
    # 10 + 11. Build sequences & save (500 MB chunks — no RAM spike)
    from preprocessing.sequence_fix import build_sequences, load_split

    print("\\n[10] Building sliding-window sequences (500 MB chunks)...")
    kw = dict(
        feature_cols       = feature_cols,
        out_dir            = DATA_PROCESSED,
        target_col         = 'AQI_poly',
        label_col          = 'AQI_label',
        window_size        = WINDOW_SIZE,
        forecast_horizons  = FORECAST_HOURS,
        city_to_idx        = city_to_idx,
        chunk_mb           = 500,
    )
    train_ds = build_sequences(train_df, split_name='train', **kw)
    val_ds   = build_sequences(val_df,   split_name='val',   **kw)
    test_ds  = build_sequences(test_df,  split_name='test',  **kw)

    # Save shared metadata
    meta = {
        'feature_cols':   feature_cols,
        'city_to_idx':    city_to_idx,
        'adj':            adj,
        'n_features':     len(feature_cols),
        'forecast_hours': FORECAST_HOURS,
        'window_size':    WINDOW_SIZE,
        'n_train':        len(train_ds),
        'n_val':          len(val_ds),
        'n_test':         len(test_ds),
    }
    joblib.dump(meta,    DATA_PROCESSED / 'meta.joblib')
    joblib.dump(scalers, DATA_PROCESSED / 'scalers.joblib')
    df.to_parquet(DATA_PROCESSED / 'hourly_clean.parquet', index=False)

    print("\\n" + "=" * 60)
    print("  Preprocessing complete.")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Train    : {len(train_ds):,} sequences")
    print(f"  Val      : {len(val_ds):,} sequences")
    print(f"  Test     : {len(test_ds):,} sequences")
    print(f"  Saved to : {DATA_PROCESSED}")
    print("=" * 60)
    return meta
"""