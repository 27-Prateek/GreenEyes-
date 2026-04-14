"""
models/model.py
===============
GreenEyes++ — Full model architecture.

Components:
  1. Input projection
  2. Stacked WaveNet blocks (dilated causal convolutions)
  3. Graph Attention layer (spatial city dependencies)
  4. Bidirectional LSTM
  5. Custom Temporal Attention (from paper, with learnable temperature)
  6. Regression heads (one per forecast horizon)
  7. Classification head (AQI category)

Also includes:
  - LSTMBaseline and TransformerBaseline for ablation comparison
  - ConformalPredictor for calibrated uncertainty intervals
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# WAVENET COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class WaveNetLayer(nn.Module):
    """
    Single dilated causal convolution with gated activation.
    Mirrors DeepMind WaveNet: output = tanh(W_f * x) ⊙ σ(W_g * x)
    """
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.dilated_conv = nn.Conv1d(
            channels, channels * 2, kernel_size=kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.res_conv  = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self.norm      = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.dilated_conv(x)
        out = out[:, :, :x.size(2)]          # trim future leak from causal padding
        tanh_out = torch.tanh(out[:, :out.size(1)//2])
        gate_out = torch.sigmoid(out[:, out.size(1)//2:])
        gated    = tanh_out * gate_out
        skip     = self.skip_conv(gated)
        res      = self.norm(self.res_conv(gated) + residual)
        return res, skip


class WaveNetBlock(nn.Module):
    """Stack of WaveNetLayers with exponentially growing dilation rates."""
    def __init__(self, channels: int, n_layers: int, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            WaveNetLayer(channels, dilation=2**i, kernel_size=kernel_size)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor):
        skip_sum = torch.zeros_like(x)
        for layer in self.layers:
            x, skip = layer(x)
            skip_sum = skip_sum + skip
        return x, skip_sum


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ATTENTION (spatial city-level message passing)
# ══════════════════════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """
    Multi-head graph attention — pure PyTorch, no torch_geometric needed.
    Each city aggregates information from its neighbours weighted by
    learned attention scores × pre-computed distance weights.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = out_dim // n_heads
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.out = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N_cities, F)
        adj : (N_cities, N_cities) — row-normalised adjacency
        """
        B, N, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)   # (B,H,N,D)
        K = self.W_k(x).view(B, N, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B,H,N,N)

        # Mask non-edges
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)              # cities with no edges
        attn = self.drop(attn * adj.unsqueeze(0).unsqueeze(0))

        out = torch.matmul(attn, V)                          # (B,H,N,D)
        out = out.transpose(1, 2).contiguous().view(B, N, H * D)
        out = self.out(out)
        return self.norm(out + x[:, :, :out.shape[-1]])      # residual


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL ATTENTION (custom from GreenEyes paper, improved)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """
    Custom attention from the GreenEyes paper.
    Formula: Attention(V) = exp(tanh(WV + b)) ⊙ V  [summed over time]
    
    Replaces softmax with exp(tanh(·)) which the paper showed better
    handles periodic temporal data. Added learnable temperature τ.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W           = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        # V : (B, T, dim)
        scores  = self.W(V) / (self.temperature.abs() + 1e-6)
        weights = torch.exp(torch.tanh(scores))              # (B, T, dim)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return (weights * V).sum(dim=1)                      # (B, dim)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL: GreenEyes++
# ══════════════════════════════════════════════════════════════════════════════

class GreenEyesPlus(nn.Module):
    """
    GreenEyes++ Model.

    Forward:
        x        : (B, T, n_features)
        adj      : (N_cities, N_cities) adjacency, optional
        city_idx : (B,) city index for each sample, optional

    Returns:
        reg_preds : list of (B, 1) — one per forecast horizon
        cat_logits: (B, n_categories)
    """
    def __init__(
        self,
        n_features:    int,
        n_cities:      int   = 26,
        hidden:        int   = 128,
        wavenet_layers: list = None,   # e.g. [8, 5, 3]
        kernel_size:   int   = 3,
        lstm_layers:   int   = 2,
        lstm_dropout:  float = 0.2,
        gnn_heads:     int   = 4,
        n_categories:  int   = 6,
        horizons:      list  = None,
        dropout:       float = 0.15,
    ):
        super().__init__()
        self.horizons      = horizons or [1, 24, 72]
        self.hidden        = hidden
        wavenet_layers     = wavenet_layers or [8, 5, 3]

        # 1. Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Stacked WaveNet blocks
        self.wn_blocks = nn.ModuleList([
            WaveNetBlock(hidden, n_layers=nl, kernel_size=kernel_size)
            for nl in wavenet_layers
        ])
        self.wn_dropout = nn.Dropout(dropout)

        # 3. Graph attention (spatial)
        self.gnn = GraphAttentionLayer(hidden, hidden, n_heads=gnn_heads, dropout=dropout)

        # 4. Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden, hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = lstm_dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = hidden * 2

        # 5. Temporal Attention
        self.attn = TemporalAttention(lstm_out)

        # 6. Shared MLP
        self.shared = nn.Sequential(
            nn.Linear(lstm_out, lstm_out),
            nn.LayerNorm(lstm_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, lstm_out // 2),
            nn.LayerNorm(lstm_out // 2),
            nn.GELU(),
        )
        ctx_dim = lstm_out // 2

        # 7. Regression heads
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ctx_dim, ctx_dim // 2),
                nn.GELU(),
                nn.Linear(ctx_dim // 2, 1),
            )
            for _ in self.horizons
        ])

        # 8. Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_dim // 2, n_categories),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        x:        torch.Tensor,
        adj:      torch.Tensor  = None,
        city_idx: torch.Tensor  = None,
    ):
        B, T, F = x.shape

        # 1. Project features
        h = self.input_proj(x)            # (B, T, hidden)
        h = h.transpose(1, 2)             # (B, hidden, T) for conv

        # 2. WaveNet
        skip_total = torch.zeros_like(h)
        for block in self.wn_blocks:
            h, skip = block(h)
            skip_total = skip_total + skip
        h = self.wn_dropout(h + skip_total)
        h = h.transpose(1, 2)             # (B, T, hidden)

        # 3. GNN (if graph provided)
        if adj is not None and city_idx is not None:
            N = adj.shape[0]
            city_repr = h.mean(dim=1)                         # (B, hidden)
            city_mat  = torch.zeros(1, N, self.hidden, device=h.device)
            for b in range(B):
                city_mat[0, city_idx[b]] += city_repr[b]
            city_out = self.gnn(city_mat, adj)                # (1, N, hidden)
            spatial  = city_out[0, city_idx]                  # (B, hidden)
            h = h + spatial.unsqueeze(1)

        # 4. BiLSTM
        lstm_out, _ = self.lstm(h)        # (B, T, hidden*2)

        # 5. Temporal Attention
        ctx = self.attn(lstm_out)         # (B, hidden*2)

        # 6. Shared MLP
        ctx = self.shared(ctx)            # (B, ctx_dim)

        # 7. Heads
        reg  = [head(ctx) for head in self.reg_heads]   # list of (B,1)
        cls_ = self.cls_head(ctx)                        # (B, n_cat)

        return reg, cls_

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE MODELS (for ablation comparison)
# ══════════════════════════════════════════════════════════════════════════════

class LSTMBaseline(nn.Module):
    """Stacked BiLSTM — simplest strong baseline."""
    def __init__(self, n_features, hidden=128, n_layers=3,
                 horizons=None, n_categories=6, dropout=0.2):
        super().__init__()
        self.horizons = horizons or [1, 24, 72]
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        dim = hidden * 2
        self.reg_heads = nn.ModuleList([nn.Linear(dim, 1) for _ in self.horizons])
        self.cls_head  = nn.Linear(dim, n_categories)

    def forward(self, x, **kw):
        out, (h, _) = self.lstm(x)
        ctx = torch.cat([h[-2], h[-1]], dim=1)
        return [head(ctx) for head in self.reg_heads], self.cls_head(ctx)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUBaseline(nn.Module):
    """Stacked BiGRU baseline."""
    def __init__(self, n_features, hidden=128, n_layers=3,
                 horizons=None, n_categories=6, dropout=0.2):
        super().__init__()
        self.horizons = horizons or [1, 24, 72]
        self.gru = nn.GRU(n_features, hidden, n_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        dim = hidden * 2
        self.reg_heads = nn.ModuleList([nn.Linear(dim, 1) for _ in self.horizons])
        self.cls_head  = nn.Linear(dim, n_categories)

    def forward(self, x, **kw):
        out, h = self.gru(x)
        ctx = torch.cat([h[-2], h[-1]], dim=1)
        return [head(ctx) for head in self.reg_heads], self.cls_head(ctx)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBaseline(nn.Module):
    """Transformer encoder baseline."""
    def __init__(self, n_features, hidden=128, n_heads=8, n_layers=4,
                 horizons=None, n_categories=6, dropout=0.1):
        super().__init__()
        self.horizons   = horizons or [1, 24, 72]
        self.input_proj = nn.Linear(n_features, hidden)
        enc_layer       = nn.TransformerEncoderLayer(
            hidden, n_heads, hidden * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, n_layers)
        self.reg_heads  = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.horizons])
        self.cls_head   = nn.Linear(hidden, n_categories)

    def forward(self, x, **kw):
        h   = self.input_proj(x)
        h   = self.encoder(h)
        ctx = h.mean(dim=1)
        return [head(ctx) for head in self.reg_heads], self.cls_head(ctx)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WaveNetOnlyBaseline(nn.Module):
    """WaveNet without LSTM or attention — ablation: w/o LSTM+Attention."""
    def __init__(self, n_features, hidden=128, wavenet_layers=None,
                 horizons=None, n_categories=6, dropout=0.15):
        super().__init__()
        self.horizons   = horizons or [1, 24, 72]
        wavenet_layers  = wavenet_layers or [8, 5, 3]
        self.proj       = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU())
        self.wn_blocks  = nn.ModuleList([
            WaveNetBlock(hidden, n_layers=nl) for nl in wavenet_layers
        ])
        self.reg_heads  = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.horizons])
        self.cls_head   = nn.Linear(hidden, n_categories)

    def forward(self, x, **kw):
        h = self.proj(x).transpose(1, 2)
        skip_total = torch.zeros_like(h)
        for block in self.wn_blocks:
            h, skip = block(h)
            skip_total = skip_total + skip
        ctx = (h + skip_total).mean(dim=2)
        return [head(ctx) for head in self.reg_heads], self.cls_head(ctx)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

class ConformalPredictor:
    """
    Post-hoc conformal prediction — provides statistically valid intervals.
    Requires NO retraining. Calibrate on validation set, apply to test.

    Coverage guarantee: P(y ∈ [ŷ - q, ŷ + q]) ≥ 1 - α
    """
    def __init__(self, model: nn.Module, alpha: float = 0.10):
        self.model     = model
        self.alpha     = alpha          # 0.10 → 90% coverage
        self.quantiles = {}             # per horizon index

    @torch.no_grad()
    def calibrate(self, val_loader, device, adj=None):
        """Compute non-conformity scores on validation set."""
        self.model.eval()
        all_res = {i: [] for i in range(len(self.model.horizons))}

        for X, y_reg, y_cls, city_idx in val_loader:
            X, y_reg = X.to(device), y_reg.to(device)
            city_idx = city_idx.to(device)
            reg_preds, _ = self.model(X, adj=adj, city_idx=city_idx)
            for i, pred in enumerate(reg_preds):
                residuals = (y_reg[:, i] - pred.squeeze()).abs().cpu()
                all_res[i].extend(residuals.tolist())

        for i, residuals in all_res.items():
            r = torch.tensor(residuals)
            n = len(r)
            q_level = math.ceil((1 - self.alpha) * (n + 1)) / n
            self.quantiles[i] = float(torch.quantile(r, min(q_level, 1.0)))

        print(f"  Conformal quantiles (α={self.alpha}, {int((1-self.alpha)*100)}% coverage):")
        for i, q in self.quantiles.items():
            h = self.model.horizons[i]
            print(f"    t+{h:3d}h: ±{q:.4f}")

    @torch.no_grad()
    def predict_with_intervals(self, X, adj=None, city_idx=None):
        self.model.eval()
        reg_preds, cat = self.model(X, adj=adj, city_idx=city_idx)
        intervals = []
        for i, pred in enumerate(reg_preds):
            q = self.quantiles.get(i, 0.0)
            p = pred.squeeze()
            intervals.append({'point': p, 'lower': p - q, 'upper': p + q,
                               'width': 2 * q, 'horizon': self.model.horizons[i]})
        return intervals, cat
