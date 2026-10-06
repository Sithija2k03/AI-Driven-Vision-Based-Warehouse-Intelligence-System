"""
f4-routing/training/train_predictor.py

Trains a multi-output neural network to predict BOTH:
  item_pph  : items picked per hour (routing quality)
  order_pph : complete orders per hour (batching + routing quality)

Both outputs are weighted equally in the loss function.
This reflects the equal operational importance of item
throughput and order completion speed.

Architecture:
  Input  : 8 features (warehouse, batch size, order size,
                        aisle spread, ergo, gait, demo, distance)
  Hidden : 128 → 64 → 32 with ReLU + Dropout
  Output : 2 units [item_pph_norm, order_pph_norm]
  Loss   : 0.5 * MSE(item_pph) + 0.5 * MSE(order_pph)
  Optim  : Adam lr=1e-3

Output files:
  data/predictor_weights.pt    trained model + norm params
  data/predictor_metrics.json  training history + eval metrics
"""

import sys
import os
import csv
import json
import numpy as np

_F4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(_F4_ROOT, "data")
CSV_PATH  = os.path.join(DATA_DIR, "training_dataset.csv")
WT_PATH   = os.path.join(DATA_DIR, "predictor_weights.pt")
MET_PATH  = os.path.join(DATA_DIR, "predictor_metrics.json")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load dataset ───────────────────────────────────────────────────────────
print(f"Loading dataset from {CSV_PATH}")
rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
print(f"Loaded {len(rows)} samples")

# ── Features and labels ────────────────────────────────────────────────────
FEATURE_COLS = [
    "warehouse_enc",        # warehouse size
    "n_orders_in_batch",    # how many orders in this trip
    "mean_order_size",      # average order complexity
    "total_items_in_batch", # total picker workload
    "aisle_spread",         # spatial spread of items
    "ergo_score",           # picker risk level
    "gait_score",           # fall risk
    "total_distance_m",     # physical route length
]
LABEL_COLS = ["item_pph", "order_pph"]   # equal weight targets

X = np.array([[float(r[c]) for c in FEATURE_COLS]
               for r in rows], dtype=np.float32)
Y = np.array([[float(r[c]) for c in LABEL_COLS]
               for r in rows], dtype=np.float32)

print(f"Features: {X.shape}   Labels: {Y.shape}")
print(f"item_pph  range: {Y[:,0].min():.1f} – {Y[:,0].max():.1f}")
print(f"order_pph range: {Y[:,1].min():.2f} – {Y[:,1].max():.2f}")

# ── Normalise ──────────────────────────────────────────────────────────────
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_rng = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
X_norm = (X - X_min) / X_rng

Y_min = Y.min(axis=0)
Y_max = Y.max(axis=0)
Y_rng = np.where(Y_max - Y_min == 0, 1.0, Y_max - Y_min)
Y_norm = (Y - Y_min) / Y_rng

norm_params = {
    "X_min": X_min.tolist(), "X_max": X_max.tolist(),
    "Y_min": Y_min.tolist(), "Y_max": Y_max.tolist(),
    "feature_cols": FEATURE_COLS, "label_cols": LABEL_COLS
}

# ── Train / val split ──────────────────────────────────────────────────────
n      = len(X_norm)
idx    = np.random.permutation(n)
split  = int(n * 0.8)
tr_idx, va_idx = idx[:split], idx[split:]

Xtr = torch.tensor(X_norm[tr_idx])
Ytr = torch.tensor(Y_norm[tr_idx])
Xva = torch.tensor(X_norm[va_idx])
Yva = torch.tensor(Y_norm[va_idx])

print(f"Train: {len(tr_idx)}   Val: {len(va_idx)}")

train_dl = DataLoader(
    TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True
)

# ── Model ──────────────────────────────────────────────────────────────────
class DualOutputPredictor(nn.Module):
    """
    Predicts both item_pph and order_pph simultaneously.
    Shared feature extraction layers capture warehouse-level
    routing patterns common to both metrics.
    Separate output heads let each metric specialise.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,  32),         nn.ReLU(),
        )
        # Separate heads — item_pph and order_pph can specialise
        self.head_item  = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_order = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        shared = self.shared(x)
        return torch.cat([self.head_item(shared),
                          self.head_order(shared)], dim=1)


model     = DualOutputPredictor(n_features=len(FEATURE_COLS))
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
mse       = nn.MSELoss()

def dual_loss(pred, target):
    """
    Equal-weight dual-objective loss.
    Both item_pph and order_pph contribute equally to training signal.
    This ensures the model is genuinely optimised for both metrics,
    not biased toward whichever has larger numerical range.
    """
    loss_item  = mse(pred[:, 0], target[:, 0])
    loss_order = mse(pred[:, 1], target[:, 1])
    return 0.5 * loss_item + 0.5 * loss_order

print(f"\nModel parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS   = 100
history  = {"train_loss": [], "val_loss": [],
            "val_mae_item": [], "val_mae_order": []}
best_val = float("inf")
best_wts = None

print(f"\nTraining {EPOCHS} epochs...\n")
print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} "
      f"{'MAE item_pph':>14} {'MAE order_pph':>14}")
print("-" * 58)

for epoch in range(1, EPOCHS + 1):

    model.train()
    tl = []
    for xb, yb in train_dl:
        optimiser.zero_grad()
        loss = dual_loss(model(xb), yb)
        loss.backward()
        optimiser.step()
        tl.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_pred = model(Xva)
        vl       = dual_loss(val_pred, Yva).item()

        # Denormalise for MAE in real units
        vp_np = val_pred.numpy()
        yt_np = Yva.numpy()

        mae_item  = float(np.mean(np.abs(
            vp_np[:,0] * Y_rng[0] + Y_min[0] -
            (yt_np[:,0] * Y_rng[0] + Y_min[0])
        )))
        mae_order = float(np.mean(np.abs(
            vp_np[:,1] * Y_rng[1] + Y_min[1] -
            (yt_np[:,1] * Y_rng[1] + Y_min[1])
        )))

    tl_mean = float(np.mean(tl))
    history["train_loss"].append(round(tl_mean, 6))
    history["val_loss"].append(round(vl, 6))
    history["val_mae_item"].append(round(mae_item, 3))
    history["val_mae_order"].append(round(mae_order, 4))

    if vl < best_val:
        best_val = vl
        best_wts = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6} {tl_mean:>10.6f} {vl:>10.6f} "
              f"{mae_item:>13.2f}  {mae_order:>13.4f}")

# ── Restore best weights ───────────────────────────────────────────────────
model.load_state_dict(best_wts)

# ── Final evaluation ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    fp      = model(Xva).numpy()
    yt      = Yva.numpy()
    fp_real = fp * Y_rng + Y_min
    yt_real = yt * Y_rng + Y_min

final_mae_item  = float(np.mean(np.abs(fp_real[:,0] - yt_real[:,0])))
final_mae_order = float(np.mean(np.abs(fp_real[:,1] - yt_real[:,1])))
final_r2_item   = float(1 - np.sum((yt_real[:,0]-fp_real[:,0])**2)
                           / np.sum((yt_real[:,0]-yt_real[:,0].mean())**2))
final_r2_order  = float(1 - np.sum((yt_real[:,1]-fp_real[:,1])**2)
                           / np.sum((yt_real[:,1]-yt_real[:,1].mean())**2))

print(f"\n{'='*58}")
print(f"Final evaluation ({len(va_idx)} validation samples):")
print(f"  item_pph  — MAE: {final_mae_item:.2f} picks/hr   "
      f"R²: {final_r2_item:.4f}")
print(f"  order_pph — MAE: {final_mae_order:.4f} orders/hr  "
      f"R²: {final_r2_order:.4f}")
print(f"{'='*58}")

# ── Save ───────────────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": best_wts,
    "norm_params":      norm_params,
    "n_features":       len(FEATURE_COLS),
    "architecture":     "128-64-32 shared + dual heads"
}, WT_PATH)
print(f"\nModel saved → {WT_PATH}")

metrics = {
    "training_samples":    len(tr_idx),
    "validation_samples":  len(va_idx),
    "epochs":              EPOCHS,
    "best_val_loss":       round(best_val, 6),
    "final_mae_item_pph":  round(final_mae_item, 3),
    "final_mae_order_pph": round(final_mae_order, 4),
    "final_r2_item_pph":   round(final_r2_item, 4),
    "final_r2_order_pph":  round(final_r2_order, 4),
    "feature_cols":        FEATURE_COLS,
    "label_cols":          LABEL_COLS,
    "history":             history,
    "norm_params":         norm_params
}
with open(MET_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {MET_PATH}")

print(f"\nSummary:")
print(f"  item_pph  predictor: ±{final_mae_item:.1f} picks/hr  "
      f"(R²={final_r2_item:.2f})")
print(f"  order_pph predictor: ±{final_mae_order:.3f} orders/hr "
      f"(R²={final_r2_order:.2f})")
print(f"\nNext step: python training/train_ppo.py")