"""
f4-routing/training/generate_training_data.py

Generates the supervised training dataset for the route quality predictor.

Runs all three baseline heuristics across all three warehouse sizes.
Combines results into one dataset so the predictor generalises
across warehouse layouts.

CHANGELOG:
  v1: demo_weight permanently tied to task_tier (bug)
  v2: Decoupled demo_weight from task_tier (partial fix)
  v3: task_tier masking disabled during measurement
  v4: Distance-based time model (current)
      Previous model used fixed SIM_SECONDS_PER_STEP = 45 per pick,
      which made picks_per_hour always equal 80.0 regardless of
      route quality. All baselines were identical — the predictor
      had nothing to learn.

      Fix: warehouse_env.py now computes sim_time as:
        step_time = (physical_distance / 1.2 m/s) + 15 seconds
      Efficient routes (short physical distance) produce more picks/hr.
      Inefficient routes (long physical distance) produce fewer picks/hr.
      Baselines are now genuinely distinguishable.

Target: picks_per_hour should vary meaningfully across baselines.
        Expected range: 100-250 picks/hr depending on route and order size.
        S-shape should consistently outperform random routing.

v4 changes:
  - Order-aware episodes: each episode is one picker trip with
    one or more orders in the batch (matching real warehouse operation)
  - Two target labels: item_pph AND order_pph (equal weight)
  - New features: n_orders_in_batch, mean_order_size, batch_type
  - task_tier fixed to Heavy during data collection (no masking)
  - Distance-based time model throughout
"""

import sys
import os
import csv
import random
import numpy as np

_F4_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENV_DIR  = os.path.join(_F4_ROOT, "gym-environment")
_SRC_DIR  = os.path.join(_F4_ROOT, "src")
_EVAL_DIR = os.path.join(_F4_ROOT, "evaluation")
_DATA_DIR = os.path.join(_F4_ROOT, "data", "orders")

for d in [_ENV_DIR, _SRC_DIR, _EVAL_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from warehouse_env import WarehouseEnv
from baselines import (sshape_route, largest_gap_route,
                       random_route, run_baseline_episode)

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(_F4_ROOT, "data")
OUT_PATH  = os.path.join(DATA_DIR, "training_dataset.csv")

WAREHOUSES = [
    {
        "name":        "warehouse_small",
        "enc":         0,
        "path":        os.path.join(DATA_DIR, "graphs", "warehouse_small.json"),
        "orders_path": os.path.join(DATA_DIR, "orders", "orders_small.json")
    },
    {
        "name":        "warehouse_medium",
        "enc":         1,
        "path":        os.path.join(DATA_DIR, "graphs", "warehouse_medium.json"),
        "orders_path": os.path.join(DATA_DIR, "orders", "orders_medium.json")
    },
    {
        "name":        "warehouse_large",
        "enc":         2,
        "path":        os.path.join(DATA_DIR, "graphs", "warehouse_large.json"),
        "orders_path": os.path.join(DATA_DIR, "orders", "orders_large.json")
    },
]

BASELINES = [
    ("sshape",      sshape_route),
    ("largest_gap", largest_gap_route),
    ("random",      random_route),
]

EPISODES_PER_CONFIG = 300

ERGO_RANGE = list(range(1, 11))
GAIT_RANGE = list(range(1, 11))
DEMO_RANGE = [1.0, 1.3, 1.6]
TIER_RANGE = ["Heavy", "Moderate", "Light"]

CAT_ENC  = {"Small": 0, "Medium": 1, "Large": 2}
TIER_ENC = {"Heavy": 0, "Moderate": 1, "Light": 2}

# Batch composition rules
LARGE_ORDER_THRESHOLD = 15   # orders above this are always solo
MAX_BATCH_ORDERS      = 3    # maximum orders per batch during training

FIELDNAMES = [
    # Warehouse context
    "warehouse", "warehouse_enc",
    # Baseline used
    "baseline",
    # Batch features — what the predictor learns FROM
    "n_orders_in_batch",      # 1 = solo trip, 2-3 = multi-order batch
    "mean_order_size",        # average items per order in this batch
    "total_items_in_batch",   # total items the picker must collect
    "batch_type",             # "solo_large", "solo_small", "multi"
    "batch_type_enc",         # 0=solo_large, 1=solo_small, 2=multi
    "aisle_spread",           # unique aisles / total items
    # Picker state features
    "ergo_score", "gait_score", "demo_weight",
    "task_tier",  "task_tier_enc",
    # Physical outcome
    "total_distance_m", "return_distance_m",
    # Primary target labels — BOTH equally important
    "item_pph",             # items per hour (routing quality)
    "order_pph",            # complete orders per hour (batching + routing)
    # Secondary metrics
    "mean_cycle_time_sec",  # responsiveness
    "dist_per_item_m",      # travel efficiency
    "completion_rate",      # fraction of items successfully picked
    "total_reward"
]


def aisle_spread(pick_items: list) -> float:
    """Unique aisles used divided by total items in batch."""
    if not pick_items:
        return 0.0
    aisles = set()
    for item in pick_items:
        loc = item.get("shelf_location", "")
        if loc.startswith("A"):
            try:
                aisles.add(int(loc.split("-")[0][1:]))
            except ValueError:
                pass
    return round(len(aisles) / max(len(pick_items), 1), 4)


def sample_batch(all_orders: list, large_threshold: int,
                 max_batch_orders: int) -> list:
    """
    Samples a realistic batch of orders for one training episode.

    Rules matching real operation:
      - Large orders (> large_threshold items): always solo
      - Small/Medium: batch 1-3 together randomly
        (in training we randomise batch size to cover all scenarios)
    """
    if not all_orders:
        return []

    # Pick a random seed order
    seed_order = random.choice(all_orders)

    if seed_order["order_size"] > large_threshold:
        # Large order — always solo
        return [seed_order]

    # Small/Medium — decide batch size randomly (1, 2, or 3 orders)
    n_batch = random.randint(1, max_batch_orders)
    if n_batch == 1:
        return [seed_order]

    # Add spatially compatible orders to the batch
    batch = [seed_order]
    seed_aisle = _order_centroid(seed_order)

    candidates = [o for o in all_orders
                  if o["order_id"] != seed_order["order_id"]
                  and o["order_size"] <= large_threshold]
    random.shuffle(candidates)

    for candidate in candidates:
        if len(batch) >= n_batch:
            break
        cand_aisle = _order_centroid(candidate)
        if abs(cand_aisle - seed_aisle) <= 2.5:   # proximity threshold
            batch.append(candidate)

    return batch


def _order_centroid(order: dict) -> float:
    aisles = []
    for item in order.get("items", []):
        loc = item.get("shelf_location", "")
        if loc.startswith("A"):
            try:
                aisles.append(int(loc.split("-")[0][1:]))
            except ValueError:
                pass
    return float(np.mean(aisles)) if aisles else 0.0


def classify_batch(orders: list, large_threshold: int) -> tuple:
    """Returns (batch_type_str, batch_type_enc)."""
    if len(orders) == 1:
        if orders[0]["order_size"] > large_threshold:
            return "solo_large", 0
        return "solo_small", 1
    return "multi", 2


# ── Main loop ──────────────────────────────────────────────────────────────
all_rows      = []
total_planned = len(WAREHOUSES) * len(BASELINES) * EPISODES_PER_CONFIG

print("=" * 65)
print("F4 Training Data Generator — v4 (order-aware dual metrics)")
print("=" * 65)
print(f"Generating {total_planned} training samples")
print(f"  Target labels: item_pph AND order_pph (equal weight)")
print(f"  Batch types:   solo_large | solo_small | multi-order")
print()

for wh in WAREHOUSES:

    for label, path in [("Graph",  wh["path"]),
                        ("Orders", wh["orders_path"])]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    print(f"{'=' * 55}")
    print(f"Warehouse: {wh['name']}")
    print(f"{'=' * 55}")

    env = WarehouseEnv(
        floor_plan_path=wh["path"],
        orders_path=wh["orders_path"],
        max_episode_steps=200
    )

    # Load order pool
    import json
    with open(wh["orders_path"]) as f:
        order_data = json.load(f)
    order_pool = order_data["orders"]

    for baseline_name, route_fn in BASELINES:

        print(f"\n  Baseline: {baseline_name.upper()}")
        item_pphs    = []
        order_pphs   = []
        cycle_times  = []
        zero_item    = 0
        zero_order   = 0

        for ep in range(EPISODES_PER_CONFIG):

            # Randomise F3 state
            ergo = random.choice(ERGO_RANGE)
            gait = random.choice(GAIT_RANGE)
            demo = random.choice(DEMO_RANGE)
            tier = random.choice(TIER_RANGE)   # feature only

            env.last_ergo_score  = ergo
            env.last_gait_score  = gait
            env.last_demo_weight = demo
            env.last_task_tier   = "Heavy"   # no masking during collection

            # Sample a realistic batch
            batch_orders = sample_batch(
                order_pool, LARGE_ORDER_THRESHOLD, MAX_BATCH_ORDERS
            )

            # Load batch into environment
            env.load_batch({
                "batch_id": f"TRAIN-{ep}",
                "is_solo":  len(batch_orders) == 1,
                "orders":   batch_orders
            })

            info = run_baseline_episode(env, route_fn)

            # Batch characteristics for features
            all_items      = [item for o in batch_orders
                              for item in o["items"]]
            spread         = aisle_spread(all_items)
            mean_ord_size  = np.mean([o["order_size"]
                                      for o in batch_orders])
            total_items    = sum(o["order_size"] for o in batch_orders)
            btype, benc    = classify_batch(
                batch_orders, LARGE_ORDER_THRESHOLD
            )

            ipph  = info["item_pph"]
            opph  = info["order_pph"]
            ctime = info["mean_cycle_time_sec"]

            if ipph  == 0.0: zero_item  += 1
            if opph  == 0.0: zero_order += 1

            item_pphs.append(ipph)
            order_pphs.append(opph)
            cycle_times.append(ctime)

            all_rows.append({
                "warehouse":           wh["name"],
                "warehouse_enc":       wh["enc"],
                "baseline":            baseline_name,
                "n_orders_in_batch":   len(batch_orders),
                "mean_order_size":     round(float(mean_ord_size), 2),
                "total_items_in_batch": total_items,
                "batch_type":          btype,
                "batch_type_enc":      benc,
                "aisle_spread":        spread,
                "ergo_score":          ergo,
                "gait_score":          gait,
                "demo_weight":         demo,
                "task_tier":           tier,
                "task_tier_enc":       TIER_ENC[tier],
                "total_distance_m":    info["total_distance_m"],
                "return_distance_m":   info["return_distance_m"],
                "item_pph":            ipph,
                "order_pph":           opph,
                "mean_cycle_time_sec": ctime,
                "dist_per_item_m":     info["dist_per_item_m"],
                "completion_rate":     info["completion_rate"],
                "total_reward":        info["total_reward"]
            })

            if (ep + 1) % 100 == 0:
                print(f"    [{ep+1}/{EPISODES_PER_CONFIG}] "
                      f"item_pph={np.mean(item_pphs):.1f}  "
                      f"order_pph={np.mean(order_pphs):.2f}  "
                      f"cycle={np.mean(cycle_times):.0f}s")

        print(f"  {baseline_name} done | "
              f"item_pph: {np.mean(item_pphs):.1f} | "
              f"order_pph: {np.mean(order_pphs):.2f} | "
              f"cycle: {np.mean(cycle_times):.0f}s | "
              f"zero_item={zero_item} zero_order={zero_order}")

    print()

# ── Save ───────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved {len(all_rows)} samples → {OUT_PATH}")
print()

# ── Validation ─────────────────────────────────────────────────────────────
item_pphs_all  = [float(r["item_pph"])  for r in all_rows]
order_pphs_all = [float(r["order_pph"]) for r in all_rows]

print("Dataset health check:")
print(f"  item_pph  : mean={np.mean(item_pphs_all):.1f}  "
      f"std={np.std(item_pphs_all):.1f}  "
      f"min={np.min(item_pphs_all):.1f}  "
      f"max={np.max(item_pphs_all):.1f}")
print(f"  order_pph : mean={np.mean(order_pphs_all):.2f}  "
      f"std={np.std(order_pphs_all):.2f}  "
      f"min={np.min(order_pphs_all):.2f}  "
      f"max={np.max(order_pphs_all):.2f}")
print()

print("Baseline comparison (mean item_pph | order_pph):")
for bl in ["sshape", "largest_gap", "random"]:
    s = [r for r in all_rows if r["baseline"] == bl]
    i = [float(r["item_pph"])  for r in s]
    o = [float(r["order_pph"]) for r in s]
    print(f"  {bl:<15}: item_pph={np.mean(i):.1f}  "
          f"order_pph={np.mean(o):.2f}")
print()

print("Batch type breakdown:")
for bt in ["solo_large", "solo_small", "multi"]:
    s = [r for r in all_rows if r["batch_type"] == bt]
    if not s: continue
    i = [float(r["item_pph"])  for r in s]
    o = [float(r["order_pph"]) for r in s]
    print(f"  {bt:<12}: n={len(s)}  "
          f"item_pph={np.mean(i):.1f}  order_pph={np.mean(o):.2f}")
print()

print("Feature correlations with item_pph:")
arr_i = np.array(item_pphs_all)
feats = ["warehouse_enc","total_items_in_batch","n_orders_in_batch",
         "aisle_spread","ergo_score","total_distance_m","batch_type_enc"]
for f in feats:
    v = np.array([float(r[f]) for r in all_rows])
    if np.std(v) < 1e-9: continue
    c = float(np.corrcoef(v, arr_i)[0,1])
    if np.isnan(c): continue
    print(f"  {f:<25}: r={c:+.3f}")
print()

print("Feature correlations with order_pph:")
arr_o = np.array(order_pphs_all)
for f in feats:
    v = np.array([float(r[f]) for r in all_rows])
    if np.std(v) < 1e-9: continue
    c = float(np.corrcoef(v, arr_o)[0,1])
    if np.isnan(c): continue
    print(f"  {f:<25}: r={c:+.3f}")
print()
print("Next step: python training/train_predictor.py")