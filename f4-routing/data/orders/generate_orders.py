"""
f4-routing/data/orders/generate_orders.py

Generates 5000 synthetic pick orders in three size categories:
  Small  :  1 -  5 items  (~40% of orders)
  Medium :  6 - 15 items  (~45% of orders)
  Large  : 16 - 20 items  (~15% of orders)

Maximum order size is capped at 20 because the sample warehouse
has 20 unique items. When real WMS data arrives with more SKUs,
update LARGE_MAX to 30 and re-run this script.

Item frequencies follow an ABC demand distribution:
  A-class (top 20% of items)    — picked frequently
  B-class (middle 30% of items) — picked moderately
  C-class (bottom 50% of items) — picked rarely
"""

import json
import os
import random
import numpy as np

# ── Load item list ─────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "item_locations.json")) as f:
    item_data = json.load(f)

items     = item_data["items"]
num_items = len(items)
print(f"Loaded {num_items} items from item_locations.json")

# ── ABC demand weights ─────────────────────────────────────────────────────
def abc_weight(index: int, total: int) -> float:
    p = index / total
    if p < 0.20:  return 3.0   # A-class: fast movers
    elif p < 0.50: return 1.5  # B-class: medium movers
    else:          return 0.5  # C-class: slow movers

weights       = np.array([abc_weight(i, num_items) for i in range(num_items)],
                          dtype=float)
probabilities = weights / weights.sum()

a_count = sum(1 for i in range(num_items) if abc_weight(i, num_items) == 3.0)
b_count = sum(1 for i in range(num_items) if abc_weight(i, num_items) == 1.5)
c_count = sum(1 for i in range(num_items) if abc_weight(i, num_items) == 0.5)
print(f"ABC distribution: A={a_count} items | B={b_count} items | C={c_count} items")

# ── Order size configuration ───────────────────────────────────────────────
#
#  Category  |  Item range   |  Real-world meaning
#  ----------|---------------|----------------------------------
#  Small     |  1 – 5 items  |  Top-up orders, urgent single picks
#  Medium    |  6 – 15 items |  Standard daily replenishment orders
#  Large     | 16 – 20 items |  Bulk orders (capped at 20 for now)
#
# When real WMS data arrives (more than 20 SKUs):
#   Change LARGE_MAX = 30 and re-run to get true large orders
#
SMALL_MIN  = 1
SMALL_MAX  = 5
MEDIUM_MIN = 6
MEDIUM_MAX = 15
LARGE_MIN  = 16
LARGE_MAX  = 20   # capped at num_items until real WMS data arrives

SMALL_RATIO  = 0.40
MEDIUM_RATIO = 0.45
LARGE_RATIO  = 0.15

NUM_ORDERS = 5000


def get_size_and_category() -> tuple:
    roll = random.random()
    if roll < SMALL_RATIO:
        return random.randint(SMALL_MIN, SMALL_MAX), "Small"
    elif roll < SMALL_RATIO + MEDIUM_RATIO:
        return random.randint(MEDIUM_MIN, MEDIUM_MAX), "Medium"
    else:
        # Large: target 16-20, currently all 20 items due to small item pool
        size = random.randint(LARGE_MIN, min(LARGE_MAX, num_items))
        return size, "Large"


def generate_order(order_id: int) -> dict:
    size, category = get_size_and_category()
    size = min(size, num_items)   # safety cap

    selected = np.random.choice(num_items, size=size,
                                 replace=False, p=probabilities)
    order_items = [
        {
            "item_code":      items[i]["item_code"],
            "shelf_location": items[i]["shelf_location"],
            "task_weight":    items[i]["task_weight"]
        }
        for i in selected
    ]

    return {
        "order_id":   f"ORD-{str(order_id).zfill(5)}",
        "category":   category,
        "order_size": size,
        "items":      order_items
    }


# ── Generate ───────────────────────────────────────────────────────────────
print(f"\nGenerating {NUM_ORDERS} orders...")
orders = [generate_order(i) for i in range(1, NUM_ORDERS + 1)]

small_o  = [o for o in orders if o["category"] == "Small"]
medium_o = [o for o in orders if o["category"] == "Medium"]
large_o  = [o for o in orders if o["category"] == "Large"]
sizes    = [o["order_size"] for o in orders]

print(f"\nOrder distribution:")
print(f"  Small  ( 1- 5 items): {len(small_o):>5} "
      f"({len(small_o)/NUM_ORDERS*100:.1f}%)")
print(f"  Medium ( 6-15 items): {len(medium_o):>5} "
      f"({len(medium_o)/NUM_ORDERS*100:.1f}%)")
print(f"  Large  (16-20 items): {len(large_o):>5} "
      f"({len(large_o)/NUM_ORDERS*100:.1f}%)")
print(f"\nSize stats: mean={np.mean(sizes):.1f} | "
      f"min={min(sizes)} | max={max(sizes)}")

# ── Save ───────────────────────────────────────────────────────────────────
output = {
    "description": (
        "5000 synthetic pick orders for F4 RL training. "
        "Categories: Small (1-5 items), Medium (6-15 items), "
        "Large (16-20 items, capped until real WMS data arrives). "
        "Item frequencies follow ABC demand distribution."
    ),
    "update_instructions": (
        "When real partner WMS data arrives: "
        "1. Replace item_locations.json with the real stock report. "
        "2. Set LARGE_MAX = 30 in this script. "
        "3. Re-run to regenerate orders with real SKUs and true large orders."
    ),
    "categories": {
        "Small":  {"range": "1-5 items",   "count": len(small_o)},
        "Medium": {"range": "6-15 items",  "count": len(medium_o)},
        "Large":  {"range": "16-30 items",
                   "note": "currently capped at 20 items",
                   "count": len(large_o)}
    },
    "statistics": {
        "total_orders":    NUM_ORDERS,
        "mean_order_size": round(float(np.mean(sizes)), 2),
        "min_order_size":  int(min(sizes)),
        "max_order_size":  int(max(sizes))
    },
    "orders": orders
}

out_path = os.path.join(script_dir, "orders.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {NUM_ORDERS} orders to orders.json")