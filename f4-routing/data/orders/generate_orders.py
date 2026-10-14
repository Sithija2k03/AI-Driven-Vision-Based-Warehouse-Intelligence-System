"""
f4-routing/data/orders/generate_orders.py

Generates warehouse-specific pick order files.
Each file uses the SKUs and locations from the matching
item_locations_{size}.json file.

Order size categories (same for all warehouse sizes):
  Small  :  1 -  5 items  (~40% of orders)
  Medium :  6 - 15 items  (~45% of orders)
  Large  : 16 - 30 items  (~15% of orders)

With 90+ SKUs available per warehouse, large orders now
genuinely contain 16-30 distinct items spread across the
full warehouse layout.

Output:
  orders_small.json
  orders_medium.json
  orders_large.json
"""

import json
import os
import random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_ORDERS = 5000

# Order size categories
SMALL_MIN,  SMALL_MAX  = 1,  5
MEDIUM_MIN, MEDIUM_MAX = 6,  15
LARGE_MIN,  LARGE_MAX  = 16, 30

SMALL_RATIO  = 0.40
MEDIUM_RATIO = 0.45
LARGE_RATIO  = 0.15

CAT_ENC = {"Small": 0, "Medium": 1, "Large": 2}


def abc_weight(index: int, total: int) -> float:
    """
    ABC demand distribution:
      A-class (top 20%): weight 3.0 — fast movers, ordered frequently
      B-class (mid 30%): weight 1.5 — medium movers
      C-class (bot 50%): weight 0.5 — slow movers, rarely ordered
    """
    p = index / total
    if p < 0.20:   return 3.0
    elif p < 0.50: return 1.5
    else:          return 0.5


def get_size_and_category(num_items: int) -> tuple:
    """Returns (order_size, category) based on target distribution."""
    roll = random.random()
    if roll < SMALL_RATIO:
        size = random.randint(SMALL_MIN, SMALL_MAX)
        return size, "Small"
    elif roll < SMALL_RATIO + MEDIUM_RATIO:
        size = random.randint(MEDIUM_MIN, min(MEDIUM_MAX, num_items))
        return size, "Medium"
    else:
        # Large: 16-30 items, capped at available SKUs
        size = random.randint(LARGE_MIN, min(LARGE_MAX, num_items))
        return size, "Large"


def generate_for_warehouse(wh_name: str, items_file: str,
                            out_file: str):
    """Generates NUM_ORDERS pick orders for one warehouse."""

    items_path = os.path.join(SCRIPT_DIR, items_file)
    if not os.path.exists(items_path):
        print(f"  ERROR: {items_path} not found.")
        print(f"  Run generate_item_locations.py first.")
        return

    with open(items_path) as f:
        item_data = json.load(f)

    items     = item_data["items"]
    num_items = len(items)

    # ABC demand probabilities
    weights       = np.array(
        [abc_weight(i, num_items) for i in range(num_items)],
        dtype=float
    )
    probabilities = weights / weights.sum()

    print(f"Generating {NUM_ORDERS} orders for {wh_name}...")
    print(f"  Available SKUs : {num_items}")
    print(f"  Order range    : Small(1-5) Medium(6-15) Large(16-30)")

    orders = []
    for i in range(1, NUM_ORDERS + 1):
        size, category = get_size_and_category(num_items)
        size           = min(size, num_items)

        selected = np.random.choice(
            num_items, size=size, replace=False, p=probabilities
        )

        order_items = [
            {
                "item_code":      items[j]["item_code"],
                "shelf_location": items[j]["shelf_location"],
                "task_weight":    items[j]["task_weight"]
            }
            for j in selected
        ]

        orders.append({
            "order_id":   f"ORD-{str(i).zfill(5)}",
            "category":   category,
            "order_size": size,
            "items":      order_items
        })

    # Statistics
    small_n  = sum(1 for o in orders if o["category"] == "Small")
    medium_n = sum(1 for o in orders if o["category"] == "Medium")
    large_n  = sum(1 for o in orders if o["category"] == "Large")
    sizes    = [o["order_size"] for o in orders]

    output = {
        "description": (
            f"Pick orders for {wh_name}. {NUM_ORDERS} orders "
            f"across Small/Medium/Large categories. "
            f"Uses {num_items} SKUs from item_locations_{wh_name.split('_')[1]}.json."
        ),
        "warehouse":   wh_name,
        "num_skus":    num_items,
        "categories": {
            "Small":  {"range": "1-5 items",   "count": small_n},
            "Medium": {"range": "6-15 items",  "count": medium_n},
            "Large":  {"range": "16-30 items", "count": large_n}
        },
        "statistics": {
            "total_orders":    NUM_ORDERS,
            "mean_order_size": round(float(np.mean(sizes)), 2),
            "min_order_size":  int(min(sizes)),
            "max_order_size":  int(max(sizes))
        },
        "orders": orders
    }

    out_path = os.path.join(SCRIPT_DIR, out_file)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Small={small_n}  Medium={medium_n}  Large={large_n}")
    print(f"  Size stats: mean={np.mean(sizes):.1f}  "
          f"min={min(sizes)}  max={max(sizes)}")
    print(f"  Saved: {out_file}")
    print()


# Generate for all three warehouse sizes
CONFIGS = [
    ("warehouse_small",  "item_locations_small.json",  "orders_small.json"),
    ("warehouse_medium", "item_locations_medium.json", "orders_medium.json"),
    ("warehouse_large",  "item_locations_large.json",  "orders_large.json"),
]

for wh_name, items_file, orders_file in CONFIGS:
    generate_for_warehouse(wh_name, items_file, orders_file)

print("All order files generated.")