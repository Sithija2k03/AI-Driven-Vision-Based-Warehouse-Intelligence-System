"""
f4-routing/data/orders/generate_item_locations.py

Generates warehouse-specific item location files.
90% of all pick nodes in each warehouse are assigned a unique SKU.
This ensures the agent trains on a realistically stocked warehouse
where almost every shelf has something to pick.

Output:
  item_locations_small.json   — 90 SKUs across 100 nodes (small)
  item_locations_medium.json  — 324 SKUs across 360 nodes (medium)
  item_locations_large.json   — 648 SKUs across 720 nodes (large)

SKU format: SKU-XXXX (e.g. SKU-0001 through SKU-0648)
Node format: A{aisle}-B{block}-L{location}

Task weight per node follows generate_benchmarks.py rules:
  block=1, loc <= locs//3         -> Heavy
  block=last, loc > 2*locs//3     -> Light
  everything else                  -> Moderate
"""

import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COVERAGE = 0.90   # 90% of nodes get a SKU

WAREHOUSES = [
    {
        "name":   "small",
        "aisles": 5,
        "blocks": 2,
        "locs":   10,
    },
    {
        "name":   "medium",
        "aisles": 8,
        "blocks": 3,
        "locs":   15,
    },
    {
        "name":   "large",
        "aisles": 12,
        "blocks": 3,
        "locs":   20,
    },
]


def get_task_weight(block: int, loc: int,
                    num_blocks: int, locs_per_seg: int) -> str:
    """
    Assigns task weight using the same rule as generate_benchmarks.py
    so item task weights match the graph node task weights exactly.
    """
    if block == 1 and loc <= locs_per_seg // 3:
        return "Heavy"
    elif block == num_blocks and loc > 2 * locs_per_seg // 3:
        return "Light"
    else:
        return "Moderate"


def generate_item_locations(name, num_aisles, num_blocks,
                             locs_per_seg, coverage):
    """
    Builds the full list of all pick nodes, then randomly selects
    coverage% of them to assign SKUs.
    """

    # Build every possible pick node in this warehouse
    all_nodes = []
    for aisle in range(1, num_aisles + 1):
        for block in range(1, num_blocks + 1):
            for loc in range(1, locs_per_seg + 1):
                all_nodes.append({
                    "node_id":     f"A{aisle}-B{block}-L{loc}",
                    "aisle":       aisle,
                    "block":       block,
                    "loc":         loc,
                    "task_weight": get_task_weight(
                        block, loc, num_blocks, locs_per_seg
                    )
                })

    total_nodes  = len(all_nodes)
    target_count = int(total_nodes * coverage)

    # Randomly select target_count nodes without replacement
    random.seed(42)   # fixed seed for reproducibility
    selected = random.sample(all_nodes, target_count)

    # Sort by aisle -> block -> loc for readable output
    selected.sort(key=lambda n: (n["aisle"], n["block"], n["loc"]))

    # Assign sequential SKU codes
    items = []
    for i, node in enumerate(selected, start=1):
        items.append({
            "item_code":      f"SKU-{str(i).zfill(4)}",
            "shelf_location": node["node_id"],
            "task_weight":    node["task_weight"]
        })

    return items, total_nodes


for wh in WAREHOUSES:
    items, total = generate_item_locations(
        wh["name"], wh["aisles"], wh["blocks"],
        wh["locs"], COVERAGE
    )

    # Coverage report
    task_counts = {"Heavy": 0, "Moderate": 0, "Light": 0}
    aisle_counts = {}
    for item in items:
        task_counts[item["task_weight"]] += 1
        aisle = int(item["shelf_location"].split("-")[0][1:])
        aisle_counts[aisle] = aisle_counts.get(aisle, 0) + 1

    coverage_pct = len(items) / total * 100

    output = {
        "description": (
            f"Item locations for warehouse_{wh['name']}. "
            f"{len(items)} unique SKUs placed across {coverage_pct:.0f}% "
            f"of {total} pick nodes. "
            f"Replace with real partner WMS stock report when available."
        ),
        "warehouse":         f"warehouse_{wh['name']}",
        "total_nodes":       total,
        "items_placed":      len(items),
        "coverage_percent":  round(coverage_pct, 1),
        "task_distribution": task_counts,
        "items":             items
    }

    out_path = os.path.join(
        SCRIPT_DIR, f"item_locations_{wh['name']}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"warehouse_{wh['name']}:")
    print(f"  Total nodes    : {total}")
    print(f"  Items placed   : {len(items)} ({coverage_pct:.0f}% coverage)")
    print(f"  Task weights   : Heavy={task_counts['Heavy']}  "
          f"Moderate={task_counts['Moderate']}  "
          f"Light={task_counts['Light']}")
    print(f"  Aisles covered : "
          f"{len(aisle_counts)}/{wh['aisles']} "
          f"(items in every aisle: "
          f"{'YES' if len(aisle_counts)==wh['aisles'] else 'NO'})")
    print(f"  Saved          : item_locations_{wh['name']}.json")
    print()

print("Done. Run generate_orders.py next.")