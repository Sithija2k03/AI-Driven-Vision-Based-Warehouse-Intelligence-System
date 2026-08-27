"""
f4-routing/data/graphs/generate_benchmarks.py

Generates warehouse floor graph JSON files based on
Roodbergen & De Koster (2001) benchmark layout parameters.

Run once to produce:
  warehouse_small.json  — 2 blocks, 5 aisles, 10 locations per segment
  warehouse_medium.json — 3 blocks, 8 aisles, 15 locations per segment
  warehouse_large.json  — 3 blocks, 12 aisles, 20 locations per segment

Reference:
  Roodbergen, K.J. and De Koster, R. (2001a).
  Routing methods for warehouses with multiple cross aisles.
  International Journal of Production Research 39(9), 1865-1883.
"""

import json
import math
import os

AISLE_WIDTH       = 3.0   # metres between aisle centres
LOCATION_DEPTH    = 1.0   # metres per shelf slot along the aisle
CROSS_AISLE_WIDTH = 3.0   # metres for horizontal cross aisles


def generate_warehouse(config_name, num_blocks, num_aisles, locs_per_segment):

    nodes = []
    edges = []
    seg_len = locs_per_segment * LOCATION_DEPTH

    # ── Pick location nodes ────────────────────────────────────────────────
    for aisle in range(1, num_aisles + 1):
        for block in range(1, num_blocks + 1):
            for loc in range(1, locs_per_segment + 1):

                node_id = f"A{aisle}-B{block}-L{loc}"
                x = (aisle - 1) * AISLE_WIDTH
                y = ((block - 1) * (seg_len + CROSS_AISLE_WIDTH)
                     + (loc - 1) * LOCATION_DEPTH)

                # Hazard: outer aisles have less turning room
                hazard = "medium" if (aisle == 1 or aisle == num_aisles) \
                         else "low"

                # Task weight: heavy items near depot (block 1, early locs)
                if block == 1 and loc <= locs_per_segment // 3:
                    task_weight = "Heavy"
                elif block == num_blocks and loc > 2 * locs_per_segment // 3:
                    task_weight = "Light"
                else:
                    task_weight = "Moderate"

                nodes.append({
                    "id":           node_id,
                    "aisle":        aisle,
                    "block":        block,
                    "location":     loc,
                    "x":            round(x, 2),
                    "y":            round(y, 2),
                    "task_weight":  task_weight,
                    "hazard_level": hazard
                })

    # ── Cross aisle waypoint nodes ─────────────────────────────────────────
    for aisle in range(1, num_aisles + 1):
        x = (aisle - 1) * AISLE_WIDTH

        # Front (before block 1)
        nodes.append({
            "id": f"CA-FRONT-A{aisle}", "aisle": aisle, "block": 0,
            "x": round(x, 2), "y": round(-CROSS_AISLE_WIDTH, 2),
            "task_weight": "Light", "hazard_level": "low"
        })

        # Between blocks
        for block in range(1, num_blocks):
            y_ca = block * (seg_len + CROSS_AISLE_WIDTH) - CROSS_AISLE_WIDTH
            nodes.append({
                "id": f"CA-{block}-A{aisle}", "aisle": aisle, "block": block,
                "x": round(x, 2), "y": round(y_ca, 2),
                "task_weight": "Light", "hazard_level": "low"
            })

        # Rear (after last block)
        y_rear = num_blocks * (seg_len + CROSS_AISLE_WIDTH)
        nodes.append({
            "id": f"CA-REAR-A{aisle}", "aisle": aisle, "block": num_blocks + 1,
            "x": round(x, 2), "y": round(y_rear, 2),
            "task_weight": "Light", "hazard_level": "low"
        })

    # Depot
    nodes.append({
        "id": "DEPOT", "aisle": 0, "block": 0,
        "x": 0.0, "y": round(-CROSS_AISLE_WIDTH - 2.0, 2),
        "task_weight": "Light", "hazard_level": "low"
    })

    # ── Build lookup for distance calculation ──────────────────────────────
    lookup = {n["id"]: n for n in nodes}

    def dist(a, b):
        n1, n2 = lookup[a], lookup[b]
        return round(math.sqrt((n1["x"]-n2["x"])**2
                               + (n1["y"]-n2["y"])**2), 2)

    def edge(f, t, hazard="low"):
        if f in lookup and t in lookup:
            edges.append({
                "from": f, "to": t,
                "distance": dist(f, t),
                "hazard_level": hazard
            })

    # Depot ↔ front aisle
    for a in range(1, num_aisles + 1):
        edge("DEPOT", f"CA-FRONT-A{a}")
        edge(f"CA-FRONT-A{a}", "DEPOT")

    # Within-aisle edges
    for a in range(1, num_aisles + 1):
        for b in range(1, num_blocks + 1):
            front_ca = f"CA-FRONT-A{a}" if b == 1 else f"CA-{b-1}-A{a}"
            rear_ca  = f"CA-REAR-A{a}"  if b == num_blocks else f"CA-{b}-A{a}"

            first = f"A{a}-B{b}-L1"
            last  = f"A{a}-B{b}-L{locs_per_segment}"

            edge(front_ca, first)
            edge(first, front_ca)
            edge(last, rear_ca)
            edge(rear_ca, last)

            for loc in range(1, locs_per_segment):
                edge(f"A{a}-B{b}-L{loc}", f"A{a}-B{b}-L{loc+1}")
                edge(f"A{a}-B{b}-L{loc+1}", f"A{a}-B{b}-L{loc}")

    # Front cross aisle horizontal connections
    for a in range(1, num_aisles):
        edge(f"CA-FRONT-A{a}", f"CA-FRONT-A{a+1}")
        edge(f"CA-FRONT-A{a+1}", f"CA-FRONT-A{a}")

    # Rear cross aisle horizontal connections
    for a in range(1, num_aisles):
        edge(f"CA-REAR-A{a}", f"CA-REAR-A{a+1}")
        edge(f"CA-REAR-A{a+1}", f"CA-REAR-A{a}")

    # Middle cross aisles horizontal connections
    for b in range(1, num_blocks):
        for a in range(1, num_aisles):
            edge(f"CA-{b}-A{a}", f"CA-{b}-A{a+1}", "medium")
            edge(f"CA-{b}-A{a+1}", f"CA-{b}-A{a}", "medium")

    total_locs = num_aisles * num_blocks * locs_per_segment

    return {
        "warehouse_id": config_name,
        "description": (
            f"Roodbergen & De Koster (2001) benchmark — "
            f"{num_blocks} blocks, {num_aisles} aisles, "
            f"{locs_per_segment} locations per segment"
        ),
        "reference": (
            "Roodbergen, K.J. and De Koster, R. (2001a). "
            "Routing methods for warehouses with multiple cross aisles. "
            "International Journal of Production Research 39(9), 1865-1883."
        ),
        "dimensions": {
            "num_blocks":                  num_blocks,
            "num_aisles":                  num_aisles,
            "locations_per_segment":       locs_per_segment,
            "total_pick_locations":        total_locs,
            "aisle_width_m":               AISLE_WIDTH,
            "location_depth_m":            LOCATION_DEPTH,
            "cross_aisle_width_m":         CROSS_AISLE_WIDTH,
            "total_warehouse_width_m":     round((num_aisles-1)*AISLE_WIDTH, 2),
            "total_warehouse_depth_m":     round(
                num_blocks * locs_per_segment * LOCATION_DEPTH
                + (num_blocks + 1) * CROSS_AISLE_WIDTH, 2)
        },
        "nodes": nodes,
        "edges": edges
    }


# ── Generate all three configurations ─────────────────────────────────────
CONFIGS = [
    {"config_name": "warehouse_small",  "num_blocks": 2,
     "num_aisles": 5,  "locs_per_segment": 10},
    {"config_name": "warehouse_medium", "num_blocks": 3,
     "num_aisles": 8,  "locs_per_segment": 15},
    {"config_name": "warehouse_large",  "num_blocks": 3,
     "num_aisles": 12, "locs_per_segment": 20},
]

out_dir = os.path.dirname(os.path.abspath(__file__))

for cfg in CONFIGS:
    print(f"Generating {cfg['config_name']}...")
    graph = generate_warehouse(
        cfg["config_name"], cfg["num_blocks"],
        cfg["num_aisles"], cfg["locs_per_segment"]
    )
    path = os.path.join(out_dir, f"{cfg['config_name']}.json")
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)

    d = graph["dimensions"]
    print(f"  Pick locations : {d['total_pick_locations']}")
    print(f"  Nodes          : {len(graph['nodes'])}")
    print(f"  Edges          : {len(graph['edges'])}")
    print(f"  Dimensions     : {d['total_warehouse_width_m']}m wide x "
          f"{d['total_warehouse_depth_m']}m deep")
    print()

print("Done. Three warehouse graphs created.")