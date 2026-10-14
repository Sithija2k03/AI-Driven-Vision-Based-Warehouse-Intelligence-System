"""
f4-routing/mock-publishers/mock_f1_publisher.py

Simulates F1 inventory verification output on the message bus.
Used for F4 development while the real F1 function is being built.

Publishes TWO message types to channel: f1.inventory

  1. inventory_snapshot  — published ONCE at session start
                           tells F4 where every item currently is
                           this is how the RL agent knows item locations

  2. inventory_update    — published every 10 seconds during picking
                           only meaningful when exception_flag = True
                           triggers mid-pick replanning in F4

Real F1 will publish the same two message types from actual
CCTV-verified shelf readings. This mock generates them synthetically.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import random
from datetime import datetime
from shared.message_bus import publish, CHANNEL_F1_INVENTORY

# ── Warehouse inventory — SKU to shelf location mapping ───────────────────
# In production this comes from the partner WMS stock report.
# Until then we use synthetic items matched to our 5-aisle floor plan.
INVENTORY = [
    {"item_code": "SKU-1001", "shelf_location": "A1-B1-S1", "task_weight": "Heavy"},
    {"item_code": "SKU-1002", "shelf_location": "A1-B1-S2", "task_weight": "Heavy"},
    {"item_code": "SKU-1003", "shelf_location": "A1-B2-S1", "task_weight": "Moderate"},
    {"item_code": "SKU-1004", "shelf_location": "A1-B2-S2", "task_weight": "Moderate"},
    {"item_code": "SKU-1005", "shelf_location": "A2-B1-S1", "task_weight": "Heavy"},
    {"item_code": "SKU-1006", "shelf_location": "A2-B1-S2", "task_weight": "Light"},
    {"item_code": "SKU-1007", "shelf_location": "A2-B2-S1", "task_weight": "Moderate"},
    {"item_code": "SKU-1008", "shelf_location": "A2-B2-S2", "task_weight": "Light"},
    {"item_code": "SKU-1009", "shelf_location": "A3-B1-S1", "task_weight": "Heavy"},
    {"item_code": "SKU-1010", "shelf_location": "A3-B1-S2", "task_weight": "Moderate"},
    {"item_code": "SKU-1011", "shelf_location": "A3-B2-S1", "task_weight": "Light"},
    {"item_code": "SKU-1012", "shelf_location": "A3-B2-S2", "task_weight": "Light"},
    {"item_code": "SKU-1013", "shelf_location": "A4-B1-S1", "task_weight": "Heavy"},
    {"item_code": "SKU-1014", "shelf_location": "A4-B1-S2", "task_weight": "Moderate"},
    {"item_code": "SKU-1015", "shelf_location": "A4-B2-S1", "task_weight": "Moderate"},
    {"item_code": "SKU-1016", "shelf_location": "A4-B2-S2", "task_weight": "Light"},
    {"item_code": "SKU-1017", "shelf_location": "A5-B1-S1", "task_weight": "Heavy"},
    {"item_code": "SKU-1018", "shelf_location": "A5-B1-S2", "task_weight": "Moderate"},
    {"item_code": "SKU-1019", "shelf_location": "A5-B2-S1", "task_weight": "Light"},
    {"item_code": "SKU-1020", "shelf_location": "A5-B2-S2", "task_weight": "Light"},
]

# ── Configuration ──────────────────────────────────────────────────────────
UPDATE_INTERVAL_SECONDS = 10

# Probability that any shelf fires a stockout exception per update cycle.
# Real warehouses typically have 2-5% misplacement rate.
# We set higher here so stockout replanning triggers frequently during testing.
STOCKOUT_PROBABILITY = 0.12

# Track which shelves are currently stocked out so we can also clear them
active_stockouts: set = set()


# ── Message builders ───────────────────────────────────────────────────────

def build_snapshot_message(item: dict) -> dict:
    """
    inventory_snapshot: published once at session start.
    Tells F4 the confirmed location of every item.
    F4 uses these to build the initial pick graph before any order arrives.
    compliance_score is high — items are confirmed present at session start.
    """
    return {
        "message_type":    "inventory_snapshot",
        "item_code":       item["item_code"],
        "shelf_location":  item["shelf_location"],
        "task_weight":     item["task_weight"],
        "compliance_score": round(random.uniform(85.0, 99.0), 2),
        "exception_flag":  False,
        "confidence":      round(random.uniform(0.88, 0.99), 3),
        "timestamp":       datetime.now().isoformat()
    }


def build_update_message(item: dict) -> dict:
    """
    inventory_update: published every UPDATE_INTERVAL_SECONDS.
    exception_flag = True means item is missing or below confidence —
    F4 must mask this node and replan the affected picker's route.
    exception_flag = False means shelf is confirmed stocked — clear any mask.
    """
    # Decide if this shelf fires an exception this cycle
    is_exception = item["shelf_location"] in active_stockouts

    # With STOCKOUT_PROBABILITY chance, toggle a new stockout
    if not is_exception and random.random() < STOCKOUT_PROBABILITY:
        is_exception = True
        active_stockouts.add(item["shelf_location"])
    elif is_exception and random.random() < 0.3:
        # 30% chance a stockout clears each cycle (item replenished)
        is_exception = False
        active_stockouts.discard(item["shelf_location"])

    compliance = round(random.uniform(5.0, 35.0), 2)  if is_exception \
                 else round(random.uniform(70.0, 99.0), 2)
    confidence = round(random.uniform(0.45, 0.70), 3) if is_exception \
                 else round(random.uniform(0.82, 0.99), 3)

    return {
        "message_type":    "inventory_update",
        "item_code":       item["item_code"],
        "shelf_location":  item["shelf_location"],
        "task_weight":     item["task_weight"],
        "compliance_score": compliance,
        "exception_flag":  is_exception,
        "confidence":      confidence,
        "timestamp":       datetime.now().isoformat()
    }


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("Mock F1 Publisher")
    print(f"Channel         : {CHANNEL_F1_INVENTORY}")
    print(f"Items in stock  : {len(INVENTORY)}")
    print(f"Update interval : {UPDATE_INTERVAL_SECONDS}s")
    print(f"Stockout rate   : {int(STOCKOUT_PROBABILITY * 100)}% per cycle")
    print("=" * 65)

    # ── Phase 1: Publish full inventory snapshot at session start ──────────
    print("\n[F1] Publishing inventory snapshot...")
    for item in INVENTORY:
        msg = build_snapshot_message(item)
        publish(CHANNEL_F1_INVENTORY, msg)
        print(f"  SNAPSHOT | {msg['shelf_location']} | "
              f"{msg['item_code']} | {msg['task_weight']} | "
              f"compliance={msg['compliance_score']}%")

    print(f"\n[F1] Snapshot complete — {len(INVENTORY)} items confirmed.")
    print("[F1] Starting live update cycle...\n")

    # ── Phase 2: Publish continuous updates ────────────────────────────────
    while True:
        exceptions_this_cycle = 0

        for item in INVENTORY:
            msg = build_update_message(item)
            publish(CHANNEL_F1_INVENTORY, msg)

            if msg["exception_flag"]:
                exceptions_this_cycle += 1
                print(f"[F1] {datetime.now().strftime('%H:%M:%S')} | "
                      f"STOCKOUT  | {msg['shelf_location']} | "
                      f"{msg['item_code']} | "
                      f"compliance={msg['compliance_score']}% | "
                      f"confidence={msg['confidence']}")
            else:
                print(f"[F1] {datetime.now().strftime('%H:%M:%S')} | "
                      f"OK        | {msg['shelf_location']} | "
                      f"{msg['item_code']} | "
                      f"compliance={msg['compliance_score']}%")

        print(f"\n--- F1 cycle complete | "
              f"exceptions={exceptions_this_cycle} | "
              f"active_stockouts={len(active_stockouts)} | "
              f"sleeping {UPDATE_INTERVAL_SECONDS}s ---\n")

        time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()