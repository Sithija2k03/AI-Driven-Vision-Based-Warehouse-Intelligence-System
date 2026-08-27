"""
Tests the full pipeline:
  orders.json → WarehouseGraph → WarehouseEnv → random agent steps
"""

import sys
import os

# Build absolute paths relative to THIS file's location
# __file__ = f4-routing/tests/test_env.py
# So f4-routing root = one level up from tests/
F4_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRAPH_PATH = os.path.join(F4_ROOT, "data", "graphs", "warehouse_small.json")
ORDERS_PATH = os.path.join(F4_ROOT, "data", "orders", "orders.json")

# Add paths so imports resolve correctly
sys.path.insert(0, os.path.join(F4_ROOT, "gym-environment"))
sys.path.insert(0, os.path.join(F4_ROOT, "src"))

from warehouse_env import WarehouseEnv

# Verify files exist before attempting to load
for label, path in [("Graph", GRAPH_PATH), ("Orders", ORDERS_PATH)]:
    if not os.path.exists(path):
        print(f"ERROR: {label} file not found: {path}")
        print("Run these first:")
        print("  python data/graphs/generate_benchmarks.py")
        print("  python data/orders/generate_orders.py")
        sys.exit(1)
    else:
        print(f"Found {label}: {path}")

print()
env = WarehouseEnv(
    floor_plan_path=GRAPH_PATH,
    orders_path=ORDERS_PATH,
    max_episode_steps=50
)

print("\n--- Running 3 test episodes with random agent ---\n")

for ep in range(3):
    obs, _ = env.reset()
    done         = False
    total_reward = 0.0

    print(f"Episode {ep+1} | "
          f"Order: {env.current_order['order_id']} | "
          f"Category: {env.current_order['category']} | "
          f"Size: {env.current_order['order_size']} items")

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    print(f"  Picked       : {info['items_picked']}/{info['items_total']}")
    print(f"  Completion   : {info['completion_rate']*100:.1f}%")
    print(f"  Picks/hour   : {info['picks_per_hour']:.1f}")
    print(f"  Total reward : {total_reward:.2f}")
    print(f"  Steps taken  : {info['total_steps']}")
    print()

print("Environment test passed.")