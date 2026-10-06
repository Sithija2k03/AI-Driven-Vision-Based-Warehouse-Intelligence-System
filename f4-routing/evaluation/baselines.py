"""
f4-routing/evaluation/baselines.py

Classical routing baseline policies.
Updated for order-aware pick lists (v3).

Each item in the pick list is now a dict with:
  shelf_location, item_code, task_weight, order_id

Baseline route functions receive a list of shelf_location strings.
run_baseline_episode handles the order-aware structure.
"""

import sys
import os
import random
import numpy as np
from collections import defaultdict

_ENV_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "gym-environment")
)
_SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, _ENV_DIR)
sys.path.insert(0, _SRC_DIR)

from warehouse_env import WarehouseEnv


def parse_node_id(node_id: str) -> tuple:
    if not node_id.startswith("A"):
        return (0, 0, 0)
    try:
        parts = node_id.split("-")
        return (int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]))
    except (IndexError, ValueError):
        return (0, 0, 0)


def group_by_aisle(location_list: list) -> dict:
    groups = defaultdict(list)
    for loc in location_list:
        aisle, _, _ = parse_node_id(loc)
        groups[aisle].append(loc)
    return groups


def sshape_route(location_list: list) -> list:
    """
    S-shape heuristic on a flat list of shelf locations.
    Traverses aisles in alternating directions.
    """
    if not location_list:
        return []
    by_aisle = group_by_aisle(location_list)
    route    = []
    for aisle_num in sorted(by_aisle.keys()):
        nodes = sorted(by_aisle[aisle_num],
                       key=lambda n: parse_node_id(n)[1:])
        if aisle_num % 2 == 1:
            route.extend(nodes)
        else:
            route.extend(reversed(nodes))
    return route


def largest_gap_route(location_list: list) -> list:
    """
    Largest Gap heuristic on a flat list of shelf locations.
    Enters each aisle from the side with the largest internal gap.
    """
    if not location_list:
        return []
    by_aisle  = group_by_aisle(location_list)
    route     = []
    from_rear = False

    for aisle_num in sorted(by_aisle.keys()):
        nodes        = by_aisle[aisle_num]
        nodes_sorted = sorted(nodes, key=lambda n: parse_node_id(n)[2])

        if len(nodes_sorted) <= 1:
            route.extend(nodes_sorted)
            continue

        locs    = [parse_node_id(n)[2] for n in nodes_sorted]
        gaps    = [locs[i+1] - locs[i] for i in range(len(locs)-1)]
        gap_idx = gaps.index(max(gaps))

        before = nodes_sorted[:gap_idx + 1]
        after  = nodes_sorted[gap_idx + 1:]

        if from_rear:
            route.extend(reversed(after))
            route.extend(reversed(before))
        else:
            route.extend(before)
            route.extend(reversed(after))

        from_rear = not from_rear
    return route


def random_route(location_list: list) -> list:
    route = list(location_list)
    random.shuffle(route)
    return route


def run_baseline_episode(env: WarehouseEnv, route_fn) -> dict:
    """
    Runs one complete trip episode using a classical routing heuristic.
    Handles the order-aware pick list structure.

    The route function receives a list of shelf_location strings.
    The env handles item-to-order mapping internally.
    """
    obs, _ = env.reset()
    info   = {}

    # Get flat list of shelf locations from the order-aware pick list
    all_locations = [item["shelf_location"]
                     for item in env.remaining_picks]

    # Compute heuristic route over locations
    planned_locations = route_fn(list(all_locations))

    total_reward = 0.0
    done         = False

    for target_location in planned_locations:
        if done:
            break

        # Get currently valid locations
        valid_locs = env.wh_graph.get_valid_nodes(
            env.picker_id,
            [r["shelf_location"] for r in env.remaining_picks]
        )
        if not valid_locs:
            break
        if target_location not in valid_locs:
            # Stocked out or task-incompatible — skip this item
            continue

        action_idx = valid_locs.index(target_location)
        obs, reward, done, _, info = env.step(action_idx)
        total_reward += reward

    # Always get final summary
    if not info:
        info = env._episode_summary()

    info["total_reward"] = round(total_reward, 3)
    return info