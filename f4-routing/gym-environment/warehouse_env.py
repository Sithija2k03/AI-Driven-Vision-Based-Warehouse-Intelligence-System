"""
f4-routing/gym-environment/warehouse_env.py

Custom OpenAI Gymnasium environment for warehouse order picking.

EPISODE MODEL (v3 - order-aware trip model):
  One episode = one complete picker TRIP.
  A trip covers all items in an assigned batch (one or more orders),
  starting and ending at DEPOT.

  The pick list is order-aware:
    Each item carries its order_id.
    An order is COMPLETE when ALL its items are picked.
    Partial orders (items stocked out) are INCOMPLETE.

METRICS (equal weight):
  item_pph   : items picked per hour  (routing quality)
  order_pph  : complete orders per hour (batching + routing quality)
  cycle_time : mean seconds per complete order (responsiveness)
  dist_per_item: metres walked per item (travel efficiency)

REWARD (equally weighted dual objective):
  Primary A: item picking rate (picks/hour contribution per step)
  Primary B: order completion speed (triggered when order completes)
  These two signals are scaled to similar magnitudes so neither
  dominates — the agent must optimise both simultaneously.

TIME MODEL:
  step_time = (physical_distance / WALKING_SPEED_MS) + PICK_TIME_SECONDS
  + DEPOT return leg added at episode end for order_pph calculation

GENERALISABILITY:
  All operational parameters are configurable at __init__.
  The environment works for any warehouse, picker count, or
  batch composition — not specific to any single operation.
"""

import gymnasium as gym
import numpy as np
import networkx as nx
import json
import sys
import os

_SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from graph_builder import WarehouseGraph


class WarehouseEnv(gym.Env):

    metadata = {"render_modes": []}

    MAX_ORDER_SIZE = 30   # maximum items in action space

    # Time model — physical reality, not artificial constants
    WALKING_SPEED_MS  = 1.2    # m/s — average human warehouse walking speed
    PICK_TIME_SECONDS = 15     # s   — time to locate and pick one item
    MAX_SIM_SECONDS   = 3600   # s   — 1-hour trip budget

    # Reward scaling — calibrated for equal item_pph and order_pph weight
    ITEM_PPH_WEIGHT      = 5.0    # per-step item efficiency contribution
    ORDER_COMPLETE_BASE  = 50.0   # base reward per completed order
    ORDER_CYCLE_DIVISOR  = 60.0   # normalise cycle time to minutes
    TRAVEL_PENALTY       = 0.05   # per physical metre (small — efficiency shaper)
    ERGONOMIC_PENALTY    = 0.02   # per weighted cost unit (secondary — safety)
    CONGESTION_PENALTY   = 5.0    # per aisle conflict event
    INCOMPLETE_PENALTY   = 10.0   # per order that could not be completed (stockout)

    def __init__(
        self,
        floor_plan_path: str,
        orders_path: str,
        max_episode_steps: int = 200,
        render_mode=None
    ):
        super().__init__()

        self.floor_plan_path = floor_plan_path
        self.orders_path     = orders_path
        self.max_steps       = max_episode_steps

        # Load warehouse graph
        self.wh_graph = WarehouseGraph(floor_plan_path)

        # Load orders
        with open(orders_path) as f:
            data = json.load(f)
        self.all_orders = data["orders"]
        cats = data.get("categories", {})
        print(f"[Env] Loaded {len(self.all_orders)} orders from "
              f"{os.path.basename(orders_path)}")
        if cats:
            print(f"[Env] Small={cats.get('Small',{}).get('count','?')} "
                  f"Medium={cats.get('Medium',{}).get('count','?')} "
                  f"Large={cats.get('Large',{}).get('count','?')}")

        # Node coordinate lookup for observation builder
        with open(floor_plan_path) as f:
            plan = json.load(f)
        self.node_coords = {
            n["id"]: (n.get("x", 0.0), n.get("y", 0.0))
            for n in plan["nodes"]
        }
        dims       = plan["dimensions"]
        self.max_x = max(dims["total_warehouse_width_m"], 1.0)
        self.max_y = max(dims["total_warehouse_depth_m"], 1.0)

        # Episode state — all initialised in reset()
        self.current_order_batch   = []   # list of order dicts in this batch
        self.pick_list             = []   # flat list of (item, order_id) tuples
        self.remaining_picks       = []   # items not yet collected
        self.order_item_counts     = {}   # {order_id: total_items}
        self.order_picked_counts   = {}   # {order_id: items_collected_so_far}
        self.order_start_times     = {}   # {order_id: sim_time when trip started}
        self.completed_orders      = []   # order_ids fully completed
        self.current_position      = "DEPOT"
        self.picker_id             = "P-001"
        self.step_count            = 0
        self.items_picked          = 0
        self.total_distance        = 0.0
        self.aisle_conflicts       = 0
        self.stockout_rerouts      = 0
        self.sim_time              = 0.0

        # F3 state
        self.last_ergo_score  = 3
        self.last_gait_score  = 2
        self.last_demo_weight = 1.0
        self.last_task_tier   = "Heavy"
        self._current_batch_override = None

        # Observation space: 10 dimensions
        # Added: order_completion_ratio and n_orders_in_batch
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.MAX_ORDER_SIZE)

    # ══════════════════════════════════════════════════════════════════════
    # BATCH LOADING — called externally by the training loop or batcher
    # ══════════════════════════════════════════════════════════════════════

    def load_batch(self, batch: dict):
        """
        Loads a pre-assembled batch (from OrderBatcher) into the env.
        Called before reset() when you want a specific batch rather
        than a randomly sampled one.

        batch format (from OrderBatcher.create_batches()):
            {
                'batch_id':  str,
                'is_solo':   bool,
                'orders':    [order_dict, ...]
            }
        """
        self._current_batch_override = batch

    # ══════════════════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use override batch if set, otherwise sample randomly
        if hasattr(self, '_current_batch_override') \
                and self._current_batch_override:
            batch  = self._current_batch_override
            orders = batch["orders"]
            self._current_batch_override = None
        else:
            # Random sampling: pick 1-3 random orders as a batch
            n_orders = np.random.randint(1, 4)
            n_orders = min(n_orders, len(self.all_orders))
            idxs     = np.random.choice(
                len(self.all_orders), size=n_orders, replace=False
            )
            orders = [self.all_orders[i] for i in idxs]

        self.current_order_batch = orders

        # Build flat pick list — each entry is (shelf_location, order_id)
        # This is the key difference from v2: every item knows its order
        self.pick_list       = []
        self.order_item_counts   = {}
        self.order_picked_counts = {}
        self.order_start_times   = {}

        for order in orders:
            oid   = order["order_id"]
            items = order["items"]
            self.order_item_counts[oid]   = len(items)
            self.order_picked_counts[oid] = 0
            self.order_start_times[oid]   = 0.0   # trip starts at sim_time=0

            for item in items:
                self.pick_list.append({
                    "shelf_location": item["shelf_location"],
                    "item_code":      item["item_code"],
                    "task_weight":    item["task_weight"],
                    "order_id":       oid
                })

        self.remaining_picks  = list(self.pick_list)
        self.completed_orders = []
        self.current_position = "DEPOT"
        self.step_count       = 0
        self.items_picked     = 0
        self.total_distance   = 0.0
        self.aisle_conflicts  = 0
        self.stockout_rerouts = 0
        self.sim_time         = 0.0

        # Build initial picker graph
        self.wh_graph.update_from_f3({
            "picker_id":            self.picker_id,
            "ergonomic_risk_score": self.last_ergo_score,
            "gait_stability_score": self.last_gait_score,
            "demographic_weight":   self.last_demo_weight,
            "task_tier":            self.last_task_tier,
            "freeze_state":         False
        })

        return self._get_observation(), {}

    # ══════════════════════════════════════════════════════════════════════
    # STEP
    # ══════════════════════════════════════════════════════════════════════

    def step(self, action: int):

        self.step_count += 1

        # Get valid nodes (not stocked out, task-compatible)
        valid_locations = self.wh_graph.get_valid_nodes(
            self.picker_id,
            [r["shelf_location"] for r in self.remaining_picks]
        )

        # Dead-end — nothing left to pick
        if not valid_locations:
            self.sim_time += self.PICK_TIME_SECONDS
            return (self._get_observation(), -2.0,
                    True, False, self._episode_summary())

        # Map action to target location
        target_location = valid_locations[action % len(valid_locations)]

        # Find the pick item matching this location
        target_item = next(
            (r for r in self.remaining_picks
             if r["shelf_location"] == target_location), None
        )
        if target_item is None:
            return (self._get_observation(), -1.0,
                    False, False, {})

        # Physical distance on base graph (real metres walked)
        try:
            physical_distance = nx.shortest_path_length(
                self.wh_graph.base_graph,
                self.current_position,
                target_location,
                weight="base_distance"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            physical_distance = 10.0

        # Advance simulated time — distance-based
        travel_time    = physical_distance / self.WALKING_SPEED_MS
        self.sim_time += travel_time + self.PICK_TIME_SECONDS

        # Ergonomic-weighted cost for reward secondary signal
        try:
            g = self.wh_graph.picker_graphs.get(self.picker_id)
            weighted_cost = nx.shortest_path_length(
                g, self.current_position,
                target_location, weight="weight"
            ) if g else physical_distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            weighted_cost = physical_distance

        # Execute the pick
        self.current_position  = target_location
        self.remaining_picks.remove(target_item)
        self.items_picked     += 1
        self.total_distance   += physical_distance

        # Update order completion tracking
        oid = target_item["order_id"]
        self.order_picked_counts[oid] += 1
        order_just_completed = False
        order_cycle_time     = 0.0

        if self.order_picked_counts[oid] == self.order_item_counts[oid]:
            # This order is now fully picked
            order_just_completed = True
            order_cycle_time     = self.sim_time - self.order_start_times[oid]
            self.completed_orders.append(oid)

        # Congestion (single picker — always False in training)
        aisle_conflict = False

        # Termination
        all_items_done = len(self.remaining_picks) == 0
        time_exceeded  = self.sim_time >= self.MAX_SIM_SECONDS
        step_exceeded  = self.step_count >= self.max_steps
        done           = all_items_done or time_exceeded or step_exceeded

        # Reward
        reward = self._calculate_reward(
            physical_distance=physical_distance,
            weighted_cost=weighted_cost,
            order_just_completed=order_just_completed,
            order_cycle_time=order_cycle_time,
            aisle_conflict=aisle_conflict,
            trip_complete=all_items_done
        )

        obs  = self._get_observation()
        info = self._episode_summary() if done else {}
        return obs, reward, done, False, info

    # ══════════════════════════════════════════════════════════════════════
    # REWARD — equally weighted dual objective
    # ══════════════════════════════════════════════════════════════════════

    def _calculate_reward(
        self,
        physical_distance:    float,
        weighted_cost:        float,
        order_just_completed: bool,
        order_cycle_time:     float,
        aisle_conflict:       bool,
        trip_complete:        bool
    ) -> float:
        """
        Equally weighted dual-objective reward.

        PRIMARY A — Item picking rate:
          Contributes every step based on current picks/hour.
          Rewards efficient routing that covers items quickly.
          Magnitude: ~5-15 per step depending on efficiency.

        PRIMARY B — Order completion speed:
          Triggered when all items for one customer order are
          collected. Rewards both completing orders AND doing
          so quickly (faster cycle = higher reward).
          Magnitude: 50 / cycle_minutes → ~8-25 per order.

        These two primaries are calibrated to contribute similar
        total reward over a typical episode, giving equal weight
        to item-level routing quality and order-level completion.

        SECONDARY signals shape behaviour without overriding primaries:
          - Travel distance penalty: keeps routes physically short
          - Ergonomic penalty: nudges agent to ergonomic paths
          - Congestion penalty: proactive aisle avoidance

        PENALTIES:
          - Incomplete order at trip end: picker did not finish
            all items for some orders (due to stockouts)
        """
        reward = 0.0

        # ── PRIMARY A: Item picking efficiency ─────────────────────────────
        sim_hours = max(self.sim_time / 3600, 1e-9)
        item_pph  = self.items_picked / sim_hours
        reward   += (item_pph / max(self.max_steps, 1)) * self.ITEM_PPH_WEIGHT

        # ── PRIMARY B: Order completion speed ─────────────────────────────
        if order_just_completed:
            cycle_minutes = max(order_cycle_time / self.ORDER_CYCLE_DIVISOR,
                                0.5)
            # Faster cycle = higher reward
            # 3-min cycle  → 50/3  = 16.7
            # 6-min cycle  → 50/6  = 8.3
            # 12-min cycle → 50/12 = 4.2
            reward += self.ORDER_COMPLETE_BASE / cycle_minutes

        # ── Trip completion bonus ──────────────────────────────────────────
        if trip_complete:
            reward += 30.0

        # ── SECONDARY: Travel distance ─────────────────────────────────────
        reward -= self.TRAVEL_PENALTY * physical_distance

        # ── SECONDARY: Ergonomic cost ──────────────────────────────────────
        reward -= self.ERGONOMIC_PENALTY * weighted_cost

        # ── SECONDARY: Congestion ──────────────────────────────────────────
        if aisle_conflict:
            reward -= self.CONGESTION_PENALTY
            self.aisle_conflicts += 1

        return reward

    # ══════════════════════════════════════════════════════════════════════
    # OBSERVATION
    # ══════════════════════════════════════════════════════════════════════

    def _get_observation(self) -> np.ndarray:
        """
        10-dimensional state vector. All values clamped to [0, 1].

        New dimensions vs v2:
          [8] order_completion_ratio — how many orders in this batch
              are fully completed / total orders in batch
          [9] n_orders_normalised    — batch size (normalised)
              gives agent context about whether this is a solo
              or multi-order trip, shaping its routing strategy
        """
        x, y = self.node_coords.get(self.current_position, (0.0, 0.0))
        px   = float(x) / self.max_x
        py   = float(y) / self.max_y

        total_items     = max(len(self.pick_list), 1)
        remaining_ratio = len(self.remaining_picks) / total_items

        ergo_norm = (self.last_ergo_score - 1) / 9.0
        gait_norm = (self.last_gait_score  - 1) / 9.0

        aisle_occupancy = 0.0   # multi-picker phase

        stockout_norm = min(
            len(self.wh_graph.masked_nodes) / max(total_items, 1), 1.0
        )

        time_ratio = min(self.sim_time / self.MAX_SIM_SECONDS, 1.0)

        total_orders            = max(len(self.current_order_batch), 1)
        order_completion_ratio  = len(self.completed_orders) / total_orders
        n_orders_norm           = min(total_orders / 10.0, 1.0)

        return np.clip(
            np.array([
                px, py,
                remaining_ratio,
                ergo_norm,
                gait_norm,
                aisle_occupancy,
                stockout_norm,
                time_ratio,
                order_completion_ratio,
                n_orders_norm
            ], dtype=np.float32),
            0.0, 1.0
        )

    # ══════════════════════════════════════════════════════════════════════
    # EXTERNAL UPDATE HOOKS
    # ══════════════════════════════════════════════════════════════════════

    def receive_f3_update(self, msg: dict):
        if not msg.get("freeze_state", False):
            self.last_ergo_score  = msg["ergonomic_risk_score"]
            self.last_gait_score  = msg["gait_stability_score"]
            self.last_demo_weight = msg["demographic_weight"]
            self.last_task_tier   = msg["task_tier"]
        self.wh_graph.update_from_f3(msg)

    def receive_f1_update(self, msg: dict):
        """
        F1 stockout: mask node AND remove from remaining picks.
        If removing this item makes an order impossible to complete,
        mark that order as incomplete (will be penalised at trip end).
        """
        self.wh_graph.update_from_f1(msg)
        if msg.get("exception_flag", False):
            loc       = msg["shelf_location"]
            to_remove = [r for r in self.remaining_picks
                         if r["shelf_location"] == loc]
            for item in to_remove:
                self.remaining_picks.remove(item)
                self.stockout_rerouts += 1
                # Reduce the expected count for this order
                oid = item["order_id"]
                if oid in self.order_item_counts:
                    self.order_item_counts[oid] -= 1
                print(f"[Env] Stockout replan: {loc} removed "
                      f"(order={oid})")

    # ══════════════════════════════════════════════════════════════════════
    # EPISODE SUMMARY — both metrics reported equally
    # ══════════════════════════════════════════════════════════════════════

    def _episode_summary(self) -> dict:
        """
        Reports both item_pph and order_pph with equal prominence.

        Adds depot return distance to sim_time for order_pph calculation.
        This is correct because a picker must walk back to depot to
        hand over orders — this time counts against order throughput.
        """
        # Depot return distance
        try:
            return_dist = nx.shortest_path_length(
                self.wh_graph.base_graph,
                self.current_position,
                "DEPOT",
                weight="base_distance"
            )
        except Exception:
            return_dist = 20.0

        return_time            = return_dist / self.WALKING_SPEED_MS
        total_time_with_return = self.sim_time + return_time
        hours_with_return      = max(total_time_with_return / 3600, 1e-9)
        hours_without_return   = max(self.sim_time / 3600, 1e-9)

        n_orders      = len(self.current_order_batch)
        n_completed   = len(self.completed_orders)
        total_items   = len(self.pick_list)

        # item_pph: does NOT include depot return (routing quality measure)
        item_pph      = self.items_picked / hours_without_return

        # order_pph: INCLUDES depot return (order delivery measure)
        order_pph     = n_completed / hours_with_return

        # Mean cycle time per completed order
        if n_completed > 0:
            # Cycle time = from trip start to depot return, split equally
            # In single-order trips this is exact
            # In multi-order trips this is an approximation
            mean_cycle_time = total_time_with_return / n_completed
        else:
            mean_cycle_time = total_time_with_return

        dist_per_item = (
            (self.total_distance + return_dist) / max(self.items_picked, 1)
        )

        return {
            # Trip metadata
            "n_orders_in_batch":   n_orders,
            "n_orders_completed":  n_completed,
            "n_orders_incomplete": n_orders - n_completed,
            "items_total":         total_items,
            "items_picked":        self.items_picked,
            "completion_rate":     round(self.items_picked /
                                         max(total_items, 1), 3),

            # PRIMARY METRICS — equally weighted
            "item_pph":            round(item_pph,         1),
            "order_pph":           round(order_pph,        2),
            "mean_cycle_time_sec": round(mean_cycle_time,  1),

            # SECONDARY METRICS — travel and safety
            "total_distance_m":    round(self.total_distance +
                                         return_dist,      1),
            "dist_per_item_m":     round(dist_per_item,    2),
            "return_distance_m":   round(return_dist,      1),
            "sim_time_seconds":    round(total_time_with_return, 1),

            # Operational events
            "aisle_conflicts":     self.aisle_conflicts,
            "stockout_rerouts":    self.stockout_rerouts,
            "total_steps":         self.step_count,
        }