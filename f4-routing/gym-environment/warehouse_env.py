"""
f4-routing/gym-environment/warehouse_env.py

Custom OpenAI Gymnasium environment for warehouse order picking.

The RL agent lives inside this environment. At each step it:
  1. Observes the current warehouse state (state)
  2. Chooses which item node to visit next (action)
  3. Receives a reward based on efficiency + safety
  4. The environment updates and returns the new state

Primary objective : maximise picks per hour (efficiency)
Secondary         : ergonomic safety via F3-modulated edge weights

Key design decisions:
  - Simulated time replaces real clock time so picks-per-hour
    reflects realistic warehouse operation (~50-120 picks/hour)
    rather than CPU execution speed (millions per hour).
  - sim_seconds_per_step models the average time a human picker
    takes to walk to and collect one item. Default = 45 seconds.
  - Reward magnitudes are deliberately asymmetric:
    primary efficiency signals (+10x, +50) dominate ergonomic
    secondary signals (-0.1 per cost unit) by a factor of 5-50x
    so the agent always prioritises picking efficiency first.
"""

import gymnasium as gym
import numpy as np
import networkx as nx
import json
import sys
import os

# ── Absolute import — works regardless of run location ────────────────────
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from graph_builder import WarehouseGraph


class WarehouseEnv(gym.Env):
    """
    Warehouse order picking Gymnasium environment.

    Episode lifecycle:
        reset()  → random order sampled from orders.json
                 → picker placed at DEPOT
                 → simulated clock set to 0
        step()   → agent selects next pick node
                 → travel cost calculated on weighted graph
                 → simulated time advances by sim_seconds_per_step
                 → reward returned
                 → done when all items picked or max_steps reached

    Observation space (8-dimensional, all values in [0, 1]):
        [0] picker_x              current x position (normalised)
        [1] picker_y              current y position (normalised)
        [2] items_remaining_ratio remaining picks / total picks
        [3] ergo_norm             F3 ergonomic score (1-10 → 0-1)
        [4] gait_norm             F3 gait score (1-10 → 0-1)
        [5] aisle_occupancy       congestion 0-1 (0 in single-picker mode)
        [6] stockout_norm         active stockouts / total pick list
        [7] time_ratio            sim time used / max sim time budget

    Action space:
        Discrete(MAX_ORDER_SIZE)
        Agent selects index into current valid pick list.
        Invalid nodes (stockout / task-incompatible) are pre-filtered.
    """

    metadata = {"render_modes": []}

    # Maximum order size supported — matches order generator cap
    MAX_ORDER_SIZE = 20

    # Simulated time budget per episode
    # 20 items × 45 seconds = 900 seconds = 15 minutes per trip
    # Realistic for a warehouse pick run
    SIM_SECONDS_PER_STEP = 45     # seconds per pick (walk + collect)
    MAX_SIM_SECONDS      = 1800   # 30-minute episode time budget

    def __init__(
        self,
        floor_plan_path: str,
        orders_path: str,
        max_pickers: int = 1,
        max_episode_steps: int = 100,
        render_mode=None
    ):
        super().__init__()

        self.floor_plan_path  = floor_plan_path
        self.orders_path      = orders_path
        self.max_pickers      = max_pickers
        self.max_steps        = max_episode_steps

        # ── Load warehouse graph ───────────────────────────────────────────
        self.wh_graph = WarehouseGraph(floor_plan_path)

        # ── Load orders ────────────────────────────────────────────────────
        with open(orders_path) as f:
            data = json.load(f)
        self.all_orders = data["orders"]
        print(f"[Env] Loaded {len(self.all_orders)} orders")
        print(f"[Env] Order categories: "
              f"Small={data['categories']['Small']['count']} | "
              f"Medium={data['categories']['Medium']['count']} | "
              f"Large={data['categories']['Large']['count']}")

        # ── Load node coordinates for observation builder ──────────────────
        with open(floor_plan_path) as f:
            plan = json.load(f)
        self.node_coords = {
            n["id"]: (n.get("x", 0.0), n.get("y", 0.0))
            for n in plan["nodes"]
        }

        # Normalisation bounds
        dims       = plan["dimensions"]
        self.max_x = max(dims["total_warehouse_width_m"], 1.0)
        self.max_y = max(dims["total_warehouse_depth_m"], 1.0)

        # ── Episode state — initialised properly in reset() ────────────────
        self.current_order    = None
        self.full_pick_list   = []
        self.remaining_picks  = []
        self.picker_id        = "P-001"
        self.current_position = "DEPOT"
        self.step_count       = 0
        self.items_picked     = 0
        self.aisle_conflicts  = 0
        self.stockout_rerouts = 0

        # Simulated time — replaces real clock
        self.sim_time         = 0.0   # seconds elapsed in simulation

        # Last known F3 values — updated via receive_f3_update()
        self.last_ergo_score  = 3     # default: low-moderate risk
        self.last_gait_score  = 2     # default: stable gait
        self.last_demo_weight = 1.0   # default: young adult
        self.last_task_tier   = "Heavy"  # default: full task capability

        # ── Gym spaces ─────────────────────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.MAX_ORDER_SIZE)

    # ══════════════════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a random order from the pool
        idx = np.random.randint(len(self.all_orders))
        self.current_order = self.all_orders[idx]

        self.full_pick_list   = [
            item["shelf_location"]
            for item in self.current_order["items"]
        ]
        self.remaining_picks  = list(self.full_pick_list)
        self.current_position = "DEPOT"
        self.step_count       = 0
        self.items_picked     = 0
        self.aisle_conflicts  = 0
        self.stockout_rerouts = 0
        self.sim_time         = 0.0   # reset simulated clock

        # Build initial picker graph with current F3 state
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

        # ── Get valid pick nodes ───────────────────────────────────────────
        valid_picks = self.wh_graph.get_valid_nodes(
            self.picker_id, self.remaining_picks
        )

        # ── Dead-end: no valid picks left ─────────────────────────────────
        if not valid_picks:
            self.sim_time += self.SIM_SECONDS_PER_STEP
            info   = self._episode_summary()
            return self._get_observation(), -2.0, True, False, info

        # ── Map action index to target node ───────────────────────────────
        # Modulo ensures action always maps to a valid index
        # even if action > len(valid_picks)
        target_node = valid_picks[action % len(valid_picks)]

        # ── Advance simulated time ─────────────────────────────────────────
        # This is the critical fix — time advances by a fixed realistic
        # amount per pick rather than real CPU execution time.
        # Effect: picks_per_hour stays in the realistic 50-120 range.
        self.sim_time += self.SIM_SECONDS_PER_STEP

        # ── Calculate weighted travel cost ─────────────────────────────────
        g = self.wh_graph.picker_graphs.get(self.picker_id)
        try:
            travel_cost = nx.shortest_path_length(
                g, self.current_position, target_node, weight="weight"
            ) if g else 5.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            travel_cost = 10.0   # penalty for unreachable node

        # ── Execute the pick ───────────────────────────────────────────────
        self.current_position = target_node
        self.remaining_picks.remove(target_node)
        self.items_picked += 1

        # ── Check congestion (single picker — always false for now) ────────
        # Extended in multi-picker phase with shared occupancy map
        aisle_conflict = False

        # ── Termination check ──────────────────────────────────────────────
        order_complete = len(self.remaining_picks) == 0
        time_exceeded  = self.sim_time >= self.MAX_SIM_SECONDS
        step_exceeded  = self.step_count >= self.max_steps
        done = order_complete or time_exceeded or step_exceeded

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._calculate_reward(
            travel_cost=travel_cost,
            aisle_conflict=aisle_conflict,
            order_complete=order_complete
        )

        obs  = self._get_observation()
        info = self._episode_summary() if done else {}
        return obs, reward, done, False, info

    # ══════════════════════════════════════════════════════════════════════
    # REWARD FUNCTION
    # ══════════════════════════════════════════════════════════════════════

    def _calculate_reward(
        self,
        travel_cost: float,
        aisle_conflict: bool,
        order_complete: bool
    ) -> float:
        """
        Multi-objective reward. Efficiency signals dominate ergonomic
        signals by design — the agent learns efficiency first.

        PRIMARY (high magnitude):
          +picks_per_hour_contribution  core efficiency metric
          +50.0                         order completion bonus
          -0.5 × remaining_items        urgency pressure

        SECONDARY (low magnitude — shape behaviour, never override efficiency):
          -0.1 × travel_cost            penalise long routes
          -5.0                          aisle conflict (congestion)

        Simulated time keeps picks_per_hour in realistic range (50-120/hr).
        At 45s per pick:
          10 items picked in 450s (7.5 min) → 80 picks/hour
          5 items in 225s (3.75 min)        → 80 picks/hour
        """
        reward = 0.0

        # ── PRIMARY: Efficiency ────────────────────────────────────────────
        sim_hours = max(self.sim_time / 3600, 1e-6)
        picks_per_hour = self.items_picked / sim_hours
        # Scale to a per-step contribution
        # Dividing by max_steps keeps reward magnitude stable across
        # episodes of different lengths
        reward += (picks_per_hour / max(self.max_steps, 1)) * 10.0

        # ── PRIMARY: Completion bonus ──────────────────────────────────────
        if order_complete:
            reward += 50.0

        # ── PRIMARY: Urgency ───────────────────────────────────────────────
        reward -= 0.5 * len(self.remaining_picks)

        # ── SECONDARY: Travel cost ─────────────────────────────────────────
        # Small penalty — pushes agent toward shorter ergonomic routes
        # but never enough to override efficiency
        reward -= 0.1 * travel_cost

        # ── SECONDARY: Congestion ──────────────────────────────────────────
        if aisle_conflict:
            reward -= 5.0
            self.aisle_conflicts += 1

        return reward

    # ══════════════════════════════════════════════════════════════════════
    # OBSERVATION
    # ══════════════════════════════════════════════════════════════════════

    def _get_observation(self) -> np.ndarray:
        """
        Returns the 8-dimensional state vector.
        All values clamped to [0, 1] for stable PPO training.

        The agent uses this to decide its next action.
        Richer state → better decisions, but larger state →
        slower training. 8 dimensions is a pragmatic balance
        for a final year project scope.
        """
        # ── [0, 1] Position ────────────────────────────────────────────────
        x, y = self.node_coords.get(self.current_position, (0.0, 0.0))
        px   = float(x) / self.max_x
        py   = float(y) / self.max_y

        # ── [2] Progress ───────────────────────────────────────────────────
        total              = max(len(self.full_pick_list), 1)
        remaining_ratio    = len(self.remaining_picks) / total

        # ── [3, 4] F3 ergonomic signals ────────────────────────────────────
        # Normalised from 1-10 integer scale to 0-1 float
        ergo_norm = (self.last_ergo_score - 1) / 9.0
        gait_norm = (self.last_gait_score  - 1) / 9.0

        # ── [5] Aisle occupancy / congestion ───────────────────────────────
        # Placeholder — always 0.0 in single-picker mode
        # Will be populated from shared occupancy map in multi-picker phase
        aisle_occupancy = 0.0

        # ── [6] Stockout pressure ──────────────────────────────────────────
        stockout_norm = min(
            len(self.wh_graph.masked_nodes) / max(total, 1), 1.0
        )

        # ── [7] Time budget used ───────────────────────────────────────────
        time_ratio = min(self.sim_time / max(self.MAX_SIM_SECONDS, 1), 1.0)

        return np.clip(
            np.array([
                px, py,
                remaining_ratio,
                ergo_norm,
                gait_norm,
                aisle_occupancy,
                stockout_norm,
                time_ratio
            ], dtype=np.float32),
            0.0, 1.0
        )

    # ══════════════════════════════════════════════════════════════════════
    # EXTERNAL UPDATE HOOKS — called by message bus subscribers
    # ══════════════════════════════════════════════════════════════════════

    def receive_f3_update(self, msg: dict):
        """
        Called when F3 publishes an ergonomic update on the message bus.
        Frozen readings are ignored — previous valid scores are held.
        This updates both the environment state AND the graph edge weights.
        """
        if not msg.get("freeze_state", False):
            self.last_ergo_score  = msg["ergonomic_risk_score"]
            self.last_gait_score  = msg["gait_stability_score"]
            self.last_demo_weight = msg["demographic_weight"]
            self.last_task_tier   = msg["task_tier"]

        self.wh_graph.update_from_f3(msg)

    def receive_f1_update(self, msg: dict):
        """
        Called when F1 publishes an inventory update on the message bus.
        On stockout exception: masks the node AND removes it from the
        active pick list so the agent is forced to replan immediately.
        This is the vision-triggered mid-pick replanning novelty.
        """
        self.wh_graph.update_from_f1(msg)

        if msg.get("exception_flag", False):
            loc = msg["shelf_location"]
            if loc in self.remaining_picks:
                self.remaining_picks.remove(loc)
                self.stockout_rerouts += 1
                print(f"[Env] Mid-pick replan triggered: {loc} removed "
                      f"| item={msg['item_code']} "
                      f"| rerouts this episode={self.stockout_rerouts}")

    # ══════════════════════════════════════════════════════════════════════
    # EPISODE SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    def _episode_summary(self) -> dict:
        """
        Returns a dictionary of episode metrics written to the
        evaluation_runs table in Supabase at the end of each episode.
        picks_per_hour is the headline metric compared against baselines.
        """
        sim_hours      = max(self.sim_time / 3600, 1e-6)
        picks_per_hour = self.items_picked / sim_hours
        total          = max(len(self.full_pick_list), 1)

        return {
            "order_id":         self.current_order["order_id"],
            "order_category":   self.current_order["category"],
            "order_size":       self.current_order["order_size"],
            "items_picked":     self.items_picked,
            "items_total":      total,
            "completion_rate":  round(self.items_picked / total, 3),
            "picks_per_hour":   round(picks_per_hour, 1),
            "sim_time_seconds": round(self.sim_time, 1),
            "total_steps":      self.step_count,
            "aisle_conflicts":  self.aisle_conflicts,
            "stockout_rerouts": self.stockout_rerouts,
        }