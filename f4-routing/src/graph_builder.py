"""
f4-routing/src/graph_builder.py

Loads a warehouse floor plan JSON into a NetworkX directed graph.
Applies dynamic modifiers from F1 and F3 on each update cycle.

The graph is the foundation the Gymnasium environment and RL agent
sit on. Every routing decision the agent makes is a path through
this graph.
"""

import json
import networkx as nx


def ergonomic_multiplier(score: int) -> float:
    """Score 1 → 1.0x (no penalty). Score 10 → 2.0x (double cost)."""
    return 1.0 + (score - 1) / 9.0


def gait_hazard_multiplier(gait_score: int, edge_hazard: str) -> float:
    """Extra cost on hazardous edges for fall-risk pickers (gait > 7)."""
    if gait_score <= 7:
        return 1.0
    return {"low": 1.1, "medium": 1.4, "high": 1.8}.get(edge_hazard, 1.0)


def demographic_floor(demo_weight: float) -> float:
    """Baseline cost floor for vulnerable demographic groups."""
    return 1.0 + (demo_weight - 1.0) * 0.5


def edge_cost(base_dist, ergo, gait, demo, hazard="low"):
    return round(
        base_dist
        * ergonomic_multiplier(ergo)
        * gait_hazard_multiplier(gait, hazard)
        * demographic_floor(demo), 3
    )


class WarehouseGraph:

    def __init__(self, floor_plan_path: str):
        self.path = floor_plan_path
        self.base_graph = nx.DiGraph()
        self.masked_nodes: set = set()
        self.picker_graphs: dict = {}
        self.last_f3: dict = {}
        self._load()

    def _load(self):
        with open(self.path) as f:
            plan = json.load(f)

        for n in plan["nodes"]:
            self.base_graph.add_node(
                n["id"],
                x=n.get("x", 0), y=n.get("y", 0),
                task_weight=n.get("task_weight", "Moderate"),
                hazard_level=n.get("hazard_level", "low")
            )

        for e in plan["edges"]:
            kw = dict(base_distance=e["distance"],
                      hazard_level=e.get("hazard_level", "low"))
            self.base_graph.add_edge(e["from"], e["to"],   **kw)
            self.base_graph.add_edge(e["to"],   e["from"], **kw)

        print(f"[Graph] {plan['warehouse_id']} loaded — "
              f"{self.base_graph.number_of_nodes()} nodes, "
              f"{self.base_graph.number_of_edges()} edges")

    def update_from_f3(self, msg: dict):
        pid = msg["picker_id"]
        if msg.get("freeze_state", False):
            return   # hold last valid scores

        self.last_f3[pid] = msg
        ergo = msg["ergonomic_risk_score"]
        gait = msg["gait_stability_score"]
        demo = msg["demographic_weight"]
        tier = msg["task_tier"]

        g = self.base_graph.copy()
        tier_rank = {"Heavy": 3, "Moderate": 2, "Light": 1}
        picker_rank = tier_rank.get(tier, 2)

        for u, v, data in g.edges(data=True):
            g[u][v]["weight"] = edge_cost(
                data["base_distance"], ergo, gait, demo,
                data.get("hazard_level", "low")
            )

        for nid, ndata in g.nodes(data=True):
            node_rank = tier_rank.get(ndata.get("task_weight", "Moderate"), 2)
            g.nodes[nid]["task_masked"]    = picker_rank < node_rank
            g.nodes[nid]["stockout_masked"] = nid in self.masked_nodes

        self.picker_graphs[pid] = g

    def update_from_f1(self, msg: dict):
        nid = msg["shelf_location"]
        if msg["exception_flag"]:
            self.masked_nodes.add(nid)
            for g in self.picker_graphs.values():
                if nid in g:
                    g.nodes[nid]["stockout_masked"] = True
            print(f"[Graph] STOCKOUT: {nid} ({msg['item_code']})")
        else:
            self.masked_nodes.discard(nid)
            for g in self.picker_graphs.values():
                if nid in g:
                    g.nodes[nid]["stockout_masked"] = False

    def get_valid_nodes(self, picker_id: str, pick_list: list) -> list:
        if picker_id not in self.picker_graphs:
            return pick_list
        g = self.picker_graphs[picker_id]
        return [
            n for n in pick_list
            if n in g
            and not g.nodes[n].get("stockout_masked", False)
            and not g.nodes[n].get("task_masked", False)
        ]

    def get_route(self, picker_id: str, start: str,
                  pick_list: list) -> list:
        """Nearest-neighbour route on the personalised weighted graph."""
        if picker_id not in self.picker_graphs:
            return pick_list
        g = self.picker_graphs[picker_id]
        remaining, route, current = list(pick_list), [], start
        while remaining:
            best, best_cost = None, float("inf")
            for node in remaining:
                try:
                    c = nx.shortest_path_length(g, current, node,
                                                weight="weight")
                    if c < best_cost:
                        best_cost, best = c, node
                except nx.NetworkXNoPath:
                    continue
            if best is None:
                break
            route.append(best)
            remaining.remove(best)
            current = best
        return route