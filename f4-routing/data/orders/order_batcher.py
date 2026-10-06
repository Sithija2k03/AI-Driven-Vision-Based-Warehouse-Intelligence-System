"""
f4-routing/data/orders/order_batcher.py

Generalised order batching module for warehouse picking operations.

Batching rules (configurable, not hard-coded):
  Large orders  (size > LARGE_THRESHOLD): always solo trips.
                These are never mixed with other orders due to
                volume, error risk, and cart capacity constraints.

  Small/Medium orders: batched together by spatial proximity.
                Orders whose items cluster in the same aisles
                are grouped so the picker does not traverse the
                full warehouse for unrelated items.

  Batch size cap: total items across all orders in one batch
                must not exceed MAX_ITEMS_PER_BATCH. This is
                configurable because it depends on picker
                strength, cart type, and item weight — factors
                that vary across warehouses and even across
                pickers in the same warehouse.

  Fairness: batches are distributed across active pickers
            using round-robin assignment so each picker
            receives approximately equal number of orders.

This module is warehouse-agnostic. All thresholds are
parameters that can be adjusted per deployment.

Usage:
    from order_batcher import OrderBatcher
    batcher = OrderBatcher(n_pickers=5, max_items_per_batch=20)
    assignments = batcher.batch_and_assign(orders)
    # assignments: {picker_id: [batch1, batch2, ...], ...}
    # Each batch is a list of order dicts with their items
"""

import numpy as np
from collections import defaultdict


class OrderBatcher:
    """
    Batches incoming orders and assigns batches to pickers.

    Parameters
    ----------
    n_pickers : int
        Number of active pickers this shift. Variable — not fixed.
        Call update_picker_count() if pickers join or leave mid-shift.

    max_items_per_batch : int
        Maximum total items a picker can carry in one trip.
        Default 20 is conservative — works for most cart types.
        Set higher for large trolleys, lower for hand baskets.
        Does NOT apply to Large orders (they are always solo).

    large_order_threshold : int
        Orders with more items than this are always solo trips.
        Default 15 matches the Medium/Large boundary in orders.json.

    aisle_proximity_threshold : float
        Maximum aisle centroid distance for two orders to be
        considered spatially compatible for batching.
        Value of 2.0 means orders whose average aisle positions
        differ by more than 2 aisles are not batched together.
        Increase for wider warehouses with more aisles.
    """

    def __init__(
        self,
        n_pickers: int = 1,
        max_items_per_batch: int = 20,
        large_order_threshold: int = 15,
        aisle_proximity_threshold: float = 2.0
    ):
        self.n_pickers                = max(n_pickers, 1)
        self.max_items_per_batch      = max_items_per_batch
        self.large_order_threshold    = large_order_threshold
        self.aisle_proximity_threshold = aisle_proximity_threshold

    def update_picker_count(self, n_pickers: int):
        """Update picker count if availability changes mid-shift."""
        self.n_pickers = max(n_pickers, 1)

    # ── Spatial analysis helpers ───────────────────────────────────────────

    def _aisle_centroid(self, order: dict) -> float:
        """
        Computes the average aisle number of all items in an order.
        This is the spatial centre of the order in the warehouse.
        Orders with close centroids are good candidates for batching.

        Example:
            Order items in A1, A1, A2, A3 → centroid = (1+1+2+3)/4 = 1.75
            Order items in A2, A3         → centroid = (2+3)/2 = 2.5
            Difference = 0.75 → spatially close → good to batch together
        """
        aisles = []
        for item in order.get("items", []):
            loc = item.get("shelf_location", "")
            if loc.startswith("A"):
                try:
                    aisles.append(int(loc.split("-")[0][1:]))
                except ValueError:
                    pass
        return float(np.mean(aisles)) if aisles else 0.0

    def _are_spatially_compatible(self,
                                   order_a: dict,
                                   order_b: dict) -> bool:
        """
        Returns True if two orders are close enough in the warehouse
        to be worth batching together.
        Uses aisle centroid distance as the proximity measure.
        """
        ca = self._aisle_centroid(order_a)
        cb = self._aisle_centroid(order_b)
        return abs(ca - cb) <= self.aisle_proximity_threshold

    def _is_large_order(self, order: dict) -> bool:
        """Large orders always get solo trips — never batched."""
        return order.get("order_size", 0) > self.large_order_threshold

    # ── Core batching logic ────────────────────────────────────────────────

    def create_batches(self, orders: list) -> list:
        """
        Groups orders into batches ready for picker assignment.

        Algorithm:
          1. Separate large orders → each becomes its own solo batch
          2. Sort remaining orders by aisle centroid (spatial grouping)
          3. Greedily assign orders to batches:
             - Try to add each order to the most spatially compatible
               open batch that has not exceeded the item cap
             - If no compatible batch exists, start a new batch
          4. Return list of batches (each batch is a list of orders)

        Returns
        -------
        list of batches, where each batch is:
            {
                "batch_id":     str,
                "is_solo":      bool,   True for Large orders
                "total_items":  int,    sum of items across all orders
                "orders":       list,   list of order dicts
                "aisle_centroid": float
            }
        """
        batches   = []
        batch_num = 0

        # Step 1: Large orders → solo batches immediately
        batchable = []
        for order in orders:
            if self._is_large_order(order):
                batch_num += 1
                batches.append({
                    "batch_id":      f"BATCH-{str(batch_num).zfill(4)}",
                    "is_solo":       True,
                    "total_items":   order["order_size"],
                    "orders":        [order],
                    "aisle_centroid": self._aisle_centroid(order)
                })
            else:
                batchable.append(order)

        # Step 2: Sort batchable orders by aisle centroid
        batchable.sort(key=lambda o: self._aisle_centroid(o))

        # Step 3: Greedy spatial batching
        open_batches = []   # batches not yet finalised

        for order in batchable:
            order_centroid  = self._aisle_centroid(order)
            order_size      = order.get("order_size", 0)
            placed          = False

            # Try to fit into an existing open batch
            for ob in open_batches:
                # Check spatial compatibility
                centroid_diff = abs(ob["aisle_centroid"] - order_centroid)
                if centroid_diff > self.aisle_proximity_threshold:
                    continue

                # Check item cap
                if ob["total_items"] + order_size > self.max_items_per_batch:
                    continue

                # Compatible — add to this batch
                ob["orders"].append(order)
                ob["total_items"]   += order_size
                # Update centroid to reflect new combined spatial position
                all_centroids        = [self._aisle_centroid(o)
                                        for o in ob["orders"]]
                ob["aisle_centroid"] = float(np.mean(all_centroids))
                placed = True
                break

            if not placed:
                # No compatible batch — start a new one
                batch_num += 1
                open_batches.append({
                    "batch_id":      f"BATCH-{str(batch_num).zfill(4)}",
                    "is_solo":       False,
                    "total_items":   order_size,
                    "orders":        [order],
                    "aisle_centroid": order_centroid
                })

        batches.extend(open_batches)
        return batches

    # ── Picker assignment ──────────────────────────────────────────────────

    def batch_and_assign(self, orders: list) -> dict:
        """
        Creates batches from orders and assigns them to pickers.

        Fairness: round-robin assignment so each picker gets
        approximately equal number of batches (and therefore
        approximately equal number of orders).

        This is order-count fairness, not item-count fairness.
        Item count per picker naturally balances through the
        batch size cap — no picker's batch exceeds max_items_per_batch.

        Returns
        -------
        dict: {picker_id: [batch1, batch2, ...]}
              Each picker gets an ordered list of batches to complete
              sequentially during their shift.
        """
        batches = self.create_batches(orders)

        # Initialise assignment dict
        picker_ids  = [f"P-{str(i+1).zfill(3)}"
                       for i in range(self.n_pickers)]
        assignments = {pid: [] for pid in picker_ids}
        batch_counts = {pid: 0 for pid in picker_ids}

        # Round-robin: assign each batch to the picker with fewest batches
        for batch in batches:
            # Pick the picker with the least batches so far
            assigned_picker = min(batch_counts, key=batch_counts.get)
            assignments[assigned_picker].append(batch)
            batch_counts[assigned_picker] += 1

        return assignments

    # ── Summary reporting ──────────────────────────────────────────────────

    def summarise_assignments(self, assignments: dict) -> dict:
        """
        Returns a summary of the batch assignment for logging
        and supervisor dashboard display.
        """
        summary = {}
        for pid, batches in assignments.items():
            total_orders = sum(len(b["orders"]) for b in batches)
            total_items  = sum(b["total_items"] for b in batches)
            solo_trips   = sum(1 for b in batches if b["is_solo"])
            mixed_trips  = len(batches) - solo_trips
            summary[pid] = {
                "n_batches":    len(batches),
                "n_orders":     total_orders,
                "n_items":      total_items,
                "solo_trips":   solo_trips,
                "mixed_trips":  mixed_trips,
            }
        return summary