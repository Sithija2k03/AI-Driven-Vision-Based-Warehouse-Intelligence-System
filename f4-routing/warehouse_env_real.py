"""
F4 Gymnasium environment — real Fine Enterprises warehouse
==========================================================
One episode = one pick trip: start at the DEPOT, visit every stop of one
order, return to the DEPOT. At each step the agent picks the next
stand-point to walk to.

  Action      : Discrete(N) — index of a pick stand-point (N = 204)
                Only stand-points the order still needs are allowed
                (action_masks(), used by MaskablePPO). A slot reachable from
                several stand-points (X-3 island) can be picked from any of them.
  Observation : float32 vector, 4N + 3 values
                  needed[N]      1 if the stand-point still has to be visited
                  dist_now[N]    walking distance from where the picker is now
                  dist_home[N]   walking distance back to the DEPOT
                  stairs_now[N]  staircase trips from here to each stand-point
                  remaining, walked so far, on-mezzanine flag
  Reward      : minus metres walked / 100 on every step, including the final
                walk back to the DEPOT. The episode return is therefore
                -(trip length / 100): maximising reward = shortest trip.

Distances include staircase trips at their real length (distance_matrix.json).
Hook for the dynamic phase: pass `edge_multiplier` (N+1 x N+1 array) to scale
distances for congestion or F3 ergonomic scores without retraining code.
"""

import json
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baselines import stops_for

D_SCALE = 150.0     # metres; normalises distances to roughly 0-1
MAX_STOPS = 30.0


class RealWarehouseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, orders_path='orders_train.json', dm_path='distance_matrix.json',
                 min_stops=2, sequential=False, seed=None, edge_multiplier=None):
        super().__init__()
        dm = json.load(open(dm_path, encoding='utf-8'))
        self.node_ids = dm['nodes']                      # index 0 = DEPOT
        self.idx = {n: i for i, n in enumerate(self.node_ids)}
        D = np.array(dm['dist_m'], dtype=np.float32)
        if edge_multiplier is not None:
            D = D * np.asarray(edge_multiplier, dtype=np.float32)
        self.D = D
        self.S = np.array(dm['stairs'], dtype=np.float32)
        self.N = len(self.node_ids) - 1                   # pick stand-points
        self.mezz = np.array([n.startswith('MZ') for n in self.node_ids], dtype=np.float32)

        orders = json.load(open(orders_path, encoding='utf-8'))['orders']
        self.orders = []
        for o in orders:
            groups = [[self.idx[a] for a in s['alts']] for s in stops_for(o)]
            if len(groups) >= min_stops:
                self.orders.append({'so_no': o['so_no'], 'n_lines': o['n_lines'], 'groups': groups})
        if not self.orders:
            raise ValueError('No orders with enough stops in ' + orders_path)
        self.sequential = sequential
        self._next = 0
        self.rng = random.Random(seed)

        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(4 * self.N + 3,), dtype=np.float32)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _needed_nodes(self):
        need = np.zeros(self.N + 1, dtype=np.float32)
        for g in self.remaining:
            need[g] = 1.0
        return need

    def _obs(self):
        need = self._needed_nodes()[1:]
        cur = self.cur
        return np.concatenate([
            need,
            self.D[cur, 1:] / D_SCALE,
            self.D[1:, 0] / D_SCALE,
            np.minimum(self.S[cur, 1:], 3.0) / 3.0,
            np.array([len(self.remaining) / MAX_STOPS, self.walked / (4 * D_SCALE), self.mezz[cur]],
                     dtype=np.float32),
        ]).astype(np.float32)

    def action_masks(self):
        return self._needed_nodes()[1:].astype(bool)

    # ── gym API ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'order' in options:
            self.order = options['order']
        elif self.sequential:
            self.order = self.orders[self._next % len(self.orders)]
            self._next += 1
        else:
            self.order = self.rng.choice(self.orders)
        self.remaining = [list(g) for g in self.order['groups']]
        self.cur, self.walked, self.stairs, self.route = 0, 0.0, 0.0, []
        return self._obs(), {}

    def step(self, action):
        node = int(action) + 1
        hit = [g for g in self.remaining if node in g]
        if not hit:                                   # only reachable without masking
            return self._obs(), -5.0, False, False, {'invalid': True}
        d = float(self.D[self.cur, node])
        self.walked += d
        self.stairs += float(self.S[self.cur, node])
        self.route.append(self.node_ids[node])
        self.remaining = [g for g in self.remaining if node not in g]
        self.cur = node
        reward = -d / 100.0
        done = not self.remaining
        info = {}
        if done:
            back = float(self.D[self.cur, 0])
            self.walked += back
            self.stairs += float(self.S[self.cur, 0])
            reward -= back / 100.0
            info = {'trip_m': round(self.walked, 2), 'stairs': int(self.stairs),
                    'so_no': self.order['so_no'], 'route': list(self.route),
                    'n_stops': len(self.order['groups'])}
        return self._obs(), reward, done, False, info


def mask_fn(env):
    return env.action_masks()


if __name__ == '__main__':
    # quick self-test: nearest-neighbour policy through the env
    env = RealWarehouseEnv('orders_val.json', sequential=True)
    tot = []
    for _ in range(200):
        obs, _ = env.reset()
        done = False
        while not done:
            m = env.action_masks()
            a = int(np.where(m)[0][np.argmin(obs[env.N:2 * env.N][m])])
            obs, r, done, _, info = env.step(a)
        tot.append(info['trip_m'])
    print(f'{len(env.orders)} validation orders with 2+ stops; '
          f'nearest-neighbour through the env, first 200: mean {np.mean(tot):.1f} m')
