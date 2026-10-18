"""
F4 Step 7 — Baseline routes on the validation and test orders
=============================================================
Routes every order as one pick trip DEPOT → picks → DEPOT with eight
methods, so the RL agent has fixed numbers to beat.

  Current practice
    LOCATION ORDER     visit locations in pick-list order (A-01, A-02 … B-01 …)
                       — close to how pickers work from a printed list today
  Classic aisle rules from the literature, adapted to the real layout
  (see classic_heuristics.py for how each rule is applied here)
    S-SHAPE            walk every aisle with a pick end to end
    RETURN             enter each aisle from the front, back out the same end
    MIDPOINT           front half from the front, back half from the back
    LARGEST GAP        split each aisle at its largest empty stretch
    COMBINED           per aisle, walk through or go in and back, whichever is shorter
  Simple and exact
    NEAREST NEIGHBOUR  always walk to the closest remaining stop
    OPTIMAL            shortest possible tour
                       · up to 12 stops: exact (Held-Karp dynamic programming)
                       · more stops: OR-Tools guided local search (best found)

A "stop" is a place the picker must visit. Lines at the same stand-point
merge into one stop. A slot reachable from several stand-points (X-3, X-05)
is one stop with alternatives; every method may use whichever is best for it.

Input : orders_val.json, orders_test.json, distance_matrix.json, warehouse_graph.json
Output: baseline_results_<set>.csv   one row per order (distance + stair trips per method)
        baseline_summary_<set>.json  totals, gaps and breakdown by order size
        baseline_routes_<set>.json   the pick sequence each method chose
Run   : python baselines.py                    (validation + test, ~12 min)
        python baselines.py orders_test.json   (one set)
Needs : ortools  (pip install ortools)
"""

import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations

from classic_heuristics import ClassicRouter

EXACT_MAX = 12
OR_TIME_LIMIT_S = 2

CLASSIC = ['sshape', 'return', 'midpoint', 'largest_gap', 'combined']
METHODS = ['location_order'] + CLASSIC + ['nearest_neighbour', 'optimal']
LABEL = {'location_order': 'Location order', 'sshape': 'S-shape', 'return': 'Return', 'midpoint': 'Midpoint',
         'largest_gap': 'Largest gap', 'combined': 'Combined', 'nearest_neighbour': 'Nearest neighbour',
         'optimal': 'Optimal'}
SHORT = {'location_order': 'loc', 'sshape': 'S', 'return': 'ret', 'midpoint': 'mid',
         'largest_gap': 'gap', 'combined': 'comb', 'nearest_neighbour': 'NN', 'optimal': 'opt'}


def loc_key(code):
    sec, num = code.split('-')
    return (sec, int(re.sub(r'\D', '', num) or 0))


class Router:
    def __init__(self, dm):
        self.idx = {n: i for i, n in enumerate(dm['nodes'])}
        self.D = dm['dist_m']
        self.S = dm['stairs']

    def d(self, a, b):
        return self.D[self.idx[a]][self.idx[b]]

    def tour_cost(self, route):
        path = ['DEPOT'] + route + ['DEPOT']
        dist = sum(self.d(a, b) for a, b in zip(path, path[1:]))
        stairs = sum(self.S[self.idx[a]][self.idx[b]] for a, b in zip(path, path[1:]))
        return round(dist, 2), stairs


def stops_for(order):
    """Merge lines into stops. Each stop = tuple of alternative stand-points."""
    groups = {}
    for l in order['lines']:
        key = tuple(sorted(l['access_nodes']))
        groups.setdefault(key, []).append(l['location'])
    # a stop whose alternatives include a stand-point that is itself a stop is covered by it
    singles = {k[0] for k in groups if len(k) == 1}
    out = []
    for k, locs in groups.items():
        if len(k) > 1 and singles & set(k):
            continue
        out.append({'alts': list(k), 'first_loc': min(locs, key=loc_key)})
    return out


# ── 1. location order ────────────────────────────────────────────────────────

def location_order(R, stops):
    route = []
    for s in sorted(stops, key=lambda s: loc_key(s['first_loc'])):
        node = s['alts'][0] if not route else min(s['alts'], key=lambda a: R.d(route[-1], a))
        if not route or route[-1] != node:
            route.append(node)
    return route


# ── 2. classic aisle rules ───────────────────────────────────────────────────

def pick_sequence(stops, waypoints):
    """Order in which a classic route first reaches each stop (pick stand-points only)."""
    seq, done = [], set()
    for n in waypoints:
        for i, s in enumerate(stops):
            if i not in done and n in s['alts']:
                seq.append(n)
                done.add(i)
                break
    return seq


# ── 3. nearest neighbour ─────────────────────────────────────────────────────

def nearest_neighbour(R, stops):
    left, cur, route = list(range(len(stops))), 'DEPOT', []
    while left:
        best = min(((R.d(cur, a), i, a) for i in left for a in stops[i]['alts']))
        _, i, a = best
        route.append(a)
        cur = a
        left.remove(i)
    return route


# ── 4. optimal ───────────────────────────────────────────────────────────────

def held_karp(R, stops):
    n = len(stops)
    alts = [s['alts'] for s in stops]
    # dp[(mask, i, a)] = best cost ending at stop i using alternative a
    dp, parent = {}, {}
    for i in range(n):
        for a in alts[i]:
            dp[(1 << i, i, a)] = R.d('DEPOT', a)
    for size in range(2, n + 1):
        for subset in combinations(range(n), size):
            mask = sum(1 << i for i in subset)
            for i in subset:
                pm = mask ^ (1 << i)
                for a in alts[i]:
                    best, arg = 1e18, None
                    for j in subset:
                        if j == i:
                            continue
                        for b in alts[j]:
                            v = dp.get((pm, j, b))
                            if v is not None and v + R.d(b, a) < best:
                                best, arg = v + R.d(b, a), (pm, j, b)
                    dp[(mask, i, a)], parent[(mask, i, a)] = best, arg
    full = (1 << n) - 1
    end = min(((dp[(full, i, a)] + R.d(a, 'DEPOT'), (full, i, a)) for i in range(n) for a in alts[i]))[1]
    route, key = [], end
    while key:
        route.append(key[2])
        key = parent.get(key)
    return route[::-1]


def ortools_tour(R, stops):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    pts, group = ['DEPOT'], [None]
    for gi, s in enumerate(stops):
        for a in s['alts']:
            pts.append(a)
            group.append(gi)
    n = len(pts)
    M = [[int(round(R.d(pts[i], pts[j]) * 100)) for j in range(n)] for i in range(n)]
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitCallback(lambda i, j: M[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    by_group = defaultdict(list)
    for i in range(1, n):
        by_group[group[i]].append(mgr.NodeToIndex(i))
    for idxs in by_group.values():
        if len(idxs) > 1:
            routing.AddDisjunction(idxs, 10 ** 9, 1)   # visit exactly one alternative
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = OR_TIME_LIMIT_S
    sol = routing.SolveWithParameters(p)
    route, i = [], routing.Start(0)
    i = sol.Value(routing.NextVar(i))
    while not routing.IsEnd(i):
        route.append(pts[mgr.IndexToNode(i)])
        i = sol.Value(routing.NextVar(i))
    return route


def optimal(R, stops):
    if len(stops) <= EXACT_MAX:
        return held_karp(R, stops), 'exact'
    return ortools_tour(R, stops), 'or-tools'


# ─────────────────────────────────────────────────────────────────────────────

def bucket(k):
    return '1' if k == 1 else '2-3' if k <= 3 else '4-7' if k <= 7 else '8-15' if k <= 15 else '16+'


def run(test_path='orders_test.json', dm_path='distance_matrix.json', graph_path='warehouse_graph.json'):
    tag = test_path.replace('orders_', '').replace('.json', '')      # 'test' or 'val'
    orders = json.load(open(test_path, encoding='utf-8'))['orders']
    R = Router(json.load(open(dm_path, encoding='utf-8')))
    CR = ClassicRouter(graph_path)
    rows, routes = [], {}
    for k, o in enumerate(orders, 1):
        stops = stops_for(o)
        res, seqs = {}, {}

        seqs['location_order'] = location_order(R, stops)
        res['location_order'] = R.tour_cost(seqs['location_order'])
        for m in CLASSIC:
            c = CR.route(stops, m)
            res[m] = (c['dist_m'], c['stairs'])
            seqs[m] = pick_sequence(stops, c['waypoints'])
        seqs['nearest_neighbour'] = nearest_neighbour(R, stops)
        res['nearest_neighbour'] = R.tour_cost(seqs['nearest_neighbour'])
        seqs['optimal'], how = optimal(R, stops)
        res['optimal'] = R.tour_cost(seqs['optimal'])

        # guard: "optimal" must never lose to another method (OR-Tools has a time limit).
        # A classic route's pick sequence walked by shortest paths is never longer than the route.
        for m in METHODS[:-1]:
            alt = R.tour_cost(seqs[m])
            if alt[0] < res['optimal'][0] - 1e-6:
                seqs['optimal'], res['optimal'], how = seqs[m], alt, how + f' (improved by {m})'

        rows.append({'so_no': o['so_no'], 'date': o['date'], 'lines': o['n_lines'], 'stops': len(stops),
                     'size_bucket': bucket(o['n_lines']), 'optimal_method': how,
                     **{f'{m}_m': res[m][0] for m in METHODS},
                     **{f'{m}_stairs': res[m][1] for m in METHODS}})
        routes[o['so_no']] = seqs
        if k % 100 == 0:
            print(f'  routed {k}/{len(orders)} orders', file=sys.stderr)

    with open(f'baseline_results_{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(routes, open(f'baseline_routes_{tag}.json', 'w', encoding='utf-8'))

    def summary(rs):
        tot = {m: sum(r[f'{m}_m'] for r in rs) for m in METHODS}
        return {
            'orders': len(rs),
            'mean_stops': round(st.mean(r['stops'] for r in rs), 2),
            'mean_m': {m: round(tot[m] / len(rs), 1) for m in METHODS},
            'total_km': {m: round(tot[m] / 1000, 2) for m in METHODS},
            'pct_above_optimal': {m: round(100 * (tot[m] / tot['optimal'] - 1), 1) for m in METHODS},
            'mean_stair_trips': {m: round(st.mean(r[f'{m}_stairs'] for r in rs), 2) for m in METHODS},
            'orders_beating_nearest_neighbour': {m: sum(1 for r in rs if r[f'{m}_m'] < r['nearest_neighbour_m'] - 0.01)
                                                 for m in METHODS},
        }
    by_b = defaultdict(list)
    for r in rows:
        by_b[r['size_bucket']].append(r)
    out = {'test_set': test_path, 'methods': METHODS, 'all': summary(rows),
           'by_order_size': {b: summary(by_b[b]) for b in ['1', '2-3', '4-7', '8-15', '16+'] if by_b[b]},
           'optimal_solver': {'exact (Held-Karp)': sum(r['optimal_method'].startswith('exact') for r in rows),
                              'OR-Tools guided local search': sum(r['optimal_method'].startswith('or-tools') for r in rows),
                              'improved by another method': sum('improved' in r['optimal_method'] for r in rows)}}
    json.dump(out, open(f'baseline_summary_{tag}.json', 'w', encoding='utf-8'), indent=1)

    a = out['all']
    print('=' * 78)
    print(f"  {tag.upper()} set: {a['orders']} orders, {a['mean_stops']} stops per order on average")
    print(f"  {'method':20s} {'mean m/order':>12s} {'total km':>9s} {'vs optimal':>11s} {'stair trips':>12s} {'beats NN':>9s}")
    for m in METHODS:
        print(f"  {LABEL[m]:20s} {a['mean_m'][m]:12.1f} {a['total_km'][m]:9.2f} {a['pct_above_optimal'][m]:10.1f}% "
              f"{a['mean_stair_trips'][m]:12.2f} {a['orders_beating_nearest_neighbour'][m]:9d}")
    print('-' * 78)
    print('  % above optimal by order size')
    print(f"  {'lines':8s} {'orders':>6s} " + ' '.join(f'{SHORT[m]:>6s}' for m in METHODS[:-1]) + f" {'opt m':>7s}")
    for b, s in out['by_order_size'].items():
        print(f"  {b:8s} {s['orders']:6d} " +
              ' '.join(f"{s['pct_above_optimal'][m]:+6.1f}" for m in METHODS[:-1]) +
              f" {s['mean_m']['optimal']:7.1f}")
    print('=' * 78)
    o = out['optimal_solver']
    print(f"  Optimal solved exactly for {o['exact (Held-Karp)']} orders, OR-Tools for "
          f"{o['OR-Tools guided local search']} (improved by another method: {o['improved by another method']})")
    print(f'  Wrote baseline_results_{tag}.csv, baseline_summary_{tag}.json, baseline_routes_{tag}.json')


if __name__ == '__main__':
    for path in (sys.argv[1:] or ['orders_val.json', 'orders_test.json']):
        run(path)
