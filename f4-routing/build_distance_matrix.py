"""
F4 Step 4 — Distance matrix between pick stand-points
=====================================================
Shortest walking distance (metres, stair length included) and number of
staircase trips between every pair of {DEPOT + all pick stand-points}.
Computed once so routing methods and the RL environment cost a route with
table lookups instead of graph searches.

Input : warehouse_graph.json
Output: distance_matrix.json
          nodes  : list of node ids (index 0 = DEPOT)
          dist_m : dist_m[i][j] metres from nodes[i] to nodes[j]
          stairs : stairs[i][j] staircase trips on that shortest path
Run   : python build_distance_matrix.py
"""

import heapq
import json
from collections import defaultdict


def dijkstra(adj, src):
    dist, stairs = {src: 0.0}, {src: 0}
    pq = [(0.0, 0, src)]
    while pq:
        d, s, u = heapq.heappop(pq)
        if d > dist[u] + 1e-9:
            continue
        for v, w, is_stair in adj[u]:
            nd, ns = d + w, s + is_stair
            if nd < dist.get(v, 1e18) - 1e-9:
                dist[v], stairs[v] = nd, ns
                heapq.heappush(pq, (nd, ns, v))
    return dist, stairs


def run(graph_path='warehouse_graph.json', out='distance_matrix.json'):
    g = json.load(open(graph_path, encoding='utf-8'))
    adj = defaultdict(list)
    for e in g['edges']:
        st = 1 if e['type'] == 'stair' else 0
        adj[e['from']].append((e['to'], e['weight'], st))
        adj[e['to']].append((e['from'], e['weight'], st))

    nodes = ['DEPOT'] + sorted(n for n, v in g['nodes'].items() if v['pick_slots'])
    dist_m, stairs = [], []
    for a in nodes:
        d, s = dijkstra(adj, a)
        dist_m.append([round(d[b], 2) for b in nodes])
        stairs.append([s[b] for b in nodes])

    # symmetry check (undirected graph)
    asym = max(abs(dist_m[i][j] - dist_m[j][i]) for i in range(len(nodes)) for j in range(len(nodes)))
    far = max(range(len(nodes)), key=lambda j: dist_m[0][j])
    json.dump({'meta': {'source': graph_path, 'n_nodes': len(nodes), 'index_0': 'DEPOT',
                        'units': 'metres (stair length included)', 'max_asymmetry_m': asym},
               'nodes': nodes, 'dist_m': dist_m, 'stairs': stairs},
              open(out, 'w', encoding='utf-8'))
    print('=' * 60)
    print(f'  Nodes: {len(nodes)} (DEPOT + {len(nodes) - 1} pick stand-points)')
    print(f'  Pairs: {len(nodes) ** 2:,}   max asymmetry {asym} m')
    print(f'  Farthest from DEPOT: {nodes[far]} at {dist_m[0][far]} m ({stairs[0][far]} stair trip)')
    print(f'  Wrote {out}')
    print('=' * 60)


if __name__ == '__main__':
    run()
