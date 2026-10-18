"""
F4 Warehouse Graph Builder (v2, walkway-based) — Fine Enterprises WIS
=====================================================================
Builds warehouse_graph.json from warehouse_layout.py.

Nodes = places a picker can stand (pick stops, junctions, stair feet/tops,
        DEPOT, DOOR_2F). Each pick stop lists the slots reachable from it.
Route rule: every pick tour starts at DEPOT and returns to DEPOT.
DOOR_2F (to the 2nd floor) is kept for later but unused for now.
Edges = walking between neighbouring stand-points along a walkway
        (weight in metres) + staircase edges (stair length in metres).

Run:  python build_warehouse_graph.py
"""

import json

import warehouse_layout as L
import warehouse_model as WM


def run(output_path='warehouse_graph.json'):
    slots = WM.build_slot_index()
    nodes, edges = WM.build_graph(slots)

    adj = WM.adjacency(edges)
    dist = WM.shortest_paths(adj, L.ROUTE_START)
    unreachable = [n for n in nodes if n not in dist]

    pick_nodes = [n for n, v in nodes.items() if v['pick_slots']]
    weights = [e['weight'] for e in edges]

    graph = {
        'metadata': {
            'description': 'Fine Enterprises warehouse — F4 RL routing graph (v2, walkway-based)',
            'scale': f'warehouse {L.WAREHOUSE_WIDTH_M:g} m x {L.WAREHOUSE_DEPTH_M:g} m on an 860x780 px canvas '
                     f'({L.SCALE_X_M_PER_PX:.4f} m/px across, {L.SCALE_Y_M_PER_PX:.4f} m/px down)',
            'route_rule': 'Every pick tour starts at DEPOT and returns to DEPOT (handover to checkers).',
            'route_start': L.ROUTE_START, 'route_end': L.ROUTE_END,
            'inactive_nodes': {'DOOR_2F': 'Door to 2nd floor (TF, 2F, AA-AF) - unused while those sections are excluded'},
            'node_meaning': 'A place a picker can stand. pick_slots = slots reachable without walking.',
            'edge_types': {
                'walk': 'Walking along a ground-floor walkway between neighbouring stand-points (m)',
                'mezz_walk': 'Walking on the mezzanine (m, approximate)',
                'stair': f'Staircase ground <-> mezzanine ({L.STAIR_LENGTH_M} m per trip, effort factor {L.STAIR_EFFORT_FACTOR})',
            },
            'stair_length_m': L.STAIR_LENGTH_M, 'stair_effort_factor': L.STAIR_EFFORT_FACTOR,
            'node_count': len(nodes),
            'pick_node_count': len(pick_nodes),
            'edge_count_undirected': len(edges),
            'avg_edge_weight_m': round(sum(weights) / len(weights), 2),
            'unreachable_nodes': unreachable,
            'excluded_sections': sorted(L.EXCLUDED_SECTIONS),
            'assumptions': L.ASSUMPTIONS,
        },
        'nodes': nodes,
        'edges': edges,   # undirected: each edge usable both ways
        'slots': {s['code']: {k: v for k, v in s.items() if k != 'stops'}
                  for s in slots.values()},
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print('=' * 60)
    print(f'  Output:            {output_path}')
    print(f'  Nodes:             {len(nodes)}  (pick stops: {len(pick_nodes)})')
    print(f'  Edges:             {len(edges)}  (undirected)')
    print(f'  Slots modelled:    {len(slots)}')
    print(f'  Avg edge weight:   {graph["metadata"]["avg_edge_weight_m"]} m')
    print(f'  Unreachable nodes: {len(unreachable)}')
    print('=' * 60)
    print('\nExample stand-points:')
    for nid in ['DEPOT', 'W1-02', 'W1-03', 'W4-02', 'W7-03']:
        if nid in nodes:
            n = nodes[nid]
            print(f'  {nid:10s} picks {", ".join(n["pick_slots"][:8]) or "-"}')
    return graph


if __name__ == '__main__':
    run()
