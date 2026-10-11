"""
F4 Warehouse Graph Builder — Fine Enterprises WIS
==================================================
Builds warehouse_graph.json from item_locations_real.json.

Graph structure:
  Nodes: every unique graph_node (A01-B03-L02, M04-B05-L07, etc.)
  Edges: adjacency between walkable positions with physical distance as weight

Edge types:
  - loc_step:   adjacent loc within same block (L01↔L02), weight ≈ 0.3m (shelf)
  - block_step: adjacent block within same aisle (B01↔B02), weight ≈ 1.5m (bay width)
  - aisle_step: connection between aisles on same floor (weight = Euclidean px distance × scale)
  - stair_step: staircase connection ground ↔ mezzanine (fixed penalty = 12m)
  - entrance:   virtual node "ENTRANCE" connected to A07-B01-L01 (nearest G node), weight 0
  - exit:       virtual node "EXIT"     connected to A05-B05-L01 (nearest X-05 area node), weight 0

Physical scale: canvas is 860×780 px representing approx 60m × 55m warehouse → ~0.07m/px
"""

import json
import math
from collections import defaultdict

SCALE_M_PER_PX = 60 / 860   # metres per pixel (approx)
STAIR_PENALTY  = 12.0        # metres for staircase travel

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ITEM LOCATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_items(path):
    with open(path) as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD NODE SET
# ─────────────────────────────────────────────────────────────────────────────

def build_nodes(items_data):
    """Collect all unique, routable graph nodes with metadata."""
    nodes = {}
    for it in items_data['items']:
        if not it['routable']:
            continue
        gn = it['graph_node']
        if gn not in nodes:
            nodes[gn] = {
                'node_id':      gn,
                'aisle_id':     it['aisle_id'],
                'block':        it['block'],
                'loc':          it['loc'],
                'floor':        it['floor'],
                'is_mezzanine': it['is_mezzanine'],
                'staircase':    it['staircase'],
                'x_approx':     it['x_approx'],
                'y_approx':     it['y_approx'],
                'sku_count':    0,
                'total_qty':    0,
            }
        nodes[gn]['sku_count'] += 1
        nodes[gn]['total_qty'] += (it['qty'] or 0)

    # Add virtual entrance / exit nodes
    nodes['ENTRANCE'] = {
        'node_id': 'ENTRANCE', 'aisle_id': 'ENTRANCE', 'block': 0, 'loc': 0,
        'floor': 'G', 'is_mezzanine': False, 'staircase': None,
        'x_approx': 0, 'y_approx': 362, 'sku_count': 0, 'total_qty': 0
    }
    nodes['EXIT'] = {
        'node_id': 'EXIT', 'aisle_id': 'EXIT', 'block': 0, 'loc': 0,
        'floor': 'G', 'is_mezzanine': False, 'staircase': None,
        'x_approx': 642, 'y_approx': 0, 'sku_count': 0, 'total_qty': 0
    }
    return nodes

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD EDGES
# ─────────────────────────────────────────────────────────────────────────────

def euclid_m(n1, n2):
    """Physical distance in metres based on approximate canvas coordinates."""
    x1 = n1.get('x_approx') or 0
    y1 = n1.get('y_approx') or 0
    x2 = n2.get('x_approx') or 0
    y2 = n2.get('y_approx') or 0
    px_dist = math.hypot(x2 - x1, y2 - y1)
    return max(0.3, round(px_dist * SCALE_M_PER_PX, 2))


def build_edges(nodes):
    edges = []
    node_ids = list(nodes.keys())

    # Group nodes by aisle for internal connectivity
    by_aisle = defaultdict(list)
    for nid, n in nodes.items():
        if nid not in ('ENTRANCE', 'EXIT'):
            by_aisle[n['aisle_id']].append(nid)

    # ── Intra-aisle edges (loc_step and block_step) ──────────────────────────
    for aisle, nids in by_aisle.items():
        # Sort by (block, loc)
        nids_sorted = sorted(nids, key=lambda n: (nodes[n]['block'], nodes[n]['loc']))

        # loc_step: adjacent L within same B
        block_groups = defaultdict(list)
        for nid in nids_sorted:
            block_groups[nodes[nid]['block']].append(nid)

        for block, bnids in block_groups.items():
            bnids.sort(key=lambda n: nodes[n]['loc'])
            for i in range(len(bnids) - 1):
                edges.append({
                    'from': bnids[i], 'to': bnids[i+1],
                    'type': 'loc_step', 'weight': 0.3
                })

        # block_step: adjacent B within aisle (connect min-loc of each block)
        blocks = sorted(block_groups.keys())
        for i in range(len(blocks) - 1):
            n1 = block_groups[blocks[i]][0]    # first loc of block i
            n2 = block_groups[blocks[i+1]][0]  # first loc of block i+1
            edges.append({
                'from': n1, 'to': n2,
                'type': 'block_step', 'weight': 1.5
            })

    # ── Aisle-to-aisle edges (inter-aisle on same floor) ─────────────────────
    # Connect representative "head" node of each aisle to adjacent aisles
    # Based on physical proximity from the warehouse map
    # Ground floor aisle adjacency (from map layout):
    aisle_adjacency_ground = [
        ('A01', 'A02'),  # B/AG → C/A corridor
        ('A01', 'A07'),  # left wall top ↔ bottom (via main cross-aisle)
        ('A02', 'A03'),  # C/A → D/E
        ('A03', 'A04'),  # D/E → R/Q/P
        ('A04', 'A05'),  # R/Q/P → X bins
        ('A05', 'A06'),  # X bins → O/N/M/L (right wall)
        ('A06', 'A07'),  # right wall → G bottom (via cross-aisle)
        ('A07', 'A08'),  # G → T/U
        ('A07', 'A09'),  # G → I/J/K
        ('A08', 'A09'),  # T/U → I/J/K
        ('A08', 'A05'),  # T/U → X-02
        ('A02', 'A07'),  # top cluster → bottom (main cross-aisle shortcut)
        ('A03', 'A08'),  # D/E → bottom T/U
        ('A04', 'A07'),  # R → G-01 area
        # Unmapped sections connected to nearest plausible aisle
        ('A10', 'A03'),  # F section → assumed near C/D/E area
        ('A11', 'A02'),  # H section → assumed near C/A area
        ('A12', 'A06'),  # Y section → assumed near N/M/L (slots Y-22..Y-36)
    ]
    # Mezzanine inter-aisle
    aisle_adjacency_mezz = [
        ('M01', 'M04'),  # V/S ↔ TF (same top-floor zone via SC1)
        ('M01', 'M06'),  # V/S ↔ AA-AF extensions
        ('M04', 'M06'),
        ('M02', 'M05'),  # Z ↔ 2F (same lower-mezz zone via SC2)
    ]

    all_aisle_adj = aisle_adjacency_ground + aisle_adjacency_mezz

    # Find head nodes (block=1, loc=1) for each aisle
    aisle_head = {}
    for nid, n in nodes.items():
        if nid in ('ENTRANCE', 'EXIT'):
            continue
        aisle = n['aisle_id']
        if aisle not in aisle_head or (n['block'], n['loc']) < (nodes[aisle_head[aisle]]['block'], nodes[aisle_head[aisle]]['loc']):
            aisle_head[aisle] = nid

    for a1, a2 in all_aisle_adj:
        if a1 in aisle_head and a2 in aisle_head:
            n1 = nodes[aisle_head[a1]]
            n2 = nodes[aisle_head[a2]]
            dist = euclid_m(n1, n2)
            edges.append({
                'from': aisle_head[a1], 'to': aisle_head[a2],
                'type': 'aisle_step', 'weight': dist
            })

    # ── Staircase edges (ground ↔ mezzanine) ─────────────────────────────────
    # SC1 (195, 345): ground A02/A03/A04 → mezzanine M01, M04, M06
    # SC2 (65, 555):  ground A01/A09     → mezzanine M02, M05
    # SC3 (640, 545): ground A05/A06     → mezzanine M03

    stair_connections = {
        'SC1': {
            'ground_aisles': ['A02', 'A03', 'A04'],
            'mezz_aisles':   ['M01', 'M04', 'M06'],
        },
        'SC2': {
            'ground_aisles': ['A01', 'A09'],
            'mezz_aisles':   ['M02', 'M05'],
        },
        'SC3': {
            'ground_aisles': ['A05', 'A06'],
            'mezz_aisles':   ['M03'],
        },
    }

    for stair, conn in stair_connections.items():
        ground_heads = [aisle_head[a] for a in conn['ground_aisles'] if a in aisle_head]
        mezz_heads   = [aisle_head[a] for a in conn['mezz_aisles']   if a in aisle_head]

        if ground_heads and mezz_heads:
            # Connect the closest ground head to each mezzanine head
            g_head = ground_heads[0]
            for m_head in mezz_heads:
                edges.append({
                    'from': g_head, 'to': m_head,
                    'type': 'stair_step',
                    'weight': STAIR_PENALTY,
                    'staircase': stair
                })

    # ── Entrance / exit virtual edges ─────────────────────────────────────────
    # Entrance (left wall, y≈362) → nearest A01 node (B section / AG section)
    a01_nodes = [nid for nid, n in nodes.items()
                 if n.get('aisle_id') == 'A01' and n.get('floor') == 'G']
    if a01_nodes:
        nearest_a01 = min(a01_nodes,
                          key=lambda n: abs((nodes[n].get('y_approx') or 0) - 362))
        edges.append({'from': 'ENTRANCE', 'to': nearest_a01, 'type': 'entrance_exit', 'weight': 0.5})

    # Exit (top wall, x≈642) → nearest A05 node (X-05 area)
    a05_nodes = [nid for nid, n in nodes.items()
                 if n.get('aisle_id') == 'A05' and n.get('floor') == 'G']
    if a05_nodes:
        nearest_a05 = min(a05_nodes,
                          key=lambda n: abs((nodes[n].get('x_approx') or 0) - 642))
        edges.append({'from': 'EXIT', 'to': nearest_a05, 'type': 'entrance_exit', 'weight': 0.5})

    # Also connect exit to A06 (O section is right next to exit)
    a06_nodes = [nid for nid, n in nodes.items()
                 if n.get('aisle_id') == 'A06' and n.get('floor') == 'G']
    if a06_nodes:
        nearest_a06 = min(a06_nodes, key=lambda n: abs((nodes[n].get('x_approx') or 0) - 665))
        edges.append({'from': 'EXIT', 'to': nearest_a06, 'type': 'entrance_exit', 'weight': 0.5})

    return edges

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(items_path, output_path):
    items_data = load_items(items_path)

    print("Building node set...")
    nodes = build_nodes(items_data)
    print(f"  Nodes: {len(nodes)}")

    print("Building edges...")
    edges = build_edges(nodes)
    print(f"  Edges (directed): {len(edges)}")

    # Make undirected: add reverse edges
    reverse = []
    for e in edges:
        reverse.append({'from': e['to'], 'to': e['from'],
                        'type': e['type'], 'weight': e['weight'],
                        **(({'staircase': e['staircase']} if 'staircase' in e else {}))})
    all_edges = edges + reverse
    print(f"  Edges (undirected): {len(all_edges)}")

    # Edge weight stats
    weights = [e['weight'] for e in all_edges]
    avg_w = sum(weights) / len(weights) if weights else 0

    graph = {
        'metadata': {
            'description': 'Fine Enterprises Warehouse — F4 RL routing graph',
            'scale': f'{SCALE_M_PER_PX:.4f} m/px (canvas 860×780 px ≈ 60m×55m)',
            'stair_penalty_m': STAIR_PENALTY,
            'node_count': len(nodes),
            'edge_count': len(all_edges),
            'edge_types': {
                'loc_step':    'Adjacent shelf position within same rack block (≈0.3m)',
                'block_step':  'Adjacent rack bay within same aisle (≈1.5m)',
                'aisle_step':  'Between aisles on same floor (Euclidean, varies)',
                'stair_step':  f'Ground↔Mezzanine via staircase (fixed {STAIR_PENALTY}m penalty)',
                'entrance_exit': 'Virtual ENTRANCE/EXIT connection (0.5m)',
            },
            'avg_edge_weight_m': round(avg_w, 2),
            'virtual_nodes': ['ENTRANCE', 'EXIT'],
        },
        'nodes': nodes,
        'edges': all_edges,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Nodes:               {len(nodes):,}")
    print(f"  Edges (undirected):  {len(all_edges):,}")
    print(f"  Avg edge weight:     {avg_w:.2f} m")
    print(f"{'='*60}")

    # Sample paths for sanity check
    print("\nSample nodes:")
    for nid in ['ENTRANCE', 'EXIT', 'A01-B01-L01', 'A05-B03-L01', 'M04-B01-L01', 'M02-B01-L01']:
        if nid in nodes:
            n = nodes[nid]
            print(f"  {nid:20s} floor={n['floor']} aisles={n['aisle_id']} skus={n['sku_count']} qty={n['total_qty']}")

    print("\nSample edges:")
    for e in all_edges[:10]:
        print(f"  {e['from']:22s} → {e['to']:22s}  [{e['type']:12s}]  {e['weight']}m")


if __name__ == '__main__':
    ITEMS  = 'item_locations_real.json'
    OUTPUT = 'warehouse_graph.json'
    run(ITEMS, OUTPUT)
