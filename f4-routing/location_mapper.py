"""
F4 Location Mapper — Fine Enterprises Warehouse Intelligence System
====================================================================
Converts real bin codes from ACCESSORIES-STOCK_LOCATIONS.xlsx
  Format: {SECT}-{NUMBER}  (e.g.  B-01, X-03, G-25, TF-55)
To F4 warehouse graph node format:
  Format: A{aisle}-B{block}-L{loc}  (e.g. A01-B03-L02)

Graph node anatomy:
  aisle  (A##) = physical corridor the rack faces
  block  (B##) = rack unit / column within the aisle (derived from section + slot range)
  loc    (L##) = shelf level / position within the block

Floor encoding:
  Ground floor sections → aisle A01–A12
  Mezzanine sections    → aisle M01–M06 (M = mezzanine, travel via staircase)
  Unknown / special     → aisle U00

Staircase travel penalty (in edges, not in node id):
  SC1 (x≈195, y≈345): connects ground main-aisle ↔ mezzanine V, S, X05-m
  SC2 (x≈65,  y≈555): connects ground ↔ mezzanine X04, T-upper, U-upper, Z
  SC3 (x≈640, y≈545): connects ground ↔ mezzanine W
"""

import openpyxl
import json
import math
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 1. SECTION → AISLE  MAPPING
#    Each entry: aisle_id, floor ('G' or 'M'), staircase (for mezzanine),
#                x_pos/y_pos (approximate canvas position in v3, 860×780 px)
#                max_slots (total slot count for block/loc split)
# ─────────────────────────────────────────────────────────────────────────────

SECTION_META = {
    # ── Ground floor ─────────────────────────────────────────────────────────
    # Top cluster (y < 340 on map)
    'B':  {'aisle': 'A01', 'floor': 'G', 'x': 65,  'y': 0,   'max_slots': 36, 'block_size': 6},
    'C':  {'aisle': 'A02', 'floor': 'G', 'x': 105, 'y': 0,   'max_slots': 34, 'block_size': 6},
    'A':  {'aisle': 'A02', 'floor': 'G', 'x': 195, 'y': 0,   'max_slots': 31, 'block_size': 6},
    'D':  {'aisle': 'A03', 'floor': 'G', 'x': 235, 'y': 0,   'max_slots': 33, 'block_size': 6},
    'E':  {'aisle': 'A03', 'floor': 'G', 'x': 325, 'y': 0,   'max_slots': 27, 'block_size': 6},
    'R':  {'aisle': 'A04', 'floor': 'G', 'x': 370, 'y': 0,   'max_slots': 18, 'block_size': 6},
    'Q':  {'aisle': 'A04', 'floor': 'G', 'x': 370, 'y': 190, 'max_slots': 3,  'block_size': 3},
    'P':  {'aisle': 'A04', 'floor': 'G', 'x': 370, 'y': 265, 'max_slots': 7,  'block_size': 4},
    # X-series bins (X-01..X-09 are all in the X zone)
    'X':  {'aisle': 'A05', 'floor': 'G', 'x': 470, 'y': 0,   'max_slots': 9,  'block_size': 1,
           'note': 'X-01..X-09 are distinct bin units; slot number IS the block'},
    'X03':{'aisle': 'A05', 'floor': 'G', 'x': 470, 'y': 0,   'max_slots': 1,  'block_size': 1,
           'note': 'X03 bin cabinet (legacy code without dash)'},
    'O':  {'aisle': 'A06', 'floor': 'G', 'x': 665, 'y': 0,   'max_slots': 17, 'block_size': 6},
    'N':  {'aisle': 'A06', 'floor': 'G', 'x': 720, 'y': 175, 'max_slots': 12, 'block_size': 4},
    'M':  {'aisle': 'A06', 'floor': 'G', 'x': 720, 'y': 295, 'max_slots': 12, 'block_size': 4},
    'L':  {'aisle': 'A06', 'floor': 'G', 'x': 720, 'y': 440, 'max_slots': 12, 'block_size': 4},
    # Bottom cluster (y > 385 on map)
    'G':  {'aisle': 'A07', 'floor': 'G', 'x': 185, 'y': 385, 'max_slots': 30, 'block_size': 6},
    'T':  {'aisle': 'A08', 'floor': 'G', 'x': 375, 'y': 385, 'max_slots': 8,  'block_size': 4},
    'U':  {'aisle': 'A08', 'floor': 'G', 'x': 375, 'y': 480, 'max_slots': 10, 'block_size': 5},
    # Floor-level rows (low racks / bins)
    'I':  {'aisle': 'A09', 'floor': 'G', 'x': 65,  'y': 590, 'max_slots': 6,  'block_size': 3},
    'J':  {'aisle': 'A09', 'floor': 'G', 'x': 65,  'y': 730, 'max_slots': 15, 'block_size': 5},
    'K':  {'aisle': 'A09', 'floor': 'G', 'x': 415, 'y': 730, 'max_slots': 6,  'block_size': 3},
    'F':  {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Excluded — position not on map yet'},
    'H':  {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Excluded — position not on map yet'},
    'Y':  {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Excluded — position not on map yet'},
    'AG': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Excluded — position not on map yet'},
    'OSL':{'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Out-of-System Location — not routable'},
    'NEW':{'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'New unassigned stock — not routable'},
    # ── Mezzanine floor ──────────────────────────────────────────────────────
    # Via SC1 (x≈195, y≈345 on ground map)
    'V':  {'aisle': 'M01', 'floor': 'M', 'stair': 'SC1', 'x': 195, 'y': 0,   'max_slots': 4,  'block_size': 2},
    'S':  {'aisle': 'M01', 'floor': 'M', 'stair': 'SC1', 'x': 360, 'y': 0,   'max_slots': 9,  'block_size': 3},
    # Via SC2 (x≈65, y≈555)
    'Z':  {'aisle': 'M02', 'floor': 'M', 'stair': 'SC2', 'x': 65,  'y': 730, 'max_slots': 52, 'block_size': 8},
    # Via SC3 (x≈640, y≈545)
    'W':  {'aisle': 'M03', 'floor': 'M', 'stair': 'SC3', 'x': 810, 'y': 0,   'max_slots': 27, 'block_size': 6},
    # TF (Top Floor) and 2F (2nd Floor) — large mezzanine zones
    # TF slots 1-108, likely spread across multiple mezzanine aisles
    'TF': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    '2F': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    'AA': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    'AB': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    'AC': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    'AD': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    'AE': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
    'AF': {'aisle': 'U00', 'floor': 'G', 'max_slots': 1, 'block_size': 1, 'note': 'Separate 2F area — excluded from routing for now'},
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. SLOT → BLOCK + LOC  FORMULA
#    slot_number → (block_index, loc_index)
#    block_size = number of loc positions per block (shelf faces per bay)
#    block = ceil(slot / block_size)
#    loc   = ((slot - 1) % block_size) + 1
# ─────────────────────────────────────────────────────────────────────────────

def slot_to_block_loc(slot_num: int, block_size: int) -> tuple[int, int]:
    """Convert a slot number to (block, loc) using fixed block size."""
    block = math.ceil(slot_num / block_size)
    loc   = ((slot_num - 1) % block_size) + 1
    return block, loc


def parse_location(raw_loc: str) -> dict:
    """
    Parse a raw location string like 'B-07', 'X-03', 'TF-55', '2F-A09', 'X03'
    Returns dict with keys: section, slot_raw, slot_num, is_valid
    """
    raw_loc = str(raw_loc).strip()

    # Special cases first
    if raw_loc in ('', 'None', 'NEW', 'OSL'):
        return {'section': raw_loc or 'UNKNOWN', 'slot_raw': '0', 'slot_num': 0, 'is_valid': False}

    if raw_loc == 'X03':  # legacy code without dash
        return {'section': 'X03', 'slot_raw': '1', 'slot_num': 1, 'is_valid': True}

    if raw_loc == '2F':   # just "2F" with no slot
        return {'section': '2F', 'slot_raw': '0', 'slot_num': 0, 'is_valid': True}

    if raw_loc == 'OSL':
        return {'section': 'OSL', 'slot_raw': '0', 'slot_num': 0, 'is_valid': True}

    parts = raw_loc.split('-', 1)
    if len(parts) < 2:
        return {'section': raw_loc, 'slot_raw': '0', 'slot_num': 0, 'is_valid': False}

    section = parts[0].strip()
    slot_raw = parts[1].strip()

    # Handle 2F-A09 style: strip the letter prefix from slot
    # 2F-A01 → slot_num = 1, 2F-31 → slot_num = 31
    slot_clean = slot_raw.lstrip('A')  # remove leading 'A' if present
    try:
        slot_num = int(slot_clean)
    except ValueError:
        # Handle ranges like "2F-30/2F-31" → use first
        try:
            slot_num = int(slot_clean.split('/')[0])
        except:
            slot_num = 0

    return {
        'section': section,
        'slot_raw': slot_raw,
        'slot_num': slot_num,
        'is_valid': True
    }


def location_to_graph_node(raw_loc: str) -> dict:
    """
    Convert a raw location code to a graph node descriptor.
    Returns dict with: graph_node, aisle_id, block, loc, floor, section,
                       is_mezzanine, staircase, routable, original_loc
    """
    parsed = parse_location(raw_loc)
    section = parsed['section']
    slot_num = parsed['slot_num']

    meta = SECTION_META.get(section)

    if not meta or not parsed['is_valid'] or section in ('UNKNOWN', 'OSL', 'NEW') or meta.get('aisle') == 'U00':
        return {
            'graph_node': 'U00-B00-L00',
            'aisle_id': 'U00',
            'block': 0, 'loc': 0,
            'floor': 'G',
            'section': section,
            'is_mezzanine': False,
            'staircase': None,
            'routable': False,
            'original_loc': raw_loc,
            'note': f'Unroutable: {section}'
        }

    aisle_id   = meta['aisle']
    block_size = meta['block_size']
    floor      = meta['floor']
    is_mezz    = (floor == 'M')
    staircase  = meta.get('stair')

    # Special handling for X-section: slot number IS the unit (X-01..X-09)
    if section in ('X', 'X03'):
        block = slot_num  # X-01 → block 1, X-03 → block 3
        loc   = 1
    elif slot_num == 0:
        block, loc = 0, 0
    else:
        block, loc = slot_to_block_loc(slot_num, block_size)

    graph_node = f"{aisle_id}-B{block:02d}-L{loc:02d}"

    return {
        'graph_node': graph_node,
        'aisle_id':   aisle_id,
        'block':      block,
        'loc':        loc,
        'floor':      floor,
        'section':    section,
        'is_mezzanine': is_mezz,
        'staircase':  staircase,
        'routable':   True,
        'original_loc': raw_loc,
        'x_approx':   meta.get('x'),
        'y_approx':   meta.get('y'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN — READ XLSX AND PRODUCE item_locations_real.json
# ─────────────────────────────────────────────────────────────────────────────

def run(xlsx_path: str, output_path: str):
    print(f"Reading: {xlsx_path}")
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active

    items = []
    errors = []
    section_stats = defaultdict(int)

    for r in range(2, ws.max_row + 1):
        group    = ws.cell(r, 1).value
        code     = ws.cell(r, 2).value
        name     = ws.cell(r, 3).value
        qty_raw  = ws.cell(r, 4).value
        loc_raw  = ws.cell(r, 5).value

        if not code:
            continue

        code = str(code).strip()
        loc_raw_str = str(loc_raw).strip() if loc_raw else ''

        # Parse quantity
        try:
            qty = int(float(str(qty_raw))) if qty_raw and str(qty_raw).strip() not in ('', ' ') else 0
        except:
            qty = 0

        # Map location to graph node
        node_info = location_to_graph_node(loc_raw_str)

        item = {
            'sku':          code,
            'item_name':    str(name).strip() if name else '',
            'group':        str(group).strip() if group else '',
            'qty':          qty,
            'original_loc': loc_raw_str,
            'graph_node':   node_info['graph_node'],
            'section':      node_info['section'],
            'aisle_id':     node_info['aisle_id'],
            'block':        node_info['block'],
            'loc':          node_info['loc'],
            'floor':        node_info['floor'],
            'is_mezzanine': node_info['is_mezzanine'],
            'staircase':    node_info['staircase'],
            'x_approx':     node_info.get('x_approx'),
            'y_approx':     node_info.get('y_approx'),
            'routable':     node_info['routable'],
        }
        items.append(item)
        section_stats[node_info['section']] += 1

        if not node_info['routable']:
            errors.append({'row': r, 'code': code, 'loc': loc_raw_str, 'reason': node_info.get('note','')})

    # Summary stats
    total        = len(items)
    routable     = sum(1 for it in items if it['routable'])
    mezzanine    = sum(1 for it in items if it['is_mezzanine'])
    zero_stock   = sum(1 for it in items if it['qty'] == 0)

    # Unique graph nodes
    unique_nodes = set(it['graph_node'] for it in items if it['routable'])

    output = {
        'metadata': {
            'source_file':     'ACCESSORIES-STOCK_LOCATIONS.xlsx',
            'total_items':     total,
            'routable_items':  routable,
            'unroutable_items':total - routable,
            'mezzanine_items': mezzanine,
            'zero_stock_items':zero_stock,
            'unique_graph_nodes': len(unique_nodes),
            'graph_node_format': 'A{aisle}-B{block:02d}-L{loc:02d}',
            'aisle_prefix': {
                'A01': 'Ground - Left-wall corridor / B section',
                'A02': 'Ground - C / A rack aisle (top cluster)',
                'A03': 'Ground - D / E rack aisle (top cluster)',
                'A04': 'Ground - R / Q / P rack aisle',
                'A05': 'Ground - X-series bins (X-01..X-09)',
                'A06': 'Ground - O / N / M / L rack aisle (right wall)',
                'A07': 'Ground - G-group rack aisle (bottom cluster)',
                'A08': 'Ground - T / U rack aisle (bottom cluster)',
                'A09': 'Ground - I / J / K floor-level racks',
                'A10': 'Ground - F section (unmapped)',
                'A11': 'Ground - H section (unmapped)',
                'A12': 'Ground - Y section (unmapped)',
                'M01': 'Mezzanine - V / S sections (via SC1)',
                'M02': 'Mezzanine - Z section (via SC2)',
                'M03': 'Mezzanine - W section (via SC3)',
                'M04': 'Mezzanine - TF Top-Floor (via SC1)',
                'M05': 'Mezzanine - 2F 2nd-Floor (via SC2)',
                'M06': 'Mezzanine - AA/AB/AC/AD/AE/AF extensions (via SC1)',
                'U00': 'Unroutable / special locations',
            },
            'staircase_access': {
                'SC1': 'Ground (195, 345) → Mezzanine V, S, TF, AA-AF',
                'SC2': 'Ground (65, 555)  → Mezzanine Z, 2F, T-upper, U-upper',
                'SC3': 'Ground (640, 545) → Mezzanine W',
            },
        },
        'items': items,
        'unroutable_entries': errors,
        'section_item_counts': dict(sorted(section_stats.items())),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Total items:         {total:,}")
    print(f"  Routable:            {routable:,}  ({100*routable/total:.1f}%)")
    print(f"  Unroutable/special:  {total-routable:,}")
    print(f"  Mezzanine items:     {mezzanine:,}")
    print(f"  Zero-stock items:    {zero_stock:,}")
    print(f"  Unique graph nodes:  {len(unique_nodes):,}")
    print(f"{'='*60}")

    print("\nSection breakdown:")
    for sect in sorted(section_stats.keys()):
        meta = SECTION_META.get(sect, {})
        aisle = meta.get('aisle', 'U00')
        floor = meta.get('floor', '?')
        cnt = section_stats[sect]
        print(f"  {sect:5s} → {aisle}  [{floor}]  {cnt:4d} items")

    print(f"\nSample graph nodes (first 15 items):")
    for it in items[:15]:
        print(f"  {it['sku']:8s}  {it['original_loc']:12s} → {it['graph_node']}")


if __name__ == '__main__':
    XLSX = 'ACCESSORIES-STOCK_LOCATIONS.xlsx'
    OUT  = 'item_locations_real.json'
    run(XLSX, OUT)
