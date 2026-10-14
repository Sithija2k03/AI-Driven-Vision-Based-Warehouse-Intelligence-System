"""
F4 Warehouse Layout Config (v2, walkway-based) — Fine Enterprises WIS
=====================================================================
Single source of truth for the physical layout. Edit THIS file to correct
the model; location_mapper.py and build_warehouse_graph.py read from it.

Coordinates are canvas pixels from warehouse_editor_v3 (860 x 780 px
~ 60 m x 55 m). y grows downward (y=0 is the top/north wall).

Core idea
---------
A picker stands in a WALKWAY in front of a rack COLUMN and picks any slot
in that column (bottom, top, B or C) without walking. So:
  * graph nodes  = places a picker can stand (pick stops + junctions)
  * graph edges  = walking along a walkway between neighbouring stops
  * a slot maps to the stop in front of its column (+ a level for reach)

Numbering patterns (confirmed by Sithija, 2026-10-11)
-----------------------------------------------------
  col2  : 2 slots per column, odd = bottom, even = top, then next column
          (A1/A2, A3/A4 ...). C sits above B: C levels are 3-4.
  row3  : T/U style. Bottom row 1-3, top row 4-6 (n under n+3).
          T-7/T-8, U-7..U-10 placed on the top (mezzanine) rack for now,
          stacked above T-4..6 / U-4..6 in the same 3 columns.
  stack : all slots in one column stacked full height (X-6..X-9)
  row1  : one slot per column, single level (V-1..V-3)
  single: a single bin / unit (X-1, X-2, X-4, X-5, V-4)
"""

SCALE_M_PER_PX = 60 / 860     # ~0.07 m per px
STAIR_PENALTY_M = 12.0        # walking cost of one staircase trip

# ---------------------------------------------------------------------------
# Sections deliberately left out of routing for now (separate 2nd floor or
# position not on the finalized map).
# ---------------------------------------------------------------------------
EXCLUDED_SECTIONS = {
    'TF', '2F', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF',   # separate 2nd floor
    '1F', '3F',                                        # other floors (Stock Report)
    'F', 'H', 'Y', 'AG',                               # position unknown
    'OSL', 'NEW',                                      # no fixed location
}

# ---------------------------------------------------------------------------
# WALKWAYS — centre-lines a picker walks along. (floor, [points])
# Walkways connect wherever an endpoint of one lies on another.
# ---------------------------------------------------------------------------
WALKWAYS = {
    # ---- Ground floor -----------------------------------------------------
    'MX':  ('G', [(0, 362), (642, 362)]),     # main cross aisle (depot at x=0)
    'W1':  ('G', [(170, 0), (170, 362)]),     # B/C rack  |  A
    'W2':  ('G', [(302, 0), (302, 362)]),     # D  |  E
    'W3':  ('G', [(442, 0), (442, 362)]),     # R/Q/P  |  X-3 island (west face)
    'NX':  ('G', [(442, 0), (642, 0)]),       # path north of X-3 island
    'W4':  ('G', [(642, 0), (642, 570)]),     # X-3 east / X-6..9  |  O/N/M/L wall
    'W5':  ('G', [(152, 362), (152, 570)]),   # G-25..30  |  G-19..24
    'W6':  ('G', [(285, 362), (285, 570)]),   # G-13..18  |  G-1..12
    'W7':  ('G', [(465, 362), (465, 570)]),   # T/U (ground)  |  X-2
    'LC':  ('G', [(32, 362), (32, 715)]),     # lower left-wall corridor (I faces it)
    'BX1': ('G', [(32, 570), (672, 570)]),    # cross aisle above X-1
    'BX2': ('G', [(32, 715), (765, 715)]),    # cross aisle above J/K
    # ---- Mezzanine (modelled approximately — see notes) ------------------
    'MZ1':  ('M', [(232, 340), (545, 340)]),  # top mezz front edge (SC1)
    'MZV':  ('M', [(257, 340), (257, 0)]),    # over A/D: V-1..V-3
    'MZS':  ('M', [(385, 340), (385, 0)]),    # over E/R: S
    'MZX5': ('M', [(545, 340), (545, 120)]),  # over X-3: X-5
    'MZV4': ('M', [(232, 340), (250, 470)]),  # over G-13..G-29 rack: V-4 (via SC1)
    'MZ2':  ('M', [(97, 555), (465, 555)]),   # lower mezz (SC2)
    'MZTU': ('M', [(465, 555), (465, 385)]),  # T/U top row  |  X-4
    'MZZ':  ('M', [(97, 555), (97, 750), (765, 750)]),  # Z over J/K
    'MZW':  ('M', [(672, 555), (785, 555), (785, 0)]),  # W (via SC3)
}

# Staircases: ground foot point -> mezzanine top point
STAIRS = {
    'SC1': {'foot': (232, 362), 'top': (232, 340)},
    'SC2': {'foot': (97, 570),  'top': (97, 555)},
    'SC3': {'foot': (672, 570), 'top': (672, 555)},
}

# DEPOT: the entrance point between B and G-25. Every pick tour STARTS here
# and RETURNS here, where collected items are handed over to the checkers.
DEPOT_XY   = (0, 362)
ROUTE_START = ROUTE_END = 'DEPOT'

# Door between X-5 and O-13 leading to the 2nd floor (TF, 2F, AA-AF).
# Kept in the graph for later, but NOT used for routing while those
# sections are excluded.
DOOR_2F_XY = (642, 0)

# ---------------------------------------------------------------------------
# RACKS
#   section  : location prefix in the stock file
#   slots    : (first, last) slot numbers held by this rack face
#   pattern  : numbering pattern (see top)
#   walkway  : walkway the picker stands in to pick from this face
#   line     : (start_xy, end_xy) along the rack; column 1 is at start_xy
#   levels   : level offset (C = 2 so its levels are 3-4, above B 1-2)
#   floor    : 'G' or 'M'
#   n_cols   : optional override of column count
#
#   ASSUMPTION A1: column 1 is at the end nearest the main cross aisle
#   (south end for top-cluster racks, north end for bottom-cluster racks),
#   matching O-13 being next to the 2nd-floor door at the top of the O run.
# ---------------------------------------------------------------------------
RACKS = [
    # ---- Walkway W1: B/C tall rack | A ------------------------------------
    dict(id='B',  section='B', slots=(1, 36), pattern='col2', walkway='W1',
         line=((85, 340), (85, 0)), levels=0, floor='G'),
    dict(id='C',  section='C', slots=(1, 34), pattern='col2', walkway='W1',
         line=((85, 340), (85, 0)), levels=2, floor='G', n_cols=18),  # same length as B
    dict(id='A',  section='A', slots=(1, 31), pattern='col2', walkway='W1',
         line=((215, 340), (215, 0)), levels=0, floor='G'),
    # ---- Walkway W2: D | E ------------------------------------------------
    dict(id='D',  section='D', slots=(1, 33), pattern='col2', walkway='W2',
         line=((255, 340), (255, 0)), levels=0, floor='G'),
    dict(id='E',  section='E', slots=(1, 27), pattern='col2', walkway='W2',
         line=((342, 340), (342, 0)), levels=0, floor='G'),
    # ---- Walkway W3: R/Q/P (one line, back of E) | X-3 island --------------
    dict(id='R',  section='R', slots=(1, 18), pattern='col2', walkway='W3',
         line=((390, 185), (390, 0)), levels=0, floor='G'),
    dict(id='Q',  section='Q', slots=(1, 3),  pattern='col2', walkway='W3',
         line=((390, 260), (390, 190)), levels=0, floor='G'),
    dict(id='P',  section='P', slots=(1, 7),  pattern='col2', walkway='W3',
         line=((390, 335), (390, 265)), levels=0, floor='G'),
    # ---- X-3 floor bin: walkable on all 4 sides ---------------------------
    dict(id='X3', section='X', slots=(3, 3), pattern='single', walkway='W3',
         line=((545, 120), (545, 120)), levels=0, floor='G',
         extra_access=['W4', 'NX', 'MX']),
    # ---- Walkway W4: right wall O/N/M/L (no access behind) ----------------
    dict(id='O_hi', section='O', slots=(13, 17), pattern='col2', walkway='W4',
         line=((665, 20), (715, 20)), levels=0, floor='G'),
    dict(id='O',  section='O', slots=(1, 12), pattern='col2', walkway='W4',
         line=((740, 170), (740, 0)), levels=0, floor='G'),
    dict(id='N',  section='N', slots=(1, 12), pattern='col2', walkway='W4',
         line=((740, 290), (740, 175)), levels=0, floor='G'),
    dict(id='M',  section='M', slots=(1, 12), pattern='col2', walkway='W4',
         line=((740, 435), (740, 295)), levels=0, floor='G'),
    dict(id='L',  section='L', slots=(1, 12), pattern='col2', walkway='W4',
         line=((740, 550), (740, 440)), levels=0, floor='G'),
    # X-6..X-9 stacked full height, face O/N/M/L across W4
    dict(id='X69', section='X', slots=(6, 9), pattern='stack', walkway='W4',
         line=((585, 472), (585, 472)), levels=0, floor='G'),
    # X-1 wide floor bin, beside X-6..X-9
    dict(id='X1', section='X', slots=(1, 1), pattern='single', walkway='BX1',
         line=((585, 640), (585, 640)), levels=0, floor='G'),
    # ---- Walkway W7: T/U ground row | X-2 ---------------------------------
    dict(id='X2', section='X', slots=(2, 2), pattern='single', walkway='W7',
         line=((532, 472), (532, 472)), levels=0, floor='G'),
    dict(id='T_lo', section='T', slots=(1, 3), pattern='row3', walkway='W7',
         line=((395, 385), (395, 475)), levels=0, floor='G'),
    dict(id='U_lo', section='U', slots=(1, 3), pattern='row3', walkway='W7',
         line=((395, 480), (395, 560)), levels=0, floor='G'),
    # ---- Walkway W6: G-13..18 | G-1..12 ------------------------------------
    dict(id='G1', section='G', slots=(1, 12), pattern='col2', walkway='W6',
         line=((345, 385), (345, 560)), levels=0, floor='G'),
    dict(id='G13', section='G', slots=(13, 18), pattern='col2', walkway='W6',
         line=((237, 410), (237, 550)), levels=0, floor='G'),
    # ---- Walkway W5: G-25..30 | G-19..24 ------------------------------------
    dict(id='G19', section='G', slots=(19, 24), pattern='col2', walkway='W5',
         line=((202, 410), (202, 550)), levels=0, floor='G'),
    dict(id='G25', section='G', slots=(25, 30), pattern='col2', walkway='W5',
         line=((90, 385), (90, 560)), levels=0, floor='G'),
    # ---- Low racks: I (left corridor), J/K (bottom wall) -------------------
    dict(id='I',  section='I', slots=(1, 6),  pattern='col2', walkway='LC',
         line=((80, 590), (80, 700)), levels=0, floor='G'),
    dict(id='J',  section='J', slots=(1, 15), pattern='col2', walkway='BX2',
         line=((65, 745), (410, 745)), levels=0, floor='G'),
    dict(id='K',  section='K', slots=(1, 6),  pattern='col2', walkway='BX2',
         line=((415, 745), (765, 745)), levels=0, floor='G'),
    # ---- Mezzanine ---------------------------------------------------------
    dict(id='V',  section='V', slots=(1, 3), pattern='row1', walkway='MZV',
         line=((257, 300), (257, 40)), levels=0, floor='M'),
    dict(id='V4', section='V', slots=(4, 4), pattern='single', walkway='MZV4',
         line=((250, 470), (250, 470)), levels=0, floor='M'),
    dict(id='S',  section='S', slots=(1, 9), pattern='col2', walkway='MZS',
         line=((385, 340), (385, 0)), levels=0, floor='M'),
    dict(id='X5', section='X', slots=(5, 5), pattern='single', walkway='MZX5',
         line=((545, 180), (545, 180)), levels=0, floor='M'),
    dict(id='X4', section='X', slots=(4, 4), pattern='single', walkway='MZTU',
         line=((532, 472), (532, 472)), levels=0, floor='M'),
    dict(id='T_hi', section='T', slots=(4, 8), pattern='row3', walkway='MZTU',
         line=((395, 385), (395, 475)), levels=0, floor='M'),
    dict(id='U_hi', section='U', slots=(4, 10), pattern='row3', walkway='MZTU',
         line=((395, 480), (395, 560)), levels=0, floor='M'),
    dict(id='W',  section='W', slots=(1, 27), pattern='col2', walkway='MZW',
         line=((835, 600), (835, 0)), levels=0, floor='M'),
    dict(id='Z',  section='Z', slots=(1, 52), pattern='col2', walkway='MZZ',
         line=((65, 750), (765, 750)), levels=0, floor='M'),
]

# Location codes known to be data errors. Items here are unroutable and the
# validation check skips them. Add a reason so the decision is traceable.
IGNORED_LOCATIONS = {
    'W-43': 'Only FE0955 (zero stock, not ordered in Sep 2026). W runs 1-27; likely a typo.',
}

# Legacy / odd codes -> (section, slot)
ALIASES = {'X03': ('X', 3)}

# Open points to confirm on the floor (shown in the validation page)
ASSUMPTIONS = [
    'A1: Column 1 of each rack is at the end nearest the main cross aisle '
    '(south end in the top cluster, north end in the bottom cluster). '
    'Horizontal racks (J, K, Z) start at the west/left end.',
    'A2: G-25..G-30 sit against the left wall facing G-19..G-24 across walkway W5 '
    '(as on the v3 map). G-30 exists in the stock file.',
    'A3: B-36 exists in the stock file (15 SKUs) although B was described as 1-35; '
    'it is kept as the bottom of column 18.',
    'A4: Mezzanine walkways are approximate straight paths from the stair landing; '
    'V-4 (on top of the G-13..G-29 rack) is reached via SC1.',
    'A5: T/U top row, T-7/T-8, U-7..U-10 and X-4 are on the mezzanine via SC2.',
]
