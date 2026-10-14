"""
f4-routing/demo/visualise_warehouse.py

Enhanced warehouse visualiser — realistic 3D rack structure.

Produces two views:
  1. warehouse_2d.png  — top-down floor plan (save and open)
  2. warehouse_3d.html — interactive 3D view (opens in browser)

The 3D view shows:
  - Physical rack structures (rectangular blocks representing shelf units)
  - Colour-coded by task weight: Red=Heavy  Amber=Moderate  Green=Light
  - Aisle walkways clearly visible between rack rows
  - Cross aisles shown as horizontal corridors
  - Depot marked prominently at the warehouse entrance
  - Walking paths shown as lines at floor level
  - Picker route example overlaid in bright yellow

Run from f4-routing folder:
    python demo/visualise_warehouse.py
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────
DEMO_DIR   = os.path.dirname(os.path.abspath(__file__))
F4_ROOT    = os.path.abspath(os.path.join(DEMO_DIR, ".."))
SRC_DIR    = os.path.join(F4_ROOT, "src")
GRAPH_PATH = os.path.join(F4_ROOT, "data", "graphs", "warehouse_small.json")

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from graph_builder import WarehouseGraph

# ── Sample F3 state (change these to see cost landscape shift) ─────────────
SAMPLE_F3 = {
    "picker_id":            "P-001",
    "ergonomic_risk_score": 7,       # 1=low risk  10=high risk
    "gait_stability_score": 5,
    "demographic_weight":   1.3,     # 1.0=young  1.3=mid  1.6=senior
    "task_tier":            "Moderate",
    "freeze_state":         False
}

# ── Load warehouse ─────────────────────────────────────────────────────────
print("Loading warehouse graph...")
wh = WarehouseGraph(GRAPH_PATH)
wh.update_from_f3(SAMPLE_F3)

with open(GRAPH_PATH) as f:
    plan = json.load(f)

dims  = plan["dimensions"]
nodes = plan["nodes"]
edges = plan["edges"]

# Physical parameters from the plan
AISLE_W  = dims["aisle_width_m"]
LOC_D    = dims["location_depth_m"]
CA_W     = dims["cross_aisle_width_m"]
N_AISLES = dims["num_aisles"]
N_BLOCKS = dims["num_blocks"]
LOCS     = dims["locations_per_segment"]
SEG_LEN  = LOCS * LOC_D

RACK_HEIGHT   = 2.5    # metres — visual height of shelf unit
RACK_DEPTH    = 0.6    # metres — depth of one rack face
RACK_WIDTH    = LOC_D  # metres — width per pick slot

# ── Node lookup ────────────────────────────────────────────────────────────
node_map = {n["id"]: n for n in nodes}

# ── Colours ────────────────────────────────────────────────────────────────
TASK_COLOR = {
    "Heavy":    "#E74C3C",
    "Moderate": "#F39C12",
    "Light":    "#27AE60"
}
FLOOR_COLOR   = "#ECF0F1"
AISLE_COLOR   = "#D5D8DC"
DEPOT_COLOR   = "#2980B9"
ROUTE_COLOR   = "#F1C40F"
WALL_COLOR    = "#BDC3C7"
CA_COLOR      = "#85C1E9"


# ══════════════════════════════════════════════════════════════════════════
# 2D FLOOR PLAN
# ══════════════════════════════════════════════════════════════════════════

print("Generating 2D floor plan...")

fig2d, ax = plt.subplots(figsize=(16, 12))
ax.set_facecolor(FLOOR_COLOR)
fig2d.patch.set_facecolor("#FFFFFF")

wh_width = (N_AISLES - 1) * AISLE_W + RACK_DEPTH * 2
wh_depth = dims["total_warehouse_depth_m"]

# Draw warehouse boundary
boundary = mpatches.FancyBboxPatch(
    (-RACK_DEPTH, -CA_W - 2),
    wh_width + RACK_DEPTH * 2,
    wh_depth + CA_W + 2 + 2,
    boxstyle="square,pad=0",
    linewidth=2, edgecolor="#2C3E50",
    facecolor="#FDFEFE"
)
ax.add_patch(boundary)

# Draw aisles (walkways between racks)
for a in range(1, N_AISLES + 1):
    x_centre = (a - 1) * AISLE_W
    for b in range(1, N_BLOCKS + 1):
        y_start = (b - 1) * (SEG_LEN + CA_W)
        # Aisle walkway
        aisle_rect = mpatches.Rectangle(
            (x_centre - AISLE_W / 2 + RACK_DEPTH, y_start),
            AISLE_W - RACK_DEPTH * 2, SEG_LEN,
            linewidth=0, facecolor=AISLE_COLOR, alpha=0.4
        )
        ax.add_patch(aisle_rect)

# Draw cross aisles
for b in range(N_BLOCKS + 1):
    if b == 0:
        y = -CA_W
    elif b == N_BLOCKS:
        y = N_BLOCKS * (SEG_LEN + CA_W)
    else:
        y = b * (SEG_LEN + CA_W) - CA_W
    ca_rect = mpatches.Rectangle(
        (-RACK_DEPTH, y), wh_width + RACK_DEPTH * 2, CA_W,
        linewidth=0, facecolor=CA_COLOR, alpha=0.35
    )
    ax.add_patch(ca_rect)
    ax.text(wh_width / 2, y + CA_W / 2,
            "Cross aisle" if b not in [0, N_BLOCKS] else
            ("Front aisle" if b == 0 else "Rear aisle"),
            ha="center", va="center", fontsize=7,
            color="#1A5276", alpha=0.7)

# Draw rack units (coloured by task weight)
for n in nodes:
    if not n["id"].startswith("A"):
        continue
    x, y = n["x"], n["y"]
    tw    = n.get("task_weight", "Moderate")
    color = TASK_COLOR[tw]
    # Left rack face
    rack_l = mpatches.Rectangle(
        (x - AISLE_W / 2, y),
        RACK_DEPTH, RACK_WIDTH,
        linewidth=0.3, edgecolor="white",
        facecolor=color, alpha=0.85
    )
    # Right rack face
    rack_r = mpatches.Rectangle(
        (x + AISLE_W / 2 - RACK_DEPTH, y),
        RACK_DEPTH, RACK_WIDTH,
        linewidth=0.3, edgecolor="white",
        facecolor=color, alpha=0.85
    )
    ax.add_patch(rack_l)
    ax.add_patch(rack_r)

# Draw depot
depot = node_map.get("DEPOT", {})
dx, dy = depot.get("x", 0), depot.get("y", -CA_W - 2)
depot_patch = mpatches.FancyBboxPatch(
    (dx - 1.2, dy - 0.8), 2.4, 1.6,
    boxstyle="round,pad=0.1",
    linewidth=2, edgecolor="#1A5276",
    facecolor=DEPOT_COLOR, alpha=0.9
)
ax.add_patch(depot_patch)
ax.text(dx, dy, "DEPOT\n(Start / End)", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white")

# Draw aisle labels
for a in range(1, N_AISLES + 1):
    x = (a - 1) * AISLE_W
    ax.text(x, wh_depth + 1.0, f"Aisle {a}",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="#2C3E50")

# Draw block labels
for b in range(1, N_BLOCKS + 1):
    y_mid = (b - 1) * (SEG_LEN + CA_W) + SEG_LEN / 2
    ax.text(-RACK_DEPTH * 2, y_mid, f"Block {b}",
            ha="right", va="center", fontsize=8,
            fontweight="bold", color="#2C3E50", rotation=90)

# Legend
legend_items = [
    mpatches.Patch(color=TASK_COLOR["Heavy"],    label="Heavy task racks"),
    mpatches.Patch(color=TASK_COLOR["Moderate"], label="Moderate task racks"),
    mpatches.Patch(color=TASK_COLOR["Light"],    label="Light task racks"),
    mpatches.Patch(color=CA_COLOR,               label="Cross aisles (walkways)"),
    mpatches.Patch(color=DEPOT_COLOR,            label="Depot (start/end point)"),
]
ax.legend(handles=legend_items, loc="upper right",
          fontsize=9, framealpha=0.9)

ax.set_xlim(-2, wh_width + 2)
ax.set_ylim(-CA_W - 4, wh_depth + 3)
ax.set_xlabel("Width (metres)", fontsize=11)
ax.set_ylabel("Depth (metres)", fontsize=11)
ax.set_title(
    f"Warehouse Floor Plan — {plan['warehouse_id']}\n"
    f"{N_AISLES} aisles  ×  {N_BLOCKS} blocks  ×  {LOCS} shelf slots per segment  |  "
    f"{dims['total_pick_locations']} total pick locations  |  "
    f"{dims['total_warehouse_width_m']}m × {dims['total_warehouse_depth_m']}m",
    fontsize=12, fontweight="bold", pad=15
)
ax.grid(True, alpha=0.15, linestyle="--")
plt.tight_layout()

out_2d = os.path.join(DEMO_DIR, "warehouse_2d.png")
plt.savefig(out_2d, dpi=150, bbox_inches="tight")
print(f"2D floor plan saved → {out_2d}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 3D REALISTIC VIEW
# ══════════════════════════════════════════════════════════════════════════

print("Generating 3D interactive view...")

traces = []

# ── Floor plane ────────────────────────────────────────────────────────────
floor_x = [0, wh_width, wh_width, 0, 0]
floor_y = [-CA_W, -CA_W, wh_depth + CA_W, wh_depth + CA_W, -CA_W]
floor_z = [0, 0, 0, 0, 0]

traces.append(go.Scatter3d(
    x=floor_x, y=floor_y, z=floor_z,
    mode="lines",
    line=dict(color="#BDC3C7", width=2),
    name="Warehouse boundary",
    showlegend=True
))

# Floor surface
xs = np.array([0, wh_width, wh_width, 0])
ys = np.array([-CA_W, -CA_W, wh_depth + CA_W, wh_depth + CA_W])
zs = np.zeros(4)
traces.append(go.Mesh3d(
    x=xs, y=ys, z=zs,
    i=[0, 0], j=[1, 2], k=[2, 3],
    color=FLOOR_COLOR,
    opacity=0.3,
    showlegend=False,
    hoverinfo="none"
))

# ── Rack structures (3D boxes) ─────────────────────────────────────────────
# Each rack slot is a small 3D box
# We build them as Mesh3d objects grouped by task weight

rack_verts = {tw: {"x":[],"y":[],"z":[],"i":[],"j":[],"k":[]} 
              for tw in ["Heavy","Moderate","Light"]}

def add_box(tw, x0, y0, z0, dx, dy, dz):
    """Add a 3D box (cuboid) to the rack vertex list."""
    rv = rack_verts[tw]
    base = len(rv["x"])
    # 8 vertices of a box
    vx = [x0,x0+dx,x0+dx,x0,  x0,x0+dx,x0+dx,x0]
    vy = [y0,y0,   y0+dy,y0+dy,y0,y0,   y0+dy,y0+dy]
    vz = [z0,z0,   z0,   z0,   z0+dz,z0+dz,z0+dz,z0+dz]
    rv["x"].extend(vx)
    rv["y"].extend(vy)
    rv["z"].extend(vz)
    # 12 triangles (6 faces × 2 triangles each)
    faces = [
        (0,1,2),(0,2,3),  # bottom
        (4,5,6),(4,6,7),  # top
        (0,1,5),(0,5,4),  # front
        (2,3,7),(2,7,6),  # back
        (0,3,7),(0,7,4),  # left
        (1,2,6),(1,6,5),  # right
    ]
    for fi, fj, fk in faces:
        rv["i"].append(base + fi)
        rv["j"].append(base + fj)
        rv["k"].append(base + fk)

RACK_BOX_H = 2.2   # height of each shelf rack unit
RACK_BOX_D = 0.55  # depth of rack (front to back)
SLOT_GAP   = 0.05  # small gap between slots for visual clarity

for n in nodes:
    if not n["id"].startswith("A"):
        continue
    cx   = n["x"]           # aisle centre x
    y0   = n["y"]           # slot start y
    tw   = n.get("task_weight", "Moderate")

    slot_len = LOC_D - SLOT_GAP

    # Left rack (negative x side of aisle)
    add_box(tw,
            cx - AISLE_W/2,            # x start
            y0,                        # y start
            0,                         # z start (floor)
            RACK_BOX_D,                # x size
            slot_len,                  # y size
            RACK_BOX_H                 # z size (height)
            )
    # Right rack (positive x side of aisle)
    add_box(tw,
            cx + AISLE_W/2 - RACK_BOX_D,
            y0,
            0,
            RACK_BOX_D,
            slot_len,
            RACK_BOX_H
            )

# Add rack mesh traces
TASK_RGBA = {
    "Heavy":    ("#E74C3C", 0.80),
    "Moderate": ("#F39C12", 0.75),
    "Light":    ("#27AE60", 0.75),
}
for tw, rv in rack_verts.items():
    if not rv["x"]:
        continue
    color, opacity = TASK_RGBA[tw]
    traces.append(go.Mesh3d(
        x=rv["x"], y=rv["y"], z=rv["z"],
        i=rv["i"], j=rv["j"], k=rv["k"],
        color=color,
        opacity=opacity,
        name=f"{tw} racks",
        showlegend=True,
        hoverinfo="name",
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.8,
                      specular=0.2, roughness=0.8)
    ))

# ── Cross aisle floor markings ─────────────────────────────────────────────
for b in range(N_BLOCKS + 1):
    if b == 0:
        y_ca = -CA_W
    elif b == N_BLOCKS:
        y_ca = N_BLOCKS * (SEG_LEN + CA_W)
    else:
        y_ca = b * (SEG_LEN + CA_W) - CA_W

    ca_xs = [0, wh_width, wh_width, 0, 0]
    ca_ys = [y_ca, y_ca, y_ca + CA_W, y_ca + CA_W, y_ca]
    ca_zs = [0.01]*5   # slightly above floor so it shows
    traces.append(go.Scatter3d(
        x=ca_xs, y=ca_ys, z=ca_zs,
        mode="lines",
        line=dict(color=CA_COLOR, width=4),
        showlegend=(b == 0),
        name="Cross aisles" if b == 0 else None,
        hoverinfo="none"
    ))

# ── Walking path edges at floor level ─────────────────────────────────────
ex, ey, ez = [], [], []
for e in edges:
    fn = node_map.get(e["from"])
    tn = node_map.get(e["to"])
    if fn and tn:
        ex += [fn["x"], tn["x"], None]
        ey += [fn["y"], tn["y"], None]
        ez += [0.05, 0.05, None]

traces.append(go.Scatter3d(
    x=ex, y=ey, z=ez,
    mode="lines",
    line=dict(color="#AAB7B8", width=1),
    name="Walkable paths",
    opacity=0.4,
    hoverinfo="none"
))

# ── Pick location markers (floating above racks) ───────────────────────────
px_pts, py_pts, pz_pts, ptxt = [], [], [], []
for n in nodes:
    if not n["id"].startswith("A"):
        continue
    tw = n.get("task_weight","Moderate")
    px_pts.append(n["x"])
    py_pts.append(n["y"] + LOC_D/2)
    pz_pts.append(RACK_BOX_H + 0.3)
    ptxt.append(f"{n['id']}<br>Task weight: {tw}")

traces.append(go.Scatter3d(
    x=px_pts, y=py_pts, z=pz_pts,
    mode="markers",
    marker=dict(size=3, color="#2C3E50",
                symbol="circle", opacity=0.5),
    name="Pick locations",
    text=ptxt,
    hoverinfo="text"
))

# ── Depot ──────────────────────────────────────────────────────────────────
depot_node = node_map.get("DEPOT", {"x": 0, "y": -CA_W - 2})
dpx, dpy   = depot_node["x"], depot_node["y"]

# Depot platform box
dv = {"x":[],"y":[],"z":[],"i":[],"j":[],"k":[]}
add_box({"Heavy":None}.get("x","Heavy") if False else "Moderate",
        dpx - 1.5, dpy - 1.0, 0, 3.0, 2.0, 0.4)

# Override with depot colour directly
traces.append(go.Mesh3d(
    x=[dpx-1.5, dpx+1.5, dpx+1.5, dpx-1.5,
       dpx-1.5, dpx+1.5, dpx+1.5, dpx-1.5],
    y=[dpy-1.0, dpy-1.0, dpy+1.0, dpy+1.0,
       dpy-1.0, dpy-1.0, dpy+1.0, dpy+1.0],
    z=[0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4],
    i=[0,0,4,4,0,1,2,3,0,1,5,4],
    j=[1,2,5,6,4,5,6,7,3,2,6,7],
    k=[2,3,6,7,5,6,7,4,4,6,7,3],
    color=DEPOT_COLOR,
    opacity=0.95,
    name="Depot (start/end point)",
    showlegend=True,
    hoverinfo="name"
))

# Depot label
traces.append(go.Scatter3d(
    x=[dpx], y=[dpy], z=[1.2],
    mode="text",
    text=["📦 DEPOT<br>(Pickers start &amp; end here)"],
    textfont=dict(size=11, color=DEPOT_COLOR),
    showlegend=False,
    hoverinfo="none"
))

# ── Example route overlay ──────────────────────────────────────────────────
# Show a sample optimal route: DEPOT → A1-B1-L1 → A2-B1-L2 → A3-B1-L1 → DEPOT
sample_route = ["DEPOT","A1-B1-L1","A2-B1-L1","A3-B1-L1","A4-B1-L1","DEPOT"]
rx, ry, rz = [], [], []
for nid in sample_route:
    nd = node_map.get(nid)
    if nd:
        rx.append(nd["x"])
        ry.append(nd.get("y", -CA_W-2) + LOC_D/2)
        rz.append(0.15)

traces.append(go.Scatter3d(
    x=rx, y=ry, z=rz,
    mode="lines+markers",
    line=dict(color=ROUTE_COLOR, width=6),
    marker=dict(size=6, color=ROUTE_COLOR,
                symbol="circle",
                line=dict(width=1, color="white")),
    name="Example pick route",
    hoverinfo="name"
))

# ── Layout ─────────────────────────────────────────────────────────────────
fig3d = go.Figure(data=traces)

fig3d.update_layout(
    title=dict(
        text=(
            f"<b>Warehouse 3D Structure — {plan['warehouse_id']}</b><br>"
            f"<sup>{N_AISLES} aisles × {N_BLOCKS} blocks × {LOCS} shelf slots "
            f"= {dims['total_pick_locations']} pick locations | "
            f"{dims['total_warehouse_width_m']}m wide × "
            f"{dims['total_warehouse_depth_m']}m deep × "
            f"{RACK_BOX_H}m rack height | "
            f"Yellow line = example picker route | "
            f"Rotate with mouse · Hover nodes for details</sup>"
        ),
        x=0.5, font=dict(size=14)
    ),
    scene=dict(
        xaxis=dict(title="Width (m)", backgroundcolor="#F4F6F7",
                   gridcolor="white", showbackground=True),
        yaxis=dict(title="Depth (m)", backgroundcolor="#EAECEE",
                   gridcolor="white", showbackground=True),
        zaxis=dict(title="Height (m)", backgroundcolor="#EBF5FB",
                   gridcolor="white", showbackground=True,
                   range=[0, RACK_BOX_H + 1.5]),
        bgcolor="#FDFEFE",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.8, y=-2.2, z=1.4)
        ),
        aspectmode="data"
    ),
    legend=dict(
        x=0.01, y=0.99,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#BDC3C7",
        borderwidth=1,
        font=dict(size=11)
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    height=750,
    paper_bgcolor="white"
)

out_3d = os.path.join(DEMO_DIR, "warehouse_3d.html")
fig3d.write_html(out_3d)
print(f"3D interactive view saved → {out_3d}")
fig3d.show()

print("\nDone.")
print("Rotate with mouse | Hover over nodes for details")
print("Click legend items to toggle layers on/off")
print(f"\nWarehouse summary:")
print(f"  Aisles         : {N_AISLES}")
print(f"  Blocks         : {N_BLOCKS}")
print(f"  Slots per seg  : {LOCS}")
print(f"  Pick locations : {dims['total_pick_locations']}")
print(f"  Rack height    : {RACK_BOX_H}m")
print(f"  Floor area     : {dims['total_warehouse_width_m']}m × "
      f"{dims['total_warehouse_depth_m']}m")