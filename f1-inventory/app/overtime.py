"""
'Live replay' page: plays a busy period of recorded CCTV (frames 3 s apart) through F1 as if it were live.
Shows the system's normal behaviour with the temporary-occlusion handling switched on:
  - a picker in front of a shelf -> the shelf shows PICKER IN THE WAY and keeps its last confirmed status;
  - statuses change only after they are confirmed over a few reliable frames.
Reads the saved results of scripts/temporal_blocking.py (outputs/temporal*/frame_shelf_readings*.csv),
so no models run during the demo. The before/after evaluation lives in docs/temporal_results.md and the slides.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import f1_engine as fe
import temporal_blocking as tb          # same logic as the experiment (scripts folder is on the path via f1_engine)

SETTINGS = argparse.Namespace(block_thr=0.10, window=5, persist=3, low=0.5, out_ratio=0.1)
PURPLE = "#7C3AED"
PURPLE_BGR = (237, 58, 124)
HEX = {"IN STOCK": fe.STATUS_COLOUR["IN"], "LOW": fe.STATUS_COLOUR["LOW"], "STOCK-OUT": fe.STATUS_COLOUR["OUT"],
       "UNKNOWN": fe.STATUS_COLOUR["UNK"], "BLOCKED": PURPLE}


def bgr(hex_colour):
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def runs():
    out = []
    for d in sorted(fe.p("outputs").glob("temporal*")):
        for f in sorted(d.glob("frame_shelf_readings*.csv")):
            cam = "X_5" if "x5" in d.name.lower() else "G_H"
            tag = f.stem.replace("frame_shelf_readings", "").strip("_") or "v1"
            out.append((cam, tag, f))
    best = {}
    for cam, tag, f in out:                      # one run per camera: the latest settings (v2 over v1)
        if cam not in best or tag > best[cam][0]:
            best[cam] = (tag, f)
    return {cam: f for cam, (tag, f) in best.items()}


@st.cache_data(show_spinner=False)
def load_run(path, mtime):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _, groups = tb.evaluate(rows, SETTINGS)
    flat = []
    for (win, shelf), seq in groups.items():
        for r in seq:
            flat.append({"window": win, "shelf": shelf, "t_sec": r["t_sec"], "frame": r["frame"],
                         "frame_path": r["frame_path"], "count": r.get("count"), "reference": r.get("reference"),
                         "level": r["fused_ratio"], "blocked": bool(r["blocked"]),
                         "blocked_share": r["blocked_share"], "status": r["proposed_status"]})
    return pd.DataFrame(flat)


def clock(t_sec):
    t = int(t_sec) + 8 * 3600                      # the recordings start at 08:00
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def events(df):
    """What the operator would see in a log: pickers arriving / leaving, confirmed status changes."""
    log = []
    for shelf, s in df.sort_values("t_sec").groupby("shelf"):
        prev_b, prev_s = False, None
        for r in s.itertuples():
            if r.blocked and not prev_b:
                log.append((r.t_sec, shelf, "picker in the way", f"holding last confirmed {r.status}", PURPLE))
            if not r.blocked and prev_b:
                log.append((r.t_sec, shelf, "view clear again", f"status {r.status}", "#64748B"))
            if prev_s is not None and r.status != prev_s and r.status in HEX:
                log.append((r.t_sec, shelf, f"status changed to {r.status}", "confirmed over 3 clear frames",
                            HEX[r.status]))
            prev_b, prev_s = r.blocked, r.status
    return sorted(log)


def frame_with_status(path, cam, rows_now):
    img = cv2.imread(str(path)) if Path(str(path)).exists() else None
    if img is None:
        return None
    locs = {l["location_id"]: l for l in fe.load_shelves().get(cam, {}).get("locations", [])}
    h, w = img.shape[:2]
    over, out = img.copy(), img.copy()
    for r in rows_now.itertuples():
        loc = locs.get(r.shelf)
        if loc is None:
            continue
        col = PURPLE_BGR if r.blocked else bgr(HEX.get(r.status, "#94A3B8"))
        pts = np.array([[int(x * w), int(y * h)] for x, y in fe.cs.loc_polygon(loc)], np.int32)
        cv2.fillPoly(over, [pts], col)
        cv2.polylines(out, [pts], True, col, 2)
    out = cv2.addWeighted(over, 0.22, out, 0.78, 0)
    for r in rows_now.itertuples():
        loc = locs.get(r.shelf)
        if loc is None:
            continue
        pts = np.array([[int(x * w), int(y * h)] for x, y in fe.cs.loc_polygon(loc)], np.int32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        (tw, th), _ = cv2.getTextSize(r.shelf, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (cx - tw // 2 - 4, cy - th - 5), (cx + tw // 2 + 4, cy + 4), (54, 23, 11), -1)
        cv2.putText(out, r.shelf, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return fe.rgb(out)


def card(r):
    if r.blocked:
        colour, title = PURPLE, "PICKER IN THE WAY"
        body = (f"A picker covers {r.blocked_share:.0%} of this shelf, so this picture is not used.<br>"
                f"Last confirmed: <b style='color:{HEX.get(r.status, '#0F172A')}'>{r.status}</b>")
    else:
        colour, title = HEX.get(r.status, "#94A3B8"), r.status
        lvl = f"about {min(r.level, 1.0):.0%} of full stock" if r.level is not None and r.level == r.level else ""
        body = f"Confirmed from recent clear pictures · {lvl}"
    return (f'<div style="background:#fff;border:1px solid #E2E8F0;border-left:5px solid {colour};border-radius:10px;'
            f'padding:10px 12px;margin-bottom:10px"><div style="display:flex;justify-content:space-between;'
            f'align-items:center"><b style="font-size:15px">{r.shelf}</b><span style="font-size:11.5px;font-weight:700;'
            f'color:{colour};background:{colour}14;border:1px solid {colour}44;border-radius:999px;padding:2px 9px">'
            f'{title}</span></div><div style="font-size:12.5px;color:#475569;margin-top:6px">{body}</div></div>')


def page_overtime(page_head, kpi_row, show_image):
    rs = runs()
    if not rs:
        page_head("Live replay", "No saved replay yet.")
        st.info("Run scripts/temporal_blocking.py first (it saves outputs/temporal*/frame_shelf_readings*.csv).")
        return
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    with c1:
        cam = st.selectbox("Camera", list(rs.keys()), format_func=lambda c: fe.CAMERA_NAMES.get(c, c))
    df = load_run(str(rs[cam]), rs[cam].stat().st_mtime)
    with c2:
        win = st.selectbox("Recorded period", sorted(df["window"].unique()),
                           format_func=lambda w: f"5 Sep · {clock(df[df.window == w].t_sec.min())} – "
                                                 f"{clock(df[df.window == w].t_sec.max())}")
    d = df[df.window == win]
    times = sorted(d.t_sec.unique())
    key = f"replay_{cam}_{win}"
    if key not in st.session_state:
        st.session_state[key] = 0
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        playing = st.toggle("Play", key=f"play_{key}", help="Plays the recording, one picture every 3 seconds of video")

    page_head(f"Live replay · {fe.CAMERA_NAMES.get(cam, cam)}",
              "Recorded CCTV played through F1 as if it were live. Pickers walking in front of shelves are detected; "
              "those shelves keep their last confirmed status instead of raising an alarm.",
              f"{clock(times[st.session_state[key]])}")
    i = st.slider("Time", 0, len(times) - 1, key=key, format="%d",
                  help="Move to any moment of the recording (3 seconds between pictures)")
    now = d[d.t_sec == times[i]].sort_values("shelf")

    n_block = int(now.blocked.sum())
    vis = now[~now.blocked]
    kpi_row([("Shelves", len(now), f"camera {fe.CAMERA_NAMES.get(cam, cam)}", None),
             ("In stock", int((vis.status == "IN STOCK").sum()), "confirmed", HEX["IN STOCK"]),
             ("Low", int((vis.status == "LOW").sum()), "confirmed", HEX["LOW"]),
             ("Stock-out", int((vis.status == "STOCK-OUT").sum()), "confirmed", HEX["STOCK-OUT"]),
             ("Picker in the way", n_block, "holding last confirmed status", PURPLE)])

    a, b = st.columns([1.5, 1])
    with a:
        img = frame_with_status(now.frame_path.iloc[0], cam, now)
        if img is not None:
            show_image(img, f"{clock(times[i])} · {now.frame.iloc[0]}  ·  colour = shelf status, purple = picker in the way")
        else:
            st.info(f"Picture not found on this computer: {now.frame_path.iloc[0]}")
    with b:
        order = now.sort_values(["blocked", "shelf"], ascending=[False, True])
        st.markdown("".join(card(r) for r in order.itertuples()), unsafe_allow_html=True)

    log = [e for e in events(d) if e[0] <= times[i]][-8:][::-1]
    if log:
        rows = "".join(f'<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:13px">'
                       f'<span style="color:#64748B;width:64px">{clock(t)}</span><b style="width:110px">{s}</b>'
                       f'<span style="color:{c};font-weight:600">{what}</span>'
                       f'<span style="color:#64748B">· {note}</span></div>' for t, s, what, note, c in log)
        st.markdown(f'<div style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:8px 14px;'
                    f'margin-top:6px"><b style="font-size:14px">Event log</b>{rows}</div>', unsafe_allow_html=True)

    with st.expander("How F1 handles pickers (for reviewers)"):
        st.markdown("1. **Pickers are detected** in every picture (COCO-pretrained YOLO, class *person*, no training). "
                    "If a picker covers at least 10% of a shelf area, that picture is not used for that shelf.\n"
                    "2. **Recent clear pictures are combined** (weighted median of the last 5, weighted by the "
                    "reliability score: detector confidence + occlusion trust + position).\n"
                    "3. **A new status is shown only after it holds for 3 clear pictures**, so one bad picture "
                    "cannot raise an alarm.\n\n"
                    "Sent to F4 per shelf: `blocked`, `blocked_share`, `status`, `last_confirmed_status`. "
                    "Evaluation: docs/temporal_results.md.")

    if playing:
        time.sleep(0.6)
        st.session_state[key] = (i + 1) % len(times)
        st.rerun()
