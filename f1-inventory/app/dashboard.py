r"""
F1 dashboard - Occlusion-Aware Inventory Verification (Warehouse Intelligence System, J26-DS-352)

Pages
  Overview        stock status of every shelf, per camera, with alerts (from the latest results)
  Live check      run F1 on one camera frame and see each step: frame -> cartons -> occlusion -> shelf status
  Camera coverage which shelves each camera can judge (shelf outlines + coverage sheet)

Run (from the f1-inventory folder):
  python -m pip install "streamlit>=1.45"
  streamlit run app\dashboard.py
"""
import html
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # find f1_engine.py next to this file
import f1_engine as fe  # noqa: E402

st.set_page_config(page_title="F1 · Inventory Verification", page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ====================================================================== helpers
def show_image(img, caption=None):
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)
    if caption:
        st.markdown(f'<div class="caption">{caption}</div>', unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def cam_name(cam):
    return fe.CAMERA_NAMES.get(cam, cam)


def local_time(ts):
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%d %b %H:%M")
    except Exception:
        return ts or "-"


def level_of(rec):
    r = rec.get("adjusted_ratio")
    return rec.get("ratio_to_reference") if r is None else r


def status_sentence(rec):
    k = fe.STATUS_KEY.get(rec["status"], "UNK")
    lvl = level_of(rec)
    if k == "IN":
        return f"Enough stock on this shelf (about {lvl:.0%} of full stock)." if lvl is not None else "Enough stock."
    if k == "LOW":
        return f"Stock is running low: about {lvl:.0%} of full stock."
    if k == "OUT":
        return "This shelf looks empty."
    return rec.get("status_note") or "Not enough information to judge this shelf."


def is_verify(rec):
    return bool(rec.get("verify_manually")) and rec["status"] != "UNKNOWN"


def page_head(title, sub, chip=None):
    chip_html = f'<span class="chip"><span class="dot"></span>{chip}</span>' if chip else ""
    st.markdown(f'<div class="page-head"><div><div class="page-title">{title}</div><div class="page-sub">{sub}</div></div>'
                f'{chip_html}</div>', unsafe_allow_html=True)


def kpi_row(items):
    cells = "".join(
        f'<div class="kpi"><div class="kpi-label">{"<span class=sw style=background:" + c + "></span>" if c else ""}{esc(l)}</div>'
        f'<div class="kpi-value">{v}</div><div class="kpi-sub">{esc(s)}</div></div>' for l, v, s, c in items)
    st.markdown(f'<div class="kpis" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">{cells}</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_image(path):
    img = cv2.imread(str(path)) if path and Path(path).exists() else None
    return img


# ====================================================================== shelf card + detail
def shelf_card(rec, cam, loc_cfg, overlay, camcal, prefix):
    k = fe.STATUS_KEY.get(rec["status"], "UNK")
    safe = re.sub(r"[^A-Za-z0-9]", "", rec["location_id"])
    with st.container(key=f"shelf-{k}-{prefix}-{cam}-{safe}"):
        lvl = level_of(rec)
        width = 0 if lvl is None else min(lvl, 1) * 100
        ref = rec.get("reference_count")
        verify = '<span class="badge b-VER">CHECK BY EYE</span>' if is_verify(rec) else ""
        rel = rec.get("reliability") if rec.get("reliability") is not None else rec.get("trust")
        est = rec.get("estimated_true_count")
        st.markdown(
            f'<div class="sc-top"><span class="sc-id" title="{esc(rec["location_id"])}">{esc(rec["location_id"])}</span>'
            f'<span class="badge b-{k}">{esc(rec["status"])}</span></div>'
            f'<div class="sc-count">{rec["count"]}<span class="of"> / {ref if ref is not None else "?"}</span></div>'
            f'<div class="sc-cap">cartons seen / when full</div>'
            f'<div class="bar"><div style="width:{width:.0f}%;background:{fe.STATUS_COLOUR[k]}"></div></div>'
            f'<div class="sc-foot"><span>{"≈ " + str(est) + " in total" if est is not None else ""}</span>'
            f'<span>{"Rel. " + format(rel, ".2f") if rel is not None else ""}</span></div>'
            + (f'<div class="sc-flag">{verify}</div>' if verify else ""), unsafe_allow_html=True)
        if st.button("Details", key=f"btn-{prefix}-{cam}-{safe}"):
            shelf_dialog(rec, cam, loc_cfg, overlay, camcal)


@st.dialog("Shelf details", width="large")
def shelf_dialog(rec, cam, loc_cfg, overlay, camcal):
    k = fe.STATUS_KEY.get(rec["status"], "UNK")
    st.markdown(f'<div class="page-title" style="font-size:20px">📍 Shelf {esc(rec["location_id"])} '
                f'<span style="color:#64748B;font-weight:500">· {esc(cam_name(cam))}</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="label-sm">Camera view · cartons coloured by how hidden they are</div>', unsafe_allow_html=True)
        img = read_image(overlay)
        if img is not None and loc_cfg is not None:
            show_image(fe.rgb(fe.crop_location(img, loc_cfg)),
                       "Green = clearly visible · orange = partly hidden · red = mostly hidden · white = not on a shelf")
        else:
            st.info("No camera image saved for this frame.")
        mix = rec.get("occlusion_mix") or {}
        tot = sum(mix.values()) or 1
        st.markdown('<div class="label-sm">How hidden are the cartons on this shelf?</div>', unsafe_allow_html=True)
        rows = ""
        for lvl, name, col in (("low", "Clearly visible", "#16A34A"), ("medium", "Partly hidden", "#D97706"),
                               ("high", "Mostly hidden", "#DC2626")):
            n = mix.get(lvl, 0)
            rows += (f'<div class="mix-row"><span>{name}</span><div class="bar"><div style="width:{n / tot * 100:.0f}%;'
                     f'background:{col}"></div></div><span style="text-align:right;font-weight:600">{n}</span></div>')
        st.markdown(rows, unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="label-sm">Stock status</div><div class="big-status c-{k}">{esc(rec["status"])}</div>'
                    f'<div style="font-size:13.5px;color:#334155;margin-bottom:8px">{esc(status_sentence(rec))}</div>',
                    unsafe_allow_html=True)
        lvl = level_of(rec)
        rel = rec.get("reliability")
        kv = [("Cartons the camera sees", rec["count"]),
              ("Cartons when full (reference)", rec.get("reference_count", "-")),
              ("Stock level", f"{lvl:.0%}" if lvl is not None else "-"),
              ("Estimated true cartons", rec.get("estimated_true_count", "-")),
              ("Reliability of this result", f"{rel:.2f}" if rel is not None else
               (f"trust {rec['trust']:.2f}" if rec.get("trust") is not None else "-")),
              ("F4 layout slots", ", ".join(rec.get("f4_slots") or []) or "not linked yet"),
              ("Frame", rec.get("frame", "-")), ("Scanned", local_time(rec.get("timestamp")))]
        st.markdown("".join(f'<div class="kv"><span>{esc(a)}</span><span>{esc(b)}</span></div>' for a, b in kv),
                    unsafe_allow_html=True)
        if is_verify(rec):
            reasons = fe.verify_reasons(rec, camcal)
            txt = ("; ".join(reasons) if reasons else "the combined reliability score is low")
            st.markdown(f'<div class="why warn"><b>Check this shelf by eye.</b> The count here may be wrong because '
                        f'{esc(txt)}. F1 still reports the status, but marks it as uncertain.</div>', unsafe_allow_html=True)
        elif rec["status"] == "UNKNOWN":
            st.markdown('<div class="why">This shelf has too few cartons in its full-stock picture, so one missed carton '
                        'would change the result a lot. F1 does not judge it.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="why">The result looks reliable: confidence, visibility and distance are normal '
                        'for this camera.</div>', unsafe_allow_html=True)
    with st.expander("What F1 sends to F4 (route planning) for this shelf"):
        st.json({k2: v for k2, v in rec.items()})


# ====================================================================== sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">F1</div><div><div class="brand-name">Inventory Verification'
                '</div><div class="brand-sub">Warehouse Intelligence System</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
    page = st.radio("Menu", ["📊  Overview", "🎥  Live check", "▶️  Live replay", "🗺️  Camera coverage"],
                    label_visibility="collapsed")
    st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
    with st.expander("Model files"):
        weights = st.text_input("Detector", fe.DEFAULTS["weights"])
        occ_w = st.text_input("Occlusion CNN", fe.DEFAULTS["occlusion_cnn"])
        rf_w = st.text_input("Geometry model", fe.DEFAULTS["rf_model"])
        frames_root = st.text_input("Frames root (blank = auto)", "",
                                    help=r"Folder that holds all recorded days, e.g. D:\Research\research camera data")
    st.markdown('<div class="side-foot" style="margin-top:18px">J26-DS-352 · Function 1<br>'
                'Occlusion-aware inventory verification<br>Fine Enterprises CCTV</div>', unsafe_allow_html=True)

shelves = fe.load_shelves()
calibration = fe.load_calibration()
cams = list(shelves.keys())


def camcal_for(cam):
    if not calibration:
        return None
    return calibration["cameras"].get(cam)


def loc_lookup(cam):
    return {l["location_id"]: l for l in shelves.get(cam, {}).get("locations", [])}


# ====================================================================== OVERVIEW
def page_overview():
    results = {c: fe.load_results(c) for c in cams}
    current = {}
    for c in cams:
        frames = list(results[c].keys())
        if frames:
            if st.session_state.get(f"frame_{c}") not in results[c]:
                st.session_state[f"frame_{c}"] = frames[0]          # newest frame by default
            current[c] = results[c][st.session_state[f"frame_{c}"]]
    recs = [r for c in current for r in current[c]["records"]]
    last = max((r.get("timestamp", "") for r in recs), default="")
    page_head("Warehouse overview", "Stock status of every shelf the cameras can see · counted from CCTV, adjusted for hidden cartons",
              f"Last scan {local_time(last)}" if last else None)
    if not recs:
        st.warning("No results yet. Run scripts\\count_stock.py for a camera, or use Live check.")
        return

    n = len(recs)
    cnt = {s: sum(r["status"] == s for r in recs) for s in ("IN STOCK", "LOW", "STOCK-OUT", "UNKNOWN")}
    ver = sum(is_verify(r) for r in recs)
    judged = n - cnt["UNKNOWN"]
    kpi_row([("Shelves monitored", n, f"across {len(current)} cameras", None),
             ("In stock", cnt["IN STOCK"], f"{cnt['IN STOCK'] / max(judged, 1):.0%} of judged shelves", "#16A34A"),
             ("Low stock", cnt["LOW"], "below 50% of full", "#D97706"),
             ("Stock-out", cnt["STOCK-OUT"], "below 10% of full", "#DC2626"),
             ("Check by eye", ver, "result may be unreliable", "#0B1736"),
             ("Not judged", cnt["UNKNOWN"], "too few cartons to judge", "#94A3B8")])

    opts = ["All cameras"] + [cam_name(c) for c in current]
    if hasattr(st, "segmented_control"):
        show = st.segmented_control("Camera", opts, default="All cameras", label_visibility="collapsed") or "All cameras"
    else:
        show = st.radio("Camera", opts, horizontal=True, label_visibility="collapsed")
    main, side = st.columns([3.4, 1], gap="medium")
    with main:
        for c in current:
            if show != "All cameras" and show != cam_name(c):
                continue
            with st.container(key=f"panel-cam-{c}"):
                h1, h2 = st.columns([2, 1.2])
                frames = list(results[c].keys())
                with h1:
                    st.markdown(f'<div class="section-title">{esc(cam_name(c))}</div>'
                                f'<div class="section-sub">{len(current[c]["records"])} shelves · '
                                f'{esc(current[c]["source"])}</div>', unsafe_allow_html=True)
                with h2:
                    st.selectbox("Frame", frames, key=f"frame_{c}", label_visibility="collapsed")
                cur = results[c][st.session_state[f"frame_{c}"]]
                img_col, cards_col = st.columns([1, 2.1], gap="medium")
                with img_col:
                    img = read_image(cur["overlay"])
                    if img is not None:
                        show_image(fe.rgb(img), f"Frame {esc(st.session_state[f'frame_{c}'])} · shelf outlines coloured by status")
                    else:
                        st.info("No overlay image for this frame.")
                with cards_col:
                    order = {"STOCK-OUT": 0, "LOW": 1, "IN STOCK": 2, "UNKNOWN": 3}
                    rs = sorted(cur["records"], key=lambda r: (order.get(r["status"], 4), not is_verify(r)))
                    locs = loc_lookup(c)
                    for i in range(0, len(rs), 3):
                        cols = st.columns(3, gap="small")
                        for col, r in zip(cols, rs[i:i + 3]):
                            with col:
                                shelf_card(r, c, locs.get(r["location_id"]), cur["overlay"], camcal_for(c), "ov")
            st.write("")
    with side:
        with st.container(key="panel-feed"):
            st.markdown('<div class="section-title">Alerts</div><div class="section-sub">What needs attention now</div>',
                        unsafe_allow_html=True)
            items = []
            for c in current:
                for r in current[c]["records"]:
                    lvl = level_of(r)
                    where = f'<b>{esc(r["location_id"])}</b> · {esc(cam_name(c))}'
                    if r["status"] == "STOCK-OUT":
                        items.append((0, "OUT", r, f"{where}: shelf looks <b>empty</b>. Restock / tell F4 not to route pickers here."))
                    elif r["status"] == "LOW":
                        items.append((1, "LOW", r, f"{where}: <b>low stock</b>, about {lvl:.0%} of full."))
                    if is_verify(r):
                        why = fe.verify_reasons(r, camcal_for(c))
                        items.append((2, "VER", r, f"{where}: <b>check by eye</b> — "
                                                   f"{esc(why[0] if why else 'low reliability')}."))
            items.sort(key=lambda t: t[0])
            with st.container(height=560, border=False):
                if not items:
                    st.markdown('<div class="feed-empty">Nothing to report. All judged shelves are in stock and reliable.</div>',
                                unsafe_allow_html=True)
                for _, kind, r, text in items:
                    st.markdown(f'<div class="feed-item f-{kind}"><div class="feed-time">{local_time(r.get("timestamp"))}</div>'
                                f'<div class="feed-text">{text}</div></div>', unsafe_allow_html=True)
        st.write("")
        with st.container(key="panel-legend"):
            st.markdown(
                '<div class="section-title">How to read</div>'
                '<div class="feed-text" style="line-height:1.9">'
                '<span class="badge b-IN">IN STOCK</span> 50% of full or more<br>'
                '<span class="badge b-LOW">LOW</span> 10–50% of full<br>'
                '<span class="badge b-OUT">STOCK-OUT</span> below 10%<br>'
                '<span class="badge b-UNK">UNKNOWN</span> too few cartons to judge<br>'
                '<span class="badge b-VER">CHECK BY EYE</span> count may be wrong here '
                '(hidden cartons, low confidence or far shelf)</div>', unsafe_allow_html=True)


# ====================================================================== LIVE CHECK
@st.cache_data(show_spinner="Looking for this camera's recordings...")
def cached_folders(cam, reference_frame, frames_root):
    return fe.camera_folders(shelves[cam], frames_root or None)


@st.cache_resource(show_spinner="Loading the detector and occlusion model (first time only)...")
def get_models(weights, occ_w, rf_w):
    return fe.load_models(weights, occ_w, rf_w)


def page_live():
    page_head("Live check", "Run F1 on one camera frame and see every step, from the camera picture to the shelf status")
    steps = [("Camera picture", "One frame from the warehouse CCTV."),
             ("Find cartons", "YOLOv8 (trained on our footage) draws a box on every carton."),
             ("How hidden is each?", "The occlusion model rates every carton: visible, partly or mostly hidden."),
             ("Count per shelf", "Boxes are placed on the shelf outlines and compared with the full-stock picture."),
             ("Status + trust", "IN STOCK / LOW / STOCK-OUT, and a warning when the count may be wrong.")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><span class="step-n">{i}</span><div class="step-t">{t}</div><div class="step-d">{d}</div></div>'
        for i, (t, d) in enumerate(steps, 1)) + '</div>', unsafe_allow_html=True)
    if not cams:
        st.error("configs/shelves.json has no cameras. Draw shelves with scripts\\draw_shelves.py first.")
        return

    with st.container(key="panel-controls"):
        c1, c2, c3, c4, c5 = st.columns([1, 0.95, 1.7, 1.5, 0.75], vertical_alignment="bottom")
        with c1:
            cam = st.selectbox("Camera", cams, format_func=cam_name)
        cam_cfg = shelves[cam]
        with c2:
            source = st.radio("Frame from", ["Recorded days", "Upload"], horizontal=True)
        frame_path = None
        if source == "Recorded days":
            folders = cached_folders(cam, cam_cfg.get("reference_frame", ""), frames_root)
            if folders:
                with c3:
                    fi = st.selectbox(f"Day / video ({len(folders)} found for this camera)", range(len(folders)),
                                      format_func=lambda i: ("★ " if folders[i][2] else "") + folders[i][0],
                                      help="★ = the recording the shelf outlines were drawn on. Others are the same camera on other days.")
                frames = fe.camera_frames(folders[fi][1])
                with c4:
                    if frames:
                        frame_path = st.selectbox(f"Frame ({len(frames)})", frames, format_func=lambda q: q.name,
                                                  index=len(frames) // 2)
                    else:
                        st.warning("No frames in this folder.")
            else:
                with c3:
                    st.warning("Camera folders not found on this computer. Check 'Frames root' in Settings, or upload a frame.")
        else:
            with c3:
                up = st.file_uploader("Any frame from this camera, either day (jpg / png)", type=["jpg", "jpeg", "png"])
                if up is not None:
                    tmp = Path(tempfile.gettempdir()) / f"f1_upload_{up.name}"
                    tmp.write_bytes(up.getvalue())
                    frame_path = tmp
        with c5:
            run = st.button("Run F1", type="primary", disabled=frame_path is None, width="stretch")
        st.markdown('<div class="caption">Use frames from the same camera position the shelves were drawn on. '
                    'If the camera moved between days, the shelf outlines will not line up. Note: day-1 footage was used to '
                    'train the detector, so day-2 frames give the fairer demo.</div>', unsafe_allow_html=True)

    if run and frame_path is not None:
        if not fe.p(weights).exists():
            st.error(f"Detector weights not found: {weights}")
            return
        det, occ = get_models(weights, occ_w, rf_w)
        refs_key = f"refs_{cam}_{weights}"
        with st.spinner("Running F1 on this frame..."):
            if refs_key not in st.session_state:
                st.session_state[refs_key] = fe.reference_counts(det, cam_cfg, fe.DEFAULTS)
            res = fe.run_frame(det, occ, cam, cam_cfg, frame_path, st.session_state[refs_key], fe.DEFAULTS)
        st.session_state["live"] = {"cam": cam, **res}
        if occ is None:
            st.warning("Occlusion model not found, so steps 3 and 5 are skipped (counts only).")

    res = st.session_state.get("live")
    if not res:
        st.info("Choose a camera and a frame, then press **Run F1**.")
        return
    cam = res["cam"]
    st.write("")
    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        with st.container(key="panel-steps"):
            t1, t2, t3, t4 = st.tabs(["① Camera picture", "② Cartons found", "③ How hidden", "④ Shelf status"])
            with t1:
                show_image(fe.rgb(res["raw"]), "What the CCTV camera sees.")
            with t2:
                show_image(fe.rgb(res["detections"]), f"{res['n_boxes']} cartons found by the detector (blue boxes).")
            with t3:
                if res["occlusion"] is not None:
                    m = res["mix"]
                    show_image(fe.rgb(res["occlusion"]), f"Green = clearly visible ({m['low']}) · orange = partly hidden "
                                                         f"({m['medium']}) · red = mostly hidden ({m['high']}). Hidden cartons "
                                                         f"are missed more often, so they lower the trust of a shelf.")
                else:
                    st.info("Occlusion model not loaded.")
            with t4:
                show_image(fe.rgb(res["final"]), "Shelf outlines coloured by status (green in stock, orange low, red out, grey "
                                                 "unknown). Label: shelf, count, status, reliability, V = check by eye. "
                                                 f"{res['n_loose']} boxes were not on any shelf (white).")
    with right:
        recs = res["records"]
        cnt = {s: sum(r["status"] == s for r in recs) for s in ("IN STOCK", "LOW", "STOCK-OUT", "UNKNOWN")}
        st.markdown(f'<div class="section-title">{esc(cam_name(cam))} · result</div>'
                    f'<div class="section-sub">{cnt["IN STOCK"]} in stock · {cnt["LOW"]} low · {cnt["STOCK-OUT"]} out · '
                    f'{sum(is_verify(r) for r in recs)} to check by eye · saved to Overview</div>', unsafe_allow_html=True)
        order = {"STOCK-OUT": 0, "LOW": 1, "IN STOCK": 2, "UNKNOWN": 3}
        locs = loc_lookup(cam)
        rs = sorted(recs, key=lambda r: (order.get(r["status"], 4), not is_verify(r)))
        for i in range(0, len(rs), 2):
            cols = st.columns(2, gap="small")
            for col, r in zip(cols, rs[i:i + 2]):
                with col:
                    shelf_card(r, cam, locs.get(r["location_id"]), res["overlay"], camcal_for(cam), "live")
        with st.expander("What F1 sends to F4 for this frame (JSON)"):
            st.json(recs)


# ====================================================================== COVERAGE
def page_coverage():
    page_head("Camera coverage", "The cameras do not see the whole warehouse. This shows exactly which shelves each camera can judge.")
    cov = fe.load_coverage()
    if not cams:
        st.error("configs/shelves.json has no cameras.")
        return
    tabs = st.tabs([cam_name(c) for c in cams])
    for tab, c in zip(tabs, cams):
        with tab:
            cfg = shelves[c]
            locs = cfg.get("locations", [])
            rows = [r for r in cov if r.get("camera_id") == c]
            linked = sum(bool(l.get("slots")) for l in locs)
            kpi_row([("Shelf areas drawn", len(locs), "outlines on the camera view", None),
                     ("Main view", sum(r.get("view", "").lower() == "primary" for r in rows), "this camera is the best view", "#2563EB"),
                     ("Side view", sum(r.get("view", "").lower() == "secondary" for r in rows), "another camera sees it better", "#94A3B8"),
                     ("Linked to F4 layout", f"{linked}/{len(locs)}", "slots filled when F4's layout is final", None)])
            a, b = st.columns([1.3, 1], gap="medium")
            with a:
                img = read_image(cfg.get("reference_frame"))
                if img is not None:
                    show_image(fe.rgb(fe.draw_shelf_outlines(img, locs)), "Shelf outlines drawn with scripts\\draw_shelves.py "
                                                                           "using the warehouse's shelf names.")
                else:
                    st.info("Reference frame not found on this computer.")
            with b:
                if rows:
                    body = ""
                    for r in rows:
                        v = r.get("view", "").lower()
                        badge = ('<span class="badge b-BLUE">PRIMARY</span>' if v == "primary" else
                                 '<span class="badge b-UNK">SECONDARY</span>' if v == "secondary" else esc(r.get("view", "")))
                        sub = r.get("shelves", "")
                        sub = "" if sub == r.get("zone", "") else f'<br><span style="color:#64748B;font-size:12px">{esc(sub)}</span>'
                        body += (f'<tr><td><b>{esc(r.get("zone", ""))}</b>{sub}</td><td>{badge}</td>'
                                 f'<td>{esc(r.get("visible", ""))}</td><td style="color:#475569">{esc(r.get("notes", ""))}</td></tr>')
                    st.markdown(f'<table class="cov"><tr><th>Shelf</th><th>View</th><th>Visible</th><th>Notes</th></tr>'
                                f'{body}</table>', unsafe_allow_html=True)
                else:
                    st.info("No rows for this camera in configs/camera_coverage.csv.")


def page_over_time():
    from overtime import page_overtime
    page_overtime(page_head, kpi_row, show_image)


{"📊  Overview": page_overview, "🎥  Live check": page_live, "▶️  Live replay": page_over_time,
 "🗺️  Camera coverage": page_coverage}[page]()
