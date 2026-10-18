r"""
F1 new idea: TEMPORARY occlusion (pickers in front of shelves) + reliability-weighted temporal fusion.

Permanent occlusion (cartons behind cartons) is handled by the occlusion model and trust score.
Temporary occlusion (a picker standing in front of a shelf for a few seconds) is not: in that frame the count drops
and a single-frame system can raise a false LOW / STOCK-OUT alarm. This script tests a fix on dense frames
(a few seconds apart, made with dense_frames.py):

  1. People detection (COCO-pretrained YOLO, class 'person', no training). For each shelf area:
        blocked_share = part of the shelf polygon covered by person boxes.
        blocked_share >= --block-thr  ->  TEMPORARILY BLOCKED (this frame is not used for the decision).
  2. Reliability-weighted temporal fusion: the ratio (count / reference) of the last --window usable frames is
     combined with a weighted median; weight = reliability of that reading (1 - combined unreliability from
     confidence + occlusion trust + position, the same signals as CHECK BY EYE).
  3. Hysteresis: the reported status changes only after the fused status is the same for --persist usable frames.

Evaluation without extra labels: inside one short window (5-10 minutes) the real stock of a shelf barely changes.
Reference status per shelf and window = status of the median ratio over frames with NO person in that shelf.
  - status error rate   : share of frames whose status differs from the reference (single-frame vs proposed)
  - status flips        : how often the reported status changes inside the window
  - errors on blocked vs clear frames (do pickers cause the single-frame errors?)
  - count stability     : coefficient of variation of the ratio (single-frame vs fused)

Usage (from f1-inventory; ~3-6 s per frame on CPU):
  python scripts\temporal_blocking.py --camera G_H --frames D:\Research\dense_frames\GH_day2\dense_002000_002500 ^
         D:\Research\dense_frames\GH_day2\dense_031000_032000 --out outputs\temporal
  Re-analyse with other settings without running the models again:
  python scripts\temporal_blocking.py --analyze-only --out outputs\temporal --block-thr 0.15 --window 5 --persist 3
Outputs: <out>/frame_shelf_readings.csv, temporal_metrics.json, timeline_<shelf>.png, examples_blocked.jpg
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STATUSES = ["STOCK-OUT", "LOW", "IN STOCK"]


def status_of(ratio, low, out):
    if ratio is None:
        return "UNKNOWN"
    return "STOCK-OUT" if ratio < out else ("LOW" if ratio < low else "IN STOCK")


def t_of(name):
    m = re.search(r"__t(\d+)", name)
    return int(m.group(1)) if m else None


def weighted_median(vals, w):
    vals, w = np.asarray(vals, float), np.asarray(w, float)
    if not len(vals):
        return None
    if w.sum() <= 0:
        return float(np.median(vals))
    o = np.argsort(vals); vals, w = vals[o], w[o]
    c = np.cumsum(w)
    return float(vals[np.searchsorted(c, c[-1] / 2.0)])


# ------------------------------------------------------------------ 1) run the models on every frame
def collect(a):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "app")); sys.path.insert(0, str(root / "scripts"))
    import cv2
    import f1_engine as fe
    from ultralytics import YOLO
    cams = fe.load_shelves()
    if a.camera not in cams:
        raise SystemExit(f"Camera {a.camera} not in configs/shelves.json (have: {', '.join(cams)})")
    cam = cams[a.camera]
    s = dict(fe.DEFAULTS)
    det, occ = fe.load_models(s["weights"], s["occlusion_cnn"], s["rf_model"])
    person = YOLO(a.person_model)
    refs = fe.reference_counts(det, cam, s)
    camcal, _, _ = fe.cs.load_camera_calibration(str(fe.p(s["calibration"])), a.camera)
    rows, out = [], Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in a.frames:
        fs = sorted(q for q in Path(d).iterdir() if q.suffix.lower() in (".jpg", ".png"))
        frames += [(Path(d).name, q) for q in fs if t_of(q.name) is not None]
    print(f"{len(frames)} frames, {len(cam['locations'])} shelf areas, camera {a.camera}")
    for i, (win, fp) in enumerate(frames, 1):
        img = cv2.imread(str(fp)); h, w = img.shape[:2]
        pmask, pboxes = person_mask(person, img, a)
        res = fe.run_frame(det, occ, a.camera, cam, fp, refs, s, save=False)
        by_id = {r["location_id"]: r for r in res["records"]}
        for loc in cam["locations"]:
            rec = by_id.get(loc["location_id"])
            if rec is None:
                continue
            poly = np.array([[x * w, y * h] for x, y in fe.cs.loc_polygon(loc)], np.int32)
            smask = np.zeros((h, w), np.uint8); cv2.fillPoly(smask, [poly], 1)
            area = max(int(smask.sum()), 1)
            blocked = float((smask & pmask).sum()) / area
            parts = fe.reliability_parts(rec, camcal)
            rel = 1.0 - float(np.mean(list(parts.values()))) if parts else (rec.get("trust") or 1.0)
            ref = rec.get("reference_count")
            ratio = rec.get("ratio_to_reference")
            if ratio is None and ref:
                ratio = rec.get("count", 0) / ref
            rows.append({"window": win, "frame": fp.name, "t_sec": t_of(fp.name), "shelf": rec["location_id"],
                         "count": rec.get("count"), "reference": ref,
                         "ratio": None if ratio is None else round(float(ratio), 4),
                         "pipeline_status": rec.get("status"), "reliability": round(float(rel), 4),
                         "people_in_frame": int(len(pboxes)), "blocked_share": round(blocked, 4),
                         "frame_path": str(fp)})
        if i % 20 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}")
    with open(out / "frame_shelf_readings.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"Saved {out / 'frame_shelf_readings.csv'}")
    return rows


# ------------------------------------------------------------------ 2) temporal logic + evaluation (no models)
def simulate(seq, a):
    """seq: readings of one shelf in one window, sorted by time. Adds single / fused / proposed status."""
    hist, confirmed, streak_status, streak = [], None, None, 0
    for r in seq:
        r["single_status"] = status_of(r["ratio"], a.low, a.out_ratio)
        r["blocked"] = r["blocked_share"] >= a.block_thr
        if not r["blocked"] and r["ratio"] is not None:
            hist.append((r["ratio"], max(r["reliability"], 0.05)))
            hist = hist[-a.window:]
        fused = weighted_median([v for v, _ in hist], [w for _, w in hist]) if hist else None
        r["fused_ratio"] = fused
        fstat = status_of(fused, a.low, a.out_ratio) if fused is not None else None
        if r["blocked"] or fstat is None:
            r["proposed_status"] = confirmed or "TEMPORARILY BLOCKED"
            r["shown"] = "TEMPORARILY BLOCKED"
            continue
        if confirmed is None:
            confirmed = fstat
        if fstat != confirmed:
            streak = streak + 1 if fstat == streak_status else 1
            streak_status = fstat
            if streak >= a.persist:
                confirmed, streak, streak_status = fstat, 0, None
        else:
            streak, streak_status = 0, None
        r["proposed_status"] = confirmed
        r["shown"] = confirmed
    return seq


def flips(statuses):
    s = [x for x in statuses if x]
    return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])


def evaluate(rows, a):
    for r in rows:
        r["ratio"] = None if r["ratio"] in ("", None) else float(r["ratio"])
        r["blocked_share"] = float(r["blocked_share"]); r["reliability"] = float(r["reliability"])
        r["t_sec"] = int(r["t_sec"]); r["people_in_frame"] = int(r["people_in_frame"])
    groups = defaultdict(list)
    for r in rows:
        if r["ratio"] is not None:
            groups[(r["window"], r["shelf"])].append(r)
    res = {"settings": {"block_thr": a.block_thr, "window": a.window, "persist": a.persist, "low": a.low, "out_ratio": a.out_ratio},
           "per_shelf": {}}
    tot = defaultdict(float)
    for (win, shelf), seq in sorted(groups.items()):
        seq.sort(key=lambda r: r["t_sec"])
        simulate(seq, a)
        clear = [r["ratio"] for r in seq if r["blocked_share"] == 0]
        base = clear if len(clear) >= 3 else [r["ratio"] for r in seq]
        ref_status = status_of(float(np.median(base)), a.low, a.out_ratio)
        n = len(seq)
        single_err = sum(r["single_status"] != ref_status for r in seq)
        prop_err = sum(r["proposed_status"] != ref_status for r in seq)
        blocked = [r for r in seq if r["blocked"]]; clear_r = [r for r in seq if not r["blocked"]]
        single_ratio = np.array([r["ratio"] for r in seq])
        fused_ratio = np.array([r["fused_ratio"] for r in seq if r["fused_ratio"] is not None])
        cv = lambda x: float(np.std(x) / np.mean(x)) if len(x) and np.mean(x) > 0 else float("nan")
        d = {"frames": n, "reference_status": ref_status, "reference_from_clear_frames": len(clear) >= 3,
             "blocked_frames": len(blocked),
             "single_errors": single_err, "proposed_errors": prop_err,
             "single_errors_on_blocked": sum(r["single_status"] != ref_status for r in blocked),
             "single_errors_on_clear": sum(r["single_status"] != ref_status for r in clear_r),
             "single_flips": flips([r["single_status"] for r in seq]),
             "proposed_flips": flips([r["proposed_status"] for r in seq]),
             "false_alarms_single": sum(r["single_status"] in ("LOW", "STOCK-OUT") and ref_status == "IN STOCK" for r in seq),
             "false_alarms_proposed": sum(r["proposed_status"] in ("LOW", "STOCK-OUT") and ref_status == "IN STOCK" for r in seq),
             "cv_single": round(cv(single_ratio), 4), "cv_fused": round(cv(fused_ratio), 4)}
        res["per_shelf"][f"{win} | {shelf}"] = d
        for k in ("frames", "blocked_frames", "single_errors", "proposed_errors", "single_errors_on_blocked",
                  "single_errors_on_clear", "single_flips", "proposed_flips", "false_alarms_single", "false_alarms_proposed"):
            tot[k] += d[k]
        if d["cv_single"] == d["cv_single"]:
            tot["cv_single_sum"] += d["cv_single"]; tot["cv_fused_sum"] += d["cv_fused"]; tot["cv_n"] += 1
    n, nb = tot["frames"], tot["blocked_frames"]
    summ = {"shelf_frames": int(n), "blocked_shelf_frames": int(nb), "blocked_share": round(nb / max(n, 1), 3),
            "status_error_single": round(tot["single_errors"] / max(n, 1), 3),
            "status_error_proposed": round(tot["proposed_errors"] / max(n, 1), 3),
            "single_error_rate_on_blocked": round(tot["single_errors_on_blocked"] / max(nb, 1), 3),
            "single_error_rate_on_clear": round(tot["single_errors_on_clear"] / max(n - nb, 1), 3),
            "false_alarms_single": int(tot["false_alarms_single"]), "false_alarms_proposed": int(tot["false_alarms_proposed"]),
            "status_flips_single": int(tot["single_flips"]), "status_flips_proposed": int(tot["proposed_flips"]),
            "mean_cv_single": round(tot["cv_single_sum"] / max(tot["cv_n"], 1), 3),
            "mean_cv_fused": round(tot["cv_fused_sum"] / max(tot["cv_n"], 1), 3)}
    res["summary"] = summ
    return res, groups


def report(res):
    s = res["summary"]
    print("\n=== Temporary occlusion: single-frame vs proposed (blocked detection + reliability-weighted fusion + hysteresis) ===")
    print(f"settings: {res['settings']}")
    print(f"shelf readings: {s['shelf_frames']}  (blocked by a person: {s['blocked_shelf_frames']} = {s['blocked_share']:.0%})")
    print(f"{'':34s} {'single-frame':>13s} {'proposed':>10s}")
    print(f"{'status differs from reference':34s} {s['status_error_single']:13.1%} {s['status_error_proposed']:10.1%}")
    print(f"{'false LOW/STOCK-OUT alarms':34s} {s['false_alarms_single']:13d} {s['false_alarms_proposed']:10d}")
    print(f"{'status flips inside windows':34s} {s['status_flips_single']:13d} {s['status_flips_proposed']:10d}")
    print(f"{'ratio variation (mean CV)':34s} {s['mean_cv_single']:13.3f} {s['mean_cv_fused']:10.3f}")
    print(f"single-frame error rate: blocked frames {s['single_error_rate_on_blocked']:.1%} vs clear frames "
          f"{s['single_error_rate_on_clear']:.1%}  (do pickers cause the errors?)")


def plots(groups, out, a):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    for (win, shelf), seq in sorted(groups.items()):
        t = np.array([r["t_sec"] for r in seq]); t = (t - t.min()) / 60.0
        fig, ax = plt.subplots(figsize=(8, 3))
        for r, x in zip(seq, t):
            if r["blocked"]:
                ax.axvspan(x - 0.025, x + 0.025, color="orange", alpha=0.35, lw=0)
        ax.plot(t, [r["ratio"] for r in seq], ".-", color="grey", lw=1, label="single frame")
        ax.plot(t, [r["fused_ratio"] if r["fused_ratio"] is not None else np.nan for r in seq], "-", color="#2563EB", lw=2,
                label="proposed (fused)")
        ax.axhline(a.low, ls="--", c="#D97706", lw=1); ax.axhline(a.out_ratio, ls="--", c="#DC2626", lw=1)
        ax.set_xlabel("minutes"); ax.set_ylabel("count / reference")
        ax.set_title(f"{shelf} ({win}); orange = picker blocking the shelf", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / f"timeline{getattr(a, '_tag', '')}_{win}_{shelf}.png", dpi=140); plt.close(fig)


def examples(rows, out, n=6):
    try:
        import cv2
    except Exception:
        return
    bl = sorted([r for r in rows if r.get("blocked")], key=lambda r: -r["blocked_share"])
    seen, tiles = set(), []
    for r in bl:
        if r["frame"] in seen or not Path(r["frame_path"]).exists():
            continue
        seen.add(r["frame"])
        im = cv2.imread(r["frame_path"]); im = cv2.resize(im, (320, 320))
        cv2.putText(im, f"{r['shelf']}: {r['blocked_share']:.0%} blocked, single={r['single_status']}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
        if len(tiles) == n:
            break
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(str(out / "examples_blocked.jpg"), grid)


def person_mask(person, img, a):
    import cv2
    h, w = img.shape[:2]
    r = person.predict(img, classes=[0], conf=a.person_conf, imgsz=a.person_imgsz, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    m = np.zeros((h, w), np.uint8)
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1, y1 = int(b[0] - a.person_pad * bw), int(b[1] - a.person_pad * bh)
        x2, y2 = int(b[2] + a.person_pad * bw), int(b[3] + a.person_pad * bh)
        cv2.rectangle(m, (max(x1, 0), max(y1, 0)), (min(x2, w - 1), min(y2, h - 1)), 1, -1)
    return m, boxes


def recompute_people(a, rows):
    """Redo ONLY the people detection on the frames already processed (fast): new blocked_share per shelf."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "app")); sys.path.insert(0, str(root / "scripts"))
    import cv2
    import f1_engine as fe
    from ultralytics import YOLO
    cam = fe.load_shelves()[a.camera]
    polys = {l["location_id"]: l for l in cam["locations"]}
    person = YOLO(a.person_model)
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame_path"]].append(r)
    print(f"Re-running people detection only (conf {a.person_conf}, imgsz {a.person_imgsz}, box pad {a.person_pad}) "
          f"on {len(by_frame)} frames ...")
    for i, (fp, rs) in enumerate(sorted(by_frame.items()), 1):
        img = cv2.imread(fp); h, w = img.shape[:2]
        m, boxes = person_mask(person, img, a)
        for r in rs:
            loc = polys.get(r["shelf"])
            if loc is None:
                continue
            poly = np.array([[x * w, y * h] for x, y in fe.cs.loc_polygon(loc)], np.int32)
            sm = np.zeros((h, w), np.uint8); cv2.fillPoly(sm, [poly], 1)
            r["blocked_share"] = round(float((sm & m).sum()) / max(int(sm.sum()), 1), 4)
            r["people_in_frame"] = int(len(boxes))
        if i % 50 == 0 or i == len(by_frame):
            print(f"  {i}/{len(by_frame)}")
    return rows


def unexplained_drops(groups, out, a, n=9):
    """Frames where a shelf's ratio falls well below its window median while NO person is detected on it."""
    hits = []
    for (win, shelf), seq in groups.items():
        med = float(np.median([r["ratio"] for r in seq]))
        for r in seq:
            if r["blocked_share"] == 0 and med > 0 and r["ratio"] < 0.6 * med:
                hits.append((r["ratio"] / med, win, shelf, r))
    hits.sort(key=lambda x: x[0])
    print(f"\nUnexplained drops (ratio < 60% of the shelf's usual value, no person detected on it): {len(hits)}")
    for _, win, shelf, r in hits[:12]:
        print(f"  {win} | {shelf} | {r['frame']} | ratio {r['ratio']:.2f}")
    try:
        import cv2
        tiles = []
        for _, win, shelf, r in hits[:n]:
            if Path(r["frame_path"]).exists():
                im = cv2.resize(cv2.imread(r["frame_path"]), (320, 320))
                cv2.putText(im, f"{shelf} ratio {r['ratio']:.2f}, no person found", (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(im)
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(out / "unexplained_drops.jpg"),
                        np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
            print(f"  pictures: {out / 'unexplained_drops.jpg'}")
    except Exception as e:
        print(f"  (pictures skipped: {e})")
    return len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="G_H")
    ap.add_argument("--frames", nargs="+", help="dense frame folders (from dense_frames.py extract)")
    ap.add_argument("--out", default="outputs/temporal")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--person-model", default="yolov8s.pt"); ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--person-imgsz", type=int, default=640)
    ap.add_argument("--person-pad", type=float, default=0.0, help="enlarge each person box by this share of its size")
    ap.add_argument("--recompute-people", action="store_true",
                    help="redo only the people detection on the saved readings (fast), then evaluate")
    ap.add_argument("--readings", default="frame_shelf_readings.csv", help="readings file inside --out")
    ap.add_argument("--tag", default="", help="suffix for the output files of this run, e.g. v2")
    ap.add_argument("--block-thr", type=float, default=0.10, help="share of a shelf covered by people -> blocked")
    ap.add_argument("--window", type=int, default=5, help="usable readings combined")
    ap.add_argument("--persist", type=int, default=3, help="readings a new status must last before it is reported")
    ap.add_argument("--low", type=float, default=0.5); ap.add_argument("--out-ratio", type=float, default=0.1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    a._tag = tag
    if a.analyze_only or a.recompute_people:
        with open(out / a.readings, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if a.recompute_people:
            rows = recompute_people(a, rows)
            with open(out / f"frame_shelf_readings{tag}.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
            print(f"Saved {out / f'frame_shelf_readings{tag}.csv'}")
    else:
        if not a.frames:
            raise SystemExit("Give --frames folder(s), or --analyze-only")
        rows = collect(a)
        rows = [{k: ("" if v is None else v) for k, v in r.items()} for r in rows]
    res, groups = evaluate(rows, a)
    report(res)
    res["unexplained_drops"] = unexplained_drops(groups, out, a)
    res["settings"].update({"person_conf": a.person_conf, "person_imgsz": a.person_imgsz, "person_pad": a.person_pad})
    with open(out / f"temporal_metrics{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    plots(groups, out, a)
    examples([r for g in groups.values() for r in g], out)
    print(f"\nSaved {out / f'temporal_metrics{tag}.json'}, timeline_*.png and examples_blocked.jpg in {out}")


if __name__ == "__main__":
    main()
