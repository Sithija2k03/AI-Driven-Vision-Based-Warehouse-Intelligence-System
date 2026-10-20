r"""
BEFORE vs AFTER proof for F1's occlusion handling, on the annotated day-2 frames (never used for training),
where the TRUE number of cartons on every shelf is known from the labels.

  BEFORE = plain counting: detector count -> status. No warning, ever.
  AFTER  = F1: same count + occlusion model + reliability score -> "CHECK BY EYE" on shelves it should not trust.

For every shelf in every annotated frame it measures:
  recall       = labelled cartons the detector found / labelled cartons      (how wrong the count really is)
  BAD shelf    = detector found less than half of the cartons (recall < 0.5) (count badly wrong)
and compares the two systems:
  - share of BAD shelves that get a warning        (before: 0%)
  - share of BAD shelves among the shelves reported as reliable (what a manager would trust)
  - same number of warnings given by detector confidence only (standard baseline) -> does occlusion add value?
  - flagged vs not flagged shelves: share of "mostly hidden" cartons and real recall (links the warning to occlusion)

Outputs in --out:
  shelf_readings.csv, before_after.json, before_after.png (summary chart),
  example_flagged_1..3.png (one shelf: plain counting | what is really there | F1), example_reliable.png

Usage (from the f1-inventory folder, f1-inventory environment; a few minutes):
  python scripts\occlusion_before_after.py --frames-root "D:\Research\research camera data"
Re-make the numbers and chart only (no models):
  python scripts\occlusion_before_after.py --analyze-only
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app")); sys.path.insert(0, str(ROOT / "scripts"))

BATCHES = {"G_H": "location_G_H_b2", "X_5": "location_X_5_b2", "CP2": "location_checkingPart_2_b2"}


# ------------------------------------------------------------------ helpers
def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def matched(gts, preds, thr=0.5):
    """Boolean per labelled box: found by a detection (greedy one-to-one, IoU >= thr)."""
    found = np.zeros(len(gts), bool)
    if not len(gts) or not len(preds):
        return found
    m = iou_matrix(gts, preds); used = set()
    for gi, pi in np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]:
        if m[gi, pi] < thr:
            break
        if found[gi] or pi in used:
            continue
        found[gi] = True; used.add(pi)
    return found


def read_labels(path, w, h):
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0))
    return out


# ------------------------------------------------------------------ 1) run F1 on every annotated shelf
def collect(a):
    import cv2
    import f1_engine as fe
    cs = fe.cs
    cams = fe.load_shelves()
    s = dict(fe.DEFAULTS)
    det, occ = fe.load_models(s["weights"], s["occlusion_cnn"], s["rf_model"])
    if occ is None:
        raise SystemExit(f"Occlusion model not found ({s['occlusion_cnn']}): F1 cannot run without it.")
    index = {q.name: q for q in Path(a.frames_root).rglob("*")
             if q.suffix.lower() in (".jpg", ".jpeg", ".png") and q.parent.name == "extracted_frames"}
    rows, keep = [], []
    for cam_id, batch in BATCHES.items():
        if cam_id not in cams:
            print(f"!! {cam_id}: no shelves in configs/shelves.json, skipped"); continue
        cam = cams[cam_id]
        lab_dir = ROOT / a.converted / batch / "labels"
        files = sorted(f for f in lab_dir.glob("*.txt") if (f.stem + ".jpg") in index)
        if not files:
            print(f"!! {cam_id}: no label files with matching frames in {lab_dir}"); continue
        refs = fe.reference_counts(det, cam, s)
        camcal, _, _ = cs.load_camera_calibration(str(fe.p(s["calibration"])), cam_id)
        locs = cam["locations"]
        for lf in files:
            fp = index[lf.stem + ".jpg"]
            img = cv2.imread(str(fp)); h, w = img.shape[:2]
            res = fe.run_frame(det, occ, cam_id, cam, fp, refs, s, save=False)
            by_id = {r["location_id"]: r for r in res["records"]}
            boxes, _, _ = cs.detect(det, fp, s["conf"], s["imgsz"], s["max_det"], s["iou"])
            levels = occ.predict(img, boxes).argmax(1) if boxes else np.zeros(0, int)
            lvl_of = {id(b): int(l) for b, l in zip(boxes, levels)}
            det_loc = cs.assign_boxes(boxes, locs, w, h, s["anchor"])
            gts = read_labels(lf, w, h)
            gt_loc = cs.assign_boxes(gts, locs, w, h, s["anchor"])
            all_det = np.array([b[:4] for b in boxes]).reshape(-1, 4)
            for loc in locs:
                lid = loc["location_id"]; rec = by_id.get(lid)
                g = gt_loc.get(lid, [])
                if rec is None or len(g) < a.min_gt:
                    continue
                f = matched(np.array([b[:4] for b in g]), all_det)
                d = det_loc.get(lid, [])
                lv = [lvl_of[id(b)] for b in d]
                conf_pct = cs.pct_unreliable(rec["mean_confidence"], camcal["ref_conf"]) \
                    if camcal and rec.get("mean_confidence") is not None else None
                rows.append({"camera": cam_id, "frame": fp.name, "shelf": lid,
                             "true_count": len(g), "detected": len(d), "found": int(f.sum()),
                             "recall": round(float(f.mean()), 4),
                             "count_error": round((len(d) - len(g)) / len(g), 4),
                             "share_mostly_hidden": round(float(np.mean(np.array(lv) == 2)), 4) if lv else None,
                             "share_partly_or_mostly": round(float(np.mean(np.array(lv) >= 1)), 4) if lv else None,
                             "trust": rec.get("trust"), "mean_confidence": rec.get("mean_confidence"),
                             "confidence_unreliability": None if conf_pct is None else round(conf_pct, 4),
                             "reliability": rec.get("reliability"),
                             "f1_check_by_eye": int(bool(rec.get("verify_manually"))),
                             "status": rec.get("status"), "raw_status": rec.get("raw_status", rec.get("status"))})
                keep.append({"row": len(rows) - 1, "frame_path": str(fp), "loc": loc, "det": d, "levels": lv,
                             "gt": g, "found": f, "rec": rec, "camcal": camcal})
        print(f"{cam_id}: {len(files)} frames done")
    if not rows:
        raise SystemExit("No shelf readings collected.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "shelf_readings.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    return rows, keep


# ------------------------------------------------------------------ 2) before vs after numbers
def num(v):
    return float(v) if v not in (None, "", "None", "nan") else np.nan


def analyze(rows, a):
    R = np.array([num(r["recall"]) for r in rows]); bad = R < a.bad_recall
    flag = np.array([int(num(r["f1_check_by_eye"]) or 0) for r in rows]).astype(bool)
    err = np.abs(np.array([num(r["count_error"]) for r in rows]))
    hid = np.array([num(r["share_mostly_hidden"]) for r in rows])
    confu = np.array([num(r["confidence_unreliability"]) for r in rows])
    n, k = len(rows), int(flag.sum())
    rate = lambda m: float(m.mean()) if m.size else float("nan")

    # same number of warnings from detector confidence only (standard baseline)
    conf_flag = np.zeros(n, bool)
    if k and np.isfinite(confu).any():
        order = np.argsort(-np.nan_to_num(confu, nan=-1)); conf_flag[order[:k]] = True

    res = {"shelf_readings": n, "bad_definition": f"detector found < {a.bad_recall:.0%} of the labelled cartons",
           "bad_shelves": int(bad.sum()), "bad_share": round(rate(bad), 3),
           "BEFORE_plain_counting": {
               "warnings": 0, "bad_shelves_warned": 0.0,
               "bad_share_among_shelves_reported_as_reliable": round(rate(bad), 3),
               "mean_recall_of_reported_shelves": round(rate(R), 3),
               "mean_abs_count_error_of_reported_shelves": round(rate(err), 3)},
           "AFTER_F1_occlusion_aware": {
               "warnings": k, "share_flagged": round(k / n, 3),
               "bad_shelves_warned": round(rate(flag[bad]), 3) if bad.any() else None,
               "good_shelves_warned_false_alarm": round(rate(flag[~bad]), 3) if (~bad).any() else None,
               "bad_share_among_shelves_reported_as_reliable": round(rate(bad[~flag]), 3),
               "mean_recall_of_reported_as_reliable": round(rate(R[~flag]), 3),
               "mean_abs_count_error_of_reported_as_reliable": round(rate(err[~flag]), 3)},
           "SAME_NUMBER_OF_WARNINGS_confidence_only": {
               "bad_shelves_warned": round(rate(conf_flag[bad]), 3) if bad.any() and k else None,
               "bad_share_among_shelves_reported_as_reliable": round(rate(bad[~conf_flag]), 3) if k else None},
           "RANDOM_warnings_expected_bad_warned": round(k / n, 3),
           "WHY_flagged_flagged_vs_not": {
               "flagged": {"mean_recall": round(rate(R[flag]), 3), "share_mostly_hidden": round(float(np.nanmean(hid[flag])), 3) if flag.any() else None},
               "not_flagged": {"mean_recall": round(rate(R[~flag]), 3), "share_mostly_hidden": round(float(np.nanmean(hid[~flag])), 3) if (~flag).any() else None}}}
    ok = np.isfinite(hid)
    if ok.sum() > 5:
        from scipy.stats import spearmanr
        rho, pv = spearmanr(hid[ok], R[ok])
        res["mostly_hidden_vs_recall_spearman"] = {"rho": round(float(rho), 3), "p": float(f"{pv:.2g}")}
    per = {}
    for c in sorted({r["camera"] for r in rows}):
        m = np.array([r["camera"] == c for r in rows])
        per[c] = {"shelves": int(m.sum()), "bad": int(bad[m].sum()),
                  "bad_warned": round(rate(flag[m & bad] if (m & bad).any() else np.zeros(0)), 3) if (m & bad).any() else None,
                  "false_alarm": round(rate(flag[m & ~bad]), 3) if (m & ~bad).any() else None}
    res["per_camera"] = per
    cal = ROOT / "configs" / "calibration.json"
    if cal.exists():
        rel = json.load(open(cal, encoding="utf-8")).get("reliability", {})
        res["note_threshold"] = ("The CHECK BY EYE threshold was chosen on day-2 grid cells (Youden). Fairer check, threshold "
                                 "learned on the other cameras:")
        res["leave_one_camera_out_from_calibration"] = rel.get("leave_one_camera_out")
    return res, bad, flag, conf_flag


def report(res):
    B, A, C = res["BEFORE_plain_counting"], res["AFTER_F1_occlusion_aware"], res["SAME_NUMBER_OF_WARNINGS_confidence_only"]
    pc = lambda v: "-" if v is None else f"{v:.0%}"
    print(f"\n=== Occlusion handling: BEFORE vs AFTER on {res['shelf_readings']} shelf readings (day-2, true counts known) ===")
    print(f"BAD shelf = {res['bad_definition']}: {res['bad_shelves']} shelves ({pc(res['bad_share'])})\n")
    print(f"{'':52s}{'BEFORE':>10s}{'AFTER (F1)':>12s}")
    print(f"{'Bad shelves that get a warning':52s}{pc(B['bad_shelves_warned']):>10s}{pc(A['bad_shelves_warned']):>12s}")
    print(f"{'Good shelves wrongly warned (false alarm)':52s}{'0%':>10s}{pc(A['good_shelves_warned_false_alarm']):>12s}")
    print(f"{'Bad shelves among those reported as reliable':52s}{pc(B['bad_share_among_shelves_reported_as_reliable']):>10s}"
          f"{pc(A['bad_share_among_shelves_reported_as_reliable']):>12s}")
    print(f"{'Mean share of cartons found, reliable shelves':52s}{pc(B['mean_recall_of_reported_shelves']):>10s}"
          f"{pc(A['mean_recall_of_reported_as_reliable']):>12s}")
    print(f"{'Mean count error, reliable shelves':52s}{pc(B['mean_abs_count_error_of_reported_shelves']):>10s}"
          f"{pc(A['mean_abs_count_error_of_reported_as_reliable']):>12s}")
    print(f"\nSame number of warnings ({A['warnings']}), chosen by detector confidence only: bad shelves warned "
          f"{pc(C['bad_shelves_warned'])} (F1 {pc(A['bad_shelves_warned'])}; random {pc(res['RANDOM_warnings_expected_bad_warned'])})")
    W = res["WHY_flagged_flagged_vs_not"]
    print(f"Flagged shelves: {pc(W['flagged']['share_mostly_hidden'])} of cartons mostly hidden, {pc(W['flagged']['mean_recall'])} found | "
          f"not flagged: {pc(W['not_flagged']['share_mostly_hidden'])} mostly hidden, {pc(W['not_flagged']['mean_recall'])} found")
    if "mostly_hidden_vs_recall_spearman" in res:
        sp = res["mostly_hidden_vs_recall_spearman"]
        print(f"More mostly-hidden cartons -> fewer found: Spearman rho {sp['rho']:+.2f} (p {sp['p']})")
    print("Per camera: " + " | ".join(f"{c}: {v['bad']}/{v['shelves']} bad, warned {pc(v['bad_warned'])}, false alarm {pc(v['false_alarm'])}"
                                     for c, v in res["per_camera"].items()))
    if res.get("leave_one_camera_out_from_calibration"):
        print("Threshold learned on the other cameras (fairer): " + " | ".join(
            f"{c}: caught {pc(v.get('unreliable_caught'))}, false alarms {pc(v.get('false_alarms'))}"
            for c, v in res["leave_one_camera_out_from_calibration"].items()))


def save_fig(fig, path):
    """Save a figure; if Windows refuses (file open in a viewer / being synced), save under a new name instead."""
    import time
    path = Path(path)
    try:
        fig.savefig(path, dpi=160)
        return path
    except OSError as e:
        alt = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        print(f"!! Could not write {path.name} ({e.strerror}); is it open in another program? Saved as {alt.name} instead.")
        fig.savefig(alt, dpi=160)
        return alt


# ------------------------------------------------------------------ 2b) the failure occlusion explains: UNDER-counting
def undercount(rows, a, k):
    """BAD here = count at least --under below the true count (what a stock decision actually suffers from).
    Compares warning signals with the same number of warnings, and shelf-level correlations."""
    from scipy.stats import spearmanr
    err = np.array([num(r["count_error"]) for r in rows]); under = err <= -a.under
    sig = {"Detector confidence only": np.array([num(r["confidence_unreliability"]) for r in rows]),
           "Occlusion: share mostly hidden": np.array([num(r["share_mostly_hidden"]) for r in rows]),
           "Occlusion: trust": 1 - np.array([num(r["trust"]) for r in rows]),
           "F1 current (combined)": np.array([int(num(r["f1_check_by_eye"]) or 0) for r in rows], float)}
    out = {"definition": f"count at least {a.under:.0%} below the true count", "readings": len(rows),
           "undercounted": int(under.sum()), "warnings_per_method": k, "same_number_of_warnings": {}}
    for name, s in sig.items():
        f = s.astype(bool) if name.startswith("F1") else np.isin(np.arange(len(s)), np.argsort(-np.nan_to_num(s, nan=-9))[:k])
        out["same_number_of_warnings"][name] = {
            "undercounted_warned": round(float(f[under].mean()), 3) if under.any() else None,
            "undercounted_among_reliable": round(float(under[~f].mean()), 3)}
    # one point per shelf (frames of the same shelf are near-duplicates)
    shelves = {}
    for r in rows:
        shelves.setdefault((r["camera"], r["shelf"]), []).append(r)
    S = [{"camera": c, "shelf": s, "count_error": float(np.mean([num(x["count_error"]) for x in v])),
          "mostly_hidden": float(np.nanmean([num(x["share_mostly_hidden"]) for x in v])),
          "conf_unrel": float(np.nanmean([num(x["confidence_unreliability"]) for x in v])), "n": len(v)}
         for (c, s), v in shelves.items()]
    e = np.array([x["count_error"] for x in S])
    out["per_shelf"] = S
    out["shelf_level_spearman_with_count_error"] = {
        "mostly_hidden": round(float(spearmanr([x["mostly_hidden"] for x in S], e)[0]), 3),
        "confidence_unreliability": round(float(spearmanr([x["conf_unrel"] for x in S], e)[0]), 3), "shelves": len(S)}
    return out


def report_under(u):
    pc = lambda v: "-" if v is None else f"{v:.0%}"
    print(f"\n=== The error occlusion explains: UNDER-COUNTING ({u['definition']}) ===")
    print(f"{u['undercounted']} of {u['readings']} readings under-counted. Same number of warnings ({u['warnings_per_method']}) for each method:")
    print(f"{'':34s}{'under-counted warned':>22s}{'under-counted among reliable':>30s}")
    print(f"{'Plain counting (before)':34s}{'0%':>22s}{pc(u['undercounted'] / u['readings']):>30s}")
    for n, v in u["same_number_of_warnings"].items():
        print(f"{n:34s}{pc(v['undercounted_warned']):>22s}{pc(v['undercounted_among_reliable']):>30s}")
    sp = u["shelf_level_spearman_with_count_error"]
    print(f"One point per shelf ({sp['shelves']} shelves), Spearman with count error (negative = more under-counting): "
          f"mostly hidden {sp['mostly_hidden']:+.2f}, detector confidence {sp['confidence_unreliability']:+.2f}")
    print("Caution: the under-count definition was added after the first run, and there are only "
          f"{sp['shelves']} distinct shelves (frames of one shelf are near-duplicates).")


def chart_under(u, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = u["per_shelf"]; cols = {"G_H": "#1D4ED8", "X_5": "#D97706", "CP2": "#16A34A"}
    sp = u["shelf_level_spearman_with_count_error"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axi, key, xl, rho in ((ax[0], "mostly_hidden", "Share of cartons rated MOSTLY HIDDEN by F1", sp["mostly_hidden"]),
                              (ax[1], "conf_unrel", "Detector confidence: how unusual for this camera (1 = least sure)",
                               sp["confidence_unreliability"])):
        for x in S:
            axi.scatter(x[key], x["count_error"] * 100, s=40 + 4 * x["n"], color=cols.get(x["camera"], "#64748B"), alpha=0.85,
                        edgecolor="white", linewidth=0.8)
            axi.annotate(x["shelf"], (x[key], x["count_error"] * 100), fontsize=7, xytext=(4, 3), textcoords="offset points",
                         color="#334155")
        axi.axhline(0, color="#94A3B8", lw=0.8); axi.axhline(-100 * 0.25, color="#DC2626", lw=0.8, ls=":")
        axi.set_xlabel(xl, fontsize=9); axi.set_title(f"Spearman rho {rho:+.2f}", fontsize=10)
        axi.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Count error % (below 0 = cartons missed)")
    for c, col in cols.items():
        ax[1].scatter([], [], color=col, label=c.replace("_", "-"))
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"More hidden cartons -> more under-counting; detector confidence does not show this "
                 f"(one point per shelf, {sp['shelves']} shelves, day-2)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save_fig(fig, out / "occlusion_vs_undercount.png"); plt.close(fig)


# ------------------------------------------------------------------ 3) pictures
def chart(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B, A, C = res["BEFORE_plain_counting"], res["AFTER_F1_occlusion_aware"], res["SAME_NUMBER_OF_WARNINGS_confidence_only"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    names = ["Plain counting\n(before)", "Confidence only\n(same no. of warnings)", "F1 occlusion-aware\n(after)"]
    cols = ["#94A3B8", "#60A5FA", "#1D4ED8"]
    v1 = [0, C["bad_shelves_warned"] or 0, A["bad_shelves_warned"] or 0]
    v2 = [B["bad_share_among_shelves_reported_as_reliable"], C["bad_share_among_shelves_reported_as_reliable"] or 0,
          A["bad_share_among_shelves_reported_as_reliable"]]
    for axi, vals, title in ((ax[0], v1, "Badly wrong shelves that get a warning\n(higher is better)"),
                             (ax[1], v2, "Badly wrong shelves among those\nreported as reliable (lower is better)")):
        bars = axi.bar(names, [x * 100 for x in vals], color=cols)
        for b, x in zip(bars, vals):
            axi.text(b.get_x() + b.get_width() / 2, x * 100 + 1, f"{x:.0%}", ha="center", fontsize=11, fontweight="bold")
        axi.set_title(title, fontsize=11); axi.set_ylabel("%"); axi.set_ylim(0, max(100, max(vals) * 115))
        axi.spines[["top", "right"]].set_visible(False); axi.tick_params(axis="x", labelsize=8.5)
    fig.suptitle(f"Occlusion handling, before vs after · {res['shelf_readings']} shelves on day-2 frames with known true counts",
                 fontsize=11)
    fig.tight_layout(); save_fig(fig, out / "before_after.png"); plt.close(fig)


def example(k, out, name):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import f1_engine as fe
    img = cv2.imread(k["frame_path"])
    blue, green, red = (235, 99, 37), (40, 170, 40), (40, 40, 230)
    p1 = fe.draw_boxes(img, k["det"], [blue] * len(k["det"]), 1)
    p2 = fe.draw_boxes(img, k["gt"], [green if f else red for f in k["found"]], 1)
    p3 = fe.draw_boxes(img, k["det"], [fe.LEVEL_BGR[l] for l in k["levels"]], 2)
    crops = [fe.rgb(fe.crop_location(x, k["loc"])) for x in (p1, p2, p3)]
    r, n_gt, n_f = k["rec"], len(k["gt"]), int(k["found"].sum())
    lv = np.array(k["levels"]); hid = int((lv == 2).sum()); part = int((lv == 1).sum())
    flagged = bool(r.get("verify_manually"))
    why = fe.verify_reasons(r, k["camcal"]) if flagged else []
    t1 = f"BEFORE: plain counting\n{len(k['det'])} cartons -> {r.get('raw_status', r['status'])}\nno warning"
    t2 = f"TRUTH (labels)\n{n_gt} cartons: {n_f} found (green), {n_gt - n_f} missed (red)"
    t3 = (f"AFTER: F1 occlusion-aware\n{part} partly + {hid} mostly hidden (orange/red)\n"
          + (f"CHECK BY EYE · reliability {r['reliability']:.2f}" if flagged and r.get("reliability") is not None
             else "CHECK BY EYE" if flagged else
             (f"reliable · reliability {r['reliability']:.2f}" if r.get("reliability") is not None else "reliable")))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.8))
    for axi, c, t, col in zip(ax, crops, (t1, t2, t3), ("#334155", "#166534", "#B91C1C" if flagged else "#1D4ED8")):
        axi.imshow(c); axi.set_title(t, fontsize=10, color=col); axi.axis("off")
    import textwrap
    sub = f"Shelf {r['location_id']} · {Path(k['frame_path']).name}"
    if why:
        sub = "\n".join(textwrap.wrap(sub + "  ·  reason: " + "; ".join(why), 150))
    fig.suptitle(sub, fontsize=9.5, color="#475569", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save_fig(fig, out / name); plt.close(fig)


def examples(rows, keep, bad, flag, out, n=3):
    R = np.array([num(r["recall"]) for r in rows])
    big = np.array([num(r["true_count"]) >= 10 for r in rows])
    cand = [i for i in np.argsort(R) if flag[i] and bad[i] and big[i]]
    seen, made = set(), 0
    for i in cand:                                  # different shelves, worst recall first
        sh = (rows[i]["camera"], rows[i]["shelf"])
        if sh in seen:
            continue
        seen.add(sh); made += 1
        example(next(k for k in keep if k["row"] == i), out, f"example_flagged_{made}.png")
        if made == n:
            break
    good = [i for i in np.argsort(-R) if not flag[i] and not bad[i] and big[i]]
    if good:
        example(next(k for k in keep if k["row"] == good[0]), out, "example_reliable.png")
    print(f"Saved {made} flagged example(s){' and 1 reliable example' if good else ''} in {out}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--out", default="outputs/occlusion_proof")
    ap.add_argument("--min-gt", type=int, default=5, help="skip shelves with fewer labelled cartons (status would be noise)")
    ap.add_argument("--bad-recall", type=float, default=0.5, help="BAD shelf = detector found less than this share")
    ap.add_argument("--under", type=float, default=0.25, help="UNDER-COUNTED = count at least this share below the truth")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    keep = None
    if a.analyze_only:
        rows = list(csv.DictReader(open(out / "shelf_readings.csv", encoding="utf-8")))
    else:
        if not a.frames_root:
            raise SystemExit("Give --frames-root (or --analyze-only to reuse shelf_readings.csv)")
        rows, keep = collect(a)
    res, bad, flag, _ = analyze(rows, a)
    report(res)
    res["UNDERCOUNT"] = undercount(rows, a, int(flag.sum()))
    report_under(res["UNDERCOUNT"])
    json.dump(res, open(out / "before_after.json", "w"), indent=2)
    chart_under(res["UNDERCOUNT"], out)
    chart(res, out)
    if keep:
        examples(rows, keep, bad, flag, out)
    print(f"\nSaved {out / 'before_after.json'}, before_after.png and shelf_readings.csv")


if __name__ == "__main__":
    main()
