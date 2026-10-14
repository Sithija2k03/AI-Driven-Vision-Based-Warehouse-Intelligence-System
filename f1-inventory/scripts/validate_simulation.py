r"""
Validate the SYNTHETIC stock-removal data (inpainting) before trusting any result that uses it.

Answers the panel's question "how do you know the synthetic data is valid?" with four measurable checks,
on many annotated day-2 frames (never used for training), removing 25/50/75/100% of the annotated cartons
(2 random draws each) with the same inpainting as simulate_removal.py:

  V1 Response   - does the detected count fall in line with the TRUE remaining count?
                  (linear fit detected ~ true remaining: slope, intercept, R^2; and normalised curve)
  V2 Ghosts     - does the detector still "see" cartons where they were removed?
                  (detections centred inside an erased area, per removed carton)
  V3 Side effects - does erasing damage the cartons that were kept?
                  (recall of the KEPT cartons in the edited image vs the same cartons in the original)
  V4 Status     - with the ratio rule (count / original count), how often is IN STOCK / LOW / STOCK-OUT right?

Usage (from the f1-inventory folder; ~5-10 min on CPU):
  python scripts\validate_simulation.py --weights weights\day2_s1024_best.pt ^
      --frames-root "D:\Research\research camera data" --out outputs\simulation_validation
"""
import argparse, csv, json, random
from pathlib import Path

import cv2
import numpy as np

DAY2 = ["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"]


def read_yolo(path, w, h):
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out).reshape(-1, 4)


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def matched(gts, preds, thr=0.5):
    """Boolean per GT box: was it found (greedy one-to-one, IoU >= thr)?"""
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


def remove(img, gts, idx, pad=2):
    """Same method as simulate_removal.py: erase chosen boxes, protect kept boxes, TELEA inpainting."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for i in idx:
        x1, y1, x2, y2 = gts[i].astype(int)
        cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 255, -1)
    for i in sorted(set(range(len(gts))) - set(idx)):
        x1, y1, x2, y2 = gts[i].astype(int)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA), mask


def status(ratio, low=0.5, out=0.1):
    return "STOCK-OUT" if ratio < out else "LOW" if ratio < low else "IN STOCK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--batches", nargs="+", default=DAY2)
    ap.add_argument("--frames-per-batch", type=int, default=8)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="outputs/simulation_validation")
    a = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(a.weights)
    out = Path(a.out); (out / "examples").mkdir(parents=True, exist_ok=True)
    index = {p.name: p for p in Path(a.frames_root).rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "extracted_frames"}

    def detect(img):
        r = det.predict(img, conf=a.conf, iou=a.iou, imgsz=a.imgsz, max_det=1000, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))

    trials = []
    for batch in a.batches:
        lab_dir = Path(a.converted) / batch / "labels"
        files = sorted(f for f in lab_dir.glob("*.txt") if (f.stem + ".jpg") in index)
        if not files:
            print(f"!! {batch}: no label files with matching frames in {lab_dir}"); continue
        pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(a.frames_per_batch, len(files)))]
        for fi, lf in enumerate(pick):
            img = cv2.imread(str(index[lf.stem + ".jpg"])); h, w = img.shape[:2]
            gts = read_yolo(lf, w, h)
            if len(gts) < 10:
                continue
            p0 = detect(img); found0 = matched(gts, p0)
            for frac in a.fractions:
                for s in range(a.seeds):
                    random.seed(1000 * s + fi)
                    k = round(len(gts) * frac)
                    idx = random.sample(range(len(gts)), k)
                    kept = np.array(sorted(set(range(len(gts))) - set(idx)), int)
                    im2, mask = remove(img, gts, idx)
                    p1 = detect(im2)
                    cx, cy = ((p1[:, 0] + p1[:, 2]) / 2).astype(int).clip(0, w - 1), ((p1[:, 1] + p1[:, 3]) / 2).astype(int).clip(0, h - 1)
                    ghosts = int((mask[cy, cx] > 0).sum()) if len(p1) else 0
                    kf0 = found0[kept] if len(kept) else np.zeros(0, bool)
                    kf1 = matched(gts[kept], p1) if len(kept) else np.zeros(0, bool)
                    true_ratio = len(kept) / len(gts); det_ratio = len(p1) / max(len(p0), 1)
                    trials.append({"batch": batch, "frame": lf.stem, "fraction_removed": frac, "seed": s,
                                   "gt_total": len(gts), "true_remaining": len(kept), "removed": k,
                                   "detected_original": len(p0), "detected_after": len(p1),
                                   "true_ratio": round(true_ratio, 3), "detected_ratio": round(det_ratio, 3),
                                   "ghost_detections": ghosts,
                                   "kept_recall_original": round(float(kf0.mean()), 3) if len(kept) else None,
                                   "kept_recall_after": round(float(kf1.mean()), 3) if len(kept) else None,
                                   "true_status": status(true_ratio), "detected_status": status(det_ratio)})
                    if fi == 0 and s == 0:
                        cv2.imwrite(str(out / "examples" / f"{batch}__{lf.stem}__removed{int(frac * 100)}.jpg"), im2)
        print(f"{batch}: {len(pick)} frames done")

    with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(trials[0].keys())); wr.writeheader(); wr.writerows(trials)

    T = lambda k: np.array([t[k] for t in trials], float)
    tr, dr = T("true_remaining"), T("detected_after")
    slope, icpt = np.polyfit(tr, dr, 1); r2 = np.corrcoef(tr, dr)[0, 1] ** 2
    tr_ratio, d_ratio = T("true_ratio"), T("detected_ratio")
    res = {"n_trials": len(trials), "n_frames": len({(t['batch'], t['frame']) for t in trials}),
           "V1_response": {"slope_detected_per_true_carton": round(float(slope), 3), "intercept": round(float(icpt), 2),
                           "r2": round(float(r2), 3), "by_fraction": {}}}
    print(f"\n=== Synthetic removal validation: {res['n_trials']} trials on {res['n_frames']} frames ===")
    print(f"V1 response: detected = {icpt:.1f} + {slope:.2f} x true remaining   (R^2 = {r2:.3f})")
    print(f"{'removed':>8s} {'true ratio':>10s} {'detected ratio (mean +- sd)':>28s}")
    for fr in sorted(set(T("fraction_removed"))):
        m = T("fraction_removed") == fr
        res["V1_response"]["by_fraction"][str(fr)] = {"true_ratio": round(float(tr_ratio[m].mean()), 3),
                                                      "detected_ratio_mean": round(float(d_ratio[m].mean()), 3),
                                                      "detected_ratio_sd": round(float(d_ratio[m].std()), 3)}
        print(f"{fr:8.0%} {tr_ratio[m].mean():10.2f} {d_ratio[m].mean():18.2f} +- {d_ratio[m].std():.2f}")
    ghosts, removed = T("ghost_detections"), T("removed")
    res["V2_ghosts"] = {"ghost_detections_per_removed_carton": round(float(ghosts.sum() / max(removed.sum(), 1)), 3),
                        "at_100pct_removed_mean_detections_left": round(float(dr[T("fraction_removed") == 1.0].mean()), 2)
                        if (T("fraction_removed") == 1.0).any() else None}
    print(f"V2 ghosts: {res['V2_ghosts']['ghost_detections_per_removed_carton']:.3f} detections per removed carton "
          f"inside erased areas; with 100% removed, {res['V2_ghosts']['at_100pct_removed_mean_detections_left']} detections remain on average")
    ok = [t for t in trials if t["kept_recall_after"] is not None]
    k0 = np.array([t["kept_recall_original"] for t in ok]); k1 = np.array([t["kept_recall_after"] for t in ok])
    res["V3_side_effects"] = {"kept_recall_original": round(float(k0.mean()), 3), "kept_recall_after": round(float(k1.mean()), 3),
                              "change": round(float((k1 - k0).mean()), 3)}
    print(f"V3 side effects: recall of KEPT cartons {k0.mean():.3f} before vs {k1.mean():.3f} after erasing "
          f"(change {(k1 - k0).mean():+.3f})")
    st_ok = np.array([t["true_status"] == t["detected_status"] for t in trials])
    res["V4_status_accuracy"] = {"overall": round(float(st_ok.mean()), 3), "by_true_status": {}}
    for s in ("IN STOCK", "LOW", "STOCK-OUT"):
        m = np.array([t["true_status"] == s for t in trials])
        if m.any():
            res["V4_status_accuracy"]["by_true_status"][s] = {"n": int(m.sum()), "accuracy": round(float(st_ok[m].mean()), 3)}
    print(f"V4 status accuracy: {st_ok.mean():.1%} overall; " + ", ".join(
        f"{k} {v['accuracy']:.0%} (n={v['n']})" for k, v in res["V4_status_accuracy"]["by_true_status"].items()))
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(tr_ratio, d_ratio, s=10, alpha=0.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect response")
        ax.axhline(0.5, color="orange", lw=0.8, ls=":"); ax.axhline(0.1, color="red", lw=0.8, ls=":")
        ax.set_xlabel("True share of cartons remaining"); ax.set_ylabel("Detected count / original detected count")
        ax.set_title("Synthetic removal: detector response"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "V1_response.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"\nSaved {out / 'metrics.json'}, trials.csv, V1_response.png and example images in {out / 'examples'}")


if __name__ == "__main__":
    main()
