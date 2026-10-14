r"""
F1 sub-function 2, step 2c: geometry-based occlusion model (random forest), as a comparison to the CNN.

Instead of looking at pixels, it describes each annotated box by its GEOMETRY relative to the other boxes in the
same frame (size, position, overlap, crowding, boxes in front of it) and learns low / medium / high from that.
Needs only annotations\converted\<batch>\occlusion_labels.csv: no images, no GPU; runs in a minute on a laptop.

Same split and filtering as the CNN (train = 5 day-1 batches, val = camera_18 day 1, test = 3 day-2 batches,
boxes under 6 px skipped), so the test set is identical (7,950 boxes) and results are directly comparable.

Usage (from the f1-inventory folder):
  python scripts\geometry_occlusion_rf.py --out outputs\occlusion\geometry_rf
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                             cohen_kappa_score)

LEVELS = ["low", "medium", "high"]
DAY1 = ["location_a", "location_checkingPart_2_b1", "location_checkingPart_Full_b1", "location_G_H_1", "location_X_5_b1"]
VAL = ["location_camera_18_b1"]
DAY2 = ["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"]
FEATURES = ["w", "h", "area", "aspect", "cx", "cy", "y_bottom",
            "rel_area_vs_frame_median", "rel_area_vs_neighbours",
            "n_overlapping", "max_iou", "max_cover_by_one", "sum_cover",
            "cover_by_front", "n_front_overlapping", "n_near", "bottom_rank"]


def frame_features(boxes, W, H):
    """boxes: (n,4) array x1,y1,x2,y2 in pixels. Returns (n, len(FEATURES)) array."""
    b = boxes.astype(float)
    n = len(b)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    area = w * h
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    ix1 = np.maximum(b[:, None, 0], b[None, :, 0]); iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], b[None, :, 2]); iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    np.fill_diagonal(inter, 0)
    union = area[:, None] + area[None, :] - inter
    iou = np.where(union > 0, inter / union, 0)
    cover = inter / np.maximum(area[:, None], 1e-6)            # share of box i covered by box j
    front = b[None, :, 3] > b[:, None, 3]                       # j's bottom edge is lower => j is in front of i
    diag = np.sqrt(w ** 2 + h ** 2)
    dist = np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2)
    near = (dist < 1.5 * diag[:, None]); np.fill_diagonal(near, False)
    med = np.median(area) if n else 1.0
    neigh_area = np.array([np.median(area[near[i]]) if near[i].any() else area[i] for i in range(n)])
    order = np.argsort(np.argsort(-b[:, 3]))                    # 0 = lowest bottom edge (closest to camera)
    return np.column_stack([
        w / W, h / H, area / (W * H), w / np.maximum(h, 1e-6), cx / W, cy / H, b[:, 3] / H,
        area / max(med, 1e-6), area / np.maximum(neigh_area, 1e-6),
        (inter > 0).sum(1), iou.max(1) if n > 1 else np.zeros(n), cover.max(1) if n > 1 else np.zeros(n),
        np.minimum(cover.sum(1), 3.0), np.minimum((cover * front).sum(1), 3.0), ((inter > 0) & front).sum(1),
        near.sum(1), order / max(n - 1, 1),
    ])


def load(converted, batches, img_w, img_h, min_side):
    X, y, groups = [], [], []
    for batch in batches:
        p = Path(converted) / batch / "occlusion_labels.csv"
        if not p.exists():
            print(f"!! missing {p}"); continue
        per_img = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            per_img.setdefault(r["image_name"], []).append(r)
        for rows in per_img.values():
            boxes = np.array([[float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr")] for r in rows])
            F = frame_features(boxes, img_w, img_h)   # features use ALL boxes in the frame
            for r, f, bx in zip(rows, F, boxes):
                lvl = r["occlusion_level"].strip().lower()
                if lvl not in LEVELS or min(bx[2] - bx[0], bx[3] - bx[1]) < min_side:
                    continue                          # same filtering as the CNN crops
                X.append(f); y.append(LEVELS.index(lvl)); groups.append(batch)
    return np.array(X), np.array(y), np.array(groups)


def summary(yt, pt):
    return {"accuracy": round(float(accuracy_score(yt, pt)), 4),
            "macro_f1": round(float(f1_score(yt, pt, average="macro")), 4),
            "cohen_kappa": round(float(cohen_kappa_score(yt, pt)), 4),
            "quadratic_weighted_kappa": round(float(cohen_kappa_score(yt, pt, weights="quadratic")), 4),
            "within_one_accuracy": round(float(np.mean(np.abs(yt - pt) <= 1)), 4),
            "low_vs_high_confusions": int(np.sum(np.abs(yt - pt) == 2)),
            "confusion_matrix_rows_true_cols_pred": confusion_matrix(yt, pt, labels=[0, 1, 2]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--out", default="outputs/occlusion/geometry_rf")
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--img-h", type=int, default=640)
    ap.add_argument("--min-side", type=int, default=6)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    Xtr, ytr, _ = load(a.converted, DAY1, a.img_w, a.img_h, a.min_side)
    Xva, yva, _ = load(a.converted, VAL, a.img_w, a.img_h, a.min_side)
    Xte, yte, gte = load(a.converted, DAY2, a.img_w, a.img_h, a.min_side)
    for name, yy in (("train", ytr), ("val", yva), ("test", yte)):
        c = np.bincount(yy, minlength=3)
        print(f"{name:5s}: low {c[0]}, medium {c[1]}, high {c[2]}  (total {len(yy)})")

    rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=a.seed)
    rf.fit(Xtr, ytr)
    pv, pt = rf.predict(Xva), rf.predict(Xte)

    base = np.full_like(yte, np.bincount(ytr).argmax())
    res = {"model": "random forest on box geometry", "features": FEATURES,
           "val": summary(yva, pv), "test": summary(yte, pt),
           "test_majority_baseline": {"accuracy": round(float(accuracy_score(yte, base)), 4),
                                      "macro_f1": round(float(f1_score(yte, base, average="macro")), 4)},
           "test_accuracy_per_batch": {b: round(float(np.mean(pt[gte == b] == yte[gte == b])), 4) for b in DAY2},
           "feature_importance": dict(sorted(((f, round(float(v), 4)) for f, v in zip(FEATURES, rf.feature_importances_)),
                                             key=lambda kv: -kv[1])),
           "settings": vars(a)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)

    print(f"\nval accuracy {res['val']['accuracy']}  macro-F1 {res['val']['macro_f1']}")
    print("\n=== TEST (day-2 boxes) ===")
    print(classification_report(yte, pt, target_names=LEVELS, digits=3))
    t = res["test"]
    print(f"accuracy {t['accuracy']}  macro-F1 {t['macro_f1']}  kappa {t['cohen_kappa']}  QWK {t['quadratic_weighted_kappa']}  "
          f"within-one {t['within_one_accuracy']}  low<->high confusions {t['low_vs_high_confusions']}")
    print("confusion matrix (rows = true low/medium/high, cols = predicted):")
    for row in t["confusion_matrix_rows_true_cols_pred"]:
        print("   ", row)
    print(f"majority baseline: accuracy {res['test_majority_baseline']['accuracy']}  macro-F1 {res['test_majority_baseline']['macro_f1']}")
    print("per-batch accuracy:", res["test_accuracy_per_batch"])
    print("\nmost useful features:")
    for f, v in list(res["feature_importance"].items())[:8]:
        print(f"   {f:26s} {v:.3f}")
    print(f"\nSaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
