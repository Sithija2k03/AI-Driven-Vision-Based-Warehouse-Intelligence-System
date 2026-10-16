r"""
Realism check ONLY, on an already post-processed 3D synthetic set (images\ + occlusion_labels.csv).
Use it when postprocess_synthetic_3d.py finished the images but the check did not run.

Question: does the REAL-trained detector find synthetic cartons about as often as real ones, per occlusion level?
Real day-2 recall (IoU 0.5): low 0.66 / medium 0.57 / high 0.41. Pre-set pass rule: synthetic medium >= 0.40.

Run it with the Python that has ultralytics (the f1-inventory environment), e.g. from f1-inventory:
  python scripts\realism_check_3d.py --data D:\Research\synth3d_trial\synth_500 --weights weights\day2_s1024_best.pt

Writes <data>\summary.json (dataset stats + realism result).
"""
import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = ["low", "medium", "high"]
REAL_RECALL = {"low": 0.659, "medium": 0.572, "high": 0.414}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output folder of postprocess_synthetic_3d.py")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--pass-medium", type=float, default=0.40)
    a = ap.parse_args()
    data = Path(a.data)

    gt = defaultdict(list)
    all_rows = list(csv.DictReader(open(data / "occlusion_labels.csv", encoding="utf-8")))
    for r in all_rows:
        if str(r["labelled"]) == "1" and r["occlusion_level"] in LEVELS:
            gt[r["image_name"]].append(([float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr")],
                                        r["occlusion_level"], float(r["hidden_share"])))
    names = sorted(gt)
    if a.limit:
        names = names[:a.limit]
    lab = [b for n in names for b in gt[n]]
    print(f"{len(names)} images, {len(lab)} labelled cartons: "
          + ", ".join(f"{l} {sum(b[1] == l for b in lab)}" for l in LEVELS))

    from ultralytics import YOLO
    det = YOLO(a.weights)
    found = {l: [0, 0] for l in LEVELS}
    per_image = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        res = det.predict(str(data / "images" / name), conf=a.conf, iou=0.5, imgsz=a.imgsz, max_det=1000, verbose=False)[0]
        preds = res.boxes.xyxy.cpu().numpy().tolist() if len(res.boxes) else []
        used, hit = set(), 0
        for bx, lvl, _ in sorted(gt[name], key=lambda t: t[2]):      # least hidden first, one-to-one
            best, bj = 0, -1
            for j, p in enumerate(preds):
                if j not in used:
                    v = iou(bx, p)
                    if v > best:
                        best, bj = v, j
            found[lvl][1] += 1
            if best >= 0.5:
                found[lvl][0] += 1; used.add(bj); hit += 1
        per_image.append(hit / max(len(gt[name]), 1))
        if i % 50 == 0 or i == len(names):
            print(f"  {i}/{len(names)} images ({time.time() - t0:.0f} s)")

    rec = {l: (found[l][0] / found[l][1] if found[l][1] else float("nan")) for l in LEVELS}
    print("\nRealism check - real-trained detector recall (IoU 0.5):")
    print(f"  {'level':7s} {'synthetic':>18s} {'real day-2':>11s}")
    for l in LEVELS:
        print(f"  {l:7s} {rec[l]:8.2f} ({found[l][0]}/{found[l][1]}) {REAL_RECALL[l]:11.2f}")
    print(f"  per-image recall: median {np.median(per_image):.2f}, 10th-90th pct "
          f"{np.percentile(per_image, 10):.2f}-{np.percentile(per_image, 90):.2f}")
    ok = rec["medium"] >= a.pass_medium
    print(f"\nPre-set rule: synthetic medium recall >= {a.pass_medium:.2f} -> {'PASS' if ok else 'FAIL'}")

    summary = {"images": len(names), "labelled_cartons": len(lab),
               "per_level": {l: sum(b[1] == l for b in lab) for l in LEVELS},
               "cartons_per_image": round(len(lab) / max(len(names), 1), 1),
               "mean_box_w_px": round(float(np.mean([b[0][2] - b[0][0] for b in lab])), 1),
               "mean_box_h_px": round(float(np.mean([b[0][3] - b[0][1] for b in lab])), 1),
               "realism_check_recall": {l: round(rec[l], 3) for l in LEVELS},
               "realism_check_counts": {l: found[l] for l in LEVELS},
               "real_day2_recall": REAL_RECALL,
               "pass_rule": f"medium >= {a.pass_medium}", "pass": bool(ok)}
    json.dump(summary, open(data / "summary.json", "w"), indent=2)
    print(f"Saved {data / 'summary.json'}")


if __name__ == "__main__":
    main()
