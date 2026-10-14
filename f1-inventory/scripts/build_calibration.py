r"""
Build configs\calibration.json from the research results, so count_stock.py can use the three improvements:

  1. Combined reliability score (research E1): detector confidence + occlusion trust + position in frame.
     Each signal is turned into a percentile against THAT camera's own reference cells (no training, so it does
     not suffer from camera shift), the three percentiles are averaged, and "verify manually" is raised when the
     score passes a threshold chosen by Youden's J on the 304 test cells (checked leave-one-camera-out).
  2. Per-camera count correction (research E2b): estimated true count = detected x precision / recall of that
     camera, measured on its annotated day-2 frames.
  3. Empty-shelf correction (synthetic validation): the share of detections that remain on an empty shelf from
     REAL background objects only (ghosts from inpainting are excluded, they are an artefact of the synthetic data).
     adjusted ratio = (ratio - floor) / (1 - floor).

Usage (from the f1-inventory folder; seconds, no models):
  python scripts\build_calibration.py
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

BATCH_TO_CAMERA = {"location_G_H_b2": "G_H", "location_X_5_b2": "X_5", "location_checkingPart_2_b2": "CP2"}


def pct_unreliable(x, ref):
    """Share of the camera's reference cells that look MORE reliable than x (mid-rank). 1 = worst seen."""
    ref = np.asarray(ref, float)
    return float(((ref > x).sum() + 0.5 * (ref == x).sum()) / len(ref))


def combined(conf, trust, ypos, cal):
    return float(np.mean([pct_unreliable(conf, cal["ref_conf"]), pct_unreliable(trust, cal["ref_trust"]),
                          pct_unreliable(ypos, cal["ref_ypos"])]))


def youden(score, unrel):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(unrel, score)
    k = int(np.argmax(tpr - fpr))
    return float(thr[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="outputs/research/cells_and_frames.csv")
    ap.add_argument("--trials", default="outputs/simulation_validation/trials.csv")
    ap.add_argument("--out", default="configs/calibration.json")
    ap.add_argument("--unreliable-recall", type=float, default=0.5)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    rows = list(csv.DictReader(open(a.cells, encoding="utf-8")))
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
    cells = [r for r in rows if r["cell"] != "frame" and not np.isnan(num(r, "trust"))]
    frames = [r for r in rows if r["cell"] == "frame"]
    grid = max(int(r["cell"].split(",")[1]) for r in cells) + 1

    cal = {"description": "F1 calibration built by build_calibration.py from the research results",
           "unreliable_definition": f"detector found < {a.unreliable_recall:.0%} of the annotated cartons", "cameras": {}}
    for batch, cam in BATCH_TO_CAMERA.items():
        cs = [r for r in cells if r["batch"] == batch]; fs = [r for r in frames if r["batch"] == batch]
        if not cs or not fs:
            continue
        P = sum(num(r, "tp") for r in fs) / sum(num(r, "detected") for r in fs)
        R = sum(num(r, "found") for r in fs) / sum(num(r, "gt") for r in fs)
        cal["cameras"][cam] = {
            "source_batch": batch,
            "ref_conf": [round(num(r, "conf"), 4) for r in cs],
            "ref_trust": [round(num(r, "trust"), 4) for r in cs],
            "ref_ypos": [round((int(r["cell"].split(",")[1]) + 0.5) / grid, 4) for r in cs],
            "precision": round(P, 3), "recall": round(R, 3), "count_factor": round(P / R, 3),
        }

    # empty-shelf floor from the synthetic trials, REAL background only
    if Path(a.trials).exists():
        T = list(csv.DictReader(open(a.trials, encoding="utf-8")))
        for batch, cam in BATCH_TO_CAMERA.items():
            t = [r for r in T if r["batch"] == batch and float(r["fraction_removed"]) == 1.0]
            if t and cam in cal["cameras"]:
                fl = np.mean([(float(r["detected_after"]) - float(r["ghost_detections"])) / max(float(r["detected_original"]), 1)
                              for r in t])
                ghost = np.mean([float(r["ghost_detections"]) / max(float(r["detected_original"]), 1) for r in t])
                cal["cameras"][cam]["empty_floor"] = round(float(max(fl, 0)), 3)
                cal["cameras"][cam]["ghost_share_excluded"] = round(float(ghost), 3)

    # combined score for every test cell, threshold by Youden, checked leave-one-camera-out
    score, unrel, cam_of = [], [], []
    for r in cells:
        c = BATCH_TO_CAMERA.get(r["batch"])
        if c not in cal["cameras"]:
            continue
        score.append(combined(num(r, "conf"), num(r, "trust"), (int(r["cell"].split(",")[1]) + 0.5) / grid, cal["cameras"][c]))
        unrel.append(num(r, "recall") < a.unreliable_recall); cam_of.append(c)
    score, unrel, cam_of = np.array(score), np.array(unrel), np.array(cam_of)
    thr = youden(score, unrel)
    auc = roc_auc_score(unrel, score)
    loco = {}
    for c in sorted(set(cam_of)):
        tr, te = cam_of != c, cam_of == c
        th = youden(score[tr], unrel[tr]); f = score[te] >= th
        loco[c] = {"threshold_from_others": round(th, 3),
                   "unreliable_caught": round(float(f[unrel[te]].mean()), 3) if unrel[te].any() else None,
                   "false_alarms": round(float(f[~unrel[te]].mean()), 3) if (~unrel[te]).any() else None}
    f = score >= thr
    cal["reliability"] = {"signals": ["detector confidence", "occlusion trust", "position in frame (top = far)"],
                          "auc": round(float(auc), 3), "verify_threshold": round(thr, 3),
                          "at_threshold": {"unreliable_caught": round(float(f[unrel].mean()), 3),
                                           "false_alarms": round(float(f[~unrel].mean()), 3),
                                           "share_flagged": round(float(f.mean()), 3)},
                          "leave_one_camera_out": loco}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(a.out, "w"), indent=1)

    print(f"Combined reliability score on {len(score)} test cells: AUC {auc:.3f}")
    print(f"VERIFY threshold (Youden): unreliability score >= {thr:.3f}  ->  catches {f[unrel].mean():.0%} of unreliable "
          f"cells, false alarms {f[~unrel].mean():.0%}, flags {f.mean():.0%} of cells")
    for c, v in loco.items():
        print(f"   held-out {c:4s}: threshold from other cameras {v['threshold_from_others']:.3f}, caught {v['unreliable_caught']}, "
              f"false alarms {v['false_alarms']}")
    print("\nPer camera:")
    for c, v in cal["cameras"].items():
        print(f"   {c:4s} count factor {v['count_factor']:.3f} (precision {v['precision']}, recall {v['recall']}); "
              f"empty-shelf floor {v.get('empty_floor', 'n/a')} (ghosts excluded: {v.get('ghost_share_excluded', 'n/a')})")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
