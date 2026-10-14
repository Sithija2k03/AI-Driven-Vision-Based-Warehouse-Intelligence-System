r"""
Fix the "never says STOCK-OUT" problem found by validate_simulation.py, and test the fix fairly.

Problem: even with ALL annotated cartons removed, ~18% of the original detections remain (ghosts left by
inpainting + objects that were never annotated). So the detected ratio never gets near 0 and the status rule
cannot reach STOCK-OUT.

Two corrections are learned on two cameras and tested on the third (leave-one-camera-out):
  offset  - subtract the "empty-shelf floor":  adjusted = (ratio - floor) / (1 - floor)
  linear  - straight-line calibration:          adjusted = a + b * ratio
and compared with the current rule (no correction) on status accuracy and ratio error.

Uses outputs\simulation_validation\trials.csv (no models needed, runs in seconds):
  python scripts\calibrate_ratio.py --trials outputs\simulation_validation\trials.csv --out outputs\simulation_validation
"""
import argparse, csv, json
from pathlib import Path

import numpy as np


def status(r, low=0.5, out=0.1):
    return np.where(r < out, "STOCK-OUT", np.where(r < low, "LOW", "IN STOCK"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="outputs/simulation_validation/trials.csv")
    ap.add_argument("--out", default="outputs/simulation_validation")
    ap.add_argument("--low-ratio", type=float, default=0.5)
    ap.add_argument("--out-ratio", type=float, default=0.1)
    a = ap.parse_args()

    T = list(csv.DictReader(open(a.trials, encoding="utf-8")))
    cam = np.array([t["batch"] for t in T])
    frac = np.array([float(t["fraction_removed"]) for t in T])
    true_r = np.array([float(t["true_ratio"]) for t in T])
    det_r = np.array([float(t["detected_ratio"]) for t in T])
    ghosts = np.array([float(t["ghost_detections"]) for t in T])
    det0 = np.array([float(t["detected_original"]) for t in T])
    det1 = np.array([float(t["detected_after"]) for t in T])
    true_s = status(true_r, a.low_ratio, a.out_ratio)

    res = {}
    m = frac == 1.0
    if m.any():
        left = det1[m].mean(); gh = ghosts[m].mean()
        res["empty_shelf_floor"] = {"detections_left_when_all_removed": round(float(left), 1),
                                    "of_which_ghosts_in_erased_areas": round(float(gh), 1),
                                    "of_which_never_annotated_or_background": round(float(left - gh), 1),
                                    "as_share_of_original_detections": round(float((det1[m] / det0[m]).mean()), 3)}
        f = res["empty_shelf_floor"]
        print(f"Empty-shelf floor: {f['detections_left_when_all_removed']} detections remain when every annotated carton is "
              f"removed\n   {f['of_which_ghosts_in_erased_areas']} are ghosts inside erased areas (synthetic artefact), "
              f"{f['of_which_never_annotated_or_background']} are objects never annotated / background (also in real life)")

    methods = {"current rule (no correction)": np.full(len(T), np.nan),
               "offset correction": np.full(len(T), np.nan), "linear calibration": np.full(len(T), np.nan)}
    fitted = {}
    for c in sorted(set(cam)):
        tr, te = cam != c, cam == c
        floor = det_r[tr & (frac == 1.0)].mean() if (tr & (frac == 1.0)).any() else 0.0
        b, a0 = np.polyfit(det_r[tr], true_r[tr], 1)
        fitted[c] = {"floor_from_other_cameras": round(float(floor), 3), "linear_a": round(float(a0), 3), "linear_b": round(float(b), 3)}
        methods["current rule (no correction)"][te] = det_r[te]
        methods["offset correction"][te] = np.clip((det_r[te] - floor) / (1 - floor), 0, None)
        methods["linear calibration"][te] = np.clip(a0 + b * det_r[te], 0, None)
    res["fitted_leave_one_camera_out"] = fitted

    print(f"\nTested leave-one-camera-out on {len(T)} synthetic trials ({len(set(cam))} cameras)")
    print(f"{'method':30s} {'status acc':>10s} {'IN STOCK':>9s} {'LOW':>6s} {'OUT':>6s} {'ratio MAE':>10s}")
    res["methods"] = {}
    for name, est in methods.items():
        s = status(est, a.low_ratio, a.out_ratio); ok = s == true_s
        per = {k: round(float(ok[true_s == k].mean()), 3) for k in ("IN STOCK", "LOW", "STOCK-OUT") if (true_s == k).any()}
        mae = float(np.abs(est - true_r).mean())
        res["methods"][name] = {"status_accuracy": round(float(ok.mean()), 3), "by_true_status": per, "ratio_mae": round(mae, 3)}
        print(f"{name:30s} {ok.mean():10.1%} {per.get('IN STOCK', float('nan')):9.0%} {per.get('LOW', float('nan')):6.0%} "
              f"{per.get('STOCK-OUT', float('nan')):6.0%} {mae:10.3f}")
    print("\nPer held-out camera:")
    for c in sorted(set(cam)):
        te = cam == c
        line = []
        for name, est in methods.items():
            line.append(f"{name.split(' (')[0]} {np.mean(status(est[te], a.low_ratio, a.out_ratio) == true_s[te]):.0%}")
        print(f"   {c:30s} " + " | ".join(line))
    json.dump(res, open(Path(a.out) / "ratio_calibration.json", "w"), indent=2)
    print(f"\nSaved {Path(a.out) / 'ratio_calibration.json'}")


if __name__ == "__main__":
    main()
