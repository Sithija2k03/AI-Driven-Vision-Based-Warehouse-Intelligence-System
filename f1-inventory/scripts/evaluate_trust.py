r"""
F1 sub-function 2, step 3 check: does the occlusion pipeline work on DETECTOR boxes, and does the trust score
really warn when counts are unreliable?

On the annotated day-2 frames (never used for training) it:
  A) runs the detector, matches detections to annotated cartons (IoU >= 0.5) and compares the predicted
     occlusion level of each matched detection (CNN only, and hybrid CNN + geometry) with the annotated level;
  B) splits every frame into a grid of cells; per cell it computes
        recall  = annotated cartons found by the detector / annotated cartons   (how reliable the count is)
        trust   = mean trust weight of the detected boxes in the cell            (what F1 predicts)
     and checks whether low trust goes with low recall (Spearman correlation, trust bins, flagged vs not flagged).

Usage (from the f1-inventory folder):
  python scripts\evaluate_trust.py --weights weights\day2_s1024_best.pt --occlusion-cnn weights\occlusion_r18.pt ^
      --frames-root "D:\Research\research camera data" --out outputs\occlusion\trust_eval
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occlusion_model import OcclusionEstimator, box_trust, LEVELS          # noqa: E402
from recall_by_occlusion import match                                     # noqa: E402

DAY2 = ["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"]


def match_pairs(gts, preds, thr):
    """Greedy one-to-one matching; returns list of (gt_index, pred_index)."""
    from recall_by_occlusion import iou
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds)), reverse=True)
    ug, up, out = set(), set(), []
    for v, gi, pi in pairs:
        if v < thr:
            break
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi); out.append((gi, pi))
    return out


def scores(y, p):
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "cohen_kappa": round(float(cohen_kappa_score(y, p)), 4),
            "low_vs_high_confusions": int(np.sum(np.abs(np.array(y) - np.array(p)) == 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="detector weights")
    ap.add_argument("--occlusion-cnn", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--rf-model", default="weights/occlusion_geometry_rf.joblib")
    ap.add_argument("--batches", nargs="+", default=DAY2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=3, help="cells per side (3 = 3x3 cells per frame)")
    ap.add_argument("--min-gt", type=int, default=5, help="ignore cells with fewer annotated cartons")
    ap.add_argument("--trust-threshold", type=float, default=0.80, help="starting value, compared with the data-chosen one")
    ap.add_argument("--unreliable-recall", type=float, default=0.5,
                    help="a cell counts as UNRELIABLE if the detector found less than this share of its cartons")
    ap.add_argument("--out", default="outputs/occlusion/trust_eval")
    a = ap.parse_args()

    index = {p.name: p for p in Path(a.frames_root).rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "extracted_frames"}
    from ultralytics import YOLO
    det = YOLO(a.weights)
    est = OcclusionEstimator(a.occlusion_cnn, mode="hybrid", rf_path=a.rf_model, converted=a.converted)

    y_true, y_cnn, y_hyb, cells = [], [], [], []
    for batch in a.batches:
        per_img = {}
        for r in csv.DictReader(open(Path(a.converted) / batch / "occlusion_labels.csv", encoding="utf-8")):
            per_img.setdefault(r["image_name"], []).append(r)
        for name, rows in per_img.items():
            p = index.get(Path(name).name)
            if p is None:
                continue
            img = cv2.imread(str(p)); H, W = img.shape[:2]
            gts = [tuple(float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr")) for r in rows]
            glv = [LEVELS.index(r["occlusion_level"].strip().lower()) for r in rows]
            res = det.predict(str(p), conf=a.conf, iou=a.iou, imgsz=a.imgsz, max_det=1000, verbose=False)[0]
            preds = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
            p_cnn = est.cnn_probs(img, np.array(preds)) if preds else np.zeros((0, 3))
            from geometry_occlusion_rf import frame_features
            p_geo = est.rf.predict_proba(frame_features(np.array(preds), W, H)) if preds else np.zeros((0, 3))
            p_hyb = 0.5 * p_cnn + 0.5 * p_geo
            pairs = match_pairs(gts, preds, a.match_iou)
            for gi, pi in pairs:
                y_true.append(glv[gi]); y_cnn.append(int(p_cnn[pi].argmax())); y_hyb.append(int(p_hyb[pi].argmax()))
            found = np.zeros(len(gts), bool)
            for gi, _ in pairs:
                found[gi] = True
            # grid cells
            g = a.grid
            gc = [(min(int((b[0] + b[2]) / 2 / W * g), g - 1), min(int((b[1] + b[3]) / 2 / H * g), g - 1)) for b in gts]
            pc = [(min(int((b[0] + b[2]) / 2 / W * g), g - 1), min(int((b[1] + b[3]) / 2 / H * g), g - 1)) for b in preds]
            for cx in range(g):
                for cy in range(g):
                    gi = [i for i, c in enumerate(gc) if c == (cx, cy)]
                    pi = [i for i, c in enumerate(pc) if c == (cx, cy)]
                    if len(gi) < a.min_gt or not pi:
                        continue
                    cells.append({"batch": batch, "image": Path(name).stem, "cell": f"{cx},{cy}",
                                  "gt": len(gi), "detected": len(pi), "recall": float(found[gi].mean()),
                                  "count_ratio": len(pi) / len(gi),
                                  "trust_hybrid": float(box_trust(p_hyb[pi]).mean()),
                                  "trust_cnn": float(box_trust(p_cnn[pi]).mean()),
                                  "trust_true_labels": float(np.mean([[1.0, 0.868, 0.628][glv[i]] for i in gi]))})
        print(f"{batch}: done")

    yt, yc, yh = np.array(y_true), np.array(y_cnn), np.array(y_hyb)
    res = {"A_occlusion_on_detector_boxes": {"cnn": scores(yt, yc), "hybrid": scores(yt, yh)}}
    print(f"\n=== A) Occlusion level on DETECTOR boxes (matched to annotations, n={len(yt)}) ===")
    print(f"{'model':8s} {'acc':>6s} {'mF1':>6s} {'kappa':>6s} {'low<->high':>10s}")
    for k, v in res["A_occlusion_on_detector_boxes"].items():
        print(f"{k:8s} {v['accuracy']:6.3f} {v['macro_f1']:6.3f} {v['cohen_kappa']:6.3f} {v['low_vs_high_confusions']:10d}")

    rec = np.array([c["recall"] for c in cells])
    print(f"\n=== B) Does trust predict count reliability? ({len(cells)} cells, {a.grid}x{a.grid} per frame, >= {a.min_gt} cartons) ===")
    res["B_trust_vs_recall"] = {"n_cells": len(cells)}
    for key in ("trust_hybrid", "trust_cnn", "trust_true_labels"):
        t = np.array([c[key] for c in cells])
        rho, pval = spearmanr(t, rec)
        res["B_trust_vs_recall"][key] = {"spearman_rho": round(float(rho), 3), "p_value": float(pval)}
        print(f"Spearman(trust, recall) using {key:18s}: rho = {rho:+.3f}  (p = {pval:.1e})")

    t = np.array([c["trust_hybrid"] for c in cells])
    qs = np.quantile(t, [0, 0.25, 0.5, 0.75, 1])
    print("\nHybrid trust bins (quartiles) -> mean detector recall in those cells:")
    bins = []
    for i in range(4):
        m = (t >= qs[i]) & (t <= qs[i + 1]) if i == 3 else (t >= qs[i]) & (t < qs[i + 1])
        bins.append({"trust_from": round(float(qs[i]), 3), "trust_to": round(float(qs[i + 1]), 3),
                     "cells": int(m.sum()), "mean_recall": round(float(rec[m].mean()), 3)})
        print(f"   trust {qs[i]:.3f}-{qs[i + 1]:.3f}: {m.sum():4d} cells, mean recall {rec[m].mean():.3f}")
    flag = t < a.trust_threshold
    res["B_trust_vs_recall"]["quartile_bins"] = bins
    res["B_trust_vs_recall"]["verify_flag"] = {
        "threshold": a.trust_threshold, "flagged_cells": int(flag.sum()),
        "mean_recall_flagged": round(float(rec[flag].mean()), 3) if flag.any() else None,
        "mean_recall_not_flagged": round(float(rec[~flag].mean()), 3) if (~flag).any() else None}
    vf = res["B_trust_vs_recall"]["verify_flag"]
    print(f"\n'verify manually' (trust < {a.trust_threshold}): {vf['flagged_cells']} of {len(cells)} cells flagged; "
          f"mean recall flagged {vf['mean_recall_flagged']} vs not flagged {vf['mean_recall_not_flagged']}")

    # ---- C) choose the threshold from data (ROC / Youden's J), checked leave-one-camera-out ----
    unrel = rec < a.unreliable_recall
    batches = np.array([c["batch"] for c in cells])

    def youden(tr, ur):
        fpr, tpr, thr = roc_curve(ur, -tr)          # low trust should mean unreliable
        k = int(np.argmax(tpr - fpr))
        return float(-thr[k])

    def flag_stats(tr, ur, th):
        f = tr < th
        caught = float(f[ur].mean()) if ur.any() else float("nan")      # share of unreliable cells flagged
        false_alarm = float(f[~ur].mean()) if (~ur).any() else float("nan")
        return caught, false_alarm, float(f.mean())

    print(f"\n=== C) Choosing the threshold from data ===")
    print(f"unreliable cell = detector found < {a.unreliable_recall:.0%} of its cartons: "
          f"{int(unrel.sum())} of {len(cells)} cells")
    C = {"unreliable_recall_definition": a.unreliable_recall, "n_unreliable": int(unrel.sum())}
    if 0 < unrel.sum() < len(cells):
        auc = roc_auc_score(unrel, -t)
        th_all = youden(t, unrel)
        C.update({"auc_trust_detects_unreliable": round(float(auc), 3), "youden_threshold_all": round(th_all, 3)})
        print(f"AUC (how well low trust picks out unreliable cells; 0.5 = no better than chance): {auc:.3f}")
        print(f"best threshold on all cells (Youden's J): {th_all:.3f}")
        loco = []
        for b in sorted(set(batches)):
            tr_m, te_m = batches != b, batches == b
            if unrel[tr_m].sum() in (0, tr_m.sum()) or te_m.sum() == 0:
                continue
            th = youden(t[tr_m], unrel[tr_m])
            caught, fa, share = flag_stats(t[te_m], unrel[te_m], th)
            loco.append({"held_out": b, "threshold_from_others": round(th, 3), "unreliable_caught": round(caught, 3),
                         "false_alarms": round(fa, 3), "share_flagged": round(share, 3)})
        C["leave_one_camera_out"] = loco
        print("leave-one-camera-out (threshold chosen on the other two cameras, tested on the held-out one):")
        for r in loco:
            print(f"   {r['held_out']:28s} threshold {r['threshold_from_others']:.3f}  caught {r['unreliable_caught']:.0%} "
                  f"of unreliable cells, false alarms {r['false_alarms']:.0%}, flagged {r['share_flagged']:.0%}")
        for name, th in (("starting value", a.trust_threshold), ("data-chosen", th_all)):
            caught, fa, share = flag_stats(t, unrel, th)
            C[f"at_{name.replace(' ', '_').replace('-', '_')}"] = {"threshold": round(th, 3), "unreliable_caught": round(caught, 3),
                                                                   "false_alarms": round(fa, 3), "share_flagged": round(share, 3)}
            print(f"at {name} {th:.3f}: caught {caught:.0%} of unreliable cells, false alarms {fa:.0%}, flagged {share:.0%} of cells")
    else:
        print("not enough variety (all cells reliable or all unreliable) to choose a threshold; try another --unreliable-recall")
    res["C_threshold_choice"] = C

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)
    print(f"\nSaved {out / 'metrics.json'} and {out / 'cells.csv'}")


if __name__ == "__main__":
    main()
