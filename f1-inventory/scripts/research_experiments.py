r"""
F1 research experiments - turn the pipeline into measured research findings.

E1  Is the occlusion-aware trust score a BETTER reliability signal than simple alternatives?
    Per grid cell on the annotated day-2 frames (never used for training), predict whether the count is
    UNRELIABLE (detector found < 50% of the annotated cartons) from:
        trust      - occlusion-aware trust (hybrid CNN + geometry)          <- our method
        conf       - mean detector confidence of the boxes                    <- standard baseline
        brightness - mean image brightness of the cell                        <- lighting
        crowding   - how much detected boxes overlap each other
    alone and combined (logistic regression, trained leave-one-camera-out). Metric: ROC AUC + 95% bootstrap CI,
    and a paired bootstrap test "trust vs confidence".

E2  Can occlusion make the COUNT itself more accurate (not only flag it)?
    Count estimators are FITTED on the validation camera (checkingPart_2_b2) and TESTED on the test cameras
    (G_H_b2, X_5_b2):
        raw       - number of detected boxes
        global    - raw x precision / recall              (one correction factor for everything)
        occlusion - each detected box counts precision / recall(its predicted occlusion level)
        trust     - raw x precision / recall predicted from the cell's trust
    Metric: mean absolute % error and bias against the annotated count, per frame and per cell.

Usage (from the f1-inventory folder; ~5-10 min on CPU):
  python scripts\research_experiments.py --weights weights\day2_s1024_best.pt --occlusion-cnn weights\occlusion_r18.pt ^
      --frames-root "D:\Research\research camera data" --out outputs\research
Re-analyse without running the models again (uses the saved CSV):
  python scripts\research_experiments.py --analyze-only --out outputs\research
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

DAY2 = ["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"]
VAL_BATCH = "location_checkingPart_2_b2"
LEVELS = ["low", "medium", "high"]
NUM = ["gt", "detected", "found", "tp", "recall", "trust", "conf", "brightness", "contrast", "crowding", "box_size",
       "soft_low", "soft_medium", "soft_high", "gt_low", "gt_medium", "gt_high",
       "found_low", "found_medium", "found_high"]


# ------------------------------------------------------------------ collect (runs the models)
def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(gts, preds, thr):
    if not len(gts) or not len(preds):
        return []
    m = iou_matrix(gts, preds)
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    ug, up, out = set(), set(), []
    for gi, pi in order:
        if m[gi, pi] < thr:
            break
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi); out.append((int(gi), int(pi)))
    return out


def collect(a):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from occlusion_model import OcclusionEstimator, box_trust
    from geometry_occlusion_rf import frame_features
    from ultralytics import YOLO

    index = {p.name: p for p in Path(a.frames_root).rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "extracted_frames"}
    det = YOLO(a.weights)
    est = OcclusionEstimator(a.occlusion_cnn, mode="hybrid", rf_path=a.rf_model, converted=a.converted)
    rows = []
    for batch in a.batches:
        per_img = {}
        for r in csv.DictReader(open(Path(a.converted) / batch / "occlusion_labels.csv", encoding="utf-8")):
            if r["occlusion_level"].strip().lower() in LEVELS:
                per_img.setdefault(r["image_name"], []).append(r)
        n = 0
        for name, rr in per_img.items():
            p = index.get(Path(name).name)
            if p is None:
                continue
            img = cv2.imread(str(p)); H, W = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) / 255
            gts = np.array([[float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr")] for r in rr])
            glv = np.array([LEVELS.index(r["occlusion_level"].strip().lower()) for r in rr])
            res = det.predict(str(p), conf=a.conf, iou=a.iou, imgsz=a.imgsz, max_det=1000, verbose=False)[0]
            preds = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
            if len(preds):
                p_hyb = 0.5 * est.cnn_probs(img, preds) + 0.5 * est.rf.predict_proba(frame_features(preds, W, H))
                ov = iou_matrix(preds, preds); np.fill_diagonal(ov, 0); crowd = ov.max(1)
            else:
                p_hyb, crowd = np.zeros((0, 3)), np.zeros(0)
            pairs = greedy_match(gts, preds, a.match_iou)
            found = np.zeros(len(gts), bool); tp = np.zeros(len(preds), bool)
            for gi, pi in pairs:
                found[gi] = True; tp[pi] = True
            gcx, gcy = (gts[:, 0] + gts[:, 2]) / 2, (gts[:, 1] + gts[:, 3]) / 2
            pcx, pcy = ((preds[:, 0] + preds[:, 2]) / 2, (preds[:, 1] + preds[:, 3]) / 2) if len(preds) else (np.zeros(0),) * 2
            g = a.grid
            regions = [("frame", 0, 0, W, H)] + [(f"{cx},{cy}", cx * W / g, cy * H / g, (cx + 1) * W / g, (cy + 1) * H / g)
                                                 for cx in range(g) for cy in range(g)]
            for cid, x0, y0, x1, y1 in regions:
                last = cid != "frame"
                gi = np.where((gcx >= x0) & (gcy >= y0) & ((gcx < x1) | (x1 >= W)) & ((gcy < y1) | (y1 >= H)))[0]
                pi = np.where((pcx >= x0) & (pcy >= y0) & ((pcx < x1) | (x1 >= W)) & ((pcy < y1) | (y1 >= H)))[0]
                if (last and (len(gi) < a.min_gt or not len(pi))) or not len(gi):
                    continue
                reg = gray[int(y0):int(y1), int(x0):int(x1)]
                row = {"batch": batch, "image": Path(name).stem, "cell": cid,
                       "gt": len(gi), "detected": len(pi), "found": int(found[gi].sum()), "tp": int(tp[pi].sum()),
                       "recall": float(found[gi].mean()),
                       "trust": float(box_trust(p_hyb[pi]).mean()) if len(pi) else float("nan"),
                       "conf": float(confs[pi].mean()) if len(pi) else float("nan"),
                       "brightness": float(reg.mean()), "contrast": float(reg.std()),
                       "crowding": float(crowd[pi].mean()) if len(pi) else float("nan"),
                       "box_size": float(np.median(np.sqrt((preds[pi, 2] - preds[pi, 0]) * (preds[pi, 3] - preds[pi, 1]))) / W)
                       if len(pi) else float("nan")}
                soft = p_hyb[pi].sum(0) if len(pi) else np.zeros(3)
                for k, l in enumerate(LEVELS):
                    row[f"soft_{l}"] = float(soft[k])
                    row[f"gt_{l}"] = int((glv[gi] == k).sum())
                    row[f"found_{l}"] = int(found[gi][glv[gi] == k].sum())
                rows.append(row)
            n += 1
        print(f"{batch}: {n} frames")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "cells_and_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


# ------------------------------------------------------------------ analysis (no models needed)
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(auc(y[i], s[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def paired_boot(y, s1, s2, n=2000, seed=1):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(auc(y[i], s1[i]) - auc(y[i], s2[i]))
    d = np.array(d)
    return {"auc_difference": round(float(np.mean(d)), 3),
            "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
            "share_of_resamples_where_first_is_better": round(float((d > 0).mean()), 3)}


def loco_scores(X, y, groups):
    """Logistic regression trained on the other cameras, scored on the held-out camera."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def analyze(rows, a):
    from scipy.stats import spearmanr
    out = Path(a.out); res = {}
    cells = [r for r in rows if r["cell"] != "frame" and not np.isnan(r["trust"])]
    frames = [r for r in rows if r["cell"] == "frame"]
    col = lambda rs, k: np.array([r[k] for r in rs], float)
    groups = np.array([r["batch"] for r in cells])
    rec = col(cells, "recall"); y = (rec < a.unreliable_recall).astype(int)

    # ---------------- E1
    print(f"\n=== E1) Which signal best predicts an UNRELIABLE count? ({len(cells)} cells, "
          f"{y.sum()} unreliable = detector found < {a.unreliable_recall:.0%}) ===")
    signals = {"trust (ours: occlusion-aware)": -col(cells, "trust"),
               "detector confidence (baseline)": -col(cells, "conf"),
               "brightness (lighting)": -col(cells, "brightness"),
               "crowding (box overlap)": col(cells, "crowding")}
    feats = {"trust": col(cells, "trust"), "conf": col(cells, "conf"),
             "brightness": col(cells, "brightness"), "crowding": col(cells, "crowding")}
    combos = {"combined: trust + brightness": ["trust", "brightness"],
              "combined: trust + confidence": ["trust", "conf"],
              "combined: all four": ["trust", "conf", "brightness", "crowding"],
              "combined: all except trust": ["conf", "brightness", "crowding"]}
    E1 = {"n_cells": len(cells), "n_unreliable": int(y.sum()), "signals": {}}
    print(f"{'signal':38s} {'AUC':>6s}  {'95% CI':>13s}  {'Spearman with recall':>20s}")
    plot_curves = {}
    for name, s in signals.items():
        A, ci = auc(y, s), boot_ci(y, s)
        rho = spearmanr(s if "crowding" in name else -s, rec)[0]
        E1["signals"][name] = {"auc": round(A, 3), "ci95": ci, "spearman_with_recall": round(float(rho), 3)}
        plot_curves[name] = s
        print(f"{name:38s} {A:6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  {rho:+20.3f}")
    for name, ks in combos.items():
        s = loco_scores(np.column_stack([feats[k] for k in ks]), y, groups)
        ok = ~np.isnan(s)
        A, ci = auc(y[ok], s[ok]), boot_ci(y[ok], s[ok])
        E1["signals"][name + " (leave-one-camera-out)"] = {"auc": round(A, 3), "ci95": ci}
        if name in ("combined: all four", "combined: all except trust"):
            plot_curves[name] = np.where(ok, s, np.nanmean(s))
        print(f"{name + ' (LOCO)':38s} {A:6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
    E1["trust_vs_confidence"] = paired_boot(y, signals["trust (ours: occlusion-aware)"],
                                            signals["detector confidence (baseline)"])
    t = E1["trust_vs_confidence"]
    print(f"\nTrust vs detector confidence: AUC difference {t['auc_difference']:+.3f} "
          f"(95% CI {t['ci95'][0]:+.3f} to {t['ci95'][1]:+.3f}); trust better in "
          f"{t['share_of_resamples_where_first_is_better']:.0%} of bootstrap resamples")
    b = feats["brightness"]; q1, q3 = np.quantile(b, [0.25, 0.75])
    E1["lighting"] = {"mean_recall_darkest_quarter": round(float(rec[b <= q1].mean()), 3),
                      "mean_recall_brightest_quarter": round(float(rec[b >= q3].mean()), 3)}
    print(f"Lighting: mean recall in the darkest quarter of cells {E1['lighting']['mean_recall_darkest_quarter']:.3f} "
          f"vs brightest quarter {E1['lighting']['mean_recall_brightest_quarter']:.3f}")
    # ---- E1b: what is brightness standing in for? per-camera stability; training-free combinations
    from scipy.stats import rankdata
    row_pos = np.array([int(r["cell"].split(",")[1]) for r in cells], float)     # 0 = top row of the frame
    extra = {"position (top of frame = far)": -row_pos}
    if not np.all(np.isnan(col(cells, "box_size"))):
        extra["box size (small = far)"] = -col(cells, "box_size")
    print("\nOther explanations (higher score = predicted unreliable):")
    for name, s_ in extra.items():
        A, ci = auc(y, s_), boot_ci(y, s_)
        E1["signals"][name] = {"auc": round(A, 3), "ci95": ci}
        signals[name] = s_
        print(f"{name:38s} {A:6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
    rho_bp = spearmanr(feats["brightness"], row_pos)[0]
    E1["brightness_vs_row_spearman"] = round(float(rho_bp), 3)
    print(f"Spearman(brightness, row in frame) = {rho_bp:+.3f}  (negative = bright cells are near the TOP of the frame)")

    print("\nAUC per camera (is each signal stable across cameras?):")
    cams = sorted(set(groups)); E1["auc_per_camera"] = {}
    print(f"{'signal':38s} " + " ".join(f"{c.replace('location_', '')[:14]:>15s}" for c in cams))
    for name, s_ in signals.items():
        vals = {}
        for c in cams:
            m = groups == c
            vals[c] = round(auc(y[m], s_[m]), 3) if 0 < y[m].sum() < m.sum() else None
        E1["auc_per_camera"][name] = vals
        print(f"{name:38s} " + " ".join(f"{(v if v is not None else float('nan')):15.3f}" for v in vals.values()))

    def within_cam_rank(score):
        out = np.zeros(len(score))
        for c in cams:
            m = groups == c
            out[m] = rankdata(score[m]) / m.sum()
        return out
    rk = {k: within_cam_rank(v) for k, v in signals.items()}
    conf_k, trust_k = "detector confidence (baseline)", "trust (ours: occlusion-aware)"
    combos2 = {"rank-avg: confidence + trust": [conf_k, trust_k],
               "rank-avg: confidence + position": [conf_k, "position (top of frame = far)"],
               "rank-avg: confidence + trust + position": [conf_k, trust_k, "position (top of frame = far)"]}
    if "box size (small = far)" in signals:
        combos2["rank-avg: confidence + trust + box size"] = [conf_k, trust_k, "box size (small = far)"]
    print("\nTraining-free combinations (each signal ranked within its own camera, then averaged):")
    E1["rank_average_combinations"] = {}
    for name, ks in combos2.items():
        s_ = np.mean([rk[k] for k in ks], axis=0)
        A, ci = auc(y, s_), boot_ci(y, s_)
        pb = paired_boot(y, s_, signals[conf_k])
        E1["rank_average_combinations"][name] = {"auc": round(A, 3), "ci95": ci, "vs_confidence_alone": pb}
        print(f"{name:42s} AUC {A:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]   vs confidence alone {pb['auc_difference']:+.3f} "
              f"(95% CI {pb['ci95'][0]:+.3f} to {pb['ci95'][1]:+.3f})")
    res["E1_reliability_signals"] = E1

    # ---------------- E2
    val_f = [r for r in frames if r["batch"] == a.val_batch]
    val_c = [r for r in cells if r["batch"] == a.val_batch]
    P = col(val_f, "tp").sum() / col(val_f, "detected").sum()
    R = col(val_f, "found").sum() / col(val_f, "gt").sum()
    r_l = {l: max(col(val_f, f"found_{l}").sum() / max(col(val_f, f"gt_{l}").sum(), 1), 0.05) for l in LEVELS}
    slope, icpt = np.polyfit(col(val_c, "trust"), col(val_c, "recall"), 1)
    print(f"\n=== E2) Can occlusion make the COUNT more accurate? (fitted on {a.val_batch}, tested on the other cameras) ===")
    print(f"fitted on validation: precision {P:.3f}, recall {R:.3f}, recall by level "
          + ", ".join(f"{l} {v:.3f}" for l, v in r_l.items()) + f"; recall ~ {icpt:.2f} + {slope:.2f} x trust")

    def estimates(rs):
        d = col(rs, "detected")
        occl = P * sum(col(rs, f"soft_{l}") / r_l[l] for l in LEVELS)
        rh = np.clip(icpt + slope * np.nan_to_num(col(rs, "trust"), nan=np.nanmean(col(val_c, "trust"))), 0.15, 1.0)
        return {"raw count": d, "global correction": d * P / R, "occlusion-aware correction": occl,
                "trust-calibrated correction": d * P / rh}

    E2 = {"fitted_on": a.val_batch, "precision": round(float(P), 3), "recall": round(float(R), 3),
          "recall_by_level": {k: round(float(v), 3) for k, v in r_l.items()},
          "recall_vs_trust_line": {"intercept": round(float(icpt), 3), "slope": round(float(slope), 3)}}
    bar = {}
    for unit, rs in (("frames", [r for r in frames if r["batch"] != a.val_batch]),
                     ("cells", [r for r in cells if r["batch"] != a.val_batch])):
        gt = col(rs, "gt"); E2[unit] = {"n": len(rs)}
        print(f"\nTest {unit} (n={len(rs)}): {'estimator':30s} {'MAE':>7s} {'MAPE':>7s} {'bias':>7s}")
        ests = estimates(rs)
        for name, e in ests.items():
            err = e - gt
            E2[unit][name] = {"mae": round(float(np.abs(err).mean()), 2),
                              "mape": round(float((np.abs(err) / gt).mean()), 3),
                              "bias": round(float((err / gt).mean()), 3)}
            if unit == "frames":
                bar[name] = E2[unit][name]["mape"]
            print(f"{'':22s}{name:30s} {E2[unit][name]['mae']:7.2f} {E2[unit][name]['mape']:7.1%} {E2[unit][name]['bias']:+7.1%}")
        # paired bootstrap: occlusion-aware vs global
        rng = np.random.default_rng(2)
        e_o, e_g = np.abs(ests["occlusion-aware correction"] - gt) / gt, np.abs(ests["global correction"] - gt) / gt
        d = [(e_g[i] - e_o[i]).mean() for i in (rng.integers(0, len(gt), len(gt)) for _ in range(2000))]
        E2[unit]["occlusion_vs_global"] = {"mape_reduction": round(float(np.mean(d)), 3),
                                           "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]}
        o = E2[unit]["occlusion_vs_global"]
        print(f"{'':22s}occlusion-aware vs global: MAPE reduced by {o['mape_reduction']:+.1%} "
              f"(95% CI {o['ci95'][0]:+.1%} to {o['ci95'][1]:+.1%})")
        per_cam = {}
        for bt in sorted(set(r["batch"] for r in rs)):
            sub = [r for r in rs if r["batch"] == bt]; g2 = col(sub, "gt"); e2 = estimates(sub)
            per_cam[bt] = {k: round(float((np.abs(v - g2) / g2).mean()), 3) for k, v in e2.items()}
        E2[unit]["mape_per_camera"] = per_cam
    # ---- E2b: calibrate PER CAMERA (first half of its frames), test on the second half of the same camera
    print("\n=== E2b) Per-camera calibration (fit on the first half of each camera's frames, test on the second half) ===")
    E2b, pooled = {}, {"raw count": [], "per-camera global correction": [], "per-camera occlusion-aware correction": []}
    for bt in sorted(set(r["batch"] for r in frames)):
        fr = sorted([r for r in frames if r["batch"] == bt], key=lambda r: r["image"])
        if len(fr) < 4:
            continue
        fit, test = fr[:len(fr) // 2], fr[len(fr) // 2:]
        Pc = col(fit, "tp").sum() / col(fit, "detected").sum(); Rc = col(fit, "found").sum() / col(fit, "gt").sum()
        rlc = {l: max(col(fit, f"found_{l}").sum() / max(col(fit, f"gt_{l}").sum(), 1), 0.05) for l in LEVELS}
        g2, d2 = col(test, "gt"), col(test, "detected")
        est = {"raw count": d2, "per-camera global correction": d2 * Pc / Rc,
               "per-camera occlusion-aware correction": Pc * sum(col(test, f"soft_{l}") / rlc[l] for l in LEVELS)}
        E2b[bt] = {k: round(float((np.abs(v - g2) / g2).mean()), 3) for k, v in est.items()}
        for k, v in est.items():
            pooled[k] += list(np.abs(v - g2) / g2)
        print(f"{bt:30s} " + "  ".join(f"{k}: {v:.1%}" for k, v in E2b[bt].items()))
    E2b["pooled_mape"] = {k: round(float(np.mean(v)), 3) for k, v in pooled.items() if v}
    print("pooled MAPE: " + ", ".join(f"{k} {v:.1%}" for k, v in E2b["pooled_mape"].items()))
    E2["E2b_per_camera_calibration"] = E2b
    res["E2_count_correction"] = E2

    json.dump(res, open(out / "research_metrics.json", "w"), indent=2)
    print(f"\nSaved {out / 'research_metrics.json'}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, s in plot_curves.items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=f"{name} (AUC {auc(y, s):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance (0.50)")
        ax.set_xlabel("False alarm rate (reliable cells flagged)"); ax.set_ylabel("Unreliable cells caught")
        ax.set_title("E1: predicting unreliable counts"); ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(out / "E1_roc_signals.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(bar.keys())[::-1], [v * 100 for v in list(bar.values())[::-1]])
        ax.set_xlabel("Mean absolute % error of the count (test frames)"); ax.set_title("E2: count estimators")
        fig.tight_layout(); fig.savefig(out / "E2_count_error.png", dpi=150); plt.close(fig)
        print(f"Saved figures E1_roc_signals.png and E2_count_error.png in {out}")
    except Exception as e:
        print(f"(figures skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights"); ap.add_argument("--occlusion-cnn"); ap.add_argument("--frames-root")
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--rf-model", default="weights/occlusion_geometry_rf.joblib")
    ap.add_argument("--batches", nargs="+", default=DAY2)
    ap.add_argument("--val-batch", default=VAL_BATCH)
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024); ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=3); ap.add_argument("--min-gt", type=int, default=5)
    ap.add_argument("--unreliable-recall", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/research")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.analyze_only:
        rows = list(csv.DictReader(open(Path(a.out) / "cells_and_frames.csv", encoding="utf-8")))
        for r in rows:
            for k in NUM:
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", "nan", None) else float("nan")
    else:
        if not (a.weights and a.occlusion_cnn and a.frames_root):
            raise SystemExit("--weights, --occlusion-cnn and --frames-root are needed (or use --analyze-only)")
        rows = collect(a)
    analyze(rows, a)


if __name__ == "__main__":
    main()
