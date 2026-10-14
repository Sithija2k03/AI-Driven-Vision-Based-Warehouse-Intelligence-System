r"""
Turn the saved result files into evidence figures for the slides, the report and Teams Planner.
No models needed; runs in seconds. Missing inputs are skipped with a message.

  python scripts\make_evidence_figures.py            -> outputs\figures\*.png

Figures:
  fig1_recall_by_occlusion.png     detector recall per occlusion level with 95% CI (IoU 0.5 and 0.3)
  fig2_occlusion_training_curves   training loss and validation macro-F1 per epoch for every CNN run
  fig3_classifier_comparison       test accuracy / macro-F1: baseline vs CNN runs vs layout RF vs hybrid
  fig4_confusion_matrices          CNN, layout RF and hybrid (rows = true level, % per row)
  fig5_rf_feature_importance       which box-layout features the random forest uses
  fig6_trust_threshold             ROC of trust vs unreliable cells, Youden point and 0.80, recall by trust quartile
  fig7_stockout_fix                status accuracy before/after the empty-shelf calibration
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = ["low", "medium", "high"]
RUNS = {"cnn_r18_v1": "ResNet18 v1", "cnn_r18_dedupe": "ResNet18 A (de-duplicated)",
        "cnn_r18_reg": "ResNet18 B (regularised, chosen)", "cnn_effb0_reg": "EfficientNet-B0"}


def load_json(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(out / name, dpi=160); plt.close(fig)
    print(f"  saved {name}")


def fig_recall(out, a):
    series = []
    for path, label in ((a.recall_csv, "match IoU 0.5"), (a.recall_csv.replace(".csv", "_iou03.csv"), "match IoU 0.3")):
        if Path(path).exists():
            rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["batch"] == "ALL"]
            rows = sorted(rows, key=lambda r: LEVELS.index(r["occlusion_level"]))
            series.append((label, [float(r["recall"]) for r in rows],
                           [float(r["ci95_low"]) for r in rows], [float(r["ci95_high"]) for r in rows]))
    if not series:
        print("  skip fig1 (no recall_by_occlusion.csv)"); return
    fig, ax = plt.subplots(figsize=(6, 4)); x = np.arange(3); w = 0.38
    for i, (label, r, lo, hi) in enumerate(series):
        pos = x + (i - (len(series) - 1) / 2) * w
        ax.bar(pos, r, w, label=label, yerr=[np.array(r) - lo, np.array(hi) - r], capsize=4)
        for p_, v in zip(pos, r):
            ax.text(p_, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["low", "medium", "high"]); ax.set_ylim(0, 1)
    ax.set_xlabel("Occlusion level"); ax.set_ylabel("Detector recall (share of cartons found)")
    ax.set_title("Hidden cartons are found less often (day-2 test, 95% CI)"); ax.legend()
    save(fig, out, "fig1_recall_by_occlusion.png")


def fig_training(out, a):
    runs = {k: load_json(Path(a.occ_dir) / k / "metrics.json") for k in RUNS}
    runs = {k: v for k, v in runs.items() if v and v.get("history")}
    if not runs:
        print("  skip fig2 (no CNN metrics.json with history)"); return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for k, m in runs.items():
        ep = [h["epoch"] for h in m["history"]]
        axs[0].plot(ep, [h["train_loss"] for h in m["history"]], marker="o", ms=3, label=RUNS[k])
        axs[1].plot(ep, [h["val_macro_f1"] for h in m["history"]], marker="o", ms=3, label=RUNS[k])
    axs[0].set_title("Training loss per epoch"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss")
    axs[1].set_title("Validation macro-F1 per epoch (best epoch kept)"); axs[1].set_xlabel("epoch"); axs[1].set_ylabel("macro-F1")
    axs[1].legend(fontsize=8)
    save(fig, out, "fig2_occlusion_training_curves.png")


def fusion_hybrid(fus):
    if not fus:
        return None, None
    for k, v in fus["test"].items():
        if "0.5" in k and "Fusion" in k:
            return k, v
    return None, None


def fig_comparison(out, a):
    rows = []
    base = None
    for k, lab in RUNS.items():
        m = load_json(Path(a.occ_dir) / k / "metrics.json")
        if m:
            rows.append((lab, m["test"]["accuracy"], m["test"]["macro_f1"]))
            base = base or m.get("test_majority_baseline")
    rf = load_json(Path(a.occ_dir) / "geometry_rf" / "metrics.json")
    if rf:
        rows.append(("Layout random forest", rf["test"]["accuracy"], rf["test"]["macro_f1"]))
    _, hyb = fusion_hybrid(load_json(Path(a.occ_dir) / "fusion" / "metrics.json"))
    if hyb:
        rows.append(("Hybrid 50/50 (used)", hyb["accuracy"], hyb["macro_f1"]))
    if base:
        rows.insert(0, ("Majority baseline", base["accuracy"], base["macro_f1"]))
    if not rows:
        print("  skip fig3 (no classifier metrics)"); return
    fig, ax = plt.subplots(figsize=(9, 4.2)); y = np.arange(len(rows)); h = 0.38
    ax.barh(y - h / 2, [r[1] for r in rows], h, label="accuracy")
    ax.barh(y + h / 2, [r[2] for r in rows], h, label="macro-F1")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.005, i - h / 2, f"{r[1]:.3f}", va="center", fontsize=8)
        ax.text(r[2] + 0.005, i + h / 2, f"{r[2]:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 0.85)
    ax.set_title("Occlusion classifier, day-2 test crops"); ax.legend(loc="upper right")
    save(fig, out, "fig3_classifier_comparison.png")


def fig_confusion(out, a):
    mats = []
    m = load_json(Path(a.occ_dir) / "cnn_r18_reg" / "metrics.json")
    if m:
        mats.append(("CNN (ResNet18 B)", m["test"]["confusion_matrix_rows_true_cols_pred"]))
    rf = load_json(Path(a.occ_dir) / "geometry_rf" / "metrics.json")
    if rf:
        mats.append(("Layout random forest", rf["test"]["confusion_matrix_rows_true_cols_pred"]))
    _, hyb = fusion_hybrid(load_json(Path(a.occ_dir) / "fusion" / "metrics.json"))
    if hyb:
        mats.append(("Hybrid 50/50", hyb["confusion_matrix_rows_true_cols_pred"]))
    if not mats:
        print("  skip fig4"); return
    fig, axs = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 3.8))
    axs = np.atleast_1d(axs)
    for ax, (title, cm) in zip(axs, mats):
        cm = np.array(cm, float); pc = cm / cm.sum(1, keepdims=True)
        ax.imshow(pc, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{pc[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                        color="white" if pc[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(3), LEVELS); ax.set_yticks(range(3), LEVELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    save(fig, out, "fig4_confusion_matrices.png")


def fig_importance(out, a):
    rf = load_json(Path(a.occ_dir) / "geometry_rf" / "metrics.json")
    if not rf or "feature_importance" not in rf:
        print("  skip fig5"); return
    fi = list(rf["feature_importance"].items())[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(fi) + 1))
    ax.barh([f for f, _ in fi], [v for _, v in fi])
    ax.set_xlabel("importance"); ax.set_title("Layout model: which box-layout features matter")
    save(fig, out, "fig5_rf_feature_importance.png")


def fig_trust(out, a):
    p = Path(a.trust_dir) / "cells.csv"
    if not p.exists():
        print("  skip fig6 (no trust_eval/cells.csv)"); return
    from sklearn.metrics import roc_curve, roc_auc_score
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    t = np.array([float(r["trust_hybrid"]) for r in rows]); rec = np.array([float(r["recall"]) for r in rows])
    y = rec < 0.5
    fpr, tpr, thr = roc_curve(y, -t); k = int(np.argmax(tpr - fpr))
    f80 = t < 0.80
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(fpr, tpr, label=f"trust (AUC {roc_auc_score(y, -t):.2f})"); axs[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axs[0].scatter([fpr[k]], [tpr[k]], color="red", zorder=3, label=f"Youden best: trust < {-thr[k]:.3f}")
    axs[0].scatter([f80[~y].mean()], [f80[y].mean()], color="orange", marker="s", zorder=3, label="chosen: trust < 0.80")
    axs[0].set_xlabel("false alarm rate"); axs[0].set_ylabel("unreliable cells caught")
    axs[0].set_title(f"Choosing the verify threshold ({len(rows)} cells)"); axs[0].legend(fontsize=8)
    qs = np.quantile(t, [0, .25, .5, .75, 1]); means = []
    for i in range(4):
        m = (t >= qs[i]) & ((t <= qs[i + 1]) if i == 3 else (t < qs[i + 1])); means.append(rec[m].mean())
    axs[1].bar([f"Q{i + 1}\n{qs[i]:.2f}-{qs[i + 1]:.2f}" for i in range(4)], means)
    axs[1].set_ylabel("mean detector recall"); axs[1].set_title("Recall by trust quartile (low trust -> lower recall)")
    save(fig, out, "fig6_trust_threshold.png")


def fig_stockout(out, a):
    rc = load_json(a.ratio_calibration)
    if not rc:
        print("  skip fig7 (no ratio_calibration.json)"); return
    names = list(rc["methods"].keys()); cats = ["IN STOCK", "LOW", "STOCK-OUT"]
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(cats) + 1); w = 0.8 / len(names)
    for i, n in enumerate(names):
        m = rc["methods"][n]
        vals = [m["by_true_status"].get(c, 0) for c in cats] + [m["status_accuracy"]]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=n)
    ax.set_xticks(x, cats + ["overall"]); ax.set_ylim(0, 1.05); ax.set_ylabel("status correct")
    ax.set_title("Synthetic depletion test: status accuracy before/after calibration (leave-one-camera-out)")
    ax.legend(fontsize=8)
    save(fig, out, "fig7_stockout_fix.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall-csv", default="outputs/recall_by_occlusion.csv")
    ap.add_argument("--occ-dir", default="outputs/occlusion")
    ap.add_argument("--trust-dir", default="outputs/occlusion/trust_eval")
    ap.add_argument("--ratio-calibration", default="outputs/simulation_validation/ratio_calibration.json")
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out}")
    for f in (fig_recall, fig_training, fig_comparison, fig_confusion, fig_importance, fig_trust, fig_stockout):
        try:
            f(out, a)
        except Exception as e:
            print(f"  {f.__name__} failed: {e}")
    print("Also use: outputs\\research\\E1_roc_signals.png, E2_count_error.png and "
          "outputs\\simulation_validation\\V1_response.png")


if __name__ == "__main__":
    main()
