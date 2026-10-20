r"""
WITH vs WITHOUT occlusion handling (ablation study) for F1, scored against TRUE counts.

The same shelves are evaluated twice: once with every occlusion part switched off, once with it switched on.
Nothing else changes (same detector, same frames, same shelves), so any difference comes from occlusion handling.

Inputs (already produced earlier, no models are run here):
  outputs/occlusion_proof/shelf_readings.csv     (from occlusion_before_after.py: 185 day-2 shelf readings, true counts)
  outputs/temporal*/temporal_metrics*.json       (from temporal_blocking.py: picker study), optional

Three parts:
  1. COUNT  - how many cartons are on the shelf?
       WITHOUT: A0 raw detector count; A1 raw x per-camera factor (best non-occlusion correction)
       WITH   : B1 A1 x (camera's usual trust / this shelf's occlusion trust)   <- shelves with more hidden cartons are scaled up
                B2 each carton counted 1 / its occlusion weight (1.00 / 0.87 / 0.63), scaled per camera
       Fair test: corrections are learned on the OTHER shelves of the same camera and tested on the held-out shelf
       (leave-one-shelf-out). A second check learns on the first half of each camera's frames, tests on the second.
  2. WARNING - does the system warn when the count is wrong?  (same number of warnings for every method)
  3. PICKERS - temporary occlusion: single-frame vs picker-aware (from the picker study)

Usage (f1-inventory folder, seconds):
  python scripts\occlusion_ablation.py
Outputs in outputs\occlusion_ablation\: ablation_table.md (paste into slides/report), ablation.json, ablation_counts.csv,
  ablation_chart.png
"""
import argparse, csv, glob, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = {"low": 1.00, "med": 0.87, "high": 0.63}       # trust weights from measured recall 0.659 / 0.572 / 0.414


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in ("true_count", "detected", "recall", "count_error", "share_mostly_hidden", "share_partly_or_mostly",
                  "trust", "confidence_unreliability", "reliability", "f1_check_by_eye"):
            r[k] = num(r.get(k))
        det = r["detected"]
        r["n_high"] = r["share_mostly_hidden"] * det
        r["n_med"] = (r["share_partly_or_mostly"] - r["share_mostly_hidden"]) * det
        r["n_low"] = det - r["n_high"] - r["n_med"]
        r["fnum"] = int("".join(ch for ch in r["frame"].split("__")[-1] if ch.isdigit()) or 0)
    return [r for r in rows if r["true_count"] > 0 and np.isfinite(r["trust"])]


def inv_weight(r):
    return r["n_low"] / WEIGHT["low"] + r["n_med"] / WEIGHT["med"] + r["n_high"] / WEIGHT["high"]


def estimates(fit, test):
    """Learn on `fit` rows, return {method: estimate} for each test row."""
    f_cam = sum(r["true_count"] for r in fit) / max(sum(r["detected"] for r in fit), 1e-9)
    t_ref = float(np.mean([r["trust"] for r in fit]))
    f_inv = sum(r["true_count"] for r in fit) / max(sum(inv_weight(r) for r in fit), 1e-9)
    out = []
    for r in test:
        out.append({"A0": r["detected"], "A1": r["detected"] * f_cam,
                    "B1": r["detected"] * f_cam * t_ref / r["trust"], "B2": inv_weight(r) * f_inv})
    return out


def run_split(rows, scheme):
    preds = []
    cams = sorted({r["camera"] for r in rows})
    for c in cams:
        g = [r for r in rows if r["camera"] == c]
        if scheme == "shelf":                          # leave one shelf out
            for s in sorted({r["shelf"] for r in g}):
                fit = [r for r in g if r["shelf"] != s]; test = [r for r in g if r["shelf"] == s]
                if fit and test:
                    preds += list(zip(test, estimates(fit, test)))
        else:                                          # first half of frames -> second half
            fr = sorted({r["fnum"] for r in g}); first = set(fr[:len(fr) // 2])
            fit = [r for r in g if r["fnum"] in first]; test = [r for r in g if r["fnum"] not in first]
            if fit and test:
                preds += list(zip(test, estimates(fit, test)))
    return preds


METHODS = [("A0", "WITHOUT: raw detector count"), ("A1", "WITHOUT: raw x per-camera factor"),
           ("B1", "WITH: per-camera factor x occlusion trust"), ("B2", "WITH: each carton weighted by how hidden")]


def count_scores(preds, cut):
    res = {}
    hid = np.array([r["share_mostly_hidden"] for r, _ in preds])
    tier = np.where(hid <= cut[0], "low", np.where(hid <= cut[1], "mid", "high"))
    true = np.array([r["true_count"] for r, _ in preds])
    for m, _ in METHODS:
        est = np.array([p[m] for _, p in preds]); e = (est - true) / true
        res[m] = {"mape": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
                  "undercount_25": float(np.mean(e <= -0.25)), "n": int(len(e))}
        for t in ("low", "mid", "high"):
            k = tier == t
            res[m][f"mape_{t}"] = float(np.mean(np.abs(e[k]))) if k.any() else None
            res[m][f"bias_{t}"] = float(np.mean(e[k])) if k.any() else None
    return res


def warning_scores(rows, under=0.25):
    err = np.array([r["count_error"] for r in rows]); bad = err <= -under
    f1 = np.array([bool(r["f1_check_by_eye"]) for r in rows]); k = int(f1.sum())
    sig = {"confidence only (no occlusion)": np.array([r["confidence_unreliability"] for r in rows]),
           "occlusion: share mostly hidden": np.array([r["share_mostly_hidden"] for r in rows]),
           "occlusion: trust score": 1 - np.array([r["trust"] for r in rows])}
    out = {"under_definition": f"count at least {under:.0%} below the true count", "undercounted": int(bad.sum()),
           "readings": len(rows), "warnings_each": k,
           "plain counting (no warnings)": {"warned": 0.0, "bad_among_reliable": float(bad.mean())}}
    for name, s in sig.items():
        f = np.zeros(len(rows), bool); f[np.argsort(-np.nan_to_num(s, nan=-9))[:k]] = True
        out[name] = {"warned": float(f[bad].mean()) if bad.any() else None, "bad_among_reliable": float(bad[~f].mean())}
    out["F1 current combined score"] = {"warned": float(f1[bad].mean()) if bad.any() else None,
                                        "bad_among_reliable": float(bad[~f1].mean())}
    return out


def picker_scores():
    found = {}
    for f in sorted(glob.glob(str(ROOT / "outputs" / "temporal*" / "temporal_metrics*.json"))):
        p = Path(f); folder = p.parent.name
        tag = p.stem.replace("temporal_metrics", "").strip("_") or "v1"
        if folder not in found or tag > found[folder][0]:
            found[folder] = (tag, p)
    out = {}
    for folder, (tag, p) in found.items():
        try:
            s = json.load(open(p, encoding="utf-8"))["summary"]
        except Exception:
            continue
        out[f"{folder} ({tag})"] = {k: s.get(k) for k in ("shelf_frames", "blocked_share", "status_error_single",
                                                          "status_error_proposed", "false_alarms_single",
                                                          "false_alarms_proposed", "status_flips_single",
                                                          "status_flips_proposed", "mean_cv_single", "mean_cv_fused")}
    return out


def pc(v):
    return "-" if v is None else f"{v:.1%}"


def sg(v):
    return "-" if v is None else f"{v:+.1%}"


def markdown(C, Ct, W, P, cut):
    L = ["# F1 occlusion handling: WITHOUT vs WITH (ablation)", "",
         "Same detector, same day-2 frames, same shelves; only occlusion handling is switched off or on. Scored against labelled true counts.", "",
         "## 1. Count accuracy (unseen shelves: learned on the other shelves of the same camera)", "",
         f"Occlusion groups by share of mostly hidden cartons on the shelf: low <= {cut[0]:.0%}, mid <= {cut[1]:.0%}, high > {cut[1]:.0%}.", "",
         "| Method | Count error (MAPE) | Low-occlusion shelves | Mid | High-occlusion shelves | Bias on high-occlusion shelves | Readings under-counted by 25%+ |",
         "|---|---|---|---|---|---|---|"]
    for m, name in METHODS:
        r = C[m]
        L.append(f"| {name} | {pc(r['mape'])} | {pc(r['mape_low'])} | {pc(r['mape_mid'])} | {pc(r['mape_high'])} | "
                 f"{sg(r['bias_high'])} | {pc(r['undercount_25'])} |")
    L += ["", "Second check (learned on the first half of each camera's frames, tested on the second half):", "",
          "| Method | Count error (MAPE) | High-occlusion shelves | Bias on high-occlusion shelves |", "|---|---|---|---|"]
    for m, name in METHODS:
        r = Ct[m]
        L.append(f"| {name} | {pc(r['mape'])} | {pc(r['mape_high'])} | {sg(r['bias_high'])} |")
    L += ["", f"## 2. Warning when the count is wrong ({W['under_definition']}; {W['undercounted']} of {W['readings']} readings)", "",
          f"Every method gives the same number of warnings ({W['warnings_each']}), except plain counting, which never warns.", "",
          "| Method | Under-counted shelves warned | Under-counted among shelves reported reliable |", "|---|---|---|"]
    for name in ("plain counting (no warnings)", "confidence only (no occlusion)", "occlusion: share mostly hidden",
                 "occlusion: trust score", "F1 current combined score"):
        r = W[name]; L.append(f"| {name} | {pc(r['warned'])} | {pc(r['bad_among_reliable'])} |")
    if P:
        L += ["", "## 3. Temporary occlusion (pickers): single-frame (WITHOUT) vs picker-aware (WITH)", "",
              "| Camera run | Readings | Blocked by a picker | Status wrong, without | with | False LOW/STOCK-OUT, without | with | Status flips, without | with |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, s in P.items():
            L.append(f"| {k} | {s['shelf_frames']} | {pc(s['blocked_share'])} | {pc(s['status_error_single'])} | {pc(s['status_error_proposed'])} | "
                     f"{s['false_alarms_single']} | {s['false_alarms_proposed']} | {s['status_flips_single']} | {s['status_flips_proposed']} |")
    L += ["", "Caution: 15 distinct shelves on one test day; frames of one shelf are near-duplicates. Read differences of 1-2 points as trends."]
    return "\n".join(L)


def chart(C, W, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tiers = ["low", "mid", "high"]; x = np.arange(3); wbar = 0.26
    for i, (m, lab, col) in enumerate([("A0", "Without: raw count", "#94A3B8"), ("A1", "Without: per-camera factor", "#60A5FA"),
                                       ("B1", "With: occlusion-aware", "#1D4ED8")]):
        v = [C[m][f"mape_{t}"] * 100 for t in tiers]
        b = ax[0].bar(x + (i - 1) * wbar, v, wbar, color=col, label=lab)
        for bb, vv in zip(b, v):
            ax[0].text(bb.get_x() + bb.get_width() / 2, vv + 0.4, f"{vv:.0f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["low", "mid", "high"]); ax[0].set_xlabel("How hidden the shelf's cartons are")
    ax[0].set_ylabel("Count error % (MAPE, unseen shelves)"); ax[0].set_title("Count error by occlusion level", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False); ax[0].spines[["top", "right"]].set_visible(False)
    names = ["plain counting (no warnings)", "confidence only (no occlusion)", "occlusion: share mostly hidden"]
    labs = ["Without:\nplain counting", "Without:\nconfidence only", "With:\nocclusion"]
    v = [(W[n]["warned"] or 0) * 100 for n in names]
    b = ax[1].bar(labs, v, color=["#94A3B8", "#60A5FA", "#1D4ED8"])
    for bb, vv in zip(b, v):
        ax[1].text(bb.get_x() + bb.get_width() / 2, vv + 1, f"{vv:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("%"); ax[1].spines[["top", "right"]].set_visible(False)
    ax[1].set_title(f"Under-counted shelves that get a warning\n(same number of warnings: {W['warnings_each']})", fontsize=11)
    fig.suptitle("F1 with vs without occlusion handling, day-2 shelves with true counts", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    try:
        fig.savefig(out / "ablation_chart.png", dpi=160)
    except OSError:
        fig.savefig(out / "ablation_chart_new.png", dpi=160); print("!! ablation_chart.png is open elsewhere; saved ablation_chart_new.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default="outputs/occlusion_proof/shelf_readings.csv")
    ap.add_argument("--out", default="outputs/occlusion_ablation")
    ap.add_argument("--under", type=float, default=0.25)
    a = ap.parse_args()
    src = Path(a.readings); src = src if src.is_absolute() else ROOT / src
    out = Path(a.out); out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    hid = np.array([r["share_mostly_hidden"] for r in rows]); cut = (float(np.quantile(hid, 1 / 3)), float(np.quantile(hid, 2 / 3)))
    ps, pt = run_split(rows, "shelf"), run_split(rows, "time")
    C, Ct = count_scores(ps, cut), count_scores(pt, cut)
    W = warning_scores(rows, a.under)
    P = picker_scores()

    print(f"\n=== WITHOUT vs WITH occlusion handling: {len(rows)} shelf readings, {len({(r['camera'], r['shelf']) for r in rows})} shelves ===")
    print("\n1) COUNT ACCURACY on unseen shelves (learned on the other shelves of the same camera)")
    print(f"{'method':48s}{'MAPE':>8s}{'low':>8s}{'mid':>8s}{'high':>8s}{'bias high':>11s}{'under25':>9s}")
    for m, name in METHODS:
        r = C[m]
        print(f"{name:48s}{pc(r['mape']):>8s}{pc(r['mape_low']):>8s}{pc(r['mape_mid']):>8s}{pc(r['mape_high']):>8s}"
              f"{r['bias_high']:>+10.1%} {pc(r['undercount_25']):>8s}")
    print("   second check, first half of frames -> second half: " +
          " | ".join(f"{m} {pc(Ct[m]['mape'])} (high {pc(Ct[m]['mape_high'])})" for m, _ in METHODS))
    print(f"\n2) WARNING when the count is wrong: {W['under_definition']} ({W['undercounted']} of {W['readings']}); {W['warnings_each']} warnings each")
    for name in ("plain counting (no warnings)", "confidence only (no occlusion)", "occlusion: share mostly hidden",
                 "occlusion: trust score", "F1 current combined score"):
        r = W[name]; print(f"   {name:36s} warned {pc(r['warned']):>6s}   under-counted among 'reliable' {pc(r['bad_among_reliable'])}")
    if P:
        print("\n3) PICKERS (temporary occlusion): single-frame -> picker-aware")
        for k, s in P.items():
            print(f"   {k}: false alarms {s['false_alarms_single']} -> {s['false_alarms_proposed']}, status flips "
                  f"{s['status_flips_single']} -> {s['status_flips_proposed']}, status wrong {pc(s['status_error_single'])} -> {pc(s['status_error_proposed'])}")
    else:
        print("\n3) PICKERS: no outputs/temporal*/temporal_metrics*.json found (run temporal_blocking.py) - section left out")

    md = markdown(C, Ct, W, P, cut)
    (out / "ablation_table.md").write_text(md, encoding="utf-8")
    json.dump({"count_unseen_shelves": C, "count_time_split": Ct, "warnings": W, "pickers": P, "occlusion_cut": cut},
              open(out / "ablation.json", "w"), indent=2)
    with open(out / "ablation_counts.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["camera", "frame", "shelf", "true_count", "share_mostly_hidden"] + [m for m, _ in METHODS])
        for r, p in ps:
            wr.writerow([r["camera"], r["frame"], r["shelf"], int(r["true_count"]), round(r["share_mostly_hidden"], 3)] + [round(p[m], 1) for m, _ in METHODS])
    chart(C, W, out)
    print(f"\nSaved {out / 'ablation_table.md'} (paste into slides/report), ablation.json, ablation_counts.csv, ablation_chart.png")


if __name__ == "__main__":
    main()
