"""
build_dataset.py

Builds the training dataset for the product detector from:
  * your annotated real CCTV batches  (annotations/converted/<batch>/labels + images on disk)
  * optionally the merged public dataset (merged_dataset from merge_datasets.py)

SPLIT RULE (important): real data is split by WHOLE BATCH (= whole camera
location), never by individual frame. Frames from one camera are near-duplicates
of each other, so splitting them randomly would put almost-identical pictures in
both train and test and make the scores meaningless.

    train = every real batch you do not name in --val / --test  (+ public data)
    val   = the batches named in --val   (used to pick the best epoch)
    test  = the batches named in --test  (touched only for the final numbers)

The public dataset goes into TRAIN only; its own val/test images are not from
your warehouse, so they are not used for evaluation.

Real train images are repeated --real-repeat times (default 3) so the ~1000
public images (about 8 boxes each) don't drown out your ~60 real frames
(about 200+ boxes each). Use --no-public for a real-only run (useful as a
comparison: does the public data actually help?).

Usage (from the f1-inventory folder), for example:
    python scripts\\build_dataset.py ^
        --frames-root "D:\\Research\\research camera data" ^
        --public "D:\\Research\\dataset\\merged_dataset" ^
        --out "D:\\Research\\dataset\\f1_training_v1" ^
        --val location_checkingPart_2_b1 ^
        --test location_camera_18_b1

Output:
    <out>/train|val|test/images + labels
    <out>/data.yaml            (relative paths, so the folder can be zipped and
                                moved to Colab as-is)
    <out>/split_manifest.csv   (every file: source, batch, day, split)
"""

import re
import sys
import csv
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DAY_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def index_frames(frames_root: Path):
    index = defaultdict(list)
    for p in frames_root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.parent.name == "extracted_frames":
            index[p.name].append(p)
    return index


def day_and_camera(path: Path):
    day = next((x for x in path.parts if DAY_RE.match(x)), "unknown-day")
    return day, path.parent.parent.name


def count_boxes(label_path: Path):
    return len([l for l in label_path.read_text().splitlines() if l.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--converted", default=r"annotations\converted")
    ap.add_argument("--public", default=None, help="merged public dataset folder (has train/val/test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", nargs="+", default=[], help="batch folder name(s) to use as validation")
    ap.add_argument("--test", nargs="+", default=[], help="batch folder name(s) to use as test")
    ap.add_argument("--real-repeat", type=int, default=3, help="copies of each real TRAIN image (default 3)")
    ap.add_argument("--no-public", action="store_true", help="leave the public dataset out")
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier output folder made by this script")
    args = ap.parse_args()

    frames_root, converted, out = Path(args.frames_root), Path(args.converted), Path(args.out)
    if not frames_root.exists():
        sys.exit(f"Frames folder not found: {frames_root}")
    if not converted.exists():
        sys.exit(f"Converted folder not found: {converted}")

    batches = sorted(d for d in converted.iterdir() if (d / "occlusion_labels.csv").exists())
    names = [b.name for b in batches]
    for flag, chosen in (("--val", args.val), ("--test", args.test)):
        for n in chosen:
            if n not in names:
                sys.exit(f"{flag} batch '{n}' not found. Available batches:\n  " + "\n  ".join(names))
    if set(args.val) & set(args.test):
        sys.exit("The same batch is in both --val and --test.")
    if not args.val or not args.test:
        sys.exit("Please give at least one batch for --val and one for --test (see usage).")

    if out.exists() and any(out.iterdir()):
        if args.overwrite and (out / "split_manifest.csv").exists():
            shutil.rmtree(out)
        else:
            sys.exit(f"{out} already exists and is not empty. Pick a new --out, or use --overwrite "
                     f"(only works on a folder this script created).")
    for sp in ("train", "val", "test"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Indexing frames under {frames_root} ...")
    index = index_frames(frames_root)

    manifest = []
    stats = {sp: {"real_imgs": 0, "real_copies": 0, "public_imgs": 0, "boxes": 0} for sp in ("train", "val", "test")}
    missing = []

    # ---- real batches --------------------------------------------------
    for b in batches:
        split = "test" if b.name in args.test else "val" if b.name in args.val else "train"
        img_names = sorted({r["image_name"] for r in csv.DictReader(open(b / "occlusion_labels.csv", newline=""))})
        for name in img_names:
            paths = index.get(name, [])
            label = b / "labels" / (Path(name).stem + ".txt")
            if not paths or not label.exists():
                missing.append(f"{b.name}/{name}")
                continue
            src = paths[0]
            day, cam = day_and_camera(src)
            n_boxes = count_boxes(label)
            reps = args.real_repeat if split == "train" else 1
            for k in range(reps):
                tag = f"real__{b.name}__{Path(name).stem}" + (f"__r{k}" if reps > 1 else "")
                shutil.copy2(src, out / split / "images" / f"{tag}{src.suffix}")
                shutil.copy2(label, out / split / "labels" / f"{tag}.txt")
                manifest.append({"file": tag + src.suffix, "source": "real", "batch": b.name, "day": day,
                                 "camera_folder": cam, "split": split, "boxes": n_boxes})
                stats[split]["real_copies"] += 1
                stats[split]["boxes"] += n_boxes
            stats[split]["real_imgs"] += 1

    # ---- public dataset (train only) ------------------------------------
    if args.public and not args.no_public:
        pub = Path(args.public)
        if not pub.exists():
            sys.exit(f"Public dataset folder not found: {pub}")
        for sp in ("train", "val", "test"):
            img_dir, lbl_dir = pub / sp / "images", pub / sp / "labels"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                tag = f"public__{img.stem}"
                shutil.copy2(img, out / "train" / "images" / f"{tag}{img.suffix}")
                shutil.copy2(lbl, out / "train" / "labels" / f"{tag}.txt")
                n_boxes = count_boxes(lbl)
                manifest.append({"file": tag + img.suffix, "source": "public", "batch": "public", "day": "",
                                 "camera_folder": "", "split": "train", "boxes": n_boxes})
                stats["train"]["public_imgs"] += 1
                stats["train"]["boxes"] += n_boxes

    # ---- data.yaml (relative paths -> portable) --------------------------
    (out / "data.yaml").write_text(
        "# Paths are relative to this file's folder, so the whole folder can be zipped and moved.\n"
        "# If Ultralytics complains about paths, add an absolute 'path: <this folder>' line at the top.\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['product']\n"
    )
    with open(out / "split_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "batch", "day", "camera_folder", "split", "boxes"])
        w.writeheader()
        w.writerows(manifest)

    # ---- summary ----------------------------------------------------------
    print("\nSplit summary")
    print(f"{'split':6} {'real frames':>11} {'real copies':>11} {'public imgs':>11} {'boxes':>8}   real batches")
    for sp in ("train", "val", "test"):
        s = stats[sp]
        bl = sorted({m["batch"] for m in manifest if m["split"] == sp and m["source"] == "real"})
        print(f"{sp:6} {s['real_imgs']:>11} {s['real_copies']:>11} {s['public_imgs']:>11} {s['boxes']:>8}   {', '.join(bl)}")
    if missing:
        print(f"\nWARNING: {len(missing)} annotated frames could not be found on disk and were skipped, e.g. {missing[:3]}")
    print(f"\nDataset written to: {out}")
    print(f"Train with: {out / 'data.yaml'}   (remember max_det=1000 when validating - your frames have 200-300+ boxes)")


if __name__ == "__main__":
    main()
