r"""
Score a trained detector on the val and test splits (precision, recall, mAP50, mAP50-95).

Example:
  python eval_detector.py --weights weights/day2_s1024_best.pt --data D:\Research\dataset\f1_training_v2_realonly\data.yaml --imgsz 1024
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max-det", type=int, default=1000)
    a = ap.parse_args()

    model = YOLO(a.weights)
    for sp in a.splits:
        m = model.val(data=a.data, split=sp, imgsz=a.imgsz, max_det=a.max_det, verbose=False, plots=False)
        print(f"{sp}: P {m.box.mp:.3f}  R {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")


if __name__ == "__main__":
    main()
