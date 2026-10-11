r"""
Train the F1 carton detector (same settings as the Colab runs).

Needs a GPU (Colab T4 is fine). Run on Colab or any CUDA machine, not on a CPU laptop.

Examples (these reproduce the two runs reported for PP1):
  python train_detector.py --data /content/f1_training_v2_realonly/data.yaml --model yolov8n.pt --imgsz 640  --batch 16 --name day2_n640
  python train_detector.py --data /content/f1_training_v2_realonly/data.yaml --model yolov8s.pt --imgsz 1024 --batch 8  --name day2_s1024
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml made by build_dataset.py")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-det", type=int, default=1000, help="frames hold up to ~300 cartons; default 300 is too low")
    ap.add_argument("--project", default="runs/f1")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                max_det=a.max_det, project=a.project, name=a.name)


if __name__ == "__main__":
    main()
