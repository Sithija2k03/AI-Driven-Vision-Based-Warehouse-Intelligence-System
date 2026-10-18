# F1 carton detector: baseline results (PP1)

Single class `product` (one box per visible carton). Trained on real CCTV frames only.
Split by whole batch and day: train = six day-1 batches; val = `location_checkingPart_2_b2`;
test = `location_G_H_b2` + `location_X_5_b2` (same cameras as in train, recorded on a later day).

| Run | Model | imgsz | Val P / R / mAP50 | Test P / R / mAP50 | Test mAP50-95 |
|---|---|---|---|---|---|
| day2_n640  | YOLOv8n | 640  | 0.537 / 0.483 / 0.453 | 0.576 / 0.409 / 0.380 | 0.148 |
| day2_s1024 | YOLOv8s | 1024 | 0.592 / 0.542 / 0.487 | 0.633 / 0.486 / 0.461 | 0.199 |

Earlier setting for comparison (YOLOv8n @640, held-out camera never seen in training):
test mAP50 about 0.10. Cross-camera generalisation is weak; same-camera/new-day is the deployment case.

Chosen PP1 detector: `day2_s1024`.

Caveats
- Val is one camera (13 near-duplicate frames); test is 24 frames from two cameras. Differences of a few points are within noise.
- One training run per setting.
- Recall is about 0.49: the detector still misses about half of the cartons. This is what the occlusion-aware part of F1 addresses.
- Weights are not committed (see .gitignore); they live in Google Drive under `f1_runs/day2_s1024/weights/best.pt`.
