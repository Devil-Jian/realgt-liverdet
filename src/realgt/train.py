"""YOLOv8s: each epoch -> Val, Val-best checkpoint -> one final Test."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .metrics import evaluate


def run_yolo(dataset: str, output: str, *, epochs: int = 100, batch: int = 32,
             imgsz: int = 640, patience: int = 15, device: str = "0",
             workers: int = 8, seed: int = 42, pretrained: bool = True):
    from ultralytics import YOLO
    if epochs < 1 or batch == 0 or batch < -1 or patience < 1 or imgsz < 64:
        raise ValueError("Invalid epochs/batch/patience/imgsz")
    data = Path(dataset).resolve()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = data / "annotations" / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        anns = json.loads(path.read_text())
        if not anns["images"] or not anns["annotations"]:
            raise ValueError(f"{split} requires labeled lesions and slices")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    model = YOLO("yolov8s.pt" if pretrained else "yolov8s.yaml")
    # Val runs at the end of each epoch. Ultralytics manages best.pt and early stopping.
    model.train(data=str(data / "dataset.yaml"),
                project=str(out.parent), name=out.name, exist_ok=True,
                epochs=epochs, patience=patience, batch=batch, imgsz=imgsz,
                device=device, workers=workers, seed=seed,
                deterministic=True, val=True, save=True, verbose=True,
                mosaic=0.0, mixup=0.0, copy_paste=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                degrees=0.0, shear=0.0, fliplr=0.0, flipud=0.0,
                translate=0.02, scale=0.1)
    best = out / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Expected Val-best checkpoint: {best}")
    (out / "best_checkpoint.txt").write_text(str(best) + "\n")
    print(f"Val-best checkpoint: {best}", flush=True)
    # Absolutely no Test queries before the checkpoint has been finalized.
    test_gt = data / "annotations" / "test.json"
    gt = json.loads(test_gt.read_text())
    infer = YOLO(str(best))
    detections = []
    # Inference in small lists to avoid loading thousands of PNGs into RAM.
    test_batch = 16 if batch == -1 else min(32, batch)
    for start in range(0, len(gt["images"]), test_batch):
        imgs = gt["images"][start:start+test_batch]
        image_files = [str(data / "images" / "test" / item["file_name"]) for item in imgs]
        results = infer.predict(source=image_files, imgsz=imgsz, batch=test_batch,
                                conf=.001, iou=.7, device=device, verbose=False)
        if len(results) != len(imgs):
            raise RuntimeError("Test prediction count differs from expected input order")
        for item, result in zip(imgs, results):
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, conf, cls in zip(boxes.xyxy.cpu().tolist(),
                                       boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = xyxy
                detections.append({"image_id": item["id"], "category_id": 1,
                                   "bbox": [x1, y1, x2-x1, y2-y1], "score": float(conf)})
        print(f"Test inference: {min(start+test_batch, len(gt['images']))}/{len(gt['images'])}", flush=True)
    preds = out / "test_predictions.json"
    preds.write_text(json.dumps(detections) + "\n")
    metrics_file = out / "test_metrics.json"
    metrics = evaluate(test_gt, preds, metrics_file)
    print(f"FINAL TEST: mAP50={metrics['ap50']:.4f}, "
          f"mAP50-95={metrics['ap_50_95']:.4f}; {metrics_file}", flush=True)
    return best, metrics
