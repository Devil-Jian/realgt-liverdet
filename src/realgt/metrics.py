"""COCO bbox AP and 2D slice-level lesion FROC. No 3D lesion FROC claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0.0, min(ax+aw, bx+bw) - max(ax, bx))
    h = max(0.0, min(ay+ah, by+bh) - max(ay, by))
    intersection = w * h
    return intersection / max(aw*ah + bw*bh - intersection, 1e-10)


def slice_froc(gt, detections, thresholds=(.05, .1, .2, .3, .5, .7, .9)):
    img_ids = {v["id"] for v in gt["images"]}
    by_gt = {key: [] for key in img_ids}
    by_det = {key: [] for key in img_ids}
    for a in gt["annotations"]:
        by_gt[a["image_id"]].append(a["bbox"])
    for a in detections:
        if a["image_id"] in by_det:
            by_det[a["image_id"]].append(a)
    result = []
    for conf in thresholds:
        tp = fp = 0
        for img in img_ids:
            matched = set()
            for pred in sorted((x for x in by_det[img] if x["score"] >= conf),
                               key=lambda x: x["score"], reverse=True):
                candidates = [(_iou_xywh(pred["bbox"], box) if i not in matched else -1.)
                              for i, box in enumerate(by_gt[img])]
                best = int(np.argmax(candidates)) if candidates else -1
                if best >= 0 and candidates[best] >= .5:
                    matched.add(best)
                    tp += 1
                else:
                    fp += 1
        result.append({"threshold": conf,
                       "slice_object_sensitivity": tp / max(1, len(gt["annotations"])),
                       "false_positives_per_slice": fp / max(1, len(img_ids)),
                       "true_positives": tp, "false_positives": fp})
    return result


def evaluate(gt_file: Path, pred_file: Path, result_file: Path) -> dict:
    gt = json.loads(Path(gt_file).read_text())
    preds = json.loads(Path(pred_file).read_text())
    if not isinstance(preds, list):
        raise ValueError("Predictions must be a COCO detections list")
    if not gt["images"] or not gt["annotations"]:
        raise ValueError("Test set has no images or no annotated lesions")
    if preds:
        coco = COCO(str(gt_file))
        coco_pred = coco.loadRes(preds)
        coco_eval = COCOeval(coco, coco_pred, "bbox")
        coco_eval.params.imgIds = [i["id"] for i in gt["images"]]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap, ap50, ap75 = (float(coco_eval.stats[i]) for i in (0, 1, 2))
    else:
        ap = ap50 = ap75 = 0.0
    values = {"ap_50_95": ap, "ap50": ap50, "ap75": ap75,
              "images": len(gt["images"]), "objects": len(gt["annotations"]),
              "froc_definition": "2D object sensitivity vs false positive boxes per slice",
              "slice_froc": slice_froc(gt, preds)}
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(values, indent=2) + "\n")
    return values
