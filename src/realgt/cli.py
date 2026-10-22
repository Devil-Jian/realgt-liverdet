"""CLI for independent REAL-GT-only AP/PVP detection project."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import (HU_MIN, HU_MAX, build_splits, export_phase,
                      make_qc_preview, prepare, scan_cases)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="realgt", description="Manual-GT-only liver lesion detection")
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="Scan manual masks -> patient splits -> PNG/COCO/YOLO")
    b.add_argument("--root", default="/dataset/dataset/gx_all_registered")
    b.add_argument("--out", default="/dataset/realgt_liverdet")
    b.add_argument("--splits-file", help="Optional fixed patient_id,split CSV; prevents changing held-out Test")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--train-fraction", type=float, default=.70)
    b.add_argument("--val-fraction", type=float, default=.15)
    b.add_argument("--hu-min", type=float, default=HU_MIN)
    b.add_argument("--hu-max", type=float, default=HU_MAX)
    t = sub.add_parser("train", help="100 Epoch max, Val each epoch, automatic one-time Test")
    t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="0")
    t.add_argument("--no-pretrained", action="store_true")
    q = sub.add_parser("qc", help="Visualize GT bboxes on image tiles")
    q.add_argument("--dataset", required=True)
    q.add_argument("--split", choices=("train", "val", "test"), default="train")
    q.add_argument("--count", type=int, default=24)
    q.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.action == "build":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        cases = scan_cases(a.root, out / "manifests" / "cases.csv")
        splits = build_splits(cases, out / "manifests" / "splits.csv", seed=a.seed,
                              train_fraction=a.train_fraction, val_fraction=a.val_fraction,
                              external_csv=a.splits_file)
        frames = prepare(cases, splits, out, hu_min=a.hu_min, hu_max=a.hu_max)
        for phase in ("ap", "pvp"):
            export_phase(frames, out, phase)
        print("Done: REAL manual ground truth datasets (AP and PVP), no pseudo masks.")
    elif a.action == "train":
        from .train import run_yolo
        run_yolo(a.dataset, a.out, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
                 patience=a.patience, device=a.device, workers=a.workers,
                 seed=a.seed, pretrained=not a.no_pretrained)
    else:
        print(f"QC preview: {make_qc_preview(a.dataset, split=a.split, count=a.count, seed=a.seed)}")


if __name__ == "__main__":
    main()
