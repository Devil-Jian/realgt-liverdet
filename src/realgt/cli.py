"""CLI for independent REAL-GT-only AP/PVP detection project."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import (HU_MIN, HU_MAX, _read_fixed_val_list, build_splits, export_phase,
                      make_qc_preview, prepare, scan_cases)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="realgt", description="Manual-GT-only liver lesion detection")
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="Scan manual masks -> patient splits -> PNG/COCO/YOLO")
    plan = sub.add_parser("plan", help="Audit cohort and calculate Train/Val/Test BEFORE expensive NIfTI processing")
    for current in (b, plan):
        current.add_argument("--root", default="/dataset/dataset/gx_all_registered")
        current.add_argument("--out", default="/dataset/realgt_liverdet")
        current.add_argument("--splits-file", help="Optional complete patient_id,split CSV. Cannot combine with --fixed-val-patients.")
        current.add_argument("--fixed-val-patients",
                       help="Private TXT/CSV with Val patient_id lines; eligible IDs fixed to Val, "
                            "missing manual-GT IDs replaced from other non-Train patients.")
        current.add_argument("--all-patients",
                       help="Private TXT/CSV of entire allowed manual tumor GT patient cohort. "
                            "Patients not listed are never included.")
        current.add_argument("--priority-train-patients",
                       help="Private TXT/CSV of priority Train IDs. Always in Train if eligible; "
                            "only remaining allowed patients may enter Val/Test.")
        current.add_argument("--seed", type=int, default=42)
        current.add_argument("--train-fraction", type=float, default=.70)
        current.add_argument("--val-fraction", type=float, default=.15)
        current.add_argument("--hu-min", type=float, default=HU_MIN)
        current.add_argument("--hu-max", type=float, default=HU_MAX)
    e = sub.add_parser("export", help="Resume COCO/YOLO exports from existing manifests/slices.csv; no NIfTI reprocessing")
    e.add_argument("--out", required=True, help="Existing realgt build output root")
    e.add_argument("--slices", help="Override slices.csv path (defaults to <out>/manifests/slices.csv)")
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
    if a.action in ("build", "plan"):
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        if a.priority_train_patients and not a.all_patients:
            raise ValueError("--priority-train-patients requires --all-patients "
                             "so the full manual GT cohort is explicitly defined")
        if a.all_patients:
            all_ids = set(_read_fixed_val_list(a.all_patients))
            for label, file in (("priority Train", a.priority_train_patients),
                                ("fixed Val", a.fixed_val_patients)):
                if file and not (set(_read_fixed_val_list(file)) <= all_ids):
                    raise ValueError(f"{label} has patient IDs outside --all-patients")
        cases = scan_cases(a.root, out / "manifests" / "cases.csv",
                           allowed_patients=a.all_patients)
        splits = build_splits(cases, out / "manifests" / "splits.csv", seed=a.seed,
                              train_fraction=a.train_fraction, val_fraction=a.val_fraction,
                              external_csv=a.splits_file,
                              fixed_val_patients=a.fixed_val_patients,
                              priority_train_patients=a.priority_train_patients)
        if a.action == "plan":
            print("Cohort plan written. No NIfTI volumes read or PNG images generated.")
            return
        frames = prepare(cases, splits, out, hu_min=a.hu_min, hu_max=a.hu_max)
        for phase in ("ap", "pvp"):
            export_phase(frames, out, phase)
        print("Done: REAL manual ground truth datasets (AP and PVP), no pseudo masks.")
    elif a.action == "export":
        out = Path(a.out).resolve()
        slices_path = Path(a.slices) if a.slices else out / "manifests" / "slices.csv"
        if not slices_path.is_file():
            raise FileNotFoundError(f"Prepared manifest missing: {slices_path}; run realgt build first")
        frame = pd.read_csv(slices_path, dtype={"patient_id": str, "sample_id": str},
                            keep_default_na=False)
        required = {"patient_id", "phase", "split", "sample_id", "real_path",
                    "boxes", "label_source", "is_positive"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Invalid prepared manifest; missing columns: {sorted(missing)}")
        if not (out / "manifests" / "preprocess.json").is_file():
            raise FileNotFoundError("Missing preprocess.json; refuses export from unverified source")
        for phase in ("ap", "pvp"):
            export_phase(frame, out, phase)
        print("Export recovered from prepared slices.csv without rerunning NIfTI conversion.")
    elif a.action == "train":
        from .train import run_yolo
        run_yolo(a.dataset, a.out, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
                 patience=a.patience, device=a.device, workers=a.workers,
                 seed=a.seed, pretrained=not a.no_pretrained)
    else:
        print(f"QC preview: {make_qc_preview(a.dataset, split=a.split, count=a.count, seed=a.seed)}")


if __name__ == "__main__":
    main()
