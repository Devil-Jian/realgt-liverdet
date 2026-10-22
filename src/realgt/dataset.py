"""Manual-annotation-only 3D NIfTI -> full-slice 2D COCO/YOLO datasets.

No teacher predictions, no liver-mask clipping and no component-area filtering.
The only accepted labels are <phase>_tumor_mask_registered.nii.gz.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

PHASES = ("ap", "pvp")
SPLITS = ("train", "val", "test")
HU_MIN, HU_MAX = -160.0, 240.0


def _csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def scan_cases(root: str, dest: Path) -> pd.DataFrame:
    """Require complete manual tumor masks for both phases for each included patient."""
    source = Path(root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    records = []
    incomplete = []
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        paths = {
            p: (folder / f"{p}_registered.nii.gz",
                folder / f"{p}_tumor_mask_registered.nii.gz")
            for p in PHASES
        }
        labeled = [p for p, (_, label) in paths.items() if label.is_file()]
        if not labeled:
            continue  # All non-manual cases are excluded, even if pseudo masks exist.
        if len(labeled) != len(PHASES) or any(not im.is_file() for im, _ in paths.values()):
            incomplete.append(folder.name)
            continue
        for phase, (image_path, mask_path) in paths.items():
            records.append({
                "patient_id": folder.name, "phase": phase,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()), "label_source": "manual",
            })
    if incomplete:
        raise ValueError("Incomplete AP/PVP manual cases (not silently skipped): " +
                         ", ".join(incomplete[:20]) + f"; total={len(incomplete)}")
    if not records:
        raise ValueError(f"No *tumor_mask_registered.nii.gz manual AP/PVP pairs found under {source}")
    df = pd.DataFrame(records)
    if df.duplicated(["patient_id", "phase"]).any():
        raise ValueError("Duplicate case/phase")
    _csv(df, dest)
    print(f"Manual GT cases: {df.patient_id.nunique()} patients / {len(df)} phase records")
    return df


def build_splits(cases: pd.DataFrame, dest: Path, *, seed: int = 42,
                 train_fraction: float = .70, val_fraction: float = .15,
                 external_csv: Optional[str] = None) -> pd.DataFrame:
    patients = sorted(cases.patient_id.astype(str).unique().tolist())
    n = len(patients)
    if n < 3:
        raise ValueError("At least 3 manually labeled patients needed for Train/Val/Test")
    if external_csv:
        sf = pd.read_csv(external_csv, dtype={"patient_id": str}, keep_default_na=False)
        if not {"patient_id", "split"}.issubset(sf.columns):
            raise ValueError("External splits CSV requires patient_id,split")
        if sf.patient_id.duplicated().any() or set(sf.patient_id) != set(patients):
            raise ValueError("External splits must contain every and only eligible GT patient once")
        if not set(sf.split).issubset(set(SPLITS)):
            raise ValueError("Unknown split; use train/val/test")
        split = sf[["patient_id", "split"]].sort_values("patient_id").reset_index(drop=True)
    else:
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
                train_fraction + val_fraction < 1):
            raise ValueError("Invalid split fractions")
        rng = np.random.default_rng(seed)
        shuffled = np.array(patients, dtype=str)
        rng.shuffle(shuffled)
        a = round(n * train_fraction)
        b = round(n * val_fraction)
        if min(a, b, n - a - b) < 1:
            raise ValueError("Train, Val and Test must each contain a patient")
        split = pd.DataFrame({"patient_id": shuffled.tolist(),
                              "split": ["train"] * a + ["val"] * b +
                                       ["test"] * (n - a - b)})
        split = split.sort_values("patient_id").reset_index(drop=True)
    if set(split.split) != set(SPLITS):
        raise ValueError("All three splits must be nonempty")
    _csv(split, dest)
    print("Patient splits: " + ", ".join(f"{p}={int((split.split==p).sum())}" for p in SPLITS))
    return split


def _bboxes(mask_yx: np.ndarray) -> list[list[int]]:
    """8-connected components, all sizes included; returned xyxy, exclusive xmax/ymax."""
    labels, _ = ndi.label(mask_yx > 0, structure=np.ones((3, 3), dtype=np.uint8))
    out = []
    for component in ndi.find_objects(labels):
        if component is None:
            continue
        y, x = component
        out.append([int(x.start), int(y.start), int(x.stop), int(y.stop)])
    return out


def _hu_png(slice_yx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not lo < hi:
        raise ValueError("Invalid HU window")
    x = np.nan_to_num(np.asarray(slice_yx, dtype=np.float32),
                      nan=lo, posinf=hi, neginf=lo)
    return np.rint((np.clip(x, lo, hi) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def prepare(cases: pd.DataFrame, splits: pd.DataFrame, out: Path, *,
            hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> pd.DataFrame:
    out = Path(out).resolve()
    meta_path = out / "manifests" / "preprocess.json"
    if not hu_min < hu_max:
        raise ValueError("--hu-min must be smaller than --hu-max")
    split_pairs = sorted((str(r.patient_id), str(r.split))
                         for r in splits.itertuples(index=False))
    split_hash = hashlib.sha256(json.dumps(split_pairs).encode("utf-8")).hexdigest()
    expected = {"hu_min": float(hu_min), "hu_max": float(hu_max),
                "input": "manual_ground_truth_only", "mask_policy": "all_positive_pixels_8_connected",
                "split_sha256": split_hash}
    if meta_path.exists() and json.loads(meta_path.read_text()) != expected:
        raise ValueError("Existing output uses different HU/label processing or patient splits; use NEW workdir")
    join = cases.merge(splits, on="patient_id", how="left", validate="many_to_one")
    if join.split.isna().any() or not join.split.isin(SPLITS).all():
        raise ValueError("Missing or invalid splits")
    if not join.label_source.eq("manual").all():
        raise ValueError("This project prohibits pseudo labels in every split")
    if (join.groupby("patient_id").phase.nunique() != 2).any():
        raise ValueError("Every patient must have AP and PVP paired ground truth")
    rows = []
    for i, case in enumerate(join.itertuples(index=False), start=1):
        vol = nib.load(case.image_path)
        gt = nib.load(case.mask_path)
        if len(vol.shape) != 3 or vol.shape != gt.shape or not np.allclose(
                vol.affine, gt.affine, rtol=0, atol=1e-3):
            raise ValueError(f"Image/manual-mask shape/affine mismatch: {case.patient_id}/{case.phase}")
        # Read each .nii.gz ONCE to avoid O(D) gzip decompressions per slice.
        x = np.asarray(vol.dataobj, dtype=np.float32)
        m = np.asarray(gt.dataobj)
        if not np.isfinite(m).all():
            raise ValueError(f"Non-finite manual mask {case.mask_path}")
        pos = 0
        for k in range(x.shape[2]):
            boxes = _bboxes(m[:, :, k].T)
            pos += bool(boxes)
            sample_id = f"{case.patient_id}_{case.phase}_{k:04d}"
            png_path = out / "data" / "real_gt" / case.phase / f"{sample_id}.png"
            if not png_path.is_file():
                png_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_hu_png(x[:, :, k].T, hu_min, hu_max), mode="L").save(png_path)
            rows.append({
                "sample_id": sample_id, "patient_id": case.patient_id,
                "phase": case.phase, "split": case.split, "slice_index": int(k),
                "real_path": str(png_path), "source_volume": str(Path(case.image_path).resolve()),
                "mask_path": str(Path(case.mask_path).resolve()),
                "boxes": json.dumps(boxes, separators=(",", ":")),
                "label_source": "manual", "is_positive": bool(boxes),
            })
        print(f"[{i}/{len(join)}] {case.patient_id}/{case.phase} {case.split}: "
              f"{x.shape[2]} slices, {pos} positive", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No slices generated")
    _csv(frame, out / "manifests" / "slices.csv")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(expected, indent=2) + "\n")
    train_cols = ["sample_id", "patient_id", "phase", "slice_index", "source_volume",
                  "mask_path", "real_path", "label_source", "is_positive"]
    train_rows = frame.loc[frame.split.eq("train"), train_cols].to_dict(orient="records")
    (out / "manifests" / "train_slices.json").write_text(
        json.dumps(train_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Slice count: {len(frame)}; train={sum(frame.split=='train')} "
          f"val={sum(frame.split=='val')} test={sum(frame.split=='test')}")
    return frame


def export_phase(frame: pd.DataFrame, out: Path, phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(phase)
    base = Path(out).resolve() / "exports" / f"real_gt_{phase}"
    counts = []
    for split in SPLITS:
        sub = frame[frame.phase.eq(phase) & frame.split.eq(split)]
        annotations, images = [], []
        im_dir, label_dir = base / "images" / split, base / "labels" / split
        im_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id, row in enumerate(sub.itertuples(index=False), 1):
            src = Path(row.real_path)
            dest = im_dir / src.name
            if not src.is_file():
                raise FileNotFoundError(src)
            if not dest.exists():
                dest.symlink_to(src)
            with Image.open(src) as im:
                w, h = im.size
            images.append({"id": image_id, "file_name": dest.name, "width": w, "height": h,
                           "patient_id": row.patient_id, "sample_id": row.sample_id})
            lines = []
            for box in json.loads(row.boxes):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                    raise ValueError(f"Invalid pixel bbox for {row.sample_id}: {box}")
                annotations.append({"id": len(annotations)+1, "image_id": image_id,
                                    "category_id": 1, "bbox": [x1, y1, bw, bh],
                                    "area": bw * bh, "iscrowd": 0})
                lines.append(f"0 {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} "
                             f"{bw/w:.8f} {bh/h:.8f}")
            (label_dir / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        ann_path = base / "annotations" / f"{split}.json"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps({
            "info": {"description": "Manual GT only: patient-disjoint liver lesion detection"},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "liver_lesion", "supercategory": "lesion"}],
        }, indent=2) + "\n")
        counts.append({"phase": phase, "split": split, "patients": sub.patient_id.nunique(),
                       "images": len(images), "positive_slices": int(sub.is_positive.sum()),
                       "negative_slices": int((~sub.is_positive).sum()),
                       "boxes": len(annotations)})
    base.joinpath("dataset.yaml").write_text(
        f"path: {base}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: liver_lesion\n")
    _csv(pd.DataFrame(counts), base / "counts.csv")
    all_counts = pd.DataFrame(counts)
    if (all_counts.images == 0).any() or (all_counts.boxes == 0).any():
        raise ValueError(f"Each split needs slices and annotated lesions, see {base}/counts.csv")
    print(f"Exported {base}:\n{all_counts.to_string(index=False)}", flush=True)
    return base


def make_qc_preview(data_dir: str, *, split: str = "train", count: int = 24,
                    seed: int = 42) -> Path:
    data = Path(data_dir)
    coco = json.loads((data / "annotations" / f"{split}.json").read_text())
    gt_by_image = {}
    for a in coco["annotations"]:
        gt_by_image.setdefault(a["image_id"], []).append(a["bbox"])
    positives = [im for im in coco["images"] if im["id"] in gt_by_image]
    negatives = [im for im in coco["images"] if im["id"] not in gt_by_image]
    rng = np.random.default_rng(seed)
    chosen = []
    for pool, n in ((positives, count // 2), (negatives, count - count // 2)):
        ids = rng.choice(len(pool), size=min(n, len(pool)), replace=False) if pool else []
        chosen.extend(pool[int(i)] for i in ids)
    if not chosen:
        raise ValueError("No QC images")
    thumbs = []
    for imeta in chosen:
        with Image.open(data / "images" / split / imeta["file_name"]) as raw:
            tile = raw.convert("RGB")
        draw = ImageDraw.Draw(tile)
        for x, y, w, h in gt_by_image.get(imeta["id"], []):
            draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=3)
        draw.text((4, 4), imeta["sample_id"], fill=(255, 255, 0))
        tile.thumbnail((256, 256))
        thumbs.append(tile)
    cols = 4
    canvas = Image.new("RGB", (cols * 256, ((len(thumbs)+cols-1)//cols)*256), "black")
    for n, tile in enumerate(thumbs):
        canvas.paste(tile, ((n%cols)*256, (n//cols)*256))
    path = data / f"qc_{split}.jpg"
    canvas.save(path, quality=90)
    return path
