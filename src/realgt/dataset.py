"""Manual-annotation-only 3D NIfTI -> full-slice 2D COCO/YOLO datasets.

No teacher predictions, no liver-mask clipping and no component-area filtering.
The only accepted labels are <phase>_tumor_mask_registered.nii.gz.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
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


def scan_cases(root: str, dest: Path, *, allowed_patients: Optional[str] = None) -> pd.DataFrame:
    """Require paired manual masks, and optionally enforce an explicit allowed cohort."""
    source = Path(root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    allowed = set(_read_fixed_val_list(allowed_patients)) if allowed_patients else None
    records = []
    incomplete = []
    cohort_status = {}
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        if allowed is not None and folder.name not in allowed:
            continue
        paths = {
            p: (folder / f"{p}_registered.nii.gz",
                folder / f"{p}_tumor_mask_registered.nii.gz")
            for p in PHASES
        }
        labeled = [p for p, (_, label) in paths.items() if label.is_file()]
        if not labeled:
            if allowed is not None:
                cohort_status[folder.name] = "missing_manual_tumor_masks"
            continue  # Non-manual cases are excluded, even if pseudo masks exist.
        if len(labeled) != len(PHASES) or any(not im.is_file() for im, _ in paths.values()):
            incomplete.append(folder.name)
            if allowed is not None:
                cohort_status[folder.name] = "incomplete_ap_pvp_manual_gt_or_images"
            continue
        if allowed is not None:
            cohort_status[folder.name] = "eligible_manual_gt"
        for phase, (image_path, mask_path) in paths.items():
            records.append({
                "patient_id": folder.name, "phase": phase,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()), "label_source": "manual",
            })
    # Incomplete real-GT patients cannot enter any split. Record exclusions
    # explicitly, rather than allowing one incomplete patient to block all
    # other eligible cases or treating its missing mask as a true negative.
    excluded_path = dest.parent / "excluded_incomplete_manual_cases.csv"
    _csv(pd.DataFrame({"patient_id": incomplete,
                       "reason": ["incomplete AP/PVP manual tumor masks or images"] * len(incomplete)},
                      columns=["patient_id", "reason"]), excluded_path)
    if incomplete:
        print(f"Excluded {len(incomplete)} incomplete real-GT patients; audit: {excluded_path}", flush=True)
    if allowed is not None:
        audit = pd.DataFrame([
            {"patient_id": pid, "status": cohort_status.get(pid, "missing_patient_directory")}
            for pid in sorted(allowed)
        ])
        _csv(audit, dest.parent / "all_cohort_audit.csv")
        print(f"Allowed real-GT cohort={len(allowed)}; "
              f"eligible={int((audit.status == 'eligible_manual_gt').sum())}; "
              f"ineligible={int((audit.status != 'eligible_manual_gt').sum())}; "
              f"audit={dest.parent / 'all_cohort_audit.csv'}", flush=True)
    if not records:
        raise ValueError(f"No *tumor_mask_registered.nii.gz manual AP/PVP pairs found under {source}")
    df = pd.DataFrame(records)
    if df.duplicated(["patient_id", "phase"]).any():
        raise ValueError("Duplicate case/phase")
    _csv(df, dest)
    print(f"Manual GT cases: {df.patient_id.nunique()} patients / {len(df)} phase records")
    return df


def _read_fixed_val_list(filename: str) -> list[str]:
    """Read private text or CSV file: one exact patient_id per line."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed Val list not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "patient_id" not in df.columns:
            raise ValueError("Fixed Val CSV must contain a patient_id column")
        values = df["patient_id"].tolist()
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
        if values and values[0] == "patient_id":
            values = values[1:]
    values = [str(v).strip() for v in values]
    if not values or any(not value for value in values):
        raise ValueError("Fixed Val patient list is empty or has blank IDs")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate patient IDs in fixed Val patient list")
    return values


def build_splits(cases: pd.DataFrame, dest: Path, *, seed: int = 42,
                 train_fraction: float = .70, val_fraction: float = .15,
                 external_csv: Optional[str] = None,
                 fixed_val_patients: Optional[str] = None,
                 priority_train_patients: Optional[str] = None) -> pd.DataFrame:
    patients = sorted(cases.patient_id.astype(str).unique().tolist())
    n = len(patients)
    if n < 3:
        raise ValueError("At least 3 manually labeled patients needed for Train/Val/Test")
    if (fixed_val_patients or priority_train_patients) and external_csv:
        raise ValueError("Cannot combine --splits-file with fixed Val or priority Train lists: "
                         "external splits must be fully specified")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ValueError("Invalid train/val fractions")
    if priority_train_patients:
        requested_train = _read_fixed_val_list(priority_train_patients)
        train_set = set(requested_train)
        requested_val = _read_fixed_val_list(fixed_val_patients) if fixed_val_patients else []
        if train_set & set(requested_val):
            raise ValueError("Patient appears in both priority Train and fixed Val lists: " +
                             ", ".join(sorted(train_set & set(requested_val))[:10]))
        eligible = set(patients)
        train_ids = train_set & eligible
        val_ids = set(requested_val) & eligible
        rng = np.random.default_rng(seed)
        available = np.array(sorted(eligible - train_ids - val_ids), dtype=str)
        rng.shuffle(available)
        # When a fixed Val list is provided, fill invalid entries with
        # other manual-GT patients, but NEVER take candidates from Train.
        desired_val = (len(requested_val) if fixed_val_patients else
                       max(1, round(n * val_fraction)))
        n_fill = max(0, desired_val - len(val_ids))
        if len(available) < n_fill + 1 or not train_ids:
            raise ValueError(
                f"Insufficient eligible cases for priority Train ({len(train_ids)}), "
                f"target Val ({desired_val}) and nonempty Test. "
                "Check all_cohort_audit.csv and manual GT coverage."
            )
        added_val = [str(p) for p in available[:n_fill]]
        val_ids.update(added_val)
        test_ids = set(str(p) for p in available[n_fill:])
        if not val_ids or not test_ids:
            raise ValueError("Train/Val/Test must all be nonempty")
        split = pd.DataFrame({
            "patient_id": patients,
            "split": ["train" if pid in train_ids else
                      "val" if pid in val_ids else "test" for pid in patients],
            "split_source": [
                "priority_train" if pid in train_ids else
                "fixed_val" if pid in set(requested_val) else
                "val_replacement" if pid in added_val else
                "remaining_test" for pid in patients
            ]
        })
        status_rows = (
            [{"patient_id": pid, "requested": "train",
              "status": "priority_train_included" if pid in eligible else "priority_train_ineligible"}
             for pid in requested_train] +
            [{"patient_id": pid, "requested": "val",
              "status": "fixed_val_included" if pid in eligible else "fixed_val_ineligible"}
             for pid in requested_val] +
            [{"patient_id": pid, "requested": "val_replacement", "status": "val_replacement"}
             for pid in added_val]
        )
        _csv(pd.DataFrame(status_rows, columns=["patient_id", "requested", "status"]),
             dest.parent / "priority_split_audit.csv")
        if (missing_train := train_set - eligible):
            print("WARNING: priority Train IDs without complete manual GT: "
                  + ", ".join(sorted(missing_train)), flush=True)
        if (missing_val := set(requested_val) - eligible):
            print("WARNING: fixed Val IDs without complete manual GT: "
                  + ", ".join(sorted(missing_val)), flush=True)
        print(f"Priority splitting: train={len(train_ids)}, val={len(val_ids)} "
              f"(replacement={len(added_val)}), test={len(test_ids)}. "
              "No priority Train patient moved to Val/Test.", flush=True)
    elif external_csv:
        sf = pd.read_csv(external_csv, dtype={"patient_id": str}, keep_default_na=False)
        if not {"patient_id", "split"}.issubset(sf.columns):
            raise ValueError("External splits CSV requires patient_id,split")
        if sf.patient_id.duplicated().any() or set(sf.patient_id) != set(patients):
            raise ValueError("External splits must contain every and only eligible GT patient once")
        if not set(sf.split).issubset(set(SPLITS)):
            raise ValueError("Unknown split; use train/val/test")
        split = sf[["patient_id", "split"]].sort_values("patient_id").reset_index(drop=True)
    else:
        rng = np.random.default_rng(seed)
        shuffled = np.array(patients, dtype=str)
        rng.shuffle(shuffled)
        if fixed_val_patients:
            requested = _read_fixed_val_list(fixed_val_patients)
            eligible = set(patients)
            fixed = set(requested) & eligible
            missing = [pid for pid in requested if pid not in eligible]
            # At least as many Val patients as were requested, even if some
            # named patients have no complete manual-GT AP/PVP pair.
            target_val = max(len(requested), round(n * val_fraction))
            # Preserve the requested Test fraction rather than collapsing
            # Test to a handful of patients when fixed Val exceeds 15%.
            target_test = max(1, round(n * (1.0 - train_fraction - val_fraction)))
            if target_val + target_test >= n:
                raise ValueError(
                    f"Not enough eligible real-GT patients (n={n}) for "
                    f"fixed Val target={target_val}, Test target={target_test} "
                    "and at least one Train patient. Provide more complete GT "
                    "cases or adjust split fractions."
                )
            remainder = [pid for pid in shuffled if pid not in fixed]
            additional_count = target_val - len(fixed)
            added = [str(pid) for pid in remainder[:additional_count]]
            val_ids = fixed | set(added)
            candidates = [str(pid) for pid in remainder[additional_count:]]
            test_ids = set(candidates[:target_test])
            split = pd.DataFrame({
                "patient_id": patients,
                "split": ["val" if pid in val_ids else
                          "test" if pid in test_ids else "train" for pid in patients],
                "split_source": ["requested_val" if pid in fixed else
                                 "val_replacement" if pid in added else
                                 "heldout_test" if pid in test_ids else
                                 "random_train" for pid in patients],
            })
            audit = pd.DataFrame(
                [{"patient_id": pid, "status": "requested_val_included"
                  if pid in fixed else "requested_val_ineligible"}
                 for pid in requested] +
                [{"patient_id": pid, "status": "val_replacement"} for pid in added],
                columns=["patient_id", "status"],
            )
            audit_path = dest.parent / "fixed_val_audit.csv"
            _csv(audit, audit_path)
            print(f"Fixed-Val policy: requested={len(requested)}, "
                  f"eligible_fixed={len(fixed)}, ineligible={len(missing)}, "
                  f"replacements={len(added)}, final_val={len(val_ids)}, "
                  f"test={len(test_ids)}, train={n-len(val_ids)-len(test_ids)}; "
                  f"audit={audit_path}", flush=True)
            if missing:
                print("WARNING: Ineligible requested Val patients (missing complete "
                      "AP/PVP manual GT): " + ", ".join(missing), flush=True)
        else:
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


def _materialize_png(src: Path, dest: Path) -> str:
    """Export an image without requiring symlink permissions on /dataset.

    Prefer a zero-copy symlink, then a same-filesystem hardlink, and finally
    copy2 on restricted/mounted filesystems. A retry reuses existing files.
    """
    if dest.is_symlink() and not dest.exists():
        dest.unlink()  # Broken link from an interrupted older export.
    if dest.is_file():
        if dest.stat().st_size != src.stat().st_size:
            raise ValueError(f"Existing export image differs from source: {dest}")
        return "existing"
    if dest.exists():
        raise ValueError(f"Image export destination is not a file: {dest}")
    for operation in (lambda: dest.symlink_to(src),
                      lambda: os.link(src, dest)):
        try:
            operation()
            return "linked"
        except OSError:
            # Some containers forbid symlinks or hardlinks on /dataset.
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
    try:
        shutil.copy2(src, dest)
    except Exception:
        if dest.is_file():
            dest.unlink()
        raise
    return "copied"


def export_phase(frame: pd.DataFrame, out: Path, phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(phase)
    if not frame.label_source.eq("manual").all():
        raise ValueError("Manual-GT project refuses non-manual labels in exports")
    if (frame.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Patient split leakage in slices manifest")
    base = Path(out).resolve() / "exports" / f"real_gt_{phase}"
    counts = []
    materialized = {"linked": 0, "copied": 0, "existing": 0}
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
            materialized[_materialize_png(src, dest)] += 1
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
    print(f"Exported {base}:\n{all_counts.to_string(index=False)}\n"
          f"Images: {materialized['linked']} linked, "
          f"{materialized['copied']} copied, "
          f"{materialized['existing']} reused.", flush=True)
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
