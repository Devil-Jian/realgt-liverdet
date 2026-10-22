"""Tiny synthetic NIfTI regression tests; no private clinical data required."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from realgt.cli import main, parser
from realgt.dataset import build_splits, prepare, scan_cases, export_phase, _bboxes


def _nii(path, array, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine if affine is not None else np.eye(4)), str(path))


def _dataset(tmp_path, n=6):
    root = tmp_path / "source"
    for i in range(n):
        for phase in ("ap", "pvp"):
            x = np.zeros((16, 16, 3), dtype=np.float32)
            mask = np.zeros_like(x)
            mask[4, 7, 1] = 1  # one-pixel manually annotated lesion must survive
            _nii(root / f"P{i:03d}" / f"{phase}_registered.nii.gz", x)
            _nii(root / f"P{i:03d}" / f"{phase}_tumor_mask_registered.nii.gz", mask)
    # Pseudo-mask-only patient should never be included.
    _nii(root / "P999" / "ap_registered.nii.gz", np.ones((16, 16, 3)))
    _nii(root / "P999" / "pseudo_mask.nii.gz", np.ones((16, 16, 3)))
    return root


def test_manual_only_build_and_all_slices(tmp_path):
    root = _dataset(tmp_path)
    out = tmp_path / "results"
    main(["build", "--root", str(root), "--out", str(out)])
    cases = pd.read_csv(out / "manifests" / "cases.csv")
    splits = pd.read_csv(out / "manifests" / "splits.csv")
    slices = pd.read_csv(out / "manifests" / "slices.csv")
    assert cases.patient_id.nunique() == 6
    assert set(cases.patient_id) == set(splits.patient_id)
    assert "P999" not in set(cases.patient_id)
    assert set(cases.label_source) == {"manual"}
    assert len(slices) == 6*2*3
    assert all(slices.groupby("patient_id").split.nunique() == 1)
    assert set(slices.phase) == {"ap", "pvp"}
    assert slices.is_positive.sum() == 6*2
    assert (~slices.is_positive).sum() == 6*2*2
    assert json.loads(slices.iloc[1].boxes) == [[4, 7, 5, 8]]
    assert json.loads((out / "manifests" / "preprocess.json").read_text())["hu_min"] == -160.0
    for phase in ("ap", "pvp"):
        base = out / "exports" / f"real_gt_{phase}"
        for split in ("train", "val", "test"):
            coco = json.loads((base / "annotations" / f"{split}.json").read_text())
            assert len(coco["annotations"]) == int((splits.split == split).sum())
            assert len(coco["images"]) == 3 * int((splits.split == split).sum())
        assert (base / "dataset.yaml").exists()
        assert (base / "counts.csv").exists()
    selected = slices.iloc[0]
    with Image.open(selected.real_path) as im:
        assert im.mode == "L" and im.size == (16, 16)
        assert np.unique(np.asarray(im)).tolist() == [102]
    index = json.loads((out / "manifests" / "train_slices.json").read_text())
    assert index and all(rec["label_source"] == "manual" for rec in index)
    assert all(type(rec["slice_index"]) is int for rec in index)


def test_bbox_8_connectivity_keeps_one_pixel():
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[1, 2] = mask[2, 3] = 1
    mask[6, 7] = 1
    assert _bboxes(mask) == [[2, 1, 4, 3], [7, 6, 8, 7]]


def test_incomplete_manual_phase_audited_and_excluded(tmp_path):
    root = _dataset(tmp_path)
    _nii(root / "INCOMPLETE" / "ap_registered.nii.gz", np.zeros((5,5,5)))
    _nii(root / "INCOMPLETE" / "ap_tumor_mask_registered.nii.gz", np.zeros((5,5,5)))
    cases = scan_cases(str(root), tmp_path / "manifests" / "cases.csv")
    assert "INCOMPLETE" not in set(cases.patient_id)
    audit = pd.read_csv(tmp_path / "manifests" / "excluded_incomplete_manual_cases.csv")
    assert audit.patient_id.tolist() == ["INCOMPLETE"]


def test_fixed_val_27_and_manual_gt_replacements(tmp_path):
    from realgt.dataset import _read_fixed_val_list
    # Simulate 108 full-GT patients. Three named IDs lack full real-GT
    # and must be replaced with other eligible manual-GT patients.
    ids = [f"G{i:03d}" for i in range(108)]
    cases = pd.DataFrame([
        {"patient_id": pid, "phase": phase, "label_source": "manual"}
        for pid in ids for phase in ("ap", "pvp")
    ])
    requested = [f"G{i:03d}" for i in range(24)] + [
        "INELIGIBLE_1", "INELIGIBLE_2", "INELIGIBLE_3"
    ]
    private_file = tmp_path / "private_val.txt"
    private_file.write_text("\n".join(requested) + "\n")
    assert _read_fixed_val_list(str(private_file)) == requested
    dest = tmp_path / "manifests" / "splits.csv"
    splits = build_splits(
        cases, dest, seed=42, train_fraction=.70, val_fraction=.15,
        fixed_val_patients=str(private_file)
    )
    assert splits.split.value_counts().to_dict() == {
        "train": 65, "val": 27, "test": 16
    }
    assert set(requested[:24]).issubset(
        set(splits.loc[splits.split == "val", "patient_id"])
    )
    assert not (set(requested[:24]) & set(splits.loc[splits.split != "val", "patient_id"]))
    assert (splits.split_source == "requested_val").sum() == 24
    assert (splits.split_source == "val_replacement").sum() == 3
    audit = pd.read_csv(dest.parent / "fixed_val_audit.csv")
    assert (audit.status == "requested_val_ineligible").sum() == 3
    assert (audit.status == "val_replacement").sum() == 3
    assert all(splits.groupby("patient_id").split.nunique() == 1)
    again = build_splits(cases, dest.parent / "repeat.csv", seed=42,
                         train_fraction=.70, val_fraction=.15,
                         fixed_val_patients=str(private_file))
    pd.testing.assert_frame_equal(splits, again)


def test_fixed_val_insufficient_eligible_patients_raises(tmp_path):
    cases = pd.DataFrame([
        {"patient_id": f"P{i}", "phase": phase, "label_source": "manual"}
        for i in range(10) for phase in ("ap", "pvp")
    ])
    val = tmp_path / "fixed_val.csv"
    pd.DataFrame({"patient_id": [f"A{i}" for i in range(9)]}).to_csv(val, index=False)
    with pytest.raises(ValueError, match="Not enough eligible"):
        build_splits(cases, tmp_path / "splits.csv", fixed_val_patients=str(val))


def test_fixed_val_and_full_external_splits_conflict(tmp_path):
    cases = pd.DataFrame([
        {"patient_id": f"P{i}", "phase": phase, "label_source": "manual"}
        for i in range(6) for phase in ("ap", "pvp")
    ])
    val = tmp_path / "fixed.txt"
    val.write_text("P0\n")
    with pytest.raises(ValueError, match="Cannot combine"):
        build_splits(cases, tmp_path / "splits.csv",
                     fixed_val_patients=str(val), external_csv="unused.csv")


def test_external_splits_preserve_test(tmp_path):
    cases = scan_cases(str(_dataset(tmp_path)), tmp_path / "cases.csv")
    external = tmp_path / "fixed.csv"
    pd.DataFrame({"patient_id": ["P000", "P001", "P002", "P003", "P004", "P005"],
                  "split": ["test", "test", "val", "train", "train", "train"]}).to_csv(external,index=False)
    sf = build_splits(cases, tmp_path / "splits.csv", external_csv=str(external))
    assert set(sf.loc[sf.split == "test", "patient_id"]) == {"P000", "P001"}
    cases = cases.copy()
    cases.loc[cases.patient_id == "P000", "label_source"] = "pseudo"
    with pytest.raises(ValueError, match="prohibits pseudo"):
        prepare(cases, sf, tmp_path / "bad")


def test_cli_train_100_epochs_auto_test_once(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    out = tmp_path / "out"
    main(["build", "--root", str(root), "--out", str(out)])
    data = out / "exports" / "real_gt_ap"
    run = tmp_path / "runs" / "ap"
    actions = []
    class FakeYOLO:
        def __init__(self, weight):
            actions.append(("load", weight))
        def train(self, **kw):
            actions.append(("train", kw["epochs"], kw["patience"], kw["val"]))
            (run / "weights").mkdir(parents=True, exist_ok=True)
            (run / "weights" / "best.pt").write_text("weights")
        def predict(self, source, **kw):
            actions.append(("predict", len(source)))
            class Boxes:
                class FakeArray:
                    def __init__(self, value): self.value=value
                    def cpu(self): return self
                    def tolist(self): return self.value
                xyxy = FakeArray([[4., 7., 5., 8.]])
                conf = FakeArray([.9])
                cls = FakeArray([0])
            return [SimpleNamespace(boxes=Boxes()) for _ in source]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    main(["train", "--dataset", str(data), "--out", str(run),
          "--epochs", "100", "--batch", "32", "--patience", "15"])
    assert ("train", 100, 15, True) in actions
    assert actions.index(("train", 100, 15, True)) < next(i for i, x in enumerate(actions) if x[0] == "predict")
    assert (run / "test_metrics.json").is_file()
    assert (run / "test_predictions.json").is_file()
    assert (run / "best_checkpoint.txt").read_text().strip() == str(run / "weights" / "best.pt")
    assert len([a for a in actions if a[0] == "predict"]) == 1


def test_reject_changed_window_same_workdir(tmp_path):
    root = _dataset(tmp_path)
    cases = scan_cases(str(root), tmp_path / "cases.csv")
    split = build_splits(cases, tmp_path / "splits.csv")
    out = tmp_path / "out"
    prepare(cases, split, out)
    with pytest.raises(ValueError, match="different HU"):
        prepare(cases, split, out, hu_min=-100, hu_max=300)


def test_cli_uses_epochs_not_steps():
    args = parser().parse_args(["train", "--dataset", "d", "--out", "o"])
    assert (args.epochs, args.batch, args.imgsz, args.patience) == (100, 32, 640, 15)
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--dataset", "d", "--out", "o", "--steps", "2000"])


def test_export_fallback_to_copy_without_symlink_or_hardlink(tmp_path, monkeypatch):
    """Simulates container/bind mount EPERM on both link variants."""
    import os
    import shutil

    from realgt.dataset import _materialize_png
    source = tmp_path / "source.png"
    dest = tmp_path / "dest.png"
    Image.new("L", (16, 16), 73).save(source)

    def blocked(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(Path, "symlink_to", blocked)
    monkeypatch.setattr(os, "link", blocked)
    assert _materialize_png(source, dest) == "copied"
    assert dest.read_bytes() == source.read_bytes()
    # Re-running after a partially completed export must be idempotent.
    assert _materialize_png(source, dest) == "existing"


def test_cli_export_resume_does_not_reload_nifti(tmp_path, monkeypatch):
    import os
    import shutil

    root = _dataset(tmp_path)
    out = tmp_path / "out"
    main(["build", "--root", str(root), "--out", str(out)])
    shutil.rmtree(out / "exports")

    def blocked(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    # Export-only must use slices.csv and not reprocess expensive NIfTI.
    monkeypatch.setattr("realgt.dataset.nib.load",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("read NIfTI")))
    monkeypatch.setattr(Path, "symlink_to", blocked)
    monkeypatch.setattr(os, "link", blocked)
    main(["export", "--out", str(out)])
    for phase in ("ap", "pvp"):
        base = out / "exports" / f"real_gt_{phase}"
        counts = pd.read_csv(base / "counts.csv")
        assert counts.images.sum() == 6 * 3
        assert counts.boxes.sum() == 6
        assert len(list((base / "images" / "train").glob("*.png"))) > 0
    # Repeated export reuses copies without failure.
    main(["export", "--out", str(out)])


def test_prioritized_train_110_fixed_val_27_remaining_test_40(tmp_path):
    """Private all/train/val lists: train wins before anything is randomized."""
    patients = [f"G{i:03d}" for i in range(177)]
    cases = pd.DataFrame([
        {"patient_id": pid, "phase": phase, "label_source": "manual"}
        for pid in patients for phase in ("ap", "pvp")
    ])
    def write_ids(filename, ids):
        file = tmp_path / filename
        file.write_text("\n".join(ids) + "\n")
        return str(file)
    train = patients[:110]
    val = patients[110:137]
    train_path = write_ids("train.txt", train)
    val_path = write_ids("val.txt", val)
    file = tmp_path / "manifests" / "splits.csv"
    result = build_splits(
        cases, file, seed=42, train_fraction=.70, val_fraction=.15,
        priority_train_patients=train_path, fixed_val_patients=val_path
    )
    assert result.split.value_counts().to_dict() == {"train": 110, "val": 27, "test": 40}
    assert set(result.loc[result.split == "train", "patient_id"]) == set(train)
    assert set(result.loc[result.split == "val", "patient_id"]) == set(val)
    assert not (set(train) & set(result.loc[result.split != "train", "patient_id"]))
    assert all(result.groupby("patient_id").split.nunique() == 1)
    audit = pd.read_csv(file.parent / "priority_split_audit.csv")
    assert (audit.status == "priority_train_included").sum() == 110
    assert (audit.status == "fixed_val_included").sum() == 27
    repeat = build_splits(
        cases, file.parent / "repeat.csv", seed=42,
        priority_train_patients=train_path, fixed_val_patients=val_path
    )
    pd.testing.assert_frame_equal(result, repeat)


def test_priority_train_missing_and_val_missing_never_move_train(tmp_path):
    ids = [f"P{i:03d}" for i in range(20)]
    cases = pd.DataFrame([
        {"patient_id": pid, "phase": phase, "label_source": "manual"}
        for pid in ids for phase in ("ap", "pvp")
    ])
    tr = tmp_path / "train.txt"
    tr.write_text("\n".join(ids[:10] + ["TRAIN_MISSING"]) + "\n")
    val = tmp_path / "val.txt"
    val.write_text("P010\nVAL_MISSING\n")
    result = build_splits(cases, tmp_path / "manifest" / "splits.csv",
                          priority_train_patients=str(tr), fixed_val_patients=str(val))
    assert result.split.value_counts().to_dict() == {"train": 10, "val": 2, "test": 8}
    assert set(result.loc[result.split == "train", "patient_id"]) == set(ids[:10])
    audit = pd.read_csv(tmp_path / "manifest" / "priority_split_audit.csv")
    assert (audit.status == "priority_train_ineligible").sum() == 1
    assert (audit.status == "fixed_val_ineligible").sum() == 1
    assert (audit.status == "val_replacement").sum() == 1


def test_priority_train_val_overlap_is_rejected(tmp_path):
    ids = [f"P{i:03d}" for i in range(8)]
    cases = pd.DataFrame([
        {"patient_id": pid, "phase": phase, "label_source": "manual"}
        for pid in ids for phase in ("ap", "pvp")
    ])
    tr = tmp_path / "train.txt"
    tr.write_text("P000\nP001\n")
    va = tmp_path / "val.txt"
    va.write_text("P001\nP002\n")
    with pytest.raises(ValueError, match="both priority Train and fixed Val"):
        build_splits(cases, tmp_path / "splits.csv",
                     priority_train_patients=str(tr), fixed_val_patients=str(va))


def test_allowed_all_cohort_excludes_unlisted_and_audits_missing(tmp_path):
    root = _dataset(tmp_path, n=6)
    all_ids = tmp_path / "all.txt"
    all_ids.write_text("P000\nP001\nP_MISSING\nP999\n")
    result = scan_cases(
        str(root), tmp_path / "manifest" / "cases.csv",
        allowed_patients=str(all_ids)
    )
    assert set(result.patient_id) == {"P000", "P001"}
    audit = pd.read_csv(tmp_path / "manifest" / "all_cohort_audit.csv")
    status = dict(zip(audit.patient_id, audit.status))
    assert status["P000"] == status["P001"] == "eligible_manual_gt"
    assert status["P_MISSING"] == "missing_patient_directory"
    assert status["P999"] == "missing_manual_tumor_masks"


def test_build_cli_private_all_priority_train_fixed_val(tmp_path):
    root = _dataset(tmp_path, n=9)
    filenames = []
    for name, values in [
        ("all.txt", [f"P{i:03d}" for i in range(9)]),
        ("train.txt", ["P000", "P001", "P002", "P003"]),
        ("val.txt", ["P004", "P005"]),
    ]:
        fn = tmp_path / name
        fn.write_text("\n".join(values) + "\n")
        filenames.append(fn)
    out = tmp_path / "final"
    main([
        "build", "--root", str(root), "--out", str(out),
        "--all-patients", str(filenames[0]),
        "--priority-train-patients", str(filenames[1]),
        "--fixed-val-patients", str(filenames[2]),
        "--seed", "42",
    ])
    splits = pd.read_csv(out / "manifests" / "splits.csv")
    assert splits.split.value_counts().to_dict() == {"train": 4, "test": 3, "val": 2}
    assert set(splits.loc[splits.split == "train", "patient_id"]) == set(
        ["P000", "P001", "P002", "P003"]
    )
    assert (out / "manifests" / "all_cohort_audit.csv").exists()
    assert (out / "manifests" / "priority_split_audit.csv").exists()


def test_cohort_plan_only_no_volume_processing(tmp_path, monkeypatch):
    root = _dataset(tmp_path, n=9)
    def write_ids(name, values):
        p = tmp_path / name
        p.write_text("\n".join(values) + "\n")
        return str(p)
    all_path = write_ids("all.txt", [f"P{i:03d}" for i in range(9)])
    train_path = write_ids("train.txt", [f"P{i:03d}" for i in range(4)])
    val_path = write_ids("val.txt", ["P004", "P005"])
    monkeypatch.setattr("realgt.dataset.nib.load",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("NIfTI should not be read")))
    dest = tmp_path / "plan"
    main(["plan", "--root", str(root), "--out", str(dest),
          "--all-patients", all_path, "--priority-train-patients", train_path,
          "--fixed-val-patients", val_path])
    assert (dest / "manifests" / "splits.csv").is_file()
    assert not (dest / "manifests" / "slices.csv").exists()
    assert not (dest / "data").exists()
    splits = pd.read_csv(dest / "manifests" / "splits.csv")
    assert splits.split.value_counts().to_dict() == {"train": 4, "val": 2, "test": 3}
