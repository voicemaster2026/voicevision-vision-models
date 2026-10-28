#!/usr/bin/env python3
"""CPU fine-tune the bucket-pinned Depth Anything V2 metric model on rebar Z maps.

The loader intentionally uses left-camera RGB with the matching raw-grid gt_depth
maps. Training/validation are held-out camera-ring groups from lattice and pile;
the complete synthetic pier scene remains an untouched test group. The raw
physical COLMAP sequence is excluded because its undistorted RGB files are absent
from the pinned snapshot and raw captures do not align to its depth maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


SOURCE_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
DATASET_REPO = "sal0-h/industrial-rebar-metric-depth"
DATASET_REVISION = "f3e5145a34e68a37cb5c5027367fa3db12db66f1"
MODEL_REPO = "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small"
MODEL_REVISION = "3bc65d4e14a6786a61acec16453c50e12bf5f338"
MODEL_SHA256 = "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545"
SCENES = ("lattice_zed_504", "pile_zed_504")
TEST_SCENE = "pier_504"
VALIDATION_ELEVATION_GROUPS = {25.0, 60.0}
POSE_BOUNDARY_GUARD_FRAMES = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset_files(root: Path, manifest: dict) -> dict:
    """Verify every selected local file before training or recording provenance."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Dataset root is not a directory: {root}")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != manifest.get("file_count"):
        raise RuntimeError("Dataset manifest file list/count is incomplete")
    total_bytes = 0
    for record in records:
        relative_text = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative_text, str) or "\\" in relative_text:
            raise RuntimeError("Malformed path in dataset integrity manifest")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError(f"Unsafe dataset manifest path: {relative_text}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
            raise RuntimeError(f"Missing, unsafe or non-regular dataset file: {relative_text}")
        expected_size, expected_hash = record.get("size_bytes"), record.get("sha256")
        if type(expected_size) is not int or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(f"Malformed integrity metadata for {relative_text}")
        if path.stat().st_size != expected_size or sha256(path) != expected_hash:
            raise RuntimeError(f"Dataset size/hash mismatch: {relative_text}")
        total_bytes += expected_size
    if total_bytes != manifest.get("total_bytes"):
        raise RuntimeError("Dataset manifest total_bytes does not match selected files")
    return {"status": "verified", "files_checked": len(records), "bytes_checked": total_bytes}


def camera_pose_groups(scene_root: Path) -> list[dict]:
    """Recover contiguous radius/elevation groups from the scene manifest."""
    manifest_path = scene_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("num_frames") != 504 or manifest.get("depth_convention") != "optical_axis_z_meters":
        raise RuntimeError(f"Unexpected camera/depth manifest for {scene_root.name}")
    stems = manifest.get("stems")
    if stems != [f"{index:06d}" for index in range(504)]:
        raise RuntimeError(f"Unexpected frame ordering in {manifest_path}")
    camera_path = manifest.get("camera_path")
    if not isinstance(camera_path, dict) or not isinstance(camera_path.get("shells"), list):
        raise RuntimeError(f"No shell/ring camera path in {manifest_path}")
    groups, offset = [], 0
    for shell_index, shell in enumerate(camera_path["shells"]):
        for ring_index, ring in enumerate(shell.get("rings", [])):
            views = ring.get("views")
            if type(views) is not int or views <= 0:
                raise RuntimeError(f"Invalid ring view count in {manifest_path}")
            groups.append({"shell_index": shell_index, "radius_m": shell["radius_m"],
                           "ring_index": ring_index, "elevation_deg": ring["elevation_deg"],
                           "start": offset, "end": offset + views})
            offset += views
    if offset != 504 or len(groups) != 24:
        raise RuntimeError(f"Expected 24 camera pose groups covering 504 frames in {manifest_path}")
    return groups


def build_records(root: Path, size: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    train, val, test = [], [], []
    split_details = {}
    for scene in SCENES + (TEST_SCENE,):
        base = root / "paper" / scene
        groups = camera_pose_groups(base) if scene != TEST_SCENE else []
        validation_groups = [group for group in groups
                             if group["elevation_deg"] in VALIDATION_ELEVATION_GROUPS]
        if groups and len(validation_groups) != 8:
            raise RuntimeError(f"Expected two validation elevations per camera shell in {scene}")
        val_indices = {index for group in validation_groups for index in range(group["start"], group["end"])}
        guard_indices = set()
        for group in validation_groups:
            guard_indices.update(range(max(0, group["start"] - POSE_BOUNDARY_GUARD_FRAMES), group["start"]))
            guard_indices.update(range(group["end"], min(504, group["end"] + POSE_BOUNDARY_GUARD_FRAMES)))
        scene_counts = {"train": 0, "validation": 0, "guard_excluded": 0, "test": 0}
        for frame in range(504):
            stem = f"{frame:06d}"
            image = base / "left" / f"{stem}.png"
            depth = base / "gt_depth" / f"{stem}_depth.npy"
            mask = base / "gt_depth" / f"{stem}_mask.npy"
            if not image.is_file() or not depth.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing aligned synthetic RGB/depth/mask for {scene}/{stem}")
            target = {"image": image, "depth": depth, "mask": mask, "scene": scene, "stem": stem}
            if scene == TEST_SCENE:
                test.append(target)
                scene_counts["test"] += 1
            elif frame in val_indices:
                val.append(target)
                scene_counts["validation"] += 1
            elif frame in guard_indices:
                scene_counts["guard_excluded"] += 1
            else:
                train.append(target)
                scene_counts["train"] += 1
        if groups:
            split_details[scene] = {**scene_counts,
                                    "validation_groups": [{"shell_index": g["shell_index"],
                                                           "radius_m": g["radius_m"],
                                                           "ring_index": g["ring_index"],
                                                           "elevation_deg": g["elevation_deg"],
                                                           "frame_indices": f"{g['start']}..{g['end'] - 1}"}
                                                          for g in validation_groups],
                                    "guard_frames_each_side": POSE_BOUNDARY_GUARD_FRAMES}

    if len(train) != 608 or len(val) != 336 or len(test) != 504:
        raise RuntimeError(f"Unexpected split sizes: train={len(train)}, val={len(val)}, "
                           f"test={len(test)}")
    return train, val, test, split_details


class RebarDepth(Dataset):
    def __init__(self, records: list[dict], input_size: int, train: bool = False):
        self.records = records
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        bgr = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode image {row['image']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.load(row["depth"], allow_pickle=False).astype(np.float32, copy=False)
        if depth.ndim != 2 or depth.shape != image.shape[:2]:
            raise RuntimeError(f"Image/depth grid mismatch for {row['scene']}/{row['stem']}: "
                               f"{image.shape[:2]} vs {depth.shape}")
        if not np.isfinite(depth).all():
            raise RuntimeError(f"Non-finite metric depth values in {row['depth']}")
        if row["mask"] is None:
            mask = (depth > 0).astype(np.uint8)
        else:
            mask = np.load(row["mask"], allow_pickle=False).astype(np.uint8, copy=False)
        if mask.shape != depth.shape:
            raise RuntimeError(f"Depth/mask grid mismatch in {row['scene']}/{row['stem']}")
        if not np.isin(mask, (0, 1)).all():
            raise RuntimeError(f"Mask is not boolean/binary in {row['scene']}/{row['stem']}")
        if np.any(depth[mask.astype(bool)] < 0):
            raise RuntimeError(f"Negative depth inside the valid mask in {row['depth']}")

        # Mirror the official Depth Anything resize: preserve aspect and make
        # both dimensions multiples of the ViT patch size (14 px).
        src_h, src_w = depth.shape
        target_h = max(self.input_size, int(round(self.input_size / 14) * 14))
        scale = target_h / src_h
        target_w = max(14, int(round((src_w * scale) / 14) * 14))
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)

        if self.train and random.random() < 0.5:
            image = image[:, ::-1].copy()
            depth = depth[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = np.ascontiguousarray(((image - mean) / std).transpose(2, 0, 1), dtype=np.float32)
        valid = mask & np.isfinite(depth) & (depth >= 0.001) & (depth <= 20.0)
        if valid.sum() < 16:
            raise RuntimeError(f"Too few supervised rebar pixels for {row['scene']}/{row['stem']}")
        return (torch.from_numpy(image), torch.from_numpy(depth.copy()),
                torch.from_numpy(valid.copy()), row["scene"], row["stem"])


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    pred = pred[valid].clamp(0.001, 20.0).float()
    target = target[valid].float()
    ratio = torch.maximum(target / pred, pred / target)
    dlog = torch.log(pred) - torch.log(target)
    silog_variance = (torch.square(dlog).mean() - 0.5 * torch.square(dlog.mean())).clamp_min(0)
    return {
        "pixels": int(target.numel()),
        "abs_rel": float((torch.abs(pred - target) / target).mean()),
        "rmse_m": float(torch.sqrt(torch.square(pred - target).mean())),
        "silog": float(torch.sqrt(silog_variance)),
        "delta1": float((ratio < 1.25).float().mean()),
        "delta2": float((ratio < 1.25 ** 2).float().mean()),
        "delta3": float((ratio < 1.25 ** 3).float().mean()),
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    values = []
    with torch.inference_mode():
        for images, depths, valid, _scene, _stem in loader:
            images, depths, valid = images.to(device), depths.to(device), valid.to(device)
            prediction = model(images)
            for i in range(images.shape[0]):
                values.append(compute_metrics(prediction[i], depths[i], valid[i]))
    keys = ["abs_rel", "rmse_m", "silog", "delta1", "delta2", "delta3"]
    return {"images": len(values), "valid_pixels": sum(row["pixels"] for row in values),
            "macro": {key: float(np.mean([row[key] for row in values])) for key in keys}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="pinned upstream Depth-Anything-V2 repository checkout")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if (args.input_size < 224 or args.input_size % 14 or args.batch_size < 1 or args.epochs < 1
            or args.threads < 1 or args.threads > max(1, os.cpu_count() or 1)):
        raise ValueError("input-size must be >=224 and divisible by 14; batch-size/threads/epochs must be positive and threads must not exceed available CPUs")
    if torch.cuda.is_available():
        print("CUDA is available but this training run is explicitly CPU-only.")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    source_root = args.source_dir.resolve()
    metric_root = source_root / "metric_depth"
    if not (metric_root / "depth_anything_v2/dpt.py").is_file():
        raise FileNotFoundError(f"Not an official Depth Anything V2 checkout: {source_root}")
    source_commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected official source commit {SOURCE_COMMIT}, got {source_commit}")
    for diff_args in (["diff", "--quiet", "HEAD", "--"], ["diff", "--cached", "--quiet", "HEAD", "--"]):
        clean = subprocess.run(["git", "-C", str(source_root), *diff_args], check=False)
        if clean.returncode != 0:
            raise RuntimeError("Pinned Depth Anything V2 checkout has tracked modifications")
    tracked_python = set(subprocess.check_output(
        ["git", "-C", str(source_root), "ls-files", "--", "metric_depth"], text=True).splitlines())
    active_python = {path.relative_to(source_root).as_posix()
                     for path in (source_root / "metric_depth").rglob("*.py")}
    if active_python - tracked_python:
        raise RuntimeError(f"Pinned source contains untracked Python files: {sorted(active_python - tracked_python)}")
    untracked = subprocess.check_output(
        ["git", "-C", str(source_root), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    if untracked:
        raise RuntimeError(f"Pinned source checkout contains untracked files: {untracked[:10]}")
    if sha256(args.checkpoint) != MODEL_SHA256:
        raise RuntimeError("Bucket checkpoint SHA-256 differs from its pinned model manifest")
    dataset_manifest_path = args.dataset_root / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("repo_id") != DATASET_REPO or dataset_manifest.get("revision") != DATASET_REVISION:
        raise RuntimeError("Dataset path does not contain the pinned metric-depth dataset revision")
    if dataset_manifest.get("license", "").casefold() not in {"cc-by-4.0", "cc_by_4_0"}:
        raise RuntimeError("Dataset license is not the reviewed CC BY 4.0 declaration")
    if dataset_manifest.get("training_eligible") is not True:
        raise RuntimeError("Dataset manifest has not passed local training review/authorization")
    dataset_integrity = verify_dataset_files(args.dataset_root, dataset_manifest)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(metric_root))
    from depth_anything_v2.dpt import DepthAnythingV2
    from util.loss import SiLogLoss

    configs = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": 20.0}
    model = DepthAnythingV2(**configs)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    criterion = SiLogLoss(lambd=0.5)

    train_records, val_records, test_records, split_details = build_records(
        args.dataset_root, args.input_size)
    train_loader = DataLoader(RebarDepth(train_records, args.input_size, train=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(RebarDepth(val_records, args.input_size), batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(RebarDepth(test_records, args.input_size), batch_size=1, shuffle=False, num_workers=0)

    # Record a pre-fine-tuning baseline on validation only. Test sequences stay
    # untouched until the validation-selected checkpoint has been frozen.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "pretrained_baseline.pt"
    torch.save(model.state_dict(), baseline_path)
    baseline_validation = evaluate(model, val_loader, device)
    print(json.dumps({"pretrained_baseline_validation": baseline_validation}, ensure_ascii=False), flush=True)

    groups = [
        {"params": [p for n, p in model.named_parameters() if n.startswith("pretrained.")], "lr": 5e-6},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("pretrained.")], "lr": 5e-5},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)
    best_rel, best_epoch, best_path, stale = float("inf"), 0, args.output_dir / "model.pt", 0
    history = []
    start_time = time.time()
    total_steps = args.epochs * len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for step, (images, depths, valid, _scene, _stem) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + step
            fraction = global_step / max(1, total_steps - 1)
            lr_factor = (1.0 - fraction) ** 0.9
            optimizer.param_groups[0]["lr"] = 5e-6 * lr_factor
            optimizer.param_groups[1]["lr"] = 5e-5 * lr_factor
            images, depths, valid = images.to(device), depths.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(images)
            # DPT's sigmoid head already bounds predictions to (0, max_depth);
            # avoid a lower clamp here because it would zero gradients below 1 mm.
            loss = criterion(pred, depths, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, step {step + 1}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())

        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_silog_loss": loss_sum / max(1, len(train_loader)),
               "validation": validation, "elapsed_seconds": time.time() - start_time}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = validation["macro"]["abs_rel"]
        if score < best_rel:
            best_rel, best_epoch, stale = score, epoch + 1, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
    # Test groups are touched only after checkpoint selection has finished.
    selected_test = evaluate(model, test_loader, device)
    model.load_state_dict(torch.load(baseline_path, map_location="cpu", weights_only=True), strict=True)
    baseline_test = evaluate(model, test_loader, device)
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
    payload = {
        "schema": "voicevision.metric-depth-candidate/1", "model_id": "depth-anything-v2-rebar-metric-small",
        "candidate_only": True, "device": "cpu", "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_source_url": f"https://huggingface.co/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}",
        "dataset_integrity": dataset_integrity,
        "dataset_manifest_sha256": sha256(dataset_manifest_path), "dataset_license": dataset_manifest["license"],
        "dataset_attribution": {"authors": ["Salman Hajizada", "Diram Tabaa", "Gianni A. Di Caro"],
                                "title": "Benchmarking Metric Depth for Construction Perception: Industrial Rebar Scenes with Exact Synthetic Ground Truth",
                                "venue": "IEEE/RSJ IROS Workshop on the Future of Construction", "year": 2026},
        "dataset_evaluation_set_note": "The selected source scenes are the upstream paper evaluation suite; this run uses its own grouped train/validation/test split and does not reproduce the published benchmark protocol.",
        "pretrained_model_repo": MODEL_REPO, "pretrained_model_revision": MODEL_REVISION,
        "pretrained_checkpoint_sha256": MODEL_SHA256, "implementation_repo": "https://github.com/DepthAnything/Depth-Anything-V2",
        "implementation_commit": source_commit, "implementation_license": "Apache-2.0",
        "implementation_tree_clean": True, "implementation_tracked_python_files": len(active_python),
        "runner_script_sha256": sha256(Path(__file__)),
        "training_command": {"entrypoint": Path(__file__).name, "device": "cpu",
                             "input_size": args.input_size, "batch_size": args.batch_size,
                             "epochs_max": args.epochs, "patience": args.patience,
                             "threads": args.threads, "seed": args.seed},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor(), "torch": torch.__version__,
                    "numpy": np.__version__, "opencv": cv2.__version__,
                    "torchvision": torchvision.__version__,
                    "torch_threads": torch.get_num_threads()},
        "architecture": configs, "input_size_min_dimension": args.input_size,
        "input_resolution_for_1280x720": [args.input_size, int(round((1280 / 720 * args.input_size) / 14) * 14)],
        "supervision": "metric optical-axis Z in meters, rebar interior mask only",
        "split": {"training_frames": len(train_records), "validation_frames": len(val_records),
                  "training_and_validation_by_scene": split_details,
                  "split_method": "hold out 25 and 60 degree camera rings per radius shell, guard two adjacent frames on each ring boundary",
                  "synthetic_test": {"scene": TEST_SCENE, "frames": 504},
                  "physical_pseudoreference": {"scene": "physical_zed_1001", "frames": 0,
                                               "status": "not_evaluated_missing_aligned_rgb",
                                               "reason": "Pinned snapshot has 1001 raw 1280x720 RGB captures but no undistorted 1530x753 RGB counterparts in gt_depth; raw captures cannot be paired with the pseudo-reference arrays."}},
        "training": {"epochs_max": args.epochs, "best_epoch": best_epoch, "batch_size": args.batch_size,
                     "threads": args.threads, "optimizer": "AdamW", "backbone_lr": 5e-6,
                     "head_lr": 5e-5, "loss": "official SiLogLoss, lambda=0.5",
                     "checkpoint_selection": "lowest validation macro abs_rel; never selected on tests",
                     "history": history},
        "pretrained_baseline_validation": baseline_validation,
        "pretrained_baseline_synthetic_test": baseline_test,
        "held_out_synthetic_test": selected_test,
        "physical_colmap_pseudo_reference": {"status": "not_evaluated",
                                              "reason": "No aligned undistorted RGB files are present in the pinned snapshot; the raw RGB grid differs from the 1530x753 pseudo-reference grid."},
        "test_evaluated_after_selection_only": True,
        "checkpoint_sha256": sha256(best_path), "checkpoint_size_bytes": best_path.stat().st_size,
        "limitations": ["Trains/evaluates metric distance on masked rebar interiors only.",
                        "Domain-specific to three synthetic rebar scenes; not general industrial metrology.",
                        "Physical sequence uses COLMAP pseudo-reference, not surveyed ground truth.",
                        "No camera calibration or external scale is used at inference; no object-size/OK-NG claim.",
                        "Candidate only; no VVP runtime integration or promotion."],
    }
    (args.output_dir / "artifact.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "pretrained_baseline_synthetic_test": baseline_test,
                      "synthetic_test": selected_test,
                      "physical_pseudo_reference": payload["physical_colmap_pseudo_reference"],
                      "checkpoint_sha256": payload["checkpoint_sha256"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
