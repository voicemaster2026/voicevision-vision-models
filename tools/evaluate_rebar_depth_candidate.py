#!/usr/bin/env python3
"""Verify and evaluate a pinned rebar metric-depth candidate.

The evaluator reuses the exact data splitting, loader, integrity checks, and
metrics from the pinned training entrypoint. It evaluates the pretrained model
and candidate on validation, then evaluates the selected candidate once on the
untouched synthetic test scene.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch.utils.data import DataLoader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("voicevision_rebar_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="clean checkout of the pinned upstream source revision")
    parser.add_argument("--trainer-script", required=True, type=Path,
                        help="the matching pinned training entrypoint")
    parser.add_argument("--pretrained-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--baseline-test-only", action="store_true",
                        help="add pretrained-model test metrics to an existing evaluation JSON")
    args = parser.parse_args()
    if args.input_size < 224 or args.input_size % 14 or args.threads < 1:
        raise ValueError("input-size must be >=224 and divisible by 14; threads must be positive")
    torch.set_num_threads(args.threads)

    trainer_path = args.trainer_script.resolve(strict=True)
    trainer = load_trainer(trainer_path)
    source_root = args.source_dir.resolve(strict=True)
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != trainer.SOURCE_COMMIT:
        raise RuntimeError(f"Expected source {trainer.SOURCE_COMMIT}, found {source_commit}")
    for diff_args in (["diff", "--quiet", "HEAD", "--"],
                      ["diff", "--cached", "--quiet", "HEAD", "--"]):
        if subprocess.run(["git", "-C", str(source_root), *diff_args], check=False).returncode:
            raise RuntimeError("Pinned upstream source checkout has tracked modifications")
    metric_root = source_root / "metric_depth"
    sys.path.insert(0, str(metric_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    dataset_root = args.dataset_root.resolve(strict=True)
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != trainer.DATASET_REPO or manifest.get("revision") != trainer.DATASET_REVISION:
        raise RuntimeError("Dataset manifest does not match the pinned repository revision")
    integrity = trainer.verify_dataset_files(dataset_root, manifest)

    train_rows, val_rows, test_rows, split_details = trainer.build_records(dataset_root, args.input_size)
    val_loader = DataLoader(trainer.RebarDepth(val_rows, args.input_size),
                            batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(trainer.RebarDepth(test_rows, args.input_size),
                             batch_size=1, shuffle=False, num_workers=0)
    config = {"encoder": "vits", "features": 64,
              "out_channels": [48, 96, 192, 384], "max_depth": 20.0}
    device = torch.device("cpu")

    if sha256(args.pretrained_checkpoint) != trainer.MODEL_SHA256:
        raise RuntimeError("Pretrained checkpoint SHA-256 does not match the pinned model")
    model = DepthAnythingV2(**config).to(device)
    pretrained = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(pretrained, strict=True)
    if args.baseline_test_only:
        if not args.output.is_file():
            raise FileNotFoundError("--baseline-test-only requires an existing evaluation JSON in --output")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("candidate_checkpoint_sha256") != sha256(args.candidate_checkpoint):
            raise RuntimeError("Existing evaluation JSON belongs to a different candidate checkpoint")
        if result.get("pretrained_checkpoint_sha256") != sha256(args.pretrained_checkpoint):
            raise RuntimeError("Existing evaluation JSON belongs to a different pretrained checkpoint")
        baseline_test = trainer.evaluate(model, test_loader, device)
        result["pretrained_baseline_held_out_synthetic_test"] = baseline_test
        fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.output)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        print(json.dumps({"pretrained_baseline_test": baseline_test}, ensure_ascii=False), flush=True)
        return 0
    baseline_validation = trainer.evaluate(model, val_loader, device)

    candidate_state = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(candidate_state, strict=True)
    candidate_validation = trainer.evaluate(model, val_loader, device)
    # The selected test scene is evaluated only after both checkpoints have
    # been compared on validation.
    candidate_test = trainer.evaluate(model, test_loader, device)

    result = {
        "schema": "voicevision.metric-depth-evaluation/1",
        "model_id": "depth-anything-v2-rebar-metric-small",
        "candidate_only": True,
        "device": "cpu",
        "dataset_repo": trainer.DATASET_REPO,
        "dataset_revision": trainer.DATASET_REVISION,
        "dataset_license": manifest.get("license"),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_integrity": integrity,
        "source_commit": source_commit,
        "training_script_sha256": sha256(trainer_path),
        "pretrained_model_repo": trainer.MODEL_REPO,
        "pretrained_model_revision": trainer.MODEL_REVISION,
        "pretrained_checkpoint_sha256": sha256(args.pretrained_checkpoint),
        "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
        "candidate_checkpoint_size_bytes": args.candidate_checkpoint.stat().st_size,
        "architecture": config,
        "input_size_min_dimension": args.input_size,
        "split": {"training_frames": len(train_rows), "validation_frames": len(val_rows),
                  "training_and_validation_by_scene": split_details,
                  "synthetic_test": {"scene": trainer.TEST_SCENE, "frames": len(test_rows)}},
        "pretrained_baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "candidate_held_out_synthetic_test": candidate_test,
        "limitations": [
            "Metric optical-axis depth is evaluated only on masked rebar interiors.",
            "The held-out test is a synthetic scene, not surveyed physical ground truth.",
            "The physical COLMAP sequence has no aligned undistorted RGB in the pinned snapshot.",
            "No production metrology, object-size, or OK/NG claim is established.",
            "Candidate only; no VVP runtime integration or promotion.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(json.dumps({"baseline_validation": baseline_validation,
                      "candidate_validation": candidate_validation,
                      "candidate_test": candidate_test,
                      "candidate_checkpoint_sha256": result["candidate_checkpoint_sha256"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
