#!/usr/bin/env python3
"""Fetch and hash-verify the pinned public RobustAD data and SigLIP2 base."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/robustad-siglip2-anomaly-v1/1.0.0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_asset(manifest_path: Path, key: str) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = data.get(key)
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError(f"Expected one pinned asset in {manifest_path}")
    asset = items[0]
    if not isinstance(asset.get("revision"), str) or len(asset["revision"]) != 40:
        raise ValueError(f"Asset revision is not pinned in {manifest_path}")
    if not isinstance(asset.get("files"), list) or not asset["files"]:
        raise ValueError(f"Asset file list is missing in {manifest_path}")
    return asset


def verify_snapshot(directory: Path, asset: dict) -> tuple[int, int]:
    total_bytes = 0
    count = 0
    for record in asset["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative asset path: {record['path']}")
        path = directory / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Expected a regular downloaded file: {record['path']}")
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Downloaded file size mismatch: {record['path']}")
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Downloaded file SHA-256 mismatch: {record['path']}")
        total_bytes += record["size_bytes"]
        count += 1
    return count, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/RobustAD")
    parser.add_argument("--backbone-dir", type=Path,
                        default=ROOT / "models/backbones/siglip2-base-patch16-224")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 16:
        parser.error("--max-workers must be between 1 and 16")

    targets = [args.dataset_dir.expanduser().resolve(), args.backbone_dir.expanduser().resolve()]
    if targets[0] == targets[1] or targets[0].is_relative_to(targets[1]) or targets[1].is_relative_to(targets[0]):
        parser.error("Dataset and backbone destinations must be separate directories")
    for target in targets:
        if target.exists():
            parser.error(f"Refusing to overwrite an existing asset directory: {target}")
        if not target.is_relative_to(ROOT):
            parser.error("Destinations must stay inside this checkout; both download folders are gitignored")

    assets = [
        ("datasets", MODEL_DIR / "dataset-source-manifest.json", targets[0]),
        ("models", MODEL_DIR / "backbone-source-manifest.json", targets[1]),
    ]
    prepared = []
    required_bytes = 0
    for key, manifest_path, target in assets:
        asset = load_asset(manifest_path, key)
        size = sum(record["size_bytes"] for record in asset["files"])
        required_bytes += size
        prepared.append((key, asset, target))
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < required_bytes * 2 + min(required_bytes // 20, 1024**3):
        raise RuntimeError("Insufficient free space for staging and verified asset files")

    with tempfile.TemporaryDirectory(prefix=".robustad-assets-stage-", dir=ROOT) as temporary:
        stage_root = Path(temporary)
        staged = []
        for kind, asset, target in prepared:
            stage = stage_root / kind
            snapshot_download(
                repo_id=asset["id"], repo_type=asset.get("repo_type", "model"),
                revision=asset["revision"], local_dir=stage,
                max_workers=args.max_workers,
            )
            count, size = verify_snapshot(stage, asset)
            staged.append((stage, target))
            print(json.dumps({"asset": asset["id"], "revision": asset["revision"],
                              "verified_files": count, "verified_bytes": size}), flush=True)
        for stage, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite a destination created during download: {target}")
            os.replace(stage, target)

    print("Both pinned snapshots were fully hash-verified and installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
