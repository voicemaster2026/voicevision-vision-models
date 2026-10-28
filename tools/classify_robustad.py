#!/usr/bin/env python3
"""Run the experimental RobustAD subset-specific SigLIP2 anomaly heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/robustad-siglip2-anomaly-v1/1.0.0"
BACKBONE_ID = "google/siglip2-base-patch16-224"
BACKBONE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SUBSETS = ("MetalParts", "PCB", "PiledBags")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "model.pt")
    parser.add_argument("--backbone", default=BACKBONE_ID,
                        help="Pinned Hugging Face model ID or a local snapshot directory")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != "voicevision.robustad-siglip2-linear-probes/1"
            or checkpoint.get("backbone_revision") != BACKBONE_REVISION
            or args.subset not in checkpoint.get("heads", {})):
        raise ValueError("Checkpoint schema, pinned backbone, or subset is incompatible")

    processor = AutoImageProcessor.from_pretrained(
        args.backbone, revision=BACKBONE_REVISION,
        local_files_only=args.local_files_only,
    )
    encoder = AutoModel.from_pretrained(
        args.backbone, revision=BACKBONE_REVISION,
        local_files_only=args.local_files_only,
    ).eval()
    with Image.open(args.image) as source:
        inputs = processor(images=[source.convert("RGB")], return_tensors="pt")
    with torch.inference_mode():
        encoded = encoder.get_image_features(**inputs)
        pooled = getattr(encoded, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("Pinned SigLIP2 model did not return pooler_output")
        features = F.normalize(pooled.float(), dim=-1)
        head = checkpoint["heads"][args.subset]
        logits = F.linear(features, head["weight"].float(), head["bias"].float())
        anomaly_score = float(logits.softmax(dim=-1)[0, 1])

    threshold = float(checkpoint.get("thresholds", {}).get(args.subset, 0.5))
    print(json.dumps({
        "subset": args.subset,
        "estimate": "anomaly" if anomaly_score >= threshold else "normal",
        "anomaly_score_uncalibrated": anomaly_score,
        "validation_threshold": threshold,
        "decision_status": "experimental_estimate_only",
        "backbone": BACKBONE_ID,
        "backbone_revision": BACKBONE_REVISION,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
