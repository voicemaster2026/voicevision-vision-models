#!/usr/bin/env python3
"""Train frozen SigLIP2 linear probes on pinned RobustAD source domains."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from PIL import Image
import safetensors
from torch import nn
import transformers
from transformers import AutoImageProcessor, AutoModel


DATASET_ID = "AmazonScience/RobustAD"
DATASET_REVISION = "82c642b2d18be60f2dab104f8dd4dea1f4ccfacb"
BACKBONE_ID = "google/siglip2-base-patch16-224"
BACKBONE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SUBSETS = {
    "MetalParts": ("metal_parts_data_dir", 6),
    "PCB": ("pcb_data_dir", 5),
    "PiledBags": ("piled_bags_data_dir", 5),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(dataset_root: Path, subset: str, domain: str,
              fallback_root: Path | None = None) -> list[dict]:
    prefix, _ = SUBSETS[subset]
    directory = dataset_root / subset / f"{prefix}_{domain}"
    metadata = directory / "metadata.jsonl"
    if not metadata.is_file() and fallback_root is not None:
        metadata = fallback_root / subset / f"{prefix}_{domain}" / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing RobustAD metadata: {metadata}")
    rows = []
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        relative = Path(row["file_name"])
        image = directory / relative
        if not image.is_file() and fallback_root is not None:
            image = fallback_root / subset / f"{prefix}_{domain}" / relative
        label = row.get("label")
        if relative.is_absolute() or ".." in relative.parts or label not in (0, 1):
            raise ValueError(f"Unsafe path or invalid binary label in {metadata}:{line_number}")
        if not image.is_file():
            raise FileNotFoundError(f"Missing labeled image {image}")
        dataset_relative = Path(subset) / f"{prefix}_{domain}" / relative
        rows.append({"image": image, "relative": dataset_relative.as_posix(),
                     "label": int(label), "mask": row.get("mask")})
    if not rows or len({row["label"] for row in rows}) != 2:
        raise ValueError(f"Expected both normal and anomaly labels in {metadata}")
    return rows


def source_split(rows: list[dict], seed: int) -> tuple[list[int], list[int]]:
    """Stratify a deterministic 20% validation split within source domain."""
    rng = random.Random(seed)
    train_indices, validation_indices = [], []
    for label in (0, 1):
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        rng.shuffle(indices)
        count = max(1, round(len(indices) * 0.2))
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    return sorted(train_indices), sorted(validation_indices)


def extract_features(rows: list[dict], model, processor, batch_size: int,
                     device: torch.device, role: str) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = []
            for row in batch:
                with Image.open(row["image"]) as image:
                    images.append(image.convert("RGB"))
            encoded = processor(images=images, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()
                       if isinstance(value, torch.Tensor)}
            output = model.get_image_features(**encoded)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                raise RuntimeError("SigLIP2 get_image_features did not return pooler_output")
            features.append(F.normalize(pooled.float(), dim=-1).cpu())
            labels.extend(row["label"] for row in batch)
            print(json.dumps({"phase": role, "processed": min(start + len(batch), len(rows)),
                              "total": len(rows)}, ensure_ascii=False), flush=True)
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def metrics(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    predicted = (probabilities >= threshold).long()
    labels = labels.long()
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    anomaly_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    normal_precision = tn / max(1, tn + fn)
    normal_recall = specificity
    normal_f1 = 2 * normal_precision * normal_recall / max(1e-12, normal_precision + normal_recall)
    return {"samples": int(labels.numel()), "normal": int((labels == 0).sum()),
            "anomaly": int((labels == 1).sum()), "threshold": threshold,
            "accuracy": (tp + tn) / max(1, labels.numel()),
            "balanced_accuracy": (recall + specificity) / 2,
            "macro_f1": (normal_f1 + anomaly_f1) / 2,
            "anomaly_precision": precision, "anomaly_recall": recall,
            "confusion_matrix_tn_fp_fn_tp": [[tn, fp], [fn, tp]]}


def choose_threshold(probabilities: torch.Tensor, labels: torch.Tensor) -> tuple[float, dict]:
    best_threshold, best = 0.5, None
    for step in range(5, 96):
        threshold = step / 100
        result = metrics(probabilities, labels, threshold)
        key = (result["macro_f1"], result["balanced_accuracy"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold, result)
    return best[1], best[2]


def fit_head(train_features: torch.Tensor, train_labels: torch.Tensor,
             val_features: torch.Tensor, val_labels: torch.Tensor,
             epochs: int, seed: int) -> tuple[nn.Linear, int, float, dict]:
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], 2)
    counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = counts.sum() / (2 * counts.clamp_min(1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.025, weight_decay=0.02)
    best_state, best_epoch, best_rank = None, 0, None
    patience, stale = min(epochs, 12), 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(train_labels.numel(), generator=generator)
        for indices in order.split(32):
            logits = head(train_features[indices])
            loss = F.cross_entropy(logits, train_labels[indices], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(val_features)
            val_prob = logits.softmax(dim=-1)[:, 1]
            candidate_threshold, val_metrics = choose_threshold(val_prob, val_labels)
            val_loss = float(F.cross_entropy(logits, val_labels))
        rank = (val_metrics["macro_f1"], val_metrics["balanced_accuracy"], -val_loss)
        if best_state is None or rank > best_rank:
            best_rank = rank
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            best_epoch, selected_threshold, selected_metrics = epoch, candidate_threshold, val_metrics
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    head.load_state_dict(best_state)
    return head, best_epoch, selected_threshold, selected_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fallback-dataset-root", type=Path,
                        help="Use files in this verified persistent tree when absent from dataset-root.")
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--dataset-source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--subsets", nargs="+", choices=tuple(SUBSETS), default=list(SUBSETS))
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(max(1, min(args.threads, 8)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.mkldnn.enabled = True
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.backbone, local_files_only=True)
    model = AutoModel.from_pretrained(args.backbone, local_files_only=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads, report, training_manifest = {}, {}, {}
    for subset_index, subset in enumerate(args.subsets):
        _, max_test_domain = SUBSETS[subset]
        source_rows = read_rows(args.dataset_root, subset, "train", args.fallback_dataset_root)
        train_indices, val_indices = source_split(source_rows, args.seed + subset_index)
        all_source_features, all_source_labels = extract_features(
            source_rows, model, processor, args.batch_size, device, f"{subset}/train-source")
        source_features_train = all_source_features[train_indices]
        source_labels_train = all_source_labels[train_indices]
        val_features = all_source_features[val_indices]
        val_labels = all_source_labels[val_indices]
        final_head, best_epoch, threshold, val_metrics = fit_head(
            source_features_train, source_labels_train, val_features, val_labels,
            args.epochs, args.seed + subset_index)
        heads[subset] = {key: value.detach().cpu() for key, value in final_head.state_dict().items()}
        domain_results = {}
        for domain_index in range(max_test_domain + 1):
            domain = f"test{domain_index}"
            rows = read_rows(args.dataset_root, subset, domain, args.fallback_dataset_root)
            features, labels = extract_features(rows, model, processor, args.batch_size,
                                                device, f"{subset}/{domain}")
            with torch.no_grad():
                anomaly_prob = final_head(features).softmax(dim=-1)[:, 1]
            domain_results[domain] = metrics(anomaly_prob, labels, threshold)
        report[subset] = {"source_train_samples": len(source_rows),
                          "source_split_counts": {"train": len(train_indices), "validation": len(val_indices)},
                          "best_epoch": best_epoch, "validation": val_metrics,
                          "held_out_target_domains": domain_results,
                          "target_domain_macro_f1_mean": sum(m["macro_f1"] for m in domain_results.values()) / len(domain_results)}
        training_manifest[subset] = {
            "source_domain": "train", "validation_assignment": "stratified deterministic 20% within train domain",
            "train": [{"path": source_rows[i]["relative"], "label": source_rows[i]["label"]}
                      for i in train_indices],
            "validation": [{"path": source_rows[i]["relative"], "label": source_rows[i]["label"]}
                           for i in val_indices],
            "test_domains": [f"test{i}" for i in range(max_test_domain + 1)],
            "mask_usage": "Masks retained in source data but not used by this image-level classifier.",
        }

    checkpoint = {"schema": "voicevision.robustad-siglip2-linear-probes/1",
                  "backbone_id": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION,
                  "embedding_dim": next(iter(heads.values()))["weight"].shape[1],
                  "label_order": ["normal", "anomaly"], "heads": heads,
                  "thresholds": {key: report[key]["validation"].get("threshold", 0.5) for key in heads},
                  "processor_config": processor.to_dict()}
    checkpoint_path = args.output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    source_manifest_text = args.dataset_source_manifest.read_text(encoding="utf-8")
    source_manifest_sha = hashlib.sha256(source_manifest_text.encode("utf-8")).hexdigest()
    source_manifest = json.loads(source_manifest_text)
    dataset_entry = next(
        item for item in source_manifest["datasets"] if item["id"] == DATASET_ID
    )
    report_data = {"schema": "voicevision.robustad-siglip2-evaluation/1",
                   "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                   "backbone_id": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION,
                   "dataset_source_manifest_sha256": source_manifest_sha,
                   "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                               "safetensors": safetensors.__version__, "threads": torch.get_num_threads(),
                               "device": str(device)},
                   "threshold_selection": "validation only; maximize macro-F1 then balanced accuracy",
                   "results": report, "trained_at": datetime.now(timezone.utc).isoformat()}
    (args.output_dir / "evaluation.json").write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps({"schema": "voicevision.robustad-siglip2-split-manifest/1",
                    "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                    "dataset_source_manifest_sha256": source_manifest_sha,
                    "seed": args.seed, "subsets": training_manifest}, indent=2) + "\n", encoding="utf-8")
    all_test = [metric for result in report.values() for metric in result["held_out_target_domains"].values()]
    artifact = {"schema": "voicevision.model-artifact/1", "id": "robustad-siglip2-anomaly-v1",
                "version": "1.0.0", "task": "binary normal/anomaly image classification",
                "artifact_file": "model.pt", "artifact_sha256": sha256_file(checkpoint_path),
                "base_model": {"id": BACKBONE_ID, "revision": BACKBONE_REVISION,
                               "license": "Apache-2.0",
                               "source_url": f"https://huggingface.co/{BACKBONE_ID}/tree/{BACKBONE_REVISION}"},
                "training": {"dataset_repo": DATASET_ID, "dataset_revision": DATASET_REVISION,
                             "dataset_path": "datasets/AmazonScience/RobustAD", "dataset_manifest_sha256": source_manifest_sha,
                             "dataset_file_count": len(dataset_entry["files"]),
                             "source_train_validation_image_count": sum(
                                 split["train"] + split["validation"]
                                 for split in (result["source_split_counts"]
                                               for result in report.values())
                             ),
                             "held_out_test_image_count": sum(
                                 metric["samples"] for result in report.values()
                                 for metric in result["held_out_target_domains"].values()
                             ),
                             "split_counts": {subset: report[subset]["source_split_counts"] for subset in report},
                             "seed": args.seed, "epochs_run": {subset: report[subset]["best_epoch"] for subset in report}},
                "validation": {"macro_f1": sum(row["validation"]["macro_f1"] for row in report.values()) / len(report),
                               "balanced_accuracy": sum(row["validation"]["balanced_accuracy"] for row in report.values()) / len(report)},
                "test": {"macro_f1": sum(row["macro_f1"] for row in all_test) / len(all_test),
                         "balanced_accuracy": sum(row["balanced_accuracy"] for row in all_test) / len(all_test),
                         "samples": sum(row["samples"] for row in all_test),
                         "target_domains": len(all_test)},
                "runtime_status": "experimental; not integrated into an active VVP runtime",
                "limitations": [
                    "Three subset-specific linear heads classify images as normal/anomaly; they do not localize defects.",
                    "The SigLIP2 image encoder is frozen; only per-subset linear heads were trained.",
                    "Validation is a deterministic stratified image split within the official source train domain; it is not grouped by capture session.",
                    "Official target-domain test sets were held out and are not used for tuning.",
                    "Metrics on RobustAD do not establish performance on VoiceVision customer products or production lines.",
                    "Training data is CC-BY-4.0 and the base model is Apache-2.0; see the linked sources and attribution in the model README.",
                ]}
    (args.output_dir / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "results": report,
                      "model_sha256": sha256_file(checkpoint_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
