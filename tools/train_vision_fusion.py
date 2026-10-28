#!/usr/bin/env python3
"""Reproducible synthetic-only fusion baseline; no cameras or external datasets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicevisionunited.vision_fusion import COLORS, INTENTS, SHAPES, VERSION, build_model, encode_text, image_inputs, tokens

TRAIN_TEMPLATES = (
    ('inspect this region', 'please inspect the marked shape', 'zkontroluj tuto oblast', 'prosim zkontroluj oznacenou oblast'),
    ('count objects in this region', 'please count the marked objects', 'spocitej objekty v oblasti', 'prosim spocitej oznacene objekty'),
    ('measure this region', 'please measure the marked shape', 'zmer tuto oblast', 'prosim zmer oznaceny tvar'),
)
EVAL_TEMPLATES = (
    ('inspect the marked region please', 'prosim tuto oblast zkontroluj'),
    ('count the objects in the marked region please', 'objekty v oblasti prosim spocitej'),
    ('measure the marked region please', 'tuto oblast prosim zmer'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(index, split):
    import numpy as np
    from PIL import Image, ImageDraw

    seed = (202610230 if split == 'train' else 202610231) + index * 1009
    rng = random.Random(seed)
    # Independent labels and target slot; distractors defeat whole-image shortcuts.
    intent, shape, color, target = rng.randrange(3), rng.randrange(4), rng.randrange(1, 4), rng.randrange(4)
    if shape == 0:
        color = 0
    background = tuple(rng.randrange(8, 35) for _ in range(3))
    image = Image.new('RGB', (64, 64), background)
    draw = ImageDraw.Draw(image)
    palette = ((25, 25, 25), (220, 40, 45), (35, 210, 55), (40, 65, 220))
    selected_box = None
    for slot, (x, y) in enumerate(((16, 16), (48, 16), (16, 48), (48, 48))):
        x += rng.randint(-2, 2) if split == 'train' else rng.choice((-4, 4))
        y += rng.randint(-2, 2) if split == 'train' else rng.choice((-4, 4))
        half = rng.randint(6, 9) if split == 'train' else rng.choice((10, 11))
        box = (x - half, y - half, x + half, y + half)
        category, hue = (shape, color) if slot == target else (rng.randrange(1, 4), rng.randrange(1, 4))
        rgb = tuple(max(0, min(255, v + rng.randint(-15, 15))) for v in palette[hue])
        if category == 1:
            draw.ellipse(box, fill=rgb)
        elif category == 2:
            draw.rectangle((box[0], box[1] + half // 3, box[2], box[3] - half // 3), fill=rgb)
        elif category == 3:
            draw.polygon(((x, box[1]), (box[2], box[3]), (box[0], box[3])), fill=rgb)
        if slot == target:
            selected_box = tuple(max(0, min(63, value + (-2 if i < 2 else 2))) for i, value in enumerate(box))
    mask = Image.new('L', (64, 64))
    ImageDraw.Draw(mask).rectangle(selected_box, fill=255)
    sentence = rng.choice((TRAIN_TEMPLATES if split == 'train' else EVAL_TEMPLATES)[intent])
    scene, region = image_inputs(image, mask)
    record = {'id': f'{split}-{index:05}', 'seed': seed, 'split': split,
              'intent': INTENTS[intent], 'shape': SHAPES[shape], 'color': COLORS[color],
              'target_slot': target, 'roi_xyxy_inclusive': list(selected_box), 'transcript': sentence,
              'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
              'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
    return scene, region, sentence, (intent, shape, color), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/vision-fusion' / VERSION)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-size', type=int, default=3200)
    parser.add_argument('--evaluation-size', type=int, default=800)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 100 <= args.train_size <= 20000 or not 100 <= args.evaluation_size <= 5000 or not 1 <= args.cpu_threads <= 16:
        parser.error('Training dimensions exceed bounded run limits')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must stay inside this project')
    if output.exists() and any(output.iterdir()):
        parser.error('Output is nonempty; preserve existing artifact releases')
    import numpy as np
    import PIL
    import torch

    started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20261023)
    np.random.seed(20261023)
    random.seed(20261023)
    torch.use_deterministic_algorithms(True)
    vocabulary = {'<pad>': 0, '<unk>': 1}
    for word in sorted({word for family in TRAIN_TEMPLATES for sentence in family for word in tokens(sentence)}):
        vocabulary[word] = len(vocabulary)
    def dataset(count, split):
        samples = [sample(i, split) for i in range(count)]
        return (torch.tensor(np.stack([x[0] for x in samples])),
                torch.tensor(np.stack([x[1] for x in samples])),
                torch.tensor([encode_text(x[2], vocabulary) for x in samples], dtype=torch.long),
                torch.tensor([x[3] for x in samples], dtype=torch.long), [x[4] for x in samples])
    train = dataset(args.train_size, 'train')
    # The evaluation set is generated separately and never used for updates or selection.
    evaluation = dataset(args.evaluation_size, 'evaluation')
    train_ids = {r['image_sha256'] + r['mask_sha256'] + r['transcript'] for r in train[4]}
    overlap = len(train_ids.intersection(r['image_sha256'] + r['mask_sha256'] + r['transcript'] for r in evaluation[4]))
    if overlap:
        raise RuntimeError('Synthetic split contains repeated full inputs')
    model = build_model(len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(args.train_size)
        accumulated = 0.0
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(train[0][batch], train[1][batch], train[2][batch])
            loss = sum(loss_function(value, train[3][batch, head]) for head, value in enumerate(predictions))
            loss.backward()
            optimizer.step()
            accumulated += loss.item() * len(batch)
        history.append({'epoch': epoch + 1, 'training_loss': accumulated / args.train_size})
        print(json.dumps(history[-1]), flush=True)
    model.eval()
    def evaluate(mode='all_modalities'):
        correct = [0, 0, 0]
        joint, confusion = 0, [np.zeros((n, n), dtype=np.int64) for n in (3, 4, 4)]
        with torch.inference_mode():
            for start in range(0, args.evaluation_size, 64):
                stop = min(start + 64, args.evaluation_size)
                scene, region, text = (value[start:stop].clone() for value in evaluation[:3])
                if mode == 'no_image':
                    scene[:, :3] = 0
                    region[:, :3] = 0
                elif mode == 'no_text':
                    text[:] = 0
                elif mode == 'wrong_region':
                    # Change annotation/region association while preserving text and scene.
                    shifted = (torch.arange(start, stop) + 1) % args.evaluation_size
                    region = evaluation[1][shifted]
                    scene[:, 3] = evaluation[0][shifted, 3]
                predicted = torch.stack([value.argmax(1) for value in model(scene, region, text)], dim=1)
                truth = evaluation[3][start:stop]
                joint += int(predicted.eq(truth).all(1).sum())
                for head in range(3):
                    correct[head] += int(predicted[:, head].eq(truth[:, head]).sum())
                    for actual, estimate in zip(truth[:, head].tolist(), predicted[:, head].tolist()):
                        confusion[head][actual, estimate] += 1
        return {'samples': args.evaluation_size, 'intent_accuracy': correct[0] / args.evaluation_size,
                'shape_accuracy': correct[1] / args.evaluation_size, 'color_accuracy': correct[2] / args.evaluation_size,
                'joint_accuracy': joint / args.evaluation_size,
                'confusion_matrices_actual_rows_predicted_columns': [m.tolist() for m in confusion]}
    metrics = {mode: evaluate(mode) for mode in ('all_modalities', 'no_image', 'no_text', 'wrong_region')}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'model.pt')
    with (output / 'dataset-manifest.jsonl').open('w') as stream:
        for row in train[4] + evaluation[4]:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    report = {'schema_version': 1, 'evaluation_scope': 'synthetic generated scenes only; no camera/real parts/ASR validation',
              'model_sha256': sha(output / 'model.pt'), 'metrics': metrics,
              'full_input_train_evaluation_overlap': overlap, 'evaluation_used_for_training_or_selection': False,
              'geometry_holdout': 'train half-size 6..9 and center jitter -2..2; evaluation half-size 10..11 and center jitter +/-4',
              'template_holdout': {'train': TRAIN_TEMPLATES, 'evaluation': EVAL_TEMPLATES},
              'ablations_scope': 'diagnostic interventions only; wrong_region swaps region/mask association across samples',
              'confidence_calibrated': False}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    artifact = {'schema_version': 1, 'artifact_version': VERSION, 'family': 'vision-fusion',
                'architecture': 'region-scene-text-cnn-v1', 'intents': INTENTS, 'shapes': SHAPES, 'colors': COLORS,
                'vocabulary': vocabulary, 'model_file': 'model.pt', 'model_sha256': sha(output / 'model.pt'),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'experimental_synthetic_only',
                'training_data': {'source': 'project-authored deterministic primitive renderer and CS/EN command templates',
                                  'license': 'CC0-1.0 for newly generated synthetic data and templates',
                                  'external_data': False, 'customer_data': False, 'physical_camera_images': False,
                                  'train_samples': args.train_size, 'evaluation_samples': args.evaluation_size,
                                  'manifest_sha256': sha(output / 'dataset-manifest.jsonl')},
                'training': {'seed': 20261023, 'epochs': args.epochs, 'batch_size': 64,
                             'optimizer': 'Adam', 'learning_rate': 0.001, 'history': history,
                             'elapsed_seconds': time.monotonic() - started, 'cpu_threads': args.cpu_threads},
                'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__, 'device': 'cpu'},
                'source_sha256': {'tools/train_vision_fusion.py': sha(Path(__file__)),
                                  'voicevisionunited/vision_fusion.py': sha(ROOT / 'voicevisionunited/vision_fusion.py')},
                'evaluation_sha256': sha(output / 'evaluation.json'), 'inference_api': 'voicevisionunited.vision_fusion.propose',
                'limitations': ['No real camera/part validation.', 'Input is transcript, not audio; ASR is separate.',
                                'ROI comes from vector geometry, not inferred measurement.',
                                'Only inspect/count/measure proposals; no execution or task result.',
                                'Softmax scores are uncalibrated; no deployment/promotion claim.']}
    (output / 'artifact.json').write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps({'artifact': str(output), 'metrics': metrics, 'model_sha256': artifact['model_sha256']}), flush=True)


if __name__ == '__main__':
    main()
