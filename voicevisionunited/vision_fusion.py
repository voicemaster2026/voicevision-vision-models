"""Experimental image + vector annotation + transcript proposal model.

Trained only on generated geometric scenes. No camera access, action execution,
speech recognition, measurement, OCR or real-world inspection is performed.
PyTorch/Pillow/NumPy are imported only when inference is requested.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
import unicodedata


VERSION = '0.1.0'
INTENTS = ('inspect_region', 'count_objects', 'measure_dimensions')
SHAPES = ('empty', 'circle', 'rectangle', 'triangle')
COLORS = ('background', 'red', 'green', 'blue')
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_MODEL_BYTES = 8 * 1024 * 1024


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def tokens(text):
    normalized = unicodedata.normalize('NFKD', text.casefold())
    return re.findall(r'[^\W_]+', ''.join(c for c in normalized if not unicodedata.combining(c)))


def encode_text(text, vocabulary, length=24):
    return [vocabulary.get(word, 1) for word in tokens(text)[:length]] + [0] * max(0, length - len(tokens(text)[:length]))


def build_model(vocabulary_size):
    import torch
    from torch import nn

    class FusionNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.region = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(), nn.Linear(48 * 4 * 4, 64), nn.ReLU())
            self.scene = nn.Sequential(nn.Conv2d(4, 8, 5, stride=2, padding=2), nn.ReLU(),
                                       nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                                       nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.embedding = nn.Embedding(vocabulary_size, 24, padding_idx=0)
            self.fusion = nn.Sequential(nn.Linear(64 + 16 + 24, 64), nn.ReLU())
            self.intent = nn.Linear(64, len(INTENTS))
            self.shape = nn.Linear(64, len(SHAPES))
            self.color = nn.Linear(64, len(COLORS))

        def forward(self, scene, region, text):
            valid = text.ne(0).unsqueeze(-1)
            language = (self.embedding(text) * valid).sum(1) / valid.sum(1).clamp_min(1)
            fused = self.fusion(torch.cat((self.region(region), self.scene(scene), language), dim=1))
            return self.intent(fused), self.shape(fused), self.color(fused)

    return FusionNetwork()


def image_inputs(image, mask):
    """Shared train/inference preprocessing; mask chooses a region of the image."""
    import numpy as np
    from PIL import Image

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError('Annotation mask is empty')
    image = image.convert('RGB')
    scene = np.concatenate((np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255,
                            np.asarray(mask.resize((64, 64), Image.Resampling.NEAREST), dtype=np.float32)[..., None] / 255), axis=2)
    region = np.concatenate((np.asarray(image.crop(bbox).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255,
                             np.asarray(mask.crop(bbox).resize((32, 32), Image.Resampling.NEAREST), dtype=np.float32)[..., None] / 255), axis=2)
    return scene.transpose(2, 0, 1).copy(), region.transpose(2, 0, 1).copy()


def rasterize_strokes(strokes, size):
    """Accept image-pixel vectors, preserving coordinates and the original image."""
    from PIL import Image, ImageDraw

    if not isinstance(strokes, list) or not 1 <= len(strokes) <= 32:
        raise ValueError('strokes must contain 1–32 vector strokes')
    width, height = size
    mask, normalized, total = Image.new('L', size), [], 0
    draw = ImageDraw.Draw(mask)
    for stroke in strokes:
        if not isinstance(stroke, dict):
            raise ValueError('Each stroke must be an object with points, closed and optional width')
        points = stroke.get('points')
        closed = stroke.get('closed', False)
        pen = stroke.get('width', 4)
        if not isinstance(points, list) or not 2 <= len(points) <= 4096 or type(closed) is not bool:
            raise ValueError('Stroke points/closed are invalid')
        if not _finite_number(pen) or not 1 <= pen <= 64:
            raise ValueError('Stroke width must be a finite pixel width from 1 to 64')
        total += len(points)
        if total > 4096:
            raise ValueError('At most 4096 annotation points are supported')
        clean = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError('Each point must contain x and y')
            if not all(_finite_number(v) for v in point):
                raise ValueError('Coordinates must be finite numbers')
            x, y = point
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError('Annotation coordinates must lie inside the source image')
            clean.append((float(x), float(y)))
        if closed:
            if len(clean) < 3:
                raise ValueError('Closed polygons need at least three points')
            draw.polygon(clean, fill=255)
        else:
            draw.line(clean, fill=255, width=round(pen), joint='curve')
        normalized.append({'points': [list(point) for point in clean], 'closed': closed, 'width': float(pen)})
    if mask.getbbox() is None:
        raise ValueError('Annotation mask is empty')
    return mask, normalized


def propose(image_path, strokes, transcript, *, model_root=None, image_id=None,
            aiappdata_result=None):
    """Return an unexecuted synthetic-domain proposal linked to original inputs.

    Audio must be transcribed by a separate ASR adapter; retain its provenance
    with this result. ROI is deterministic annotation geometry, not a model's
    measurement. Scores are uncalibrated and cannot establish task readiness.
    """
    if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 4096 or '\0' in transcript:
        raise ValueError('transcript must contain 1–4096 characters without NUL')
    if image_id is not None and (not isinstance(image_id, str) or len(image_id) > 512 or '\0' in image_id):
        raise ValueError('image_id must be a string of at most 512 characters without NUL')
    import numpy as np
    import torch
    from PIL import Image

    root = Path(model_root) if model_root is not None else Path(__file__).resolve().parents[1] / 'models/vision-fusion' / VERSION
    metadata_path, weight_path = root / 'artifact.json', root / 'model.pt'
    if not metadata_path.is_file() or metadata_path.stat().st_size > 128 * 1024:
        raise ValueError('Fusion model metadata is unavailable or oversized')
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError('Fusion model metadata must be an object')
    if metadata.get('artifact_version') != VERSION or metadata.get('architecture') != 'region-scene-text-cnn-v1':
        raise ValueError('Unsupported fusion model architecture/version')
    if metadata.get('intents') != list(INTENTS) or metadata.get('shapes') != list(SHAPES) or metadata.get('colors') != list(COLORS):
        raise ValueError('Fusion model label vocabulary is incompatible')
    vocabulary = metadata.get('vocabulary')
    if not isinstance(vocabulary, dict) or not 2 <= len(vocabulary) <= 4096 or sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError('Fusion token vocabulary is invalid')
    if not weight_path.is_file() or not 0 < weight_path.stat().st_size <= MAX_MODEL_BYTES:
        raise ValueError('Fusion weights are unavailable or oversized')
    weights = weight_path.read_bytes()
    digest = hashlib.sha256(weights).hexdigest()
    if digest != metadata.get('model_sha256'):
        raise ValueError('Fusion weights differ from the artifact manifest')
    path = Path(image_path).expanduser().resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError('Image must be a regular nonempty file no larger than 32 MiB')
    image_bytes = path.read_bytes()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds 32 MiB')
    with Image.open(io.BytesIO(image_bytes)) as source:
        if getattr(source, 'n_frames', 1) != 1:
            raise ValueError('Fusion accepts exactly one source frame, not an animation or multipage image')
        if source.width * source.height > MAX_PIXELS:
            raise ValueError('Image exceeds 16 million pixels')
        # Coordinates refer to stored pixel orientation; never silently rotate.
        aiappdata_input_image = source.copy()
        image = source.convert('RGB')
    upstream_visual_evidence = None
    if aiappdata_result is not None:
        from .aiappdata_adapter import normalized_model_input, validate_result
        model_input = normalized_model_input(aiappdata_input_image).detach().cpu().to(dtype=torch.float32).contiguous()
        input_digest = hashlib.sha256(model_input.numpy().tobytes()).hexdigest()
        upstream_visual_evidence = validate_result(
            aiappdata_result, expected_input_sha256=input_digest)
    mask, vectors = rasterize_strokes(strokes, image.size)
    scene, region = image_inputs(image, mask)
    network = build_model(len(vocabulary))
    network.load_state_dict(torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True), strict=True)
    network.eval()
    with torch.inference_mode():
        logits = network(torch.from_numpy(scene[None]), torch.from_numpy(region[None]),
                         torch.tensor([encode_text(transcript, vocabulary)], dtype=torch.long))
        scores = [torch.softmax(value, dim=1)[0].tolist() for value in logits]
    choices = [int(np.argmax(row)) for row in scores]
    roi = list(mask.getbbox())
    intent, shape, color = INTENTS[choices[0]], SHAPES[choices[1]], COLORS[choices[2]]
    source_hash = hashlib.sha256(image_bytes).hexdigest()
    result = {'schema_version': 1, 'status': 'proposal_only', 'execution_available': False,
            'artifact': {'family': 'vision-fusion', 'version': VERSION, 'sha256': digest,
                         'training_domain': 'synthetic geometric RGB scenes and authored CS/EN transcript templates'},
            'source_image': {'path': str(path), 'sha256': source_hash, 'image_id': image_id or source_hash,
                             'width': image.width, 'height': image.height, 'orientation': 'stored pixel orientation'},
            'transcript': transcript, 'audio_inference_performed': False,
            'annotation': {'strokes': vectors, 'coordinate_system': 'source-image pixels',
                           'sha256': hashlib.sha256(json.dumps(vectors, sort_keys=True).encode()).hexdigest()},
            'roi': {'xyxy': roi, 'right_bottom_exclusive': True, 'source': 'vector-mask bounding rectangle'},
            'proposal': {'intent': intent, 'synthetic_shape': shape, 'synthetic_color': color,
                         'action': 'review requested operation in the marked region', 'automatically_executable': False},
            'scores': {'intent': dict(zip(INTENTS, scores[0])), 'shape': dict(zip(SHAPES, scores[1])),
                       'color': dict(zip(COLORS, scores[2])), 'calibrated': False},
            'required_review': ['Confirm transcript, annotation target and requested operation.',
                                'Real image object identity and inspection semantics are not validated.',
                                'Counting/measurement require a separate calibrated and validated task engine.'],
            'limitations': ['Synthetic-only training; camera/physical-part behavior is unknown.',
                            'Region categories apply only to the generated geometric training domain.',
                            'No OCR, defect recognition, physical measurements, ASR or camera action.',
                            'Open strokes and arbitrary real scenes are outside the evaluated closed-region distribution.',
                            'Matching model hash establishes integrity relative to local metadata, not trust/authenticity.']}
    if upstream_visual_evidence is not None:
        # Parallel evidence only: it is attached after the learned forward pass
        # and never changes VisionFusion features, scores or proposal.
        result['upstream_visual_evidence'] = upstream_visual_evidence
    return result
