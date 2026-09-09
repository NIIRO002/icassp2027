import json

import torch


FEATURE_NAMES = [
    "energy",
    "energy_slope",
    "f0_slope",
    "vibrato_rate",
    "vibrato_depth",
    "breathiness",
    "brightness",
    "onset_strength",
]

FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

DEFAULT_CONTROL = {
    "global": {
        "energy": 0.0,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
        "breathiness": 0.0,
        "brightness": 0.0,
        "onset_strength": 0.0,
    }
}

CONTROL_ALIASES = {
    "vibrato": ("vibrato_rate", "vibrato_depth"),
    "accent": ("onset_strength", "energy_slope"),
    "attack": ("onset_strength",),
    "pitch_transition": ("f0_slope",),
    "portamento": ("f0_slope",),
}


def load_expression_control(control=None, control_json=None):
    if control_json:
        with open(control_json, "r", encoding="utf-8") as f:
            return json.load(f)
    if control:
        return json.loads(control)
    return None


def summarize_expression_features(features):
    """Return compact per-feature stats for logs/debugging."""
    if features.dim() != 3:
        raise ValueError(f"features must be [B, T, C], got {tuple(features.shape)}")
    stats = {}
    values = features.detach().float()
    for idx, name in enumerate(FEATURE_NAMES[:values.size(-1)]):
        item = values[..., idx]
        stats[name] = {
            "mean": float(item.mean().cpu()),
            "std": float(item.std().cpu()),
            "min": float(item.min().cpu()),
            "max": float(item.max().cpu()),
        }
    return stats


def format_expression_stats(stats):
    parts = []
    for name in FEATURE_NAMES:
        if name not in stats:
            continue
        item = stats[name]
        parts.append(
            f"{name}=mean:{item['mean']:.3f}/std:{item['std']:.3f}/range:{item['min']:.2f}..{item['max']:.2f}"
        )
    return " | ".join(parts)


def _iter_control_items(control):
    if not control:
        return []
    if "global" in control:
        return control["global"].items()
    return {
        key: value
        for key, value in control.items()
        if key not in {"sections", "verse", "chorus", "bridge"}
    }.items()


def _apply_value(features, key, value, strength):
    keys = CONTROL_ALIASES.get(key, (key,))
    for feature_name in keys:
        if feature_name not in FEATURE_INDEX:
            continue
        features[..., FEATURE_INDEX[feature_name]] += float(value) * strength


def _section_slice(section, target_len, sr, hop_length):
    if "start_frame" in section:
        start = int(section["start_frame"])
    else:
        start = int(float(section.get("start", section.get("start_sec", 0.0))) * sr / hop_length)
    if "end_frame" in section:
        end = int(section["end_frame"])
    else:
        default_end_sec = target_len * hop_length / sr
        end = int(float(section.get("end", section.get("end_sec", default_end_sec))) * sr / hop_length)
    return max(0, start), min(target_len, max(start + 1, end))


def apply_expression_controls(features, control=None, strength=1.0, sr=None, hop_length=None):
    """Apply simple inference-time expression controls to normalized features.

    control examples:
        {"energy": 0.5, "vibrato": 0.3, "breathiness": -0.2}
        {"global": {"energy": 0.3}, "sections": [{"start": 20, "end": 40, "vibrato": 0.8}]}

    The values are offsets in normalized feature space. This is intentionally
    lightweight for prototype control before adapter training.
    """
    if control is None:
        return features
    if features.dim() != 3:
        raise ValueError(f"features must be [B, T, C], got {tuple(features.shape)}")

    controlled = features.clone()
    for key, value in _iter_control_items(control):
        _apply_value(controlled, key, value, strength)

    sections = control.get("sections", []) if isinstance(control, dict) else []
    if sections:
        if sr is None or hop_length is None:
            raise ValueError("sr and hop_length are required for section controls")
        target_len = features.size(1)
        for section in sections:
            start, end = _section_slice(section, target_len, sr, hop_length)
            view = controlled[:, start:end]
            for key, value in section.items():
                if key in {"start", "end", "start_sec", "end_sec", "start_frame", "end_frame", "label"}:
                    continue
                _apply_value(view, key, value, strength)

    return torch.clamp(controlled, min=-6.0, max=6.0)
