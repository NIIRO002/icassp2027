import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, hilbert, sosfiltfilt

from modules.expression.hierarchical_features import compose_axis_targets


MULTISCALE_AXIS_NAMES = (
    "accent",
    "intensity",
    "breathiness",
    "vocal_register",
    "vibrato",
)

REGISTER_COMPONENT_NAMES = (
    "h1_h2_db",
    "cpp_db",
    "hnr_db",
    "hf_flatness",
    "spectral_tilt_db_oct",
    "high_low_balance_db",
    "spectral_centroid_hz",
)

VIBRATO_COMPONENT_NAMES = (
    "vibrato_depth_cent",
    "vibrato_rate_hz",
    "vibrato_band_ratio",
    "vibrato_periodicity",
    "vibrato_strength",
)


def _mel_frequencies(n_mels, sample_rate, device, dtype):
    mel_max = 2595.0 * math.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel = torch.linspace(0.0, mel_max, n_mels, device=device, dtype=dtype)
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def register_frame_components(log_mel, hierarchical_components, sample_rate=44100):
    """Return frame-level source/tract proxies for mixed-to-falsetto register.

    The first four values reuse the waveform-derived harmonic features already
    stored in the hierarchical cache. The remaining values are computed from
    log-mel amplitude. No single component is treated as the register target;
    their weights are fitted and validated on matched GTSinger performances.
    """

    if log_mel.dim() != 2 or hierarchical_components.dim() != 2:
        raise ValueError("Expected log_mel [M,T] and components [C,T]")
    if log_mel.size(1) != hierarchical_components.size(1):
        raise ValueError("Feature frame lengths do not match")

    log_mel = log_mel.float()
    hierarchical_components = hierarchical_components.float()
    frequencies = _mel_frequencies(
        log_mel.size(0),
        sample_rate,
        log_mel.device,
        log_mel.dtype,
    )
    db = log_mel * (20.0 / math.log(10.0))

    valid_frequency = frequencies >= 120.0
    x = torch.log2(frequencies[valid_frequency].clamp_min(120.0))
    x = x - x.mean()
    tilt = (db[valid_frequency] * x[:, None]).sum(dim=0) / x.square().sum().clamp_min(1e-6)

    low = (frequencies >= 250.0) & (frequencies < 1800.0)
    high = (frequencies >= 3500.0) & (frequencies < 10000.0)
    high_low = db[high].mean(dim=0) - db[low].mean(dim=0)

    magnitude = log_mel.exp().clamp_min(1e-6)
    centroid = (magnitude * frequencies[:, None]).sum(dim=0) / magnitude.sum(dim=0)

    return torch.stack(
        [
            hierarchical_components[8],
            hierarchical_components[6],
            hierarchical_components[7],
            hierarchical_components[9],
            tilt,
            high_low,
            centroid,
        ],
        dim=0,
    )


def _relative_interval(start, end, segment_start, segment_end):
    if end <= segment_start or start >= segment_end:
        return None
    return (
        max(float(start), segment_start) - segment_start,
        min(float(end), segment_end) - segment_start,
    )


def _interval_mask(frame_times, intervals):
    mask = torch.zeros_like(frame_times, dtype=torch.bool)
    for start, end in intervals:
        mask |= (frame_times >= start) & (frame_times <= end)
    return mask


def parse_gtsinger_multiscale_masks(
    annotation_path,
    segment_start,
    segment_end,
    frame_times,
    sustain_trim_seconds=0.08,
):
    """Build technique masks and note identities for a segmented GTSinger clip."""

    entries = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    technique_intervals = {
        "mixed": [],
        "falsetto": [],
        "vibrato": [],
        "glissando": [],
    }
    note_intervals = []
    note_index = 1

    for entry in entries:
        starts = entry.get("ph_start", [])
        ends = entry.get("ph_end", [])
        for name, json_key in (
            ("mixed", "mix"),
            ("falsetto", "falsetto"),
            ("vibrato", "vibrato"),
            ("glissando", "glissando"),
        ):
            for start, end, value in zip(starts, ends, entry.get(json_key, [])):
                if str(value) != "1":
                    continue
                relative = _relative_interval(
                    float(start),
                    float(end),
                    segment_start,
                    segment_end,
                )
                if relative is not None:
                    technique_intervals[name].append(relative)

        for note, start, end in zip(
            entry.get("note", []),
            entry.get("note_start", []),
            entry.get("note_end", []),
        ):
            if float(note) <= 0:
                continue
            relative = _relative_interval(
                float(start),
                float(end),
                segment_start,
                segment_end,
            )
            if relative is None:
                continue
            note_intervals.append((relative[0], relative[1], note_index))
            note_index += 1

    note_ids = torch.zeros_like(frame_times, dtype=torch.long)
    note_sustain = torch.zeros_like(frame_times, dtype=torch.bool)
    for start, end, index in note_intervals:
        duration = max(end - start, 0.0)
        trim = min(float(sustain_trim_seconds), 0.2 * duration)
        note_ids[(frame_times >= start) & (frame_times <= end)] = index
        if end - trim > start + trim:
            note_sustain |= (frame_times >= start + trim) & (frame_times <= end - trim)

    output = {
        name: _interval_mask(frame_times, intervals)
        for name, intervals in technique_intervals.items()
    }
    output["note_ids"] = note_ids
    output["note_sustain"] = note_sustain
    return output


def _contiguous_runs(mask, note_ids):
    runs = []
    start = None
    previous_id = None
    for index, valid in enumerate(mask):
        current_id = int(note_ids[index]) if note_ids is not None else 1
        continuation = bool(valid) and current_id > 0 and current_id == previous_id
        if bool(valid) and current_id > 0 and start is None:
            start = index
        elif start is not None and not continuation:
            runs.append((start, index))
            start = index if bool(valid) and current_id > 0 else None
        previous_id = current_id if bool(valid) else None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def vibrato_frame_components(
    f0,
    frame_rate,
    valid_mask=None,
    note_ids=None,
    low_hz=4.0,
    high_hz=8.0,
    minimum_note_seconds=0.30,
):
    """Measure local vibrato depth, rate and periodicity on note-sustain runs."""

    f0_np = torch.as_tensor(f0).float().cpu().numpy().reshape(-1)
    valid = np.isfinite(f0_np) & (f0_np > 1.0)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).bool().cpu().numpy().reshape(-1)
    note_np = None
    if note_ids is not None:
        note_np = torch.as_tensor(note_ids).long().cpu().numpy().reshape(-1)

    output = np.zeros((len(VIBRATO_COMPONENT_NAMES), f0_np.size), dtype=np.float32)
    minimum_frames = max(int(round(minimum_note_seconds * frame_rate)), 12)
    nyquist = frame_rate / 2.0
    sos_band = butter(3, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    sos_slow = butter(3, 0.8 / nyquist, btype="highpass", output="sos")

    for start, end in _contiguous_runs(valid, note_np):
        if end - start < minimum_frames:
            continue
        values = 1200.0 * np.log2(np.maximum(f0_np[start:end], 1.0))
        try:
            band = sosfiltfilt(sos_band, values)
            non_dc = sosfiltfilt(sos_slow, values)
        except ValueError:
            continue
        envelope = np.abs(hilbert(band))
        depth = np.clip(envelope, 0.0, 300.0)
        phase = np.unwrap(np.angle(hilbert(band)))
        rate = np.gradient(phase) * frame_rate / (2.0 * np.pi)
        rate = np.clip(rate, low_hz, high_hz)
        band_variance = float(np.var(band))
        residual_variance = max(float(np.var(non_dc)), 1e-6)
        band_ratio = min(band_variance / residual_variance, 1.0)

        spectrum = np.abs(np.fft.rfft(band)) ** 2
        frequencies = np.fft.rfftfreq(band.size, d=1.0 / frame_rate)
        in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
        if np.any(in_band):
            band_spectrum = spectrum[in_band]
            periodicity = float(band_spectrum.max() / max(band_spectrum.sum(), 1e-8))
        else:
            periodicity = 0.0

        output[0, start:end] = depth.astype(np.float32)
        output[1, start:end] = rate.astype(np.float32)
        output[2, start:end] = band_ratio
        output[3, start:end] = periodicity
        output[4, start:end] = (
            depth * math.sqrt(max(band_ratio, 0.0)) * (0.5 + 0.5 * periodicity)
        ).astype(np.float32)

    return torch.from_numpy(output)


def robust_masked_summary(values, mask):
    values = torch.as_tensor(values).float()
    mask = torch.as_tensor(mask).bool()
    selected = values[..., mask]
    if selected.numel() == 0:
        return torch.zeros(values.size(0), dtype=torch.float32)
    return selected.median(dim=-1).values


def vibrato_masked_summary(values, mask):
    """Summarize intermittent vibrato without erasing it with a global median."""

    values = torch.as_tensor(values).float()
    mask = torch.as_tensor(mask).bool() & (values[0] > 0.0)
    selected = values[..., mask]
    if selected.numel() == 0:
        return torch.zeros(values.size(0), dtype=torch.float32)
    output = torch.quantile(selected, 0.75, dim=-1)
    depth_weights = selected[0].clamp_min(1e-6)
    output[1] = (selected[1] * depth_weights).sum() / depth_weights.sum()
    return output


def note_level_vibrato_strength(values, note_ids, mask, quantile=0.75):
    """Assign one robust vibrato strength to every sustain frame of a note."""

    values = torch.as_tensor(values).float()
    note_ids = torch.as_tensor(note_ids).long()
    mask = torch.as_tensor(mask).bool()
    output = torch.zeros_like(values)
    for note_id in torch.unique(note_ids[mask]):
        if int(note_id) <= 0:
            continue
        note_mask = mask & (note_ids == note_id)
        selected = values[note_mask]
        if selected.numel() == 0:
            continue
        output[note_mask] = torch.quantile(selected, float(quantile))
    return output


def register_decision_contour(register_components, confounds, stats):
    register_components = torch.as_tensor(register_components).float()
    confounds = torch.as_tensor(confounds).float()
    residualizer = stats["vocal_register"]["residualizer"]
    residual_intercept = register_components.new_tensor(residualizer["intercept"])
    residual_coefficients = register_components.new_tensor(residualizer["coefficients"])
    residual = register_components - (
        residual_intercept[:, None] + residual_coefficients @ confounds
    )

    scaler = stats["vocal_register"]["scaler"]
    mean = register_components.new_tensor(scaler["mean"])[:, None]
    scale = register_components.new_tensor(scaler["scale"])[:, None].clamp_min(1e-6)
    standardized = (residual - mean) / scale
    classifier = stats["vocal_register"]["classifier"]
    weights = register_components.new_tensor(classifier["coefficients"])
    return classifier["intercept"] + (weights[:, None] * standardized).sum(dim=0)


def compose_multiscale_targets(cache, row, base_stats, multiscale_stats):
    """Compose five normalized targets and their hierarchy-aware gates."""

    components = cache["components"].float()
    old_masks = cache["masks"].bool()
    f0 = cache["f0"].float()
    old_targets = compose_axis_targets(components, old_masks, f0, base_stats)
    frame_times = cache["frame_times"].float()
    annotation = parse_gtsinger_multiscale_masks(
        row["annotation_path"],
        float(row.get("start_sec", 0.0) or 0.0),
        float(row.get("end_sec", row.get("duration_sec", 0.0)) or 0.0),
        frame_times,
    )
    active = old_masks[1] & (f0 > 1.0)
    register_mask = old_masks[2] & annotation["note_sustain"] & active
    if int(register_mask.sum()) < 8:
        register_mask = old_masks[2] & active
    vibrato_mask = active & annotation["note_sustain"] & ~annotation["glissando"]

    register = register_frame_components(cache["log_mel"], components)
    log_f0 = torch.where(
        f0 > 1.0,
        torch.log2(f0.clamp_min(1.0)),
        torch.zeros_like(f0),
    )
    breath = -components[6] - components[7] + components[8] + components[9]
    confounds = torch.stack((log_f0, components[4], breath), dim=0)
    register_raw = register_decision_contour(register, confounds, multiscale_stats)
    register_center = float(multiscale_stats["vocal_register"]["axis"]["median"])
    register_scale = max(
        float(multiscale_stats["vocal_register"]["axis"]["scale"]),
        1e-6,
    )
    register_target = ((register_raw - register_center) / register_scale).clamp(-3.0, 3.0)
    register_target = register_target * register_mask.float()

    vibrato_components = vibrato_frame_components(
        f0,
        frame_rate=float(cache.get("sample_rate", 44100)) / float(cache.get("hop_length", 512)),
        valid_mask=vibrato_mask,
        note_ids=annotation["note_ids"],
    )
    note_strength = note_level_vibrato_strength(
        vibrato_components[4],
        annotation["note_ids"],
        vibrato_mask,
    )
    vibrato_raw = torch.log1p(note_strength.clamp_min(0.0))
    vibrato_center = float(multiscale_stats["vibrato"]["axis"]["median"])
    vibrato_scale = max(float(multiscale_stats["vibrato"]["axis"]["scale"]), 1e-6)
    vibrato_target = ((vibrato_raw - vibrato_center) / vibrato_scale).clamp(-3.0, 3.0)
    vibrato_target = vibrato_target * vibrato_mask.float()

    targets = torch.cat(
        (old_targets, register_target[None], vibrato_target[None]),
        dim=0,
    )
    masks = torch.cat(
        (old_masks, register_mask[None], vibrato_mask[None]),
        dim=0,
    )
    old_gates = cache.get("structure_gates", old_masks.float()).float()
    gates = torch.cat(
        (
            old_gates,
            register_mask.float()[None],
            vibrato_mask.float()[None],
        ),
        dim=0,
    )
    labels = torch.tensor(
        [
            1.0 if row.get("group") == "Falsetto_Group" else 0.0,
            1.0 if row.get("group") == "Vibrato_Group" else 0.0,
        ],
        dtype=torch.float32,
    )
    label_masks = torch.tensor(
        [
            row.get("technique_family") == "Mixed_Voice_and_Falsetto",
            row.get("technique_family") == "Vibrato",
        ],
        dtype=torch.bool,
    )
    return {
        "targets": targets,
        "masks": masks,
        "gates": gates,
        "labels": labels,
        "label_masks": label_masks,
        "vibrato_rate_hz": vibrato_components[1],
        "note_ids": annotation["note_ids"],
    }
