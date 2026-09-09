import json
import math
from pathlib import Path

import torch


def load_onset_times(path):
    """Load syllable onset times from the shared landmark JSON format."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    onset_times = payload.get(
        "onset_times_seconds",
        payload.get("onset_times"),
    )
    if not isinstance(onset_times, list):
        raise ValueError("accent onset JSON must contain onset_times_seconds")
    return payload, [float(value) for value in onset_times]


def onset_activity(
    length,
    sample_rate,
    onset_times,
    attack_seconds=0.07,
    fade_seconds=0.008,
    device=None,
    dtype=torch.float32,
):
    """Build a smooth waveform-rate mask over short syllable attacks."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if attack_seconds <= 0 or fade_seconds < 0:
        raise ValueError("attack_seconds must be positive and fade_seconds non-negative")

    activity = torch.zeros(length, device=device, dtype=dtype)
    fade = max(round(fade_seconds * sample_rate), 1)
    attack = max(round(attack_seconds * sample_rate), 2 * fade + 1)
    for onset in onset_times:
        start = max(round(float(onset) * sample_rate), 0)
        end = min(start + attack, length)
        width = end - start
        if width <= 0:
            continue
        shape = torch.ones(width, device=device, dtype=dtype)
        edge = min(fade, width // 2)
        if edge > 0:
            ramp = torch.linspace(
                0.0,
                1.0,
                edge + 2,
                device=device,
                dtype=dtype,
            )[1:-1]
            shape[:edge] = ramp
            shape[-edge:] = ramp.flip(0)
        activity[start:end] = torch.maximum(activity[start:end], shape)
    return activity


def _rms(waveform):
    return waveform.float().square().mean().sqrt()


def _peak_safe_rms_match(waveform, target_rms, peak_limit):
    current_rms = _rms(waveform)
    if float(current_rms) <= 1e-10:
        return waveform.clone(), False

    linear = waveform * (target_rms / current_rms).to(waveform)
    if float(linear.abs().max()) <= peak_limit:
        return linear, False

    def limited(scale):
        return peak_limit * torch.tanh(waveform * scale / peak_limit)

    low = 0.0
    high = max(float(target_rms / current_rms), 1.0)
    for _ in range(14):
        if float(_rms(limited(high))) >= float(target_rms) or high >= 1e4:
            break
        high *= 2.0
    for _ in range(48):
        middle = 0.5 * (low + high)
        if float(_rms(limited(middle))) < float(target_rms):
            low = middle
        else:
            high = middle
    return limited(high), True


def apply_onset_accent(
    waveform,
    sample_rate,
    onset_times,
    control,
    max_gain_db=3.0,
    attack_seconds=0.07,
    fade_seconds=0.008,
    preserve_rms=True,
    peak_limit=0.999,
    return_details=False,
):
    """Apply normalized signed accent control only around syllable onsets.

    ``control`` uses the interpretable range [-1, 1]. Positive values emphasize
    attacks and negative values de-emphasize them. A zero value is an exact
    identity operation.
    """
    if waveform.ndim not in (1, 2):
        raise ValueError("waveform must have shape [T] or [C, T]")
    if max_gain_db < 0:
        raise ValueError("max_gain_db must be non-negative")
    if not 0 < peak_limit <= 1:
        raise ValueError("peak_limit must be in (0, 1]")
    control = float(control)
    if not math.isfinite(control) or abs(control) > 1.0:
        raise ValueError("DSP accent control must be finite and in [-1, 1]")

    gain_db = control * float(max_gain_db)
    if control == 0.0 or waveform.numel() == 0:
        output = waveform.clone()
        rms_value = float(_rms(waveform)) if waveform.numel() else 0.0
        details = {
            "control": control,
            "gain_db": gain_db,
            "activity_coverage": 0.0,
            "rms_before": rms_value,
            "rms_after": rms_value,
            "peak_after": float(waveform.abs().max()) if waveform.numel() else 0.0,
            "limited": False,
        }
        return (output, details) if return_details else output

    activity = onset_activity(
        waveform.size(-1),
        sample_rate,
        onset_times,
        attack_seconds=attack_seconds,
        fade_seconds=fade_seconds,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    multiplier = torch.pow(
        waveform.new_tensor(10.0),
        activity * (gain_db / 20.0),
    )
    output = waveform * multiplier
    rms_before = _rms(waveform)
    limited = False
    if preserve_rms:
        output, limited = _peak_safe_rms_match(output, rms_before, peak_limit)
    elif float(output.abs().max()) > peak_limit:
        output = output * (peak_limit / output.abs().max())
        limited = True

    details = {
        "control": control,
        "gain_db": gain_db,
        "activity_coverage": float((activity > 0).float().mean()),
        "rms_before": float(rms_before),
        "rms_after": float(_rms(output)),
        "peak_after": float(output.abs().max()),
        "limited": limited,
    }
    return (output, details) if return_details else output
