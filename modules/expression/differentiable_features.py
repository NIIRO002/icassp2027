import math

import torch
import torch.nn.functional as F


AXIS_NAMES = ("accent", "intensity", "breathiness")


def _moving_average(values, window):
    if window <= 1:
        return values
    padding = window // 2
    padded = F.pad(
        values[:, None],
        (padding, window - 1 - padding),
        mode="replicate",
    )
    return F.avg_pool1d(padded, kernel_size=window, stride=1)[:, 0]


def _causal_history_average(values, window):
    if window <= 1:
        return values
    padded = F.pad(values[:, None], (window, 0), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1)[:, 0, :-1]


def _forward_window_average(values, start, end):
    if not 0 <= start < end:
        raise ValueError(f"Invalid forward window: {start}:{end}")
    padded = F.pad(
        values[:, None],
        (0, end - 1),
        mode="replicate",
    )
    windows = padded.unfold(dimension=2, size=end, step=1)[:, 0]
    return windows[..., start:end].mean(dim=-1)


def differentiable_mel_axes(
    log_mel,
    *,
    accent_gates=None,
    accent_history_frames=12,
    accent_smoothing_frames=5,
    accent_attack_frames=6,
    accent_body_start_frames=8,
    accent_body_end_frames=22,
    intensity_smoothing_frames=21,
    high_frequency_fraction=0.35,
    axis_scales=(6.0, 6.0, 6.0),
):
    """Compute gain-aware expression axes directly from log-magnitude mel.

    The returned axes are differentiable with respect to ``log_mel`` and use
    approximately six-decibel units. Constant gain changes only intensity;
    accent measures onset-local contrast and breathiness uses gain-invariant
    high-frequency texture.
    """

    if log_mel.ndim != 3:
        raise ValueError(f"Expected [B, M, T] log-mel, got {log_mel.shape}")
    if not 0.0 < high_frequency_fraction < 1.0:
        raise ValueError("high_frequency_fraction must be between zero and one")

    log_power = 2.0 * log_mel.float()
    log_total_power = torch.logsumexp(log_power, dim=1)
    energy_db = (10.0 / math.log(10.0)) * log_total_power

    if accent_gates is None:
        history = _causal_history_average(energy_db, accent_history_frames)
        accent_db = _moving_average(
            energy_db - history,
            accent_smoothing_frames,
        )
    else:
        if accent_gates.shape != energy_db.shape:
            raise ValueError(
                f"accent_gates must be {energy_db.shape}, "
                f"got {accent_gates.shape}"
            )
        attack = _forward_window_average(
            energy_db,
            0,
            accent_attack_frames,
        )
        body = _forward_window_average(
            energy_db,
            accent_body_start_frames,
            accent_body_end_frames,
        )
        accent_db = attack - body
    intensity_db = _moving_average(energy_db, intensity_smoothing_frames)

    high_bins = max(
        2,
        int(round(log_mel.size(1) * high_frequency_fraction)),
    )
    high_log_power = log_power[:, -high_bins:]
    high_log_arithmetic_mean = (
        torch.logsumexp(high_log_power, dim=1) - math.log(high_bins)
    )
    high_log_geometric_mean = high_log_power.mean(dim=1)
    flatness_db = (
        10.0
        / math.log(10.0)
        * (high_log_geometric_mean - high_log_arithmetic_mean)
    )
    high_ratio_db = (
        10.0
        / math.log(10.0)
        * (torch.logsumexp(high_log_power, dim=1) - log_total_power)
    )
    breathiness_db = 0.6 * flatness_db + 0.4 * high_ratio_db

    scales = log_mel.new_tensor(axis_scales).float().clamp_min(1e-4)
    axes = torch.stack(
        (accent_db, intensity_db, breathiness_db),
        dim=1,
    )
    return axes / scales[None, :, None]


def differentiable_onset_effort_features(
    log_mel,
    *,
    attack_frames=6,
    body_start_frames=8,
    body_end_frames=22,
    low_band_end_fraction=0.42,
    high_band_start_fraction=0.48,
    high_band_end_fraction=0.82,
    feature_scales=(6.0, 3.0, 3.0),
):
    """Return differentiable local phonation cues in normalized dB units.

    Components are attack/body energy contrast, attack spectral balance, and
    attack harmonic concentration. The latter two are invariant to constant
    gain, so an accent branch cannot satisfy every target through loudness.
    """
    if log_mel.ndim != 3:
        raise ValueError(f"Expected [B, M, T] log-mel, got {log_mel.shape}")
    if not (
        0.0 < low_band_end_fraction
        < high_band_start_fraction
        < high_band_end_fraction
        <= 1.0
    ):
        raise ValueError("Invalid onset-effort mel-band fractions")

    log_power = 2.0 * log_mel.float()
    mel_bins = log_mel.size(1)
    low_end = max(2, round(mel_bins * low_band_end_fraction))
    high_start = max(low_end + 1, round(mel_bins * high_band_start_fraction))
    high_end = max(high_start + 2, round(mel_bins * high_band_end_fraction))
    high_end = min(high_end, mel_bins)

    def log_mean_power(values):
        return torch.logsumexp(values, dim=1) - math.log(values.size(1))

    log_total_power = torch.logsumexp(log_power, dim=1)
    energy_db = (10.0 / math.log(10.0)) * log_total_power
    attack_energy = _forward_window_average(energy_db, 0, attack_frames)
    body_energy = _forward_window_average(
        energy_db,
        body_start_frames,
        body_end_frames,
    )
    attack_contrast_db = attack_energy - body_energy

    low_log_mean = log_mean_power(log_power[:, :low_end])
    high_log_mean = log_mean_power(log_power[:, high_start:high_end])
    spectral_balance_db = (10.0 / math.log(10.0)) * (
        high_log_mean - low_log_mean
    )
    attack_spectral_balance_db = _forward_window_average(
        spectral_balance_db,
        0,
        attack_frames,
    )

    harmonic_band = log_power[:, :high_end]
    harmonic_log_arithmetic = log_mean_power(harmonic_band)
    harmonic_log_geometric = harmonic_band.mean(dim=1)
    harmonic_concentration_db = (10.0 / math.log(10.0)) * (
        harmonic_log_arithmetic - harmonic_log_geometric
    )
    attack_harmonic_concentration_db = _forward_window_average(
        harmonic_concentration_db,
        0,
        attack_frames,
    )

    scales = log_mel.new_tensor(feature_scales).float().clamp_min(1e-4)
    features = torch.stack(
        (
            attack_contrast_db,
            attack_spectral_balance_db,
            attack_harmonic_concentration_db,
        ),
        dim=1,
    )
    return features / scales[None, :, None]


def differentiable_duration_accent_features(
    log_mel,
    *,
    attack_frames=6,
    body_start_frames=8,
    body_end_frames=22,
    nucleus_smoothing_frames=7,
    sustain_smoothing_frames=13,
    low_band_end_fraction=0.42,
    high_band_start_fraction=0.48,
    high_band_end_fraction=0.82,
    feature_scales=(6.0, 3.0, 6.0),
):
    """Return attack, vowel-nucleus, and sustained-effort mel contours."""
    if log_mel.ndim != 3:
        raise ValueError(f"Expected [B, M, T] log-mel, got {log_mel.shape}")
    log_power = 2.0 * log_mel.float()
    mel_bins = log_mel.size(1)
    low_end = max(2, round(mel_bins * low_band_end_fraction))
    high_start = max(low_end + 1, round(mel_bins * high_band_start_fraction))
    high_end = min(
        max(high_start + 2, round(mel_bins * high_band_end_fraction)),
        mel_bins,
    )

    def log_mean_power(values):
        return torch.logsumexp(values, dim=1) - math.log(values.size(1))

    log_total_power = torch.logsumexp(log_power, dim=1)
    energy_db = (10.0 / math.log(10.0)) * log_total_power
    attack_energy = _forward_window_average(energy_db, 0, attack_frames)
    body_energy = _forward_window_average(
        energy_db,
        body_start_frames,
        body_end_frames,
    )
    attack_contrast_db = attack_energy - body_energy

    low_log_mean = log_mean_power(log_power[:, :low_end])
    high_log_mean = log_mean_power(log_power[:, high_start:high_end])
    spectral_balance_db = (10.0 / math.log(10.0)) * (
        high_log_mean - low_log_mean
    )
    nucleus_effort_db = _moving_average(
        spectral_balance_db,
        nucleus_smoothing_frames,
    )
    sustain_energy_db = _moving_average(energy_db, sustain_smoothing_frames)
    sustain_effort_db = 0.70 * sustain_energy_db + 0.30 * spectral_balance_db

    scales = log_mel.new_tensor(feature_scales).float().clamp_min(1e-4)
    features = torch.stack(
        (attack_contrast_db, nucleus_effort_db, sustain_effort_db),
        dim=1,
    )
    return features / scales[None, :, None]
