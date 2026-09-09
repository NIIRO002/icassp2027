import math

import torch
import torch.nn.functional as F


DB_TO_LOG_AMPLITUDE = math.log(10.0) / 20.0
LOG_AMPLITUDE_TO_DB = 20.0 / math.log(10.0)
EFFORT_FEATURE_NAMES = ("intensity_db", "spectral_tilt_db", "accent_db")


def _control_tensor(control, ref):
    if not torch.is_tensor(control):
        control = torch.tensor(control, device=ref.device, dtype=ref.dtype)
    control = control.to(device=ref.device, dtype=ref.dtype)
    if control.dim() == 0:
        control = control[None]
    return control.reshape(-1, 1, 1)


def mel_log_energy(log_mel):
    log_mel = log_mel.float()
    return 0.5 * (
        torch.logsumexp(2.0 * log_mel, dim=1)
        - math.log(max(log_mel.size(1), 1))
    )


def mel_active_mask(log_mel, quantile=0.35):
    energy = mel_log_energy(log_mel).detach()
    threshold = torch.quantile(energy, quantile, dim=1, keepdim=True)
    return energy >= threshold


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _masked_rms(values, mask):
    mask = mask.to(dtype=values.dtype)
    mean_square = (values.square() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return mean_square.clamp_min(1e-10).sqrt()


def mel_effort_features(log_mel, active_mask=None):
    """Differentiable [intensity, tilt, accent] descriptors in dB-like units."""

    log_mel = log_mel.float()
    active_mask = mel_active_mask(log_mel) if active_mask is None else active_mask
    energy = mel_log_energy(log_mel)
    n_mels = log_mel.size(1)
    low = log_mel[:, : max(n_mels // 3, 1)].mean(dim=1)
    high = log_mel[:, max(2 * n_mels // 3, 1) :].mean(dim=1)
    tilt = high - low

    energy_diff = energy[:, 1:] - energy[:, :-1]
    accent_mask = active_mask[:, 1:] & active_mask[:, :-1]

    intensity_db = _masked_mean(energy, active_mask) * LOG_AMPLITUDE_TO_DB
    tilt_db = _masked_mean(tilt, active_mask) * LOG_AMPLITUDE_TO_DB
    accent_db = _masked_rms(energy_diff, accent_mask) * LOG_AMPLITUDE_TO_DB
    return torch.stack([intensity_db, tilt_db, accent_db], dim=-1)


def apply_vocal_effort_transform(
    log_mel,
    control,
    intensity_db=5.0,
    tilt_db=3.0,
    accent_db=2.0,
    active_mask=None,
):
    """Construct a mel-domain pseudo target for stronger or softer vocal effort."""

    base = log_mel.float()
    control = _control_tensor(control, base)
    active_mask = mel_active_mask(base) if active_mask is None else active_mask
    n_mels = base.size(1)

    frequency_axis = torch.linspace(-1.0, 1.0, n_mels, device=base.device, dtype=base.dtype)
    frequency_axis = frequency_axis[None, :, None]

    energy = mel_log_energy(base)
    padded_energy = F.pad(energy[:, None], (4, 4), mode="replicate")
    local_trend = F.avg_pool1d(padded_energy, kernel_size=9, stride=1).squeeze(1)
    dynamics = energy - local_trend
    dynamics_scale = torch.quantile(dynamics.detach().abs(), 0.90, dim=1, keepdim=True).clamp_min(1e-4)
    dynamics_profile = (dynamics / dynamics_scale).clamp(-1.0, 1.0)
    dynamics_profile = dynamics_profile * active_mask.to(dtype=base.dtype)
    dynamics_profile = dynamics_profile - _masked_mean(dynamics_profile, active_mask)[:, None]

    intensity_shift = control * (intensity_db * DB_TO_LOG_AMPLITUDE)
    tilt_shift = control * (tilt_db * DB_TO_LOG_AMPLITUDE) * frequency_axis
    accent_shift = (
        control
        * (accent_db * DB_TO_LOG_AMPLITUDE)
        * dynamics_profile[:, None, :]
    )
    return base + intensity_shift + tilt_shift + accent_shift


def vocal_effort_feature_loss(
    pred_mel,
    reference_pred_mel,
    target_mel,
    base_mel,
    control,
    active_mask=None,
):
    active_mask = mel_active_mask(base_mel) if active_mask is None else active_mask
    pred = mel_effort_features(pred_mel, active_mask)
    reference_pred = mel_effort_features(reference_pred_mel, active_mask).detach()
    target = mel_effort_features(target_mel, active_mask).detach()
    base = mel_effort_features(base_mel, active_mask).detach()

    scales = pred.new_tensor([4.0, 4.0, 0.75])
    predicted_delta = (pred - reference_pred) / scales
    target_delta = (target - base) / scales
    regression = F.smooth_l1_loss(predicted_delta, target_delta)

    signed_control = _control_tensor(control, pred).reshape(-1, 1)
    target_direction = torch.sign(target_delta)
    direction = F.relu(0.05 * signed_control.abs() - target_direction * predicted_delta).mean()
    return regression, direction, pred, reference_pred, target, base
