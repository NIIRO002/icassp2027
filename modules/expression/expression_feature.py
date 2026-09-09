import torch
import torch.nn.functional as F


def _safe_std(x, dim, keepdim=True, eps=1e-5):
    return x.std(dim=dim, keepdim=keepdim).clamp_min(eps)


def _resize_time(x, target_len, mode="linear"):
    if x.size(1) == target_len:
        return x
    if mode == "nearest":
        return F.interpolate(x[:, None], size=target_len, mode=mode).squeeze(1)
    return F.interpolate(x[:, None], size=target_len, mode=mode, align_corners=False).squeeze(1)


def _normalize_features(features):
    mean = features.mean(dim=1, keepdim=True)
    std = _safe_std(features, dim=1, keepdim=True)
    return (features - mean) / std


def _frame_energy(wav, hop_length, target_len):
    kernel_size = max(4 * hop_length, hop_length)
    energy = F.avg_pool1d(
        wav[:, None].pow(2),
        kernel_size=kernel_size,
        stride=hop_length,
        padding=kernel_size // 2,
        count_include_pad=False,
    ).squeeze(1).clamp_min(1e-8).sqrt()
    return _resize_time(energy, target_len)


def _local_std(x, width=9):
    pad = width // 2
    mean = F.avg_pool1d(x[:, None], width, stride=1, padding=pad).squeeze(1)
    mean_sq = F.avg_pool1d(x[:, None].pow(2), width, stride=1, padding=pad).squeeze(1)
    return (mean_sq - mean.pow(2)).clamp_min(0.0).sqrt()


def _zero_crossing_rate(x, width=9):
    signs = torch.sign(x)
    crossings = (signs[:, 1:] * signs[:, :-1] < 0).float()
    crossings = F.pad(crossings, (1, 0))
    return F.avg_pool1d(crossings[:, None], width, stride=1, padding=width // 2).squeeze(1)


def _f0_expression(f0, target_len):
    if f0 is None:
        zeros = torch.zeros(1, target_len)
        return zeros, zeros, zeros

    f0 = _resize_time(f0.float(), target_len, mode="nearest")
    voiced = f0 > 1.0
    log_f0 = torch.where(voiced, torch.log(f0.clamp_min(1.0)), torch.zeros_like(f0))
    f0_slope = F.pad(log_f0[:, 1:] - log_f0[:, :-1], (1, 0))
    f0_slope = torch.where(voiced, f0_slope, torch.zeros_like(f0_slope))
    vibrato_depth = _local_std(f0_slope, width=9)
    vibrato_rate = _zero_crossing_rate(f0_slope, width=9)
    return f0_slope, vibrato_rate, vibrato_depth


def _positive_derivative(x):
    delta = F.pad(x[:, 1:] - x[:, :-1], (1, 0))
    return delta.clamp_min(0.0)


def _spectral_proxies(wav, sr, hop_length, target_len):
    n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(float(max(4 * hop_length, 512))))).item())
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav.float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window.float(),
        center=True,
        return_complex=True,
    ).abs()
    mag = spec.clamp_min(1e-8)
    freqs = torch.linspace(0, sr / 2, mag.size(1), device=mag.device)
    high_mask = freqs >= min(6000.0, sr * 0.35)
    high_ratio = mag[:, high_mask].sum(dim=1) / mag.sum(dim=1).clamp_min(1e-8)
    flatness = torch.exp(torch.log(mag).mean(dim=1)) / mag.mean(dim=1).clamp_min(1e-8)
    breathiness = 0.5 * high_ratio + 0.5 * flatness
    centroid = (mag * freqs[None, :, None]).sum(dim=1) / mag.sum(dim=1).clamp_min(1e-8)
    brightness = centroid / max(float(sr) * 0.5, 1.0)
    return _resize_time(breathiness, target_len), _resize_time(brightness, target_len)


def extract_expression_features(wav, sr: int, hop_length: int, f0=None, target_len=None, normalize=True):
    """Extract frame-level singing expression features.

    Feature order:
        [energy, energy_slope, f0_slope, vibrato_rate, vibrato_depth,
         breathiness, brightness, onset_strength].

    Args:
        wav: Tensor shaped [B, samples].
        sr: Audio sample rate.
        hop_length: Model hop size in samples.
        f0: Optional tensor shaped [B, T_f0].
        target_len: Output frame length. If omitted, it is estimated from wav.
        normalize: If true, normalize features per utterance. Use false for
            absolute output analysis.
    Returns:
        Tensor shaped [B, target_len, 8].
    """
    if wav.dim() != 2:
        raise ValueError(f"wav must be [B, samples], got {tuple(wav.shape)}")
    target_len = target_len or max(1, wav.size(-1) // hop_length)

    energy = _frame_energy(wav, hop_length, target_len)
    energy_slope = F.pad(energy[:, 1:] - energy[:, :-1], (1, 0))
    onset_strength = _positive_derivative(energy)
    f0_slope, vibrato_rate, vibrato_depth = _f0_expression(f0, target_len)
    if f0 is None and energy.size(0) != f0_slope.size(0):
        f0_slope = f0_slope.to(energy.device).expand(energy.size(0), -1)
        vibrato_rate = vibrato_rate.to(energy.device).expand(energy.size(0), -1)
        vibrato_depth = vibrato_depth.to(energy.device).expand(energy.size(0), -1)
    breathiness, brightness = _spectral_proxies(wav, sr, hop_length, target_len)

    features = torch.stack(
        [
            energy,
            energy_slope,
            f0_slope.to(energy.device),
            vibrato_rate.to(energy.device),
            vibrato_depth.to(energy.device),
            breathiness,
            brightness,
            onset_strength,
        ],
        dim=-1,
    )
    if normalize:
        return _normalize_features(features)
    return features
