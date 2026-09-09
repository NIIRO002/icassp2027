import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.05):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(F.silu(self.norm(x)))
        x = self.dropout(x)
        x = self.conv2(F.silu(x))
        return residual + x


class HierarchicalRepresentationEncoder(nn.Module):
    """Predict accent, intensity, and breathiness contours from mel and F0."""

    def __init__(self, n_mels=128, hidden_dim=96, dropout=0.05):
        super().__init__()
        self.n_mels = n_mels
        self.input_projection = nn.Conv1d(n_mels + 2, hidden_dim, kernel_size=1)
        self.shared = nn.Sequential(
            ResidualConvBlock(hidden_dim, dilation=1, dropout=dropout),
            ResidualConvBlock(hidden_dim, dilation=2, dropout=dropout),
            ResidualConvBlock(hidden_dim, dilation=4, dropout=dropout),
        )
        self.accent_head = nn.Sequential(
            ResidualConvBlock(hidden_dim, dilation=1, dropout=dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self.intensity_head = nn.Sequential(
            ResidualConvBlock(hidden_dim, dilation=4, dropout=dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self.breathiness_head = nn.Sequential(
            ResidualConvBlock(hidden_dim, dilation=2, dropout=dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, log_mel, f0):
        if log_mel.dim() != 3:
            raise ValueError(f"Expected log_mel [B,M,T], got {tuple(log_mel.shape)}")
        if f0.dim() == 2:
            f0 = f0[:, None]
        if f0.size(-1) != log_mel.size(-1):
            f0 = F.interpolate(f0, size=log_mel.size(-1), mode="nearest")

        voiced = (f0 > 0).to(dtype=log_mel.dtype)
        log_f0 = torch.where(
            voiced.bool(),
            torch.log2(f0.clamp_min(1.0)) / 10.0,
            torch.zeros_like(f0),
        )
        x = torch.cat([log_mel, log_f0, voiced], dim=1)
        x = self.shared(self.input_projection(x))

        accent = self.accent_head(x)
        intensity_context = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
        intensity = self.intensity_head(intensity_context)
        breath_context = F.avg_pool1d(x, kernel_size=9, stride=1, padding=4)
        breathiness = self.breathiness_head(breath_context)
        return torch.cat([accent, intensity, breathiness], dim=1)


def masked_axis_losses(prediction, target, masks, correlation_weight=0.1):
    losses = []
    correlations = []
    for axis in range(prediction.size(1)):
        mask = masks[:, axis].bool()
        pred_values = prediction[:, axis][mask]
        target_values = target[:, axis][mask]
        if pred_values.numel() < 2:
            losses.append(prediction.new_tensor(0.0))
            correlations.append(prediction.new_tensor(0.0))
            continue

        regression = F.smooth_l1_loss(pred_values, target_values)
        pred_centered = pred_values - pred_values.mean()
        target_centered = target_values - target_values.mean()
        correlation = (
            (pred_centered * target_centered).mean()
            / (
                pred_centered.square().mean().sqrt()
                * target_centered.square().mean().sqrt()
            ).clamp_min(1e-6)
        ).clamp(-1.0, 1.0)
        losses.append(regression + correlation_weight * (1.0 - correlation))
        correlations.append(correlation)
    return torch.stack(losses), torch.stack(correlations)
