import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.expression.hierarchical_representation import ResidualConvBlock


class MultiscaleRepresentationEncoder(nn.Module):
    """Five-axis encoder with event, frame, note and phrase receptive fields."""

    axis_names = (
        "accent",
        "intensity",
        "breathiness",
        "vocal_register",
        "vibrato",
    )

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
        self.vocal_register_head = nn.Sequential(
            ResidualConvBlock(hidden_dim, dilation=2, dropout=dropout),
            ResidualConvBlock(hidden_dim, dilation=4, dropout=dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self.vibrato_head = nn.Sequential(
            ResidualConvBlock(hidden_dim, dilation=4, dropout=dropout),
            ResidualConvBlock(hidden_dim, dilation=8, dropout=dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self.vibrato_input_projection = nn.Conv1d(
            hidden_dim + 2,
            hidden_dim,
            kernel_size=1,
        )
        with torch.no_grad():
            self.vibrato_input_projection.weight.zero_()
            self.vibrato_input_projection.bias.zero_()
            identity = torch.arange(hidden_dim)
            self.vibrato_input_projection.weight[identity, identity, 0] = 1.0

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
        shared = self.shared(self.input_projection(torch.cat((log_mel, log_f0, voiced), dim=1)))
        accent = self.accent_head(shared)
        intensity = self.intensity_head(
            F.avg_pool1d(shared, kernel_size=31, stride=1, padding=15)
        )
        breathiness = self.breathiness_head(
            F.avg_pool1d(shared, kernel_size=9, stride=1, padding=4)
        )
        vocal_register = self.vocal_register_head(
            F.avg_pool1d(shared, kernel_size=15, stride=1, padding=7)
        )
        cents = torch.where(
            voiced.bool(),
            1200.0 * torch.log2(f0.clamp_min(1.0)),
            torch.zeros_like(f0),
        )
        fast = F.avg_pool1d(cents, kernel_size=5, stride=1, padding=2)
        slow = F.avg_pool1d(cents, kernel_size=21, stride=1, padding=10)
        f0_band = ((fast - slow) / 100.0).clamp(-3.0, 3.0) * voiced
        local_depth = torch.sqrt(
            F.avg_pool1d(
                f0_band.square(),
                kernel_size=15,
                stride=1,
                padding=7,
            ).clamp_min(1e-8)
        ) * voiced
        vibrato_input = self.vibrato_input_projection(
            torch.cat((shared, f0_band, local_depth), dim=1)
        )
        vibrato = self.vibrato_head(
            F.avg_pool1d(vibrato_input, kernel_size=15, stride=1, padding=7)
        )
        return torch.cat(
            (accent, intensity, breathiness, vocal_register, vibrato),
            dim=1,
        )

    def load_three_axis_checkpoint(self, checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        incompatible = self.load_state_dict(state["representation_encoder"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        allowed_prefixes = (
            "vocal_register_head.",
            "vibrato_head.",
            "vibrato_input_projection.",
        )
        disallowed_missing = [
            key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                f"Incompatible 3-axis checkpoint: missing={disallowed_missing} unexpected={unexpected}"
            )
        return state

    def freeze_pretrained_three_axis(self):
        modules = (
            self.input_projection,
            self.shared,
            self.accent_head,
            self.intensity_head,
            self.breathiness_head,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train_vibrato_only(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.vibrato_input_projection, self.vibrato_head):
            for parameter in module.parameters():
                parameter.requires_grad = True


def masked_multiscale_losses(prediction, target, masks, correlation_weight=0.1):
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


def segment_technique_loss(prediction, masks, labels, label_masks):
    pooled = []
    for axis in (3, 4):
        weights = masks[:, axis].float()
        pooled.append(
            (prediction[:, axis] * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        )
    pooled = torch.stack(pooled, dim=1)
    valid_values = pooled[label_masks]
    valid_labels = labels[label_masks].to(dtype=pooled.dtype)
    if valid_values.numel() == 0:
        return pooled.new_tensor(0.0), pooled
    return F.binary_cross_entropy_with_logits(valid_values, valid_labels), pooled
