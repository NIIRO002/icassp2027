"""Controller for expression-specific temporal support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.expression.hierarchical_adapter import AxisResidualBranch


def contiguous_segment_mean(condition, support, threshold=0.5):
    """Broadcast each contiguous active segment's mean to its frames."""

    if condition.dim() != 3 or support.dim() != 2:
        raise ValueError(
            f"Expected condition [B,T,D] and support [B,T], got "
            f"{tuple(condition.shape)} and {tuple(support.shape)}"
        )
    if condition.shape[:2] != support.shape:
        raise ValueError("Condition and phrase support shapes do not align")

    active = support > float(threshold)
    previous = F.pad(active[:, :-1], (1, 0), value=False)
    starts = active & ~previous
    segment_ids = starts.long().cumsum(dim=1) * active.long()
    batch, frames, channels = condition.shape
    sums = condition.new_zeros(batch, frames + 1, channels)
    sums.scatter_add_(
        1,
        segment_ids[:, :, None].expand(-1, -1, channels),
        condition * active[:, :, None].to(condition),
    )
    counts = condition.new_zeros(batch, frames + 1)
    counts.scatter_add_(1, segment_ids, active.to(condition))
    means = sums / counts.clamp_min(1.0)[:, :, None]
    pooled = means.gather(
        1, segment_ids[:, :, None].expand(-1, -1, channels)
    )
    return pooled * active[:, :, None].to(condition)


class LocalTemporalBreathinessBranch(nn.Module):
    """A shallow depthwise Conv1D controller with a three-frame RF."""

    def __init__(self, cond_dim, bottleneck, style_dim=None, kernel_size=3):
        super().__init__()
        if int(kernel_size) != 3:
            raise ValueError("breathiness kernel_size must be 3")
        self.kernel_size = int(kernel_size)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, cond_dim),
        )
        self.local_conv = nn.Conv1d(
            bottleneck,
            bottleneck,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=bottleneck,
        )
        self.style_projection = (
            nn.Linear(style_dim, bottleneck, bias=False)
            if style_dim is not None
            else None
        )
        if self.style_projection is not None:
            nn.init.zeros_(self.style_projection.weight)

    def forward(self, condition, style=None):
        hidden = self.net[0](condition)
        hidden = self.local_conv(hidden.transpose(1, 2)).transpose(1, 2)
        if self.style_projection is not None:
            if style is None:
                raise ValueError("Style embedding is required by this adapter")
            hidden = hidden + self.style_projection(style)[:, None, :]
        hidden = self.net[1](hidden)
        return torch.tanh(self.net[2](hidden))


class PhraseContextIntensityBranch(nn.Module):
    """Fuse local conditioning with each active segment's mean."""

    def __init__(self, cond_dim, bottleneck, style_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, cond_dim),
        )
        self.phrase_projection = nn.Linear(cond_dim, bottleneck, bias=False)
        self.style_projection = (
            nn.Linear(style_dim, bottleneck, bias=False)
            if style_dim is not None
            else None
        )
        if self.style_projection is not None:
            nn.init.zeros_(self.style_projection.weight)

    def forward(self, condition, phrase_support, style=None):
        phrase_context = contiguous_segment_mean(condition, phrase_support)
        hidden = self.net[0](condition) + self.phrase_projection(phrase_context)
        if self.style_projection is not None:
            if style is None:
                raise ValueError("Style embedding is required by this adapter")
            hidden = hidden + self.style_projection(style)[:, None, :]
        hidden = self.net[1](hidden)
        return torch.tanh(self.net[2](hidden))


class TemporalContextExpressionAdapter(nn.Module):
    """Phrase-level intensity and local-frame breathiness controller."""

    axis_names = ("accent", "intensity", "breathiness")

    def __init__(
        self,
        cond_dim=768,
        bottleneck=64,
        residual_scale=0.15,
        axis_residual_scales=None,
        combined_residual_scale=None,
        style_dim=None,
        breathiness_kernel_size=3,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.bottleneck = int(bottleneck)
        self.residual_scale = float(residual_scale)
        self.axis_residual_scales = (
            tuple(float(value) for value in axis_residual_scales)
            if axis_residual_scales is not None
            else None
        )
        if self.axis_residual_scales is not None and len(self.axis_residual_scales) != 3:
            raise ValueError("axis_residual_scales must contain three values")
        self.combined_residual_scale = (
            float(combined_residual_scale)
            if combined_residual_scale is not None
            else None
        )
        self.style_dim = style_dim
        self.breathiness_kernel_size = int(breathiness_kernel_size)
        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)
        self.branches = nn.ModuleList(
            [
                AxisResidualBranch(cond_dim, bottleneck, style_dim=style_dim),
                PhraseContextIntensityBranch(cond_dim, bottleneck, style_dim=style_dim),
                LocalTemporalBreathinessBranch(
                    cond_dim,
                    bottleneck,
                    style_dim=style_dim,
                    kernel_size=self.breathiness_kernel_size,
                ),
            ]
        )

    def forward(
        self,
        condition,
        controls,
        gates=None,
        context_gates=None,
        style=None,
        strength=1.0,
        return_details=False,
    ):
        if controls.dim() == 1:
            controls = controls[None]
        controls = controls.to(device=condition.device, dtype=condition.dtype)
        if controls.size(-1) != 3:
            raise ValueError(f"Expected three controls, got {tuple(controls.shape)}")

        batch, frames, _ = condition.shape
        if gates is None:
            gates = condition.new_ones(batch, 3, frames)
        elif gates.dim() != 3:
            raise ValueError(f"Expected gates [B,3,T], got {tuple(gates.shape)}")
        gates = gates.to(device=condition.device, dtype=condition.dtype)
        if gates.size(-1) != frames:
            gates = F.interpolate(gates, size=frames, mode="nearest")

        if context_gates is None:
            context_gates = gates
        elif context_gates.dim() != 3:
            raise ValueError(
                f"Expected context_gates [B,3,T], got {tuple(context_gates.shape)}"
            )
        context_gates = context_gates.to(device=condition.device, dtype=condition.dtype)
        if context_gates.size(-1) != frames:
            context_gates = F.interpolate(context_gates, size=frames, mode="nearest")

        normalized = self.norm(condition)
        directions = torch.stack(
            [
                self.branches[0](normalized, style=style),
                self.branches[1](
                    normalized,
                    phrase_support=context_gates[:, 1],
                    style=style,
                ),
                self.branches[2](normalized, style=style),
            ],
            dim=1,
        )
        weights = controls[:, :, None, None] * gates[:, :, :, None]
        if self.axis_residual_scales is None:
            mixed = (directions * weights).sum(dim=1)
            axis_deltas = None
            delta = self.residual_scale * torch.tanh(float(strength) * mixed)
        else:
            scales = condition.new_tensor(self.axis_residual_scales)
            axis_deltas = scales[None, :, None, None] * torch.tanh(
                float(strength) * directions * weights
            )
            delta = axis_deltas.sum(dim=1)
            if self.combined_residual_scale is not None:
                cap = max(self.combined_residual_scale, 1e-8)
                delta = delta / torch.maximum(torch.ones_like(delta), delta.abs() / cap)
        output = condition + delta
        if return_details:
            return output, {
                "delta": delta,
                "axis_deltas": axis_deltas,
                "directions": directions,
                "weights": weights,
                "gates": gates,
                "context_gates": context_gates,
            }
        return output
