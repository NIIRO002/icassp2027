import torch
import torch.nn as nn
import torch.nn.functional as F


class AxisResidualBranch(nn.Module):
    def __init__(self, cond_dim, bottleneck, style_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, cond_dim),
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
        if self.style_projection is not None:
            if style is None:
                raise ValueError("Style embedding is required by this adapter")
            hidden = hidden + self.style_projection(style)[:, None, :]
        hidden = self.net[1](hidden)
        return torch.tanh(self.net[2](hidden))


class HierarchicalExpressionAdapter(nn.Module):
    """Bounded additive adapter with independently controlled expression axes."""

    axis_names = ("accent", "intensity", "breathiness")

    def __init__(
        self,
        cond_dim=768,
        bottleneck=64,
        residual_scale=0.15,
        axis_residual_scales=None,
        combined_residual_scale=None,
        style_dim=None,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.bottleneck = bottleneck
        self.residual_scale = residual_scale
        self.axis_residual_scales = (
            tuple(float(value) for value in axis_residual_scales)
            if axis_residual_scales is not None
            else None
        )
        if (
            self.axis_residual_scales is not None
            and len(self.axis_residual_scales) != len(self.axis_names)
        ):
            raise ValueError("axis_residual_scales must contain three values")
        self.combined_residual_scale = (
            float(combined_residual_scale)
            if combined_residual_scale is not None
            else None
        )
        self.style_dim = style_dim
        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)
        self.branches = nn.ModuleList(
            [
                AxisResidualBranch(cond_dim, bottleneck, style_dim=style_dim)
                for _ in self.axis_names
            ]
        )

    def forward(
        self,
        condition,
        controls,
        gates=None,
        style=None,
        strength=1.0,
        return_details=False,
    ):
        if controls.dim() == 1:
            controls = controls[None]
        controls = controls.to(device=condition.device, dtype=condition.dtype)
        if controls.size(-1) != len(self.axis_names):
            raise ValueError(f"Expected three controls, got {tuple(controls.shape)}")

        batch, frames, _ = condition.shape
        if gates is None:
            gates = condition.new_ones(batch, len(self.axis_names), frames)
        elif gates.dim() != 3:
            raise ValueError(f"Expected gates [B,3,T], got {tuple(gates.shape)}")
        gates = gates.to(device=condition.device, dtype=condition.dtype)
        if gates.size(-1) != frames:
            gates = F.interpolate(gates, size=frames, mode="nearest")

        normalized = self.norm(condition)
        directions = torch.stack(
            [branch(normalized, style=style) for branch in self.branches],
            dim=1,
        )
        weights = controls[:, :, None, None] * gates[:, :, :, None]
        if self.axis_residual_scales is None:
            mixed = (directions * weights).sum(dim=1)
            axis_deltas = None
            delta = self.residual_scale * torch.tanh(float(strength) * mixed)
        else:
            scales = condition.new_tensor(self.axis_residual_scales)
            axis_deltas = (
                scales[None, :, None, None]
                * torch.tanh(float(strength) * directions * weights)
            )
            delta = axis_deltas.sum(dim=1)
            if self.combined_residual_scale is not None:
                cap = max(self.combined_residual_scale, 1e-8)
                denominator = torch.maximum(
                    torch.ones_like(delta),
                    delta.abs() / cap,
                )
                delta = delta / denominator
        output = condition + delta
        if return_details:
            return output, {
                "delta": delta,
                "axis_deltas": axis_deltas,
                "directions": directions,
                "weights": weights,
                "gates": gates,
            }
        return output


def axis_direction_orthogonality_loss(directions):
    """Penalize cosine similarity between three [B,T,D] branch directions."""

    flattened = directions.float().permute(1, 0, 2, 3).flatten(1)
    flattened = F.normalize(flattened, dim=1)
    similarity = flattened @ flattened.transpose(0, 1)
    off_diagonal = ~torch.eye(
        similarity.size(0),
        device=similarity.device,
        dtype=torch.bool,
    )
    if not off_diagonal.any():
        return similarity.new_tensor(0.0)
    return similarity[off_diagonal].square().mean()
