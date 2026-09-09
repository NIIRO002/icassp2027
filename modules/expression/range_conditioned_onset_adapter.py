import math

import torch
import torch.nn as nn

from .hierarchical_adapter import AxisResidualBranch


def _voiced_log2_median(f0, fallback_hz=220.0):
    if f0.dim() == 1:
        f0 = f0[None]
    if f0.dim() != 2:
        raise ValueError(f"Expected F0 [B,T], got {tuple(f0.shape)}")
    values = []
    fallback = math.log2(float(fallback_hz))
    for row in f0.float():
        voiced = row[row > 1.0]
        values.append(
            torch.log2(voiced).median()
            if voiced.numel()
            else row.new_tensor(fallback)
        )
    return torch.stack(values)


def pitch_range_features(source_f0, reference_f0, center_hz=220.0):
    """Summarize source/reference pitch range as three bounded features."""
    source = _voiced_log2_median(source_f0, fallback_hz=center_hz)
    reference = _voiced_log2_median(reference_f0, fallback_hz=center_hz)
    center = math.log2(float(center_hz))
    source_centered = ((source - center) / 2.0).clamp(-1.5, 1.5)
    reference_centered = ((reference - center) / 2.0).clamp(-1.5, 1.5)
    difference = ((reference - source) / 2.0).clamp(-1.5, 1.5)
    return torch.stack(
        (source_centered, reference_centered, difference),
        dim=-1,
    )


class RangeConditionedOnsetAdapter(nn.Module):
    """Energy anchor with a bounded pitch-range-conditioned spectral branch."""

    component_names = ("attack_energy", "spectral_balance")

    def __init__(
        self,
        cond_dim=768,
        bottleneck=64,
        component_scales=(0.08, 0.045),
        combined_residual_scale=0.12,
        control_gain=2.5,
        style_dim=None,
        range_dim=3,
        range_hidden_dim=16,
        spectral_gain_floor=0.08,
        spectral_gain_ceiling=0.75,
        range_decay=1.5,
        spectral_correction_max=0.08,
    ):
        super().__init__()
        if len(component_scales) != len(self.component_names):
            raise ValueError("component_scales must contain two values")
        if not 0.0 <= spectral_gain_floor < spectral_gain_ceiling <= 1.0:
            raise ValueError("Invalid spectral gain bounds")
        if range_decay <= 0.0:
            raise ValueError("range_decay must be positive")
        if spectral_correction_max < 0.0:
            raise ValueError("spectral_correction_max must be non-negative")
        self.cond_dim = int(cond_dim)
        self.bottleneck = int(bottleneck)
        self.component_scales = tuple(float(value) for value in component_scales)
        self.combined_residual_scale = float(combined_residual_scale)
        self.control_gain = float(control_gain)
        self.style_dim = style_dim
        self.range_dim = int(range_dim)
        self.range_hidden_dim = int(range_hidden_dim)
        self.spectral_gain_floor = float(spectral_gain_floor)
        self.spectral_gain_ceiling = float(spectral_gain_ceiling)
        self.range_decay = float(range_decay)
        self.spectral_correction_max = float(spectral_correction_max)

        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)
        self.positive_branches = nn.ModuleList(
            [
                AxisResidualBranch(cond_dim, bottleneck, style_dim=style_dim)
                for _ in self.component_names
            ]
        )
        self.negative_branches = nn.ModuleList(
            [
                AxisResidualBranch(cond_dim, bottleneck, style_dim=style_dim)
                for _ in self.component_names
            ]
        )
        self.range_gate = nn.Sequential(
            nn.Linear(range_dim, range_hidden_dim),
            nn.SiLU(),
            nn.Linear(range_hidden_dim, 1),
        )
        nn.init.zeros_(self.range_gate[-1].weight)
        nn.init.zeros_(self.range_gate[-1].bias)

    def spectral_gain(self, range_features, return_details=False):
        if range_features.dim() != 2 or range_features.size(-1) != self.range_dim:
            raise ValueError(
                f"Expected range features [B,{self.range_dim}], "
                f"got {tuple(range_features.shape)}"
            )
        features = range_features.float()
        distance_octaves = features[:, 2].abs() * 2.0
        base_gain = self.spectral_gain_floor + (
            self.spectral_gain_ceiling - self.spectral_gain_floor
        ) * torch.exp(-self.range_decay * distance_octaves)
        correction = self.spectral_correction_max * torch.tanh(
            self.range_gate(features)[:, 0]
        )
        gain = (base_gain + correction).clamp(
            self.spectral_gain_floor,
            self.spectral_gain_ceiling,
        )
        gain = gain.to(dtype=range_features.dtype)
        if return_details:
            return (
                gain,
                base_gain.to(dtype=range_features.dtype),
                correction.to(dtype=range_features.dtype),
            )
        return gain

    def forward(
        self,
        condition,
        control,
        component_gates,
        range_features,
        style=None,
        strength=1.0,
        return_details=False,
    ):
        if control.dim() == 0:
            control = control[None]
        if control.dim() == 2 and control.size(-1) == 1:
            control = control[:, 0]
        if control.dim() != 1 or control.size(0) != condition.size(0):
            raise ValueError(f"Expected control [B], got {tuple(control.shape)}")
        if component_gates.dim() != 3 or component_gates.size(1) != 2:
            raise ValueError(
                f"Expected component gates [B,2,T], got {tuple(component_gates.shape)}"
            )
        if component_gates.size(-1) != condition.size(1):
            raise ValueError("Component gates and condition must share frame length")
        if range_features.size(0) != condition.size(0):
            raise ValueError("Range features and condition must share batch size")

        control = control.to(device=condition.device, dtype=condition.dtype)
        gates = component_gates.to(device=condition.device, dtype=condition.dtype)
        range_features = range_features.to(
            device=condition.device,
            dtype=condition.dtype,
        )
        normalized = self.norm(condition)
        positive_directions = torch.stack(
            [branch(normalized, style=style) for branch in self.positive_branches],
            dim=1,
        )
        negative_directions = torch.stack(
            [branch(normalized, style=style) for branch in self.negative_branches],
            dim=1,
        )
        positive_amount = torch.relu(control)[:, None, None, None]
        negative_amount = torch.relu(-control)[:, None, None, None]
        mixed_directions = (
            positive_amount * positive_directions
            + negative_amount * negative_directions
        )

        range_gain, base_range_gain, range_correction = self.spectral_gain(
            range_features,
            return_details=True,
        )
        effective_gates = torch.stack(
            (
                gates[:, 0],
                gates[:, 1] * range_gain[:, None],
            ),
            dim=1,
        )
        scales = condition.new_tensor(self.component_scales)
        component_deltas = (
            scales[None, :, None, None]
            * torch.tanh(float(strength) * self.control_gain * mixed_directions)
            * effective_gates[:, :, :, None]
        )
        delta = component_deltas.sum(dim=1)
        if self.combined_residual_scale > 0:
            cap = max(self.combined_residual_scale, 1e-8)
            delta = delta / torch.maximum(
                torch.ones_like(delta),
                delta.abs() / cap,
            )
        output = condition + delta
        if return_details:
            selected_directions = torch.where(
                (control >= 0)[:, None, None, None],
                positive_directions,
                negative_directions,
            )
            return output, {
                "delta": delta,
                "component_deltas": component_deltas,
                "directions": selected_directions,
                "positive_directions": positive_directions,
                "negative_directions": negative_directions,
                "gates": effective_gates,
                "base_gates": gates,
                "range_features": range_features,
                "spectral_gain": range_gain,
                "spectral_base_gain": base_range_gain,
                "spectral_gain_correction": range_correction,
            }
        return output


class RangeConditionedExpressionAdapter(nn.Module):
    """Global intensity/breathiness plus range-conditioned local accent."""

    def __init__(self, global_adapter, onset_adapter):
        super().__init__()
        self.global_adapter = global_adapter
        self.onset_adapter = onset_adapter

    def forward(
        self,
        condition,
        controls,
        gates,
        style=None,
        strength=1.0,
        return_details=False,
        component_gates=None,
        range_features=None,
    ):
        if controls.dim() == 1:
            controls = controls[None]
        if range_features is None:
            raise ValueError("Range-conditioned controller requires source/reference range features")
        global_controls = controls.clone()
        global_controls[:, 0] = 0.0
        global_output, global_details = self.global_adapter(
            condition,
            global_controls,
            gates,
            style=style,
            strength=strength,
            return_details=True,
        )
        if component_gates is None:
            component_gates = gates[:, :1].expand(-1, 2, -1)
        output, onset_details = self.onset_adapter(
            global_output,
            controls[:, 0],
            component_gates,
            range_features,
            style=style,
            strength=strength,
            return_details=True,
        )
        if return_details:
            accent_direction = onset_details["directions"].mean(dim=1)
            axis_directions = torch.stack(
                (
                    accent_direction,
                    global_details["directions"][:, 1],
                    global_details["directions"][:, 2],
                ),
                dim=1,
            )
            return output, {
                "delta": output - condition,
                "directions": axis_directions,
                "global": global_details,
                "onset": onset_details,
            }
        return output
