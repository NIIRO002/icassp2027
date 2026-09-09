import torch
import torch.nn as nn

from .hierarchical_adapter import AxisResidualBranch


class AsymmetricOnsetEffortAdapter(nn.Module):
    """Separate positive/negative branches for three local effort components."""

    component_names = (
        "attack_energy",
        "spectral_balance",
        "harmonic_concentration",
    )

    def __init__(
        self,
        cond_dim=768,
        bottleneck=64,
        component_scales=(0.08, 0.06, 0.05),
        combined_residual_scale=0.16,
        control_gain=2.5,
        style_dim=None,
    ):
        super().__init__()
        if len(component_scales) != len(self.component_names):
            raise ValueError("component_scales must contain three values")
        self.cond_dim = int(cond_dim)
        self.bottleneck = int(bottleneck)
        self.component_scales = tuple(float(value) for value in component_scales)
        self.combined_residual_scale = float(combined_residual_scale)
        self.control_gain = float(control_gain)
        self.style_dim = style_dim
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

    def forward(
        self,
        condition,
        control,
        component_gates,
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
        if component_gates.dim() != 3 or component_gates.size(1) != 3:
            raise ValueError(
                f"Expected component_gates [B,3,T], got {tuple(component_gates.shape)}"
            )
        if component_gates.size(-1) != condition.size(1):
            raise ValueError("component gates and condition must share frame length")

        control = control.to(device=condition.device, dtype=condition.dtype)
        gates = component_gates.to(device=condition.device, dtype=condition.dtype)
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
        scales = condition.new_tensor(self.component_scales)
        component_deltas = (
            scales[None, :, None, None]
            * torch.tanh(float(strength) * self.control_gain * mixed_directions)
            * gates[:, :, :, None]
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
                "gates": gates,
            }
        return output


class AsymmetricOnsetExpressionAdapter(nn.Module):
    """Asymmetric local onset-and-effort controller."""

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
    ):
        if controls.dim() == 1:
            controls = controls[None]
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
            component_gates = gates[:, :1].expand(-1, 3, -1)
        output, onset_details = self.onset_adapter(
            global_output,
            controls[:, 0],
            component_gates,
            style=style,
            strength=strength,
            return_details=True,
        )
        if return_details:
            return output, {
                "delta": output - condition,
                "directions": onset_details["directions"],
                "global": global_details,
                "onset": onset_details,
            }
        return output
