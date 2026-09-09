import math

import torch
import torch.nn as nn

from .hierarchical_adapter import AxisResidualBranch


class LanguageDurationAccentAdapter(nn.Module):
    """Language-aware accent control over attack, nucleus, and sustain spans."""

    component_names = ("attack_energy", "nucleus_effort", "sustain_effort")

    def __init__(
        self,
        cond_dim=768,
        bottleneck=64,
        component_scales=(0.08, 0.05, 0.04),
        combined_residual_scale=0.14,
        control_gain=2.5,
        style_dim=None,
        event_context_dim=4,
        language_count=4,
        language_embedding_dim=8,
        context_hidden_dim=24,
        range_dim=3,
        range_hidden_dim=16,
        spectral_gain_floor=0.10,
        spectral_gain_ceiling=0.75,
        range_decay=1.5,
        spectral_correction_max=0.08,
        context_gain_max=0.35,
        negative_output_gain=1.0,
        positive_output_gain=1.0,
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
        self.event_context_dim = int(event_context_dim)
        self.language_count = int(language_count)
        self.language_embedding_dim = int(language_embedding_dim)
        self.context_hidden_dim = int(context_hidden_dim)
        self.range_dim = int(range_dim)
        self.range_hidden_dim = int(range_hidden_dim)
        self.spectral_gain_floor = float(spectral_gain_floor)
        self.spectral_gain_ceiling = float(spectral_gain_ceiling)
        self.range_decay = float(range_decay)
        self.spectral_correction_max = float(spectral_correction_max)
        self.context_gain_max = float(context_gain_max)
        self.negative_output_gain = float(negative_output_gain)
        self.positive_output_gain = float(positive_output_gain)
        if self.negative_output_gain <= 0.0 or self.positive_output_gain <= 0.0:
            raise ValueError("Polarity output gains must be positive")

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
        self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
        self.context_gate = nn.Sequential(
            nn.Linear(event_context_dim + language_embedding_dim, context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, len(self.component_names)),
        )
        self.range_gate = nn.Sequential(
            nn.Linear(range_dim, range_hidden_dim),
            nn.SiLU(),
            nn.Linear(range_hidden_dim, 1),
        )
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
        nn.init.zeros_(self.range_gate[-1].weight)
        nn.init.zeros_(self.range_gate[-1].bias)

    def spectral_gain(self, range_features):
        distance_octaves = range_features.float()[:, 2].abs() * 2.0
        base = self.spectral_gain_floor + (
            self.spectral_gain_ceiling - self.spectral_gain_floor
        ) * torch.exp(-self.range_decay * distance_octaves)
        correction = self.spectral_correction_max * torch.tanh(
            self.range_gate(range_features.float())[:, 0]
        )
        return (base + correction).clamp(
            self.spectral_gain_floor,
            self.spectral_gain_ceiling,
        ).to(dtype=range_features.dtype)

    def _context_gains(self, event_context, language_ids):
        if event_context.dim() != 3 or event_context.size(1) != self.event_context_dim:
            raise ValueError(
                f"Expected event context [B,{self.event_context_dim},T], "
                f"got {tuple(event_context.shape)}"
            )
        language_ids = language_ids.long().clamp(0, self.language_count - 1)
        embedding = self.language_embedding(language_ids)
        embedding = embedding[:, None].expand(-1, event_context.size(-1), -1)
        inputs = torch.cat((event_context.transpose(1, 2).float(), embedding), dim=-1)
        learned = 1.0 + self.context_gain_max * torch.tanh(self.context_gate(inputs))

        salience = event_context[:, 0].float().clamp(0.0, 1.0)
        duration = event_context[:, 1].float().clamp(0.0, 1.0)
        phrase_initial = event_context[:, 2].float().clamp(0.0, 1.0)
        selective_salience = salience.square()
        base = torch.stack(
            (
                (0.15 + 0.85 * selective_salience)
                * (1.0 + 0.20 * phrase_initial),
                0.20 + 0.80 * selective_salience,
                (0.25 + 0.75 * salience) * (0.35 + 0.65 * duration),
            ),
            dim=-1,
        )
        return (base * learned).transpose(1, 2)

    def forward(
        self,
        condition,
        control,
        component_gates,
        event_context,
        language_ids,
        range_features,
        style=None,
        strength=1.0,
        return_details=False,
    ):
        if control.dim() == 0:
            control = control[None]
        if control.dim() == 2 and control.size(-1) == 1:
            control = control[:, 0]
        if control.shape != (condition.size(0),):
            raise ValueError(f"Expected control [B], got {tuple(control.shape)}")
        if component_gates.shape != (condition.size(0), 3, condition.size(1)):
            raise ValueError(
                f"Expected component gates [B,3,T], got {tuple(component_gates.shape)}"
            )
        if range_features.shape != (condition.size(0), self.range_dim):
            raise ValueError(
                f"Expected range features [B,{self.range_dim}], "
                f"got {tuple(range_features.shape)}"
            )

        dtype = condition.dtype
        normalized = self.norm(condition)
        positive_directions = torch.stack(
            [branch(normalized, style=style) for branch in self.positive_branches],
            dim=1,
        )
        negative_directions = torch.stack(
            [branch(normalized, style=style) for branch in self.negative_branches],
            dim=1,
        )
        positive_amount = torch.relu(control.to(dtype=dtype))[:, None, None, None]
        negative_amount = torch.relu(-control.to(dtype=dtype))[:, None, None, None]
        directions = (
            positive_amount * positive_directions
            + negative_amount * negative_directions
        )

        context_gains = self._context_gains(event_context, language_ids).to(dtype=dtype)
        spectral_gain = self.spectral_gain(range_features.to(dtype=dtype))
        base_effective_gates = component_gates.to(dtype=dtype) * context_gains
        effective_gates = torch.stack(
            (
                base_effective_gates[:, 0],
                base_effective_gates[:, 1] * spectral_gain[:, None],
                base_effective_gates[:, 2],
            ),
            dim=1,
        )
        scales = condition.new_tensor(self.component_scales)
        polarity_gain = torch.where(
            (control >= 0).to(dtype=torch.bool),
            condition.new_tensor(self.positive_output_gain),
            condition.new_tensor(self.negative_output_gain),
        )[:, None, None, None]
        component_deltas = (
            scales[None, :, None, None]
            * torch.tanh(float(strength) * self.control_gain * directions)
            * effective_gates[:, :, :, None]
            * polarity_gain
        )
        delta = component_deltas.sum(dim=1)
        if self.combined_residual_scale > 0:
            cap = max(self.combined_residual_scale, 1e-8)
            delta = delta / torch.maximum(torch.ones_like(delta), delta.abs() / cap)
        output = condition + delta
        if not return_details:
            return output
        selected_directions = torch.where(
            (control >= 0)[:, None, None, None],
            positive_directions,
            negative_directions,
        )
        return output, {
            "delta": delta,
            "component_deltas": component_deltas,
            "directions": selected_directions,
            "gates": effective_gates,
            "base_gates": component_gates,
            "context_gains": context_gains,
            "event_context": event_context,
            "language_ids": language_ids,
            "spectral_gain": spectral_gain,
            "polarity_gain": polarity_gain[:, 0, 0, 0],
            "range_features": range_features,
        }


class LanguageDurationExpressionAdapter(nn.Module):
    """Frozen global controls plus language-duration-aware local accent."""

    def __init__(self, global_adapter, accent_adapter):
        super().__init__()
        self.global_adapter = global_adapter
        self.accent_adapter = accent_adapter

    def forward(
        self,
        condition,
        controls,
        gates,
        style=None,
        strength=1.0,
        return_details=False,
        component_gates=None,
        event_context=None,
        language_ids=None,
        range_features=None,
    ):
        if controls.dim() == 1:
            controls = controls[None]
        required = (component_gates, event_context, language_ids, range_features)
        if any(value is None for value in required):
            raise ValueError("Language-duration controller requires event gates, context, language, and pitch range")
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
        output, accent_details = self.accent_adapter(
            global_output,
            controls[:, 0],
            component_gates,
            event_context,
            language_ids,
            range_features,
            style=style,
            strength=strength,
            return_details=True,
        )
        if not return_details:
            return output
        accent_direction = accent_details["directions"].mean(dim=1)
        return output, {
            "delta": output - condition,
            "directions": torch.stack(
                (
                    accent_direction,
                    global_details["directions"][:, 1],
                    global_details["directions"][:, 2],
                ),
                dim=1,
            ),
            "global": global_details,
            "accent": accent_details,
        }
