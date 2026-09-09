import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_note_ids_and_sustain_gate(
    f0,
    frame_rate=44100 / 512,
    boundary_threshold_cent=50.0,
    sustain_trim_seconds=0.08,
    minimum_note_seconds=0.25,
):
    """Infer conservative note regions from F0 for annotation-free inference."""

    if f0.dim() == 1:
        f0 = f0[None]
    voiced = f0 > 1.0
    cents = torch.where(
        voiced,
        1200.0 * torch.log2(f0.clamp_min(1.0)),
        torch.zeros_like(f0),
    )
    smoothed = F.avg_pool1d(cents[:, None], kernel_size=9, stride=1, padding=4)[:, 0]
    note_ids = torch.zeros_like(f0, dtype=torch.long)
    sustain = torch.zeros_like(f0, dtype=torch.float32)
    minimum_frames = max(int(round(minimum_note_seconds * frame_rate)), 3)
    trim_frames = max(int(round(sustain_trim_seconds * frame_rate)), 1)
    for batch_index in range(f0.size(0)):
        start = None
        note_id = 1
        for frame in range(f0.size(1) + 1):
            valid = frame < f0.size(1) and bool(voiced[batch_index, frame])
            boundary = not valid
            if valid and frame > 0 and bool(voiced[batch_index, frame - 1]):
                boundary |= bool(
                    (smoothed[batch_index, frame] - smoothed[batch_index, frame - 1]).abs()
                    > boundary_threshold_cent
                )
            if valid and start is None:
                start = frame
            elif start is not None and boundary:
                end = frame
                if end - start >= minimum_frames:
                    note_ids[batch_index, start:end] = note_id
                    sustain_start = min(start + trim_frames, end)
                    sustain_end = max(end - trim_frames, sustain_start)
                    sustain[batch_index, sustain_start:sustain_end] = 1.0
                    note_id += 1
                start = frame if valid else None
    return note_ids, sustain


class _RegisterResidualBranch(nn.Module):
    def __init__(self, cond_dim, bottleneck):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cond_dim, bottleneck, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(bottleneck, cond_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition):
        return self.net(condition.transpose(1, 2)).transpose(1, 2)


class VocalRegisterAdapter(nn.Module):
    """Asymmetric mixed/chest-to-falsetto condition residual adapter."""

    def __init__(
        self,
        cond_dim=768,
        bottleneck=64,
        residual_scale=0.12,
        control_gain=2.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(cond_dim)
        self.positive_branch = _RegisterResidualBranch(cond_dim, bottleneck)
        self.negative_branch = _RegisterResidualBranch(cond_dim, bottleneck)
        self.residual_scale = float(residual_scale)
        self.control_gain = float(control_gain)

    def forward(self, condition, control, gate=None, strength=1.0, return_details=False):
        if condition.dim() != 3:
            raise ValueError(f"Expected condition [B,T,C], got {tuple(condition.shape)}")
        control = torch.as_tensor(control, device=condition.device, dtype=condition.dtype)
        if control.dim() == 0:
            control = control.expand(condition.size(0))
        control = control.reshape(condition.size(0), 1, 1).clamp(-1.0, 1.0)
        if gate is None:
            gate = condition.new_ones(condition.size(0), condition.size(1))
        gate = gate.to(device=condition.device, dtype=condition.dtype)
        if gate.dim() == 2:
            gate = gate[..., None]
        if gate.size(1) != condition.size(1):
            gate = F.interpolate(
                gate.transpose(1, 2),
                size=condition.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        normalized = self.norm(condition)
        positive = self.positive_branch(normalized)
        negative = self.negative_branch(normalized)
        positive_weight = torch.tanh(self.control_gain * F.relu(control))
        negative_weight = torch.tanh(self.control_gain * F.relu(-control))
        direction = positive_weight * positive + negative_weight * negative
        delta = float(strength) * self.residual_scale * gate * direction
        output = condition + delta
        if not return_details:
            return output
        return output, {
            "delta": delta,
            "positive_direction": positive,
            "negative_direction": negative,
            "gate": gate,
        }


class FormulaVibratoF0Adapter(nn.Module):
    """Note-gated F0 vibrato control with dataset-calibrated rate and depth."""

    def __init__(
        self,
        frame_rate=44100 / 512,
        rate_hz=5.5,
        positive_depth_cent=36.0,
        negative_suppression_gain=1.0,
        maximum_depth_cent=140.0,
    ):
        super().__init__()
        self.frame_rate = float(frame_rate)
        self.rate_hz = nn.Parameter(torch.tensor(float(rate_hz)))
        self.positive_depth_cent = nn.Parameter(torch.tensor(float(positive_depth_cent)))
        self.negative_suppression_gain = nn.Parameter(
            torch.tensor(float(negative_suppression_gain))
        )
        self.maximum_depth_cent = float(maximum_depth_cent)
        taps = 31
        offsets = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2.0
        high = 2.0 * 8.0 / self.frame_rate * torch.sinc(
            2.0 * 8.0 / self.frame_rate * offsets
        )
        low = 2.0 * 4.0 / self.frame_rate * torch.sinc(
            2.0 * 4.0 / self.frame_rate * offsets
        )
        kernel = (high - low) * torch.hamming_window(taps, periodic=False)
        gain = (
            kernel
            * torch.cos(2.0 * math.pi * 6.0 * offsets / self.frame_rate)
        ).sum().abs().clamp_min(1e-6)
        self.register_buffer("vibrato_kernel", (kernel / gain)[None, None])

    @staticmethod
    def _smooth(values, width):
        width = max(int(width) | 1, 3)
        return F.avg_pool1d(values, kernel_size=width, stride=1, padding=width // 2)

    def _note_bandpass(self, cents, note_ids):
        output = torch.zeros_like(cents)
        radius = self.vibrato_kernel.size(-1) // 2
        for batch_index in range(cents.size(0)):
            for note_id in torch.unique(note_ids[batch_index]):
                if int(note_id) <= 0:
                    continue
                indices = torch.where(note_ids[batch_index] == note_id)[0]
                if indices.numel() <= radius + 2:
                    continue
                values = cents[batch_index, indices][None, None]
                padded = F.pad(values, (radius, radius), mode="reflect")
                filtered = F.conv1d(
                    padded,
                    self.vibrato_kernel.to(device=cents.device, dtype=cents.dtype),
                )[0, 0]
                output[batch_index, indices] = filtered
        return output

    def forward(self, f0, control, gate=None, note_ids=None, return_details=False):
        if f0.dim() == 1:
            f0 = f0[None]
        control = torch.as_tensor(control, device=f0.device, dtype=f0.dtype)
        if control.dim() == 0:
            control = control.expand(f0.size(0))
        # Negative suppression did not pass paired validation; the final
        # prototype intentionally exposes neutral-to-additive vibrato only.
        control = control.reshape(f0.size(0), 1).clamp(0.0, 1.0)
        voiced = f0 > 1.0
        if note_ids is None:
            note_ids, inferred_gate = infer_note_ids_and_sustain_gate(
                f0,
                frame_rate=self.frame_rate,
            )
            if gate is None:
                gate = inferred_gate
        if gate is None:
            gate = voiced.float()
        gate = gate.to(device=f0.device, dtype=f0.dtype) * voiced.float()

        cents = torch.where(
            voiced,
            1200.0 * torch.log2(f0.clamp_min(1.0)),
            torch.zeros_like(f0),
        )
        trend = self._smooth(cents[:, None], round(0.50 * self.frame_rate))[:, 0]
        source_residual = cents - trend
        source_band = self._note_bandpass(cents, note_ids)

        positions = torch.arange(f0.size(1), device=f0.device, dtype=f0.dtype)[None]
        note_ids = note_ids.to(device=f0.device)
        phase_positions = torch.zeros_like(positions).expand_as(f0).clone()
        for batch_index in range(f0.size(0)):
            for note_id in torch.unique(note_ids[batch_index]):
                if int(note_id) <= 0:
                    continue
                indices = torch.where(note_ids[batch_index] == note_id)[0]
                phase_positions[batch_index, indices] = torch.arange(
                    indices.numel(), device=f0.device, dtype=f0.dtype
                )
        rate = self.rate_hz.clamp(4.0, 8.0)
        oscillator = torch.sin(2.0 * math.pi * rate * phase_positions / self.frame_rate)
        depth = self.positive_depth_cent.clamp(0.0, self.maximum_depth_cent)
        added = depth * oscillator
        positive = F.relu(control) * added
        suppression = self.negative_suppression_gain.clamp(0.0, 4.0)
        delta_cent = gate * positive
        output_cents = cents + delta_cent
        output = torch.where(voiced, torch.pow(2.0, output_cents / 1200.0), f0)
        if not return_details:
            return output
        return output, {
            "delta_cent": delta_cent,
            "source_residual_cent": source_residual,
            "source_band_cent": source_band,
            "rate_hz": rate,
            "depth_cent": depth,
            "negative_suppression_gain": suppression,
            "supports_negative_control": False,
            "gate": gate,
            "oscillator": oscillator,
            "note_ids": note_ids,
        }
