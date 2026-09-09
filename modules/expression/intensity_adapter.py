import torch
from torch import nn


class ScalarIntensityAdapter(nn.Module):
    """Small residual conditioning adapter for one global intensity control."""

    def __init__(self, cond_dim=768, hidden_dim=64, residual_scale=0.05):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond, intensity_control, strength=1.0):
        if not torch.is_tensor(intensity_control):
            intensity_control = torch.tensor(intensity_control, device=cond.device, dtype=cond.dtype)
        intensity_control = intensity_control.to(device=cond.device, dtype=cond.dtype)

        if intensity_control.dim() == 0:
            intensity_control = intensity_control[None, None]
        elif intensity_control.dim() == 1:
            intensity_control = intensity_control[:, None]
        elif intensity_control.dim() == 3:
            intensity_control = intensity_control.mean(dim=1)

        neutral = torch.zeros_like(intensity_control)
        delta = (self.net(intensity_control) - self.net(neutral)).unsqueeze(1)
        return cond + float(strength) * self.residual_scale * delta


class SymmetricIntensityAdapter(nn.Module):
    """A sign-symmetric residual direction for one global intensity control."""

    def __init__(self, cond_dim=768, residual_scale=0.05):
        super().__init__()
        self.residual_scale = residual_scale
        self.direction = nn.Parameter(torch.zeros(cond_dim))

    def forward(self, cond, intensity_control, strength=1.0):
        if not torch.is_tensor(intensity_control):
            intensity_control = torch.tensor(intensity_control, device=cond.device, dtype=cond.dtype)
        intensity_control = intensity_control.to(device=cond.device, dtype=cond.dtype)

        if intensity_control.dim() == 0:
            intensity_control = intensity_control[None, None]
        elif intensity_control.dim() == 1:
            intensity_control = intensity_control[:, None]
        elif intensity_control.dim() == 3:
            intensity_control = intensity_control.mean(dim=1)

        delta = intensity_control.unsqueeze(1) * self.direction.to(dtype=cond.dtype)[None, None, :]
        return cond + float(strength) * self.residual_scale * delta
