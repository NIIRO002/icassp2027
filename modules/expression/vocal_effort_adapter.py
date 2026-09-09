import torch
from torch import nn


class SymmetricVocalEffortAdapter(nn.Module):
    """Content-dependent residual adapter with exact signed-control symmetry."""

    def __init__(self, cond_dim=768, bottleneck=64, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm = nn.LayerNorm(cond_dim, elementwise_affine=False)
        self.down = nn.Linear(cond_dim, bottleneck)
        self.up = nn.Linear(bottleneck, cond_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, cond, effort_control, strength=1.0):
        if not torch.is_tensor(effort_control):
            effort_control = torch.tensor(effort_control, device=cond.device, dtype=cond.dtype)
        effort_control = effort_control.to(device=cond.device, dtype=cond.dtype)

        if effort_control.dim() == 0:
            effort_control = effort_control[None, None, None]
        elif effort_control.dim() == 1:
            effort_control = effort_control[:, None, None]
        elif effort_control.dim() == 2:
            effort_control = effort_control[:, :, None]
        elif effort_control.dim() == 3 and effort_control.size(-1) != 1:
            effort_control = effort_control.mean(dim=-1, keepdim=True)

        if effort_control.size(1) == 1:
            effort_control = effort_control.expand(-1, cond.size(1), -1)
        elif effort_control.size(1) != cond.size(1):
            effort_control = torch.nn.functional.interpolate(
                effort_control.transpose(1, 2).float(),
                size=cond.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(dtype=cond.dtype)

        direction = torch.tanh(self.up(torch.nn.functional.silu(self.down(self.norm(cond)))))
        delta = effort_control * direction
        return cond + float(strength) * self.residual_scale * delta
