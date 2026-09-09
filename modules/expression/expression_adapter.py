import torch
import torch.nn as nn


class ExpressionFiLMAdapter(nn.Module):
    """Small FiLM adapter over length-regulated conditioning frames."""

    def __init__(self, cond_dim: int, expr_dim: int, bottleneck: int = 128):
        super().__init__()
        self.to_film = nn.Sequential(
            nn.Linear(expr_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, cond_dim * 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Identity at initialization: gamma = beta = 0.
        last = self.to_film[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond: torch.Tensor, expr: torch.Tensor, strength: float = 1.0):
        if cond.shape[:2] != expr.shape[:2]:
            raise ValueError(
                f"cond and expr must share [B, T], got {tuple(cond.shape)} and {tuple(expr.shape)}"
            )
        gamma, beta = self.to_film(expr).chunk(2, dim=-1)
        return cond * (1.0 + strength * gamma) + strength * beta


class ExpressionResidualAdapter(nn.Module):
    """Conservative residual adapter over length-regulated conditioning frames.

    This adapter is intentionally weaker than FiLM. It starts as an exact
    identity mapping and can only add a small residual to SEED-VC conditioning.
    """

    def __init__(self, cond_dim: int, expr_dim: int, bottleneck: int = 64, residual_scale: float = 0.05):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(expr_dim),
            nn.Linear(expr_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, cond_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond: torch.Tensor, expr: torch.Tensor, strength: float = 1.0):
        if cond.shape[:2] != expr.shape[:2]:
            raise ValueError(
                f"cond and expr must share [B, T], got {tuple(cond.shape)} and {tuple(expr.shape)}"
            )
        delta = self.net(expr)
        return cond + (self.residual_scale * strength) * delta
