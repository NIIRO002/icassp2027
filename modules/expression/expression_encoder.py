import torch.nn as nn


class ExpressionEncoder(nn.Module):
    """Frame-level expression feature encoder."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 128, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features):
        return self.net(features)


class ExpressionFeaturePredictor(nn.Module):
    """Predict normalized expression features from expression embeddings."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 128, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embedding):
        return self.net(embedding)
