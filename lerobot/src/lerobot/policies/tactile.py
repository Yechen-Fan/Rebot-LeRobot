from __future__ import annotations

import torch
from torch import Tensor, nn


def preprocess_tactile(x: Tensor, *, baseline: float | Tensor, scale: float) -> Tensor:
    """Convert raw FlexiTac values to a clipped, channel-first contact map."""
    if scale <= 0:
        raise ValueError("tactile_scale must be positive")
    x = x.to(torch.float32)
    if isinstance(baseline, Tensor):
        baseline = baseline.to(dtype=x.dtype, device=x.device)
    x = torch.clamp((x - baseline) / scale, min=0.0, max=1.0)
    return x.unsqueeze(-3)


def tactile_baseline_key(key: str) -> str:
    return key.replace(".tactile.", ".tactile_baseline.")


class SharedTactileEncoder(nn.Module):
    """Small spatial encoder shared by every tactile pad."""

    def __init__(self, output_dim: int, num_sensors: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, output_dim),
            nn.SiLU(),
        )
        self.sensor_embedding = nn.Embedding(num_sensors, output_dim)

    def forward(self, x: Tensor, sensor_index: int) -> Tensor:
        features = self.net(x)
        sensor = self.sensor_embedding.weight[sensor_index].view(1, -1, 1, 1)
        return features + sensor.to(dtype=features.dtype, device=features.device)
