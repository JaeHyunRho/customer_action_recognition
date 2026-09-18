"""Stacked LSTM baseline for customer action recognition.

Fig. 3.2 of the thesis: three LSTM layers of 128 channels feeding
``Dense -> LeakyReLU -> Dropout -> Dense`` with a softmax on top. The input at
each timestep is the flattened skeleton, i.e. ``V * C`` values (12 joints x
three channels = 36 by default).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import LSTMConfig


class SkeletonLSTM(nn.Module):
    """Input ``(N, C, T, V)`` for interface parity with :class:`STGCN`."""

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int = 12,
        in_channels: int = 3,
        cfg: LSTMConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or LSTMConfig()
        self.cfg = cfg
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.num_classes = num_classes

        feature_dim = num_keypoints * in_channels
        self.feature_dim = feature_dim
        self.input_bn = nn.BatchNorm1d(feature_dim)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.dense_hidden),
            nn.LeakyReLU(cfg.leaky_slope, inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dense_hidden, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # (N, C, T, V) -> (N, T, V*C)
            n, c, t, v = x.size()
            x = x.permute(0, 2, 3, 1).contiguous().view(n, t, v * c)
        elif x.dim() != 3:
            raise ValueError(f"expected a 3D or 4D tensor, got {tuple(x.shape)}")

        n, t, f = x.size()
        x = self.input_bn(x.reshape(n * t, f)).view(n, t, f)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def input_layout(self) -> str:
        return "NCTV"

    def example_input(self, window: int = 30, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.in_channels, window, self.num_keypoints)


def build_lstm(
    num_classes: int,
    num_keypoints: int = 12,
    in_channels: int = 3,
    cfg: LSTMConfig | None = None,
) -> SkeletonLSTM:
    return SkeletonLSTM(
        num_classes=num_classes,
        num_keypoints=num_keypoints,
        in_channels=in_channels,
        cfg=cfg,
    )
