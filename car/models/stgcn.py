"""Lightweight ST-GCN for on-device customer action recognition.

Follows Fig. 3.1 of the thesis: each unit is
``GroupedConv (GCN) -> BN -> LeakyReLU -> GroupedConv (TCN) -> BN -> Dropout
-> LeakyReLU``, six units wide 64/64/128/128/256/256, a global average pool and
a linear classifier. Grouped convolutions and LeakyReLU are the two changes the
thesis makes to stock ST-GCN to cut parameters and avoid dead units.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from ..config import STGCNConfig
from .graph import SkeletonGraph


def _safe_groups(in_ch: int, out_ch: int, groups: int) -> int:
    """Largest divisor of both channel counts that is <= ``groups``."""
    g = max(1, int(groups))
    while g > 1 and (in_ch % g != 0 or out_ch % g != 0):
        g -= 1
    return g


class GraphConv(nn.Module):
    """Spatial graph convolution with ``K`` adjacency partitions.

    The 1x1 convolution is grouped, which is where most of the parameter
    saving comes from relative to the original ST-GCN.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int, groups: int = 1) -> None:
        super().__init__()
        self.k = k
        g = _safe_groups(in_channels, out_channels * k, groups)
        self.conv = nn.Conv2d(in_channels, out_channels * k, kernel_size=1, groups=g)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V), adj: (K, V, V)
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k, kc // self.k, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, adj)
        return x.contiguous()


class STGCNBlock(nn.Module):
    """One spatial + temporal unit (the dashed box in Fig. 3.1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.25,
        groups: int = 1,
        leaky_slope: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        pad = (temporal_kernel - 1) // 2

        self.gcn = GraphConv(in_channels, out_channels, k, groups=groups)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(leaky_slope, inplace=True)

        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(pad, 0),
            groups=_safe_groups(out_channels, out_channels, groups),
        )
        self.tcn_bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.act2 = nn.LeakyReLU(leaky_slope, inplace=True)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        res = 0.0 if self.residual is None else self.residual(x)
        x = self.act1(self.gcn_bn(self.gcn(x, adj)))
        x = self.drop(self.tcn_bn(self.tcn(x)))
        return self.act2(x + res)


class STGCN(nn.Module):
    """The full network.

    Input is ``(N, C, T, V)`` -- batch, channels (x, y, score), time (30) and
    joints (12 after head removal).
    """

    def __init__(
        self,
        num_classes: int,
        graph: SkeletonGraph | None = None,
        cfg: STGCNConfig | None = None,
        head_removal: bool = True,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or STGCNConfig()
        self.cfg = cfg
        self.graph = graph or SkeletonGraph(head_removal=head_removal)
        self.num_classes = num_classes

        adj = self.graph.torch_adjacency()
        self.register_buffer("adjacency", adj)
        k = int(adj.shape[0])
        v = int(adj.shape[1])

        c_in = int(in_channels if in_channels is not None else cfg.in_channels)
        self.in_channels = c_in
        self.data_bn = nn.BatchNorm1d(c_in * v)

        channels: Sequence[int] = tuple(cfg.channels)
        blocks = []
        prev = c_in
        for idx, ch in enumerate(channels):
            # Halve the temporal resolution when the width doubles.
            stride = 2 if idx > 0 and ch != channels[idx - 1] else 1
            blocks.append(
                STGCNBlock(
                    prev,
                    ch,
                    k,
                    temporal_kernel=cfg.temporal_kernel,
                    stride=stride,
                    dropout=cfg.dropout,
                    groups=cfg.groups,
                    leaky_slope=cfg.leaky_slope,
                    residual=idx > 0,
                )
            )
            prev = ch
        self.blocks = nn.ModuleList(blocks)

        if cfg.edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(adj)) for _ in blocks]
            )
        else:
            self.edge_importance = [1.0] * len(blocks)  # type: ignore[assignment]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(prev, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, T, V) -> normalise per joint-channel over time
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.adjacency * importance)

        x = self.pool(x).view(n, -1)
        return self.fc(x)

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def input_layout(self) -> str:
        return "NCTV"

    def example_input(self, window: int = 30, batch: int = 1) -> torch.Tensor:
        v = int(self.adjacency.shape[1])
        return torch.zeros(batch, self.in_channels, window, v)


def build_stgcn(
    num_classes: int,
    cfg: STGCNConfig | None = None,
    head_removal: bool = True,
    in_channels: int | None = None,
) -> STGCN:
    return STGCN(
        num_classes=num_classes,
        cfg=cfg,
        head_removal=head_removal,
        in_channels=in_channels,
    )
