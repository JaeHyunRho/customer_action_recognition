"""Skeleton graph and its normalised adjacency partitions for ST-GCN."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..keypoints import layout


def _hop_distance(num_nodes: int, edges: Sequence[Tuple[int, int]], max_hop: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0

    hop = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    transfer = [np.linalg.matrix_power(adj, d) for d in range(max_hop + 1)]
    arrive = (np.stack(transfer) > 0)
    for d in range(max_hop, -1, -1):
        hop[arrive[d]] = d
    return hop


def _normalize(adj: np.ndarray) -> np.ndarray:
    degree = adj.sum(axis=0)
    inv = np.zeros_like(degree)
    nonzero = degree > 0
    inv[nonzero] = degree[nonzero] ** -1.0
    return adj @ np.diag(inv)


class SkeletonGraph:
    """Spatial-configuration partitioning, as in the original ST-GCN paper.

    Each node's neighbourhood is split into three subsets -- the node itself,
    neighbours closer to the body centre, and neighbours further away -- which
    become the ``K`` adjacency slices the graph convolution sums over.
    """

    def __init__(
        self,
        head_removal: bool = True,
        strategy: str = "spatial",
        max_hop: int = 1,
        center: int | None = None,
    ) -> None:
        lay = layout(head_removal)
        self.num_nodes: int = int(lay["num_keypoints"])
        self.edges: List[Tuple[int, int]] = list(lay["edges"])  # type: ignore[arg-type]
        self.strategy = strategy
        self.max_hop = max_hop
        # Default centre: left hip for the 12-joint layout, nose otherwise.
        self.center = center if center is not None else (6 if head_removal else 0)
        self.hop = _hop_distance(self.num_nodes, self.edges, max_hop)
        self.A = self._build()

    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:
        valid = list(range(self.max_hop + 1))
        adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for hop in valid:
            adjacency[self.hop == hop] = 1.0
        normalized = _normalize(adjacency)

        if self.strategy == "uniform":
            return normalized[None]

        if self.strategy == "distance":
            stack = np.zeros((len(valid), self.num_nodes, self.num_nodes), dtype=np.float32)
            for i, hop in enumerate(valid):
                stack[i][self.hop == hop] = normalized[self.hop == hop]
            return stack

        if self.strategy != "spatial":
            raise ValueError(f"unknown partition strategy {self.strategy!r}")

        parts: List[np.ndarray] = []
        for hop in valid:
            root = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            close = np.zeros_like(root)
            further = np.zeros_like(root)
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if self.hop[j, i] != hop:
                        continue
                    di = self.hop[j, self.center]
                    dj = self.hop[i, self.center]
                    if dj == di:
                        root[j, i] = normalized[j, i]
                    elif dj > di:
                        close[j, i] = normalized[j, i]
                    else:
                        further[j, i] = normalized[j, i]
            if hop == 0:
                parts.append(root)
            else:
                parts.append(root + close)
                parts.append(further)
        return np.stack(parts)

    # ------------------------------------------------------------------
    @property
    def num_partitions(self) -> int:
        return int(self.A.shape[0])

    def torch_adjacency(self):
        """Return ``A`` as a float32 torch tensor of shape ``(K, V, V)``."""
        import torch

        return torch.from_numpy(self.A.astype(np.float32))
