"""Action-recognition model zoo."""

from __future__ import annotations

from typing import Any, Dict

from ..config import Config
from .graph import SkeletonGraph
from .lstm import SkeletonLSTM, build_lstm
from .stgcn import STGCN, build_stgcn

__all__ = [
    "SkeletonGraph",
    "SkeletonLSTM",
    "STGCN",
    "build_lstm",
    "build_stgcn",
    "build_model",
    "load_model",
]


def build_model(name: str, cfg: Config):
    """Instantiate an action model from the project config."""
    name = name.lower()
    if name == "stgcn":
        return build_stgcn(
            num_classes=cfg.num_classes,
            cfg=cfg.stgcn,
            head_removal=cfg.pose.head_removal,
            in_channels=cfg.in_channels,
        )
    if name == "lstm":
        return build_lstm(
            num_classes=cfg.num_classes,
            num_keypoints=cfg.num_keypoints,
            in_channels=cfg.in_channels,
            cfg=cfg.lstm,
        )
    raise ValueError(f"unknown action model {name!r}; expected 'stgcn' or 'lstm'")


def load_model(path: str, map_location: str = "cpu"):
    """Restore a checkpoint written by ``car.train.train_action``.

    Returns ``(model, checkpoint_dict)``. The checkpoint carries its own config
    so inference never has to guess the architecture.
    """
    import torch

    from ..config import load_config

    ckpt: Dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    cfg = load_config(overrides=None)
    saved_cfg = ckpt.get("config")
    if saved_cfg:
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(saved_cfg, fh, sort_keys=False, allow_unicode=True)
            tmp = fh.name
        cfg = load_config(tmp)

    model = build_model(ckpt.get("model_name", "stgcn"), cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
