"""Tests for the pieces that are easy to get subtly wrong.

Run with ``python3 -m pytest tests/ -v``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from car.config import ACTION_CLASSES, Config, load_config, parse_overrides  # noqa: E402
from car.keypoints import (  # noqa: E402
    BODY12_INDICES,
    COCO17_NAMES,
    drop_head,
    edge_index,
    keypoint_weights,
    oks_sigmas,
)
from car.models import build_model  # noqa: E402
from car.models.graph import SkeletonGraph  # noqa: E402
from car.pipeline.buffer import BufferBank  # noqa: E402
from car.pipeline.normalize import add_velocity, normalize_sequence, to_model_input  # noqa: E402
from car.pose.tracker import StrongSort, iou_matrix, xyah_to_xyxy, xyxy_to_xyah  # noqa: E402
from car.utils.metrics import classification_report, confusion_matrix  # noqa: E402


# ==========================================================================
# Keypoints
# ==========================================================================
def test_drop_head_removes_the_five_facial_landmarks():
    kpts = np.arange(17 * 3, dtype=np.float32).reshape(1, 17, 3)
    out = drop_head(kpts)
    assert out.shape == (1, 12, 3)
    # Row 0 of the output must be the left shoulder, COCO index 5.
    assert np.allclose(out[0, 0], kpts[0, 5])
    assert BODY12_INDICES[0] == COCO17_NAMES.index("left_shoulder")


def test_drop_head_is_idempotent():
    kpts = np.zeros((2, 12, 3), dtype=np.float32)
    assert drop_head(kpts).shape == (2, 12, 3)


def test_drop_head_rejects_wrong_joint_count():
    with pytest.raises(ValueError):
        drop_head(np.zeros((1, 9, 3), dtype=np.float32))


def test_arm_joints_carry_the_paper_bonus():
    w = keypoint_weights(head_removal=True, upper_body_bonus=0.5)
    assert w.shape == (12,)
    assert np.allclose(w[:6], 1.5)   # shoulders, elbows, wrists
    assert np.allclose(w[6:], 1.0)   # hips, knees, ankles


def test_upweighted_joints_get_tighter_oks_sigmas():
    base = oks_sigmas(head_removal=True, upper_body_bonus=0.0)
    tight = oks_sigmas(head_removal=True, upper_body_bonus=0.5)
    assert np.all(tight[:6] < base[:6])
    assert np.allclose(tight[6:], base[6:])


def test_edge_index_is_symmetric_with_self_loops():
    idx = edge_index(head_removal=True, self_loops=True)
    assert idx.shape[0] == 2
    pairs = set(map(tuple, idx.T.tolist()))
    for a, b in pairs:
        assert (b, a) in pairs
    assert all((i, i) in pairs for i in range(12))


# ==========================================================================
# Graph
# ==========================================================================
def test_spatial_partition_shape_and_normalisation():
    graph = SkeletonGraph(head_removal=True, strategy="spatial")
    assert graph.A.shape == (3, 12, 12)
    # Partitions must sum to a column-normalised adjacency.
    total = graph.A.sum(axis=0)
    assert np.all(total >= 0)
    assert np.allclose(total.sum(axis=0), 1.0, atol=1e-5)


def test_every_joint_is_connected():
    graph = SkeletonGraph(head_removal=True)
    reach = graph.A.sum(axis=0).sum(axis=0)
    assert np.all(reach > 0), "a joint with no incoming edge would never update"


# ==========================================================================
# Normalisation
# ==========================================================================
def _fake_sequence(t=30, v=12, offset=0.0, scale=1.0):
    rng = np.random.default_rng(0)
    xy = rng.normal(0, 30, (t, v, 2)).astype(np.float32) * scale + 300.0 + offset
    score = np.full((t, v, 1), 0.9, dtype=np.float32)
    return np.concatenate([xy, score], axis=-1)


def test_torso_normalisation_is_translation_invariant():
    a = _fake_sequence()
    b = a.copy()
    b[..., :2] += 150.0  # same person, elsewhere in frame
    na = normalize_sequence(a, mode="torso")
    nb = normalize_sequence(b, mode="torso")
    assert np.allclose(na, nb, atol=1e-4)


def test_torso_normalisation_is_scale_invariant():
    a = _fake_sequence()
    b = a.copy()
    b[..., :2] = (b[..., :2] - 300.0) * 2.0 + 300.0
    na = normalize_sequence(a, mode="torso")
    nb = normalize_sequence(b, mode="torso")
    assert np.allclose(na, nb, atol=1e-3)


def test_low_confidence_joints_are_zeroed():
    seq = _fake_sequence()
    seq[:, 3, 2] = 0.05
    out = normalize_sequence(seq, mode="torso", conf_threshold=0.2)
    assert np.allclose(out[:, 3, :2], 0.0)


def test_bbox_normalisation_needs_boxes():
    with pytest.raises(ValueError):
        normalize_sequence(_fake_sequence(), mode="bbox")


def test_model_input_layout_is_channels_first():
    seq = _fake_sequence(t=30, v=12)
    x = to_model_input(normalize_sequence(seq, mode="torso"))
    assert x.shape == (3, 30, 12)


def test_velocity_doubles_the_channels_and_starts_at_zero():
    seq = _fake_sequence()
    out = add_velocity(seq)
    assert out.shape[-1] == 6
    assert np.allclose(out[0, :, 3:5], 0.0)


# ==========================================================================
# Tracker
# ==========================================================================
def test_xyah_roundtrip():
    box = np.array([10.0, 20.0, 110.0, 220.0], dtype=np.float32)
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box, atol=1e-3)


def test_iou_of_identical_boxes_is_one():
    box = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    assert np.allclose(iou_matrix(box, box), 1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    b = np.array([[50.0, 50.0, 60.0, 60.0]], dtype=np.float32)
    assert np.allclose(iou_matrix(a, b), 0.0)


def test_track_is_confirmed_only_after_n_init_hits():
    from car.config import TrackerConfig

    tracker = StrongSort(TrackerConfig(n_init=3, appearance="none"))
    box = np.array([[100.0, 100.0, 150.0, 250.0]], dtype=np.float32)
    score = np.array([0.9], dtype=np.float32)
    kpts = np.zeros((1, 12, 3), dtype=np.float32)

    assert tracker.update(box, score, kpts) == []   # hit 1, tentative
    assert tracker.update(box, score, kpts) == []   # hit 2, tentative
    out = tracker.update(box, score, kpts)          # hit 3, confirmed
    assert len(out) == 1 and out[0].track_id == 1


def test_track_id_survives_a_gap_shorter_than_max_age():
    from car.config import TrackerConfig

    tracker = StrongSort(TrackerConfig(n_init=2, max_age=10, appearance="none"))
    score = np.array([0.9], dtype=np.float32)
    kpts = np.zeros((1, 12, 3), dtype=np.float32)

    for t in range(5):
        tracker.update(
            np.array([[100.0 + t, 100.0, 150.0 + t, 250.0]], np.float32), score, kpts
        )
    empty = np.zeros((0, 4), np.float32)
    for _ in range(4):  # detector misses the person
        tracker.update(empty, np.zeros((0,), np.float32), np.zeros((0, 12, 3), np.float32))

    out = tracker.update(
        np.array([[108.0, 100.0, 158.0, 250.0]], np.float32), score, kpts
    )
    assert len(out) == 1
    assert out[0].track_id == 1, "the shopper should keep their id across a short miss"


def test_two_people_get_two_ids():
    from car.config import TrackerConfig

    tracker = StrongSort(TrackerConfig(n_init=2, appearance="none"))
    boxes = np.array(
        [[100.0, 100.0, 150.0, 250.0], [400.0, 100.0, 450.0, 250.0]], np.float32
    )
    scores = np.array([0.9, 0.9], np.float32)
    kpts = np.zeros((2, 12, 3), np.float32)
    for _ in range(3):
        out = tracker.update(boxes, scores, kpts)
    assert len({t.track_id for t in out}) == 2


# ==========================================================================
# Buffer
# ==========================================================================
def test_buffer_only_classifies_once_the_window_is_full():
    bank = BufferBank(window=30, normalize="torso", min_valid_ratio=0.0)
    kpts = _fake_sequence(t=1)[0]
    box = np.array([0.0, 0.0, 100.0, 200.0], np.float32)

    for i in range(29):
        bank.update(1, kpts, box, i)
    assert bank.model_input(1) is None

    bank.update(1, kpts, box, 29)
    x = bank.model_input(1)
    assert x is not None and x.shape == (3, 30, 12)


def test_buffer_keeps_people_separate():
    bank = BufferBank(window=5, normalize="torso", min_valid_ratio=0.0)
    box = np.array([0.0, 0.0, 100.0, 200.0], np.float32)
    for i in range(5):
        bank.update(1, np.full((12, 3), 1.0, np.float32), box, i)
        bank.update(2, np.full((12, 3), 9.0, np.float32), box, i)

    ids, batch = bank.batch_inputs()
    assert ids == [1, 2]
    assert batch.shape == (2, 3, 5, 12)


def test_buffer_rejects_a_mostly_empty_window():
    bank = BufferBank(window=5, normalize="torso", min_valid_ratio=0.5, conf_threshold=0.2)
    kpts = np.zeros((12, 3), np.float32)  # every score is 0
    box = np.array([0.0, 0.0, 100.0, 200.0], np.float32)
    for i in range(5):
        bank.update(1, kpts, box, i)
    assert bank.model_input(1) is None


def test_prediction_smoothing_takes_the_majority():
    bank = BufferBank(window=3, smooth_window=5)
    box = np.array([0.0, 0.0, 10.0, 10.0], np.float32)
    bank.update(1, np.zeros((12, 3), np.float32), box, 0)
    for label, score in [(0, 0.5), (2, 0.9), (2, 0.8), (1, 0.4), (2, 0.7)]:
        bank.record_prediction(1, label, score)

    label, score = bank.smoothed(1)
    assert label == 2
    assert score == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_stale_buffers_are_pruned():
    bank = BufferBank(window=5, max_idle=10)
    box = np.array([0.0, 0.0, 10.0, 10.0], np.float32)
    bank.update(7, np.zeros((12, 3), np.float32), box, 0)
    assert len(bank) == 1
    bank.prune(frame_idx=50, alive_ids=[])
    assert len(bank) == 0


# ==========================================================================
# Models
# ==========================================================================
@pytest.mark.parametrize("name", ["stgcn", "lstm"])
def test_models_accept_the_pipeline_tensor_and_emit_one_logit_per_class(name):
    import torch

    cfg = load_config()
    model = build_model(name, cfg).eval()
    x = torch.zeros(4, cfg.in_channels, cfg.sequence.window, cfg.num_keypoints)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, cfg.num_classes)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["stgcn", "lstm"])
def test_models_handle_a_single_sample(name):
    """Batch size 1 is the common live case and trips BatchNorm in train mode."""
    import torch

    cfg = load_config()
    model = build_model(name, cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, cfg.in_channels, cfg.sequence.window, cfg.num_keypoints))
    assert out.shape == (1, cfg.num_classes)


def test_lstm_is_smaller_than_stgcn():
    """The thesis' central trade-off: LSTM is ~600k parameters lighter."""
    cfg = load_config()
    lstm = build_model("lstm", cfg).num_parameters()
    stgcn = build_model("stgcn", cfg).num_parameters()
    assert lstm < stgcn
    assert stgcn - lstm > 400_000


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        build_model("transformer", load_config())


# ==========================================================================
# Config
# ==========================================================================
def test_defaults_match_the_paper():
    cfg = Config()
    assert cfg.sequence.window == 30          # §2, 30-frame buffer
    assert cfg.train.epochs == 300            # Table 4.2
    assert cfg.train.batch_size == 128
    assert cfg.train.val_split == 0.3         # 70/30
    assert cfg.lstm.hidden_size == 128        # §3.2
    assert cfg.lstm.num_layers == 3
    assert cfg.stgcn.channels == (64, 64, 128, 128, 256, 256)  # Fig. 3.1
    assert cfg.pose.upper_body_bonus == 0.5   # §2.1
    assert cfg.num_keypoints == 12
    assert len(ACTION_CLASSES) == 5


def test_dotted_overrides_reach_nested_fields():
    cfg = load_config(overrides=parse_overrides(["pose.conf=0.55", "train.epochs=7"]))
    assert cfg.pose.conf == 0.55
    assert cfg.train.epochs == 7


def test_background_class_is_opt_in():
    cfg = load_config(overrides={"include_background": True})
    assert cfg.num_classes == 6
    assert cfg.label_names[-1] == "Standing"


def test_head_removal_off_gives_seventeen_keypoints():
    cfg = load_config(overrides={"pose": {"head_removal": False}})
    assert cfg.num_keypoints == 17


# ==========================================================================
# Metrics
# ==========================================================================
def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    report = classification_report(y, y, list(ACTION_CLASSES))
    assert report["accuracy"] == 1.0
    assert report["macro avg"]["f1-score"] == 1.0


def test_confusion_matrix_rows_are_truth():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    cm = confusion_matrix(y_true, y_pred, 2)
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_report_ignores_classes_with_no_samples():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    report = classification_report(y_true, y_pred, list(ACTION_CLASSES))
    assert report["macro avg"]["f1-score"] == 1.0
    assert report[ACTION_CLASSES[4]]["support"] == 0


# ==========================================================================
# Synthetic fixture and windowing
# ==========================================================================
def test_synthetic_store_covers_every_class():
    from car.synthetic import generate_store

    store = generate_store(num_clips=3, frames_per_clip=250, seed=1)
    assert len(store) > 0
    assert set(np.unique(store.labels)) == set(range(len(ACTION_CLASSES)))


def test_windows_never_span_two_tracks():
    from car.datasets.sequence import SkeletonStore

    records = []
    for track in (1, 2):
        for frame in range(40):
            records.append(
                {
                    "clip": "c0",
                    "frame": frame,
                    "track_id": track,
                    "keypoints": np.full((12, 3), float(track), np.float32),
                    "box": np.zeros(4, np.float32),
                    "label": track - 1,
                    "frame_size": (640, 480),
                }
            )
    store = SkeletonStore.from_records(records)
    windows = store.build_windows(window=30, stride=5)
    assert windows
    for idx, _ in windows:
        assert len(np.unique(store.track_ids[idx])) == 1


def test_windowing_breaks_a_run_on_a_long_frame_gap():
    from car.datasets.sequence import SkeletonStore

    frames = list(range(20)) + list(range(200, 220))  # a large gap in the middle
    records = [
        {
            "clip": "c0", "frame": f, "track_id": 1,
            "keypoints": np.zeros((12, 3), np.float32),
            "box": np.zeros(4, np.float32), "label": 0, "frame_size": (640, 480),
        }
        for f in frames
    ]
    store = SkeletonStore.from_records(records)
    # Neither side of the gap is 30 frames long, so nothing should be emitted.
    assert store.build_windows(window=30, stride=1) == []


def test_store_survives_a_save_load_roundtrip(tmp_path):
    from car.datasets.sequence import SkeletonStore
    from car.synthetic import generate_store

    store = generate_store(num_clips=2, frames_per_clip=60, seed=3)
    path = tmp_path / "store.npz"
    store.save(path)
    loaded = SkeletonStore.load(path)

    assert len(loaded) == len(store)
    assert np.array_equal(loaded.labels, store.labels)
    assert np.allclose(loaded.keypoints, store.keypoints)
    assert loaded.clip_names == store.clip_names
