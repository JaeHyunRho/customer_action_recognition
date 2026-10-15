#!/usr/bin/env python3
"""Data preparation and scoring for the subject-split protocol.

Subjects 1-20 train, 21-26 validate, 27-41 test. ``car.train.train_action``
does the training and the window-level test score; this tool does the rest::

    prepare    keep one person per frame (and derive a 12-joint copy of a 17-joint store)
    intersect  keep only the frames every given pose model detected
    stats      windows, bridged gaps and real time span of a store
    end2end    accuracy over every ground-truth end frame, missed detections counted wrong
    int8       dynamic weight-only INT8 copy of a checkpoint, scored on the test store
    aggregate  seeds -> mean, subject bootstrap interval, paired model difference
    report     every finished run under runs/subject_split/action in one JSON and one table
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import Config, load_config  # noqa: E402
from car.datasets.merl import discover_clips, split_official, video_frame_count  # noqa: E402
from car.datasets.sequence import (  # noqa: E402
    SkeletonStore, SkeletonWindowDataset, keep_primary_track,
)
from car.keypoints import COCO17_NAMES, HEAD_INDICES, drop_head  # noqa: E402
from car.train.train_action import _cut_windows, _subject_of, window_gap_stats  # noqa: E402
from car.utils.metrics import classification_report, confusion_matrix  # noqa: E402

LOGGER = logging.getLogger("subject_split")
NO_OUTPUT = "No output"


# --------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------
def _subset(store: SkeletonStore, keep: np.ndarray, keypoints: Optional[np.ndarray] = None) -> SkeletonStore:
    return SkeletonStore(
        keypoints=(store.keypoints if keypoints is None else keypoints)[keep],
        boxes=store.boxes[keep], labels=store.labels[keep], clip_ids=store.clip_ids[keep],
        track_ids=store.track_ids[keep], frames=store.frames[keep],
        clip_names=store.clip_names, frame_size=store.frame_size[keep],
    )


def _keys(store: SkeletonStore) -> List[Tuple[str, int]]:
    return [(store.clip_names[int(c)], int(f)) for c, f in zip(store.clip_ids, store.frames)]


def cmd_prepare(args) -> int:
    raw = SkeletonStore.load(args.raw)
    store = keep_primary_track(raw)
    keys = _keys(store)
    assert len(set(keys)) == len(keys), "more than one row per frame after keep_primary_track"
    info = {
        "raw": args.raw, "rows_raw": len(raw), "rows": len(store),
        "tracks_raw": int(len(np.unique(raw.track_ids))), "clips": len(store.clip_names),
        "joints": store.num_keypoints,
    }
    store.save(args.out)
    LOGGER.info("%s -> %s: %d -> %d rows, %d joints, track id %s",
                args.raw, args.out, len(raw), len(store), store.num_keypoints,
                np.unique(store.track_ids).tolist())

    if args.drop_head_out:
        if store.num_keypoints != 17:
            raise SystemExit(f"--drop-head-out needs a 17-joint store, got {store.num_keypoints}")
        head = store.keypoints[:, list(HEAD_INDICES), 2]
        info["head_joints"] = {
            "names": [COCO17_NAMES[i] for i in HEAD_INDICES],
            "mean_score": head.mean(0).round(4).tolist(),
            "share_zeroed_at_0.2": (head <= 0.2).mean(0).round(4).tolist(),
            "share_zeroed_at_0.2_all": round(float((head <= 0.2).mean()), 4),
        }
        body = store.keypoints[:, [i for i in range(17) if i not in HEAD_INDICES], 2]
        info["body_joints_share_zeroed_at_0.2"] = round(float((body <= 0.2).mean()), 4)
        twelve = _subset(store, np.ones(len(store), bool), keypoints=drop_head(store.keypoints))
        twelve.save(args.drop_head_out)
        info["joints_after_drop_head"] = twelve.num_keypoints
        LOGGER.info("drop_head: %d -> %d joints, saved %s; head joints zeroed at 0.2: %.1f%%",
                    store.num_keypoints, twelve.num_keypoints, args.drop_head_out,
                    100 * info["head_joints"]["share_zeroed_at_0.2_all"])
    Path(args.out).with_suffix(".prepare.json").write_text(json.dumps(info, indent=2))
    return 0


def cmd_intersect(args) -> int:
    if len(args.stores) != len(args.outs):
        raise SystemExit("--stores and --outs must have the same length")
    stores = [SkeletonStore.load(p) for p in args.stores]
    joints = {s.num_keypoints for s in stores}
    if len(joints) != 1:
        raise SystemExit(f"joint counts differ across pose models ({joints}); the comparison is void")
    common = set(_keys(stores[0]))
    for s in stores[1:]:
        common &= set(_keys(s))
    reference = None
    for path, out, store in zip(args.stores, args.outs, stores):
        keys = _keys(store)
        keep = np.array([k in common for k in keys])
        sub = _subset(store, keep)
        order = sorted(range(len(sub)), key=lambda i: (sub.clip_names[int(sub.clip_ids[i])], int(sub.frames[i])))
        sub = _subset(sub, np.array(order, dtype=np.int64))
        sub.save(out)
        frames = _keys(sub)
        if reference is None:
            reference = frames
        elif frames != reference:
            raise SystemExit(f"{out}: frames differ from the first store after intersecting")
        LOGGER.info("%s: %d -> %d rows (%d joints) -> %s", path, len(store), len(sub), sub.num_keypoints, out)
    LOGGER.info("intersection: %d frames common to %d pose models", len(common), len(stores))
    return 0


def cmd_stats(args) -> int:
    cfg = load_config(args.config, {})
    out = {}
    for path in args.stores:
        store = SkeletonStore.load(path)
        windows = _cut_windows(store, cfg)
        stats = window_gap_stats(store, windows, cfg)
        stats["rows"] = len(store)
        stats["joints"] = store.num_keypoints
        stats["class_windows"] = np.bincount([l for _, l in windows], minlength=cfg.num_classes).tolist()
        out[path] = stats
        LOGGER.info("%s: %s", path, stats)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _load(checkpoint: str):
    """Model and the config it was trained with."""
    import tempfile

    import torch
    import yaml

    from car.models import load_model

    model, ckpt = load_model(checkpoint, map_location="cpu")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(ckpt["config"], fh, sort_keys=False, allow_unicode=True)
    cfg = load_config(fh.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, cfg, device


def _predict(model, cfg: Config, store: SkeletonStore, windows, device, batch: int = 256) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    if not windows:
        return np.zeros(0, dtype=np.int64)
    ds = SkeletonWindowDataset(store, cfg, windows=windows, augment=False)
    preds = []
    model = model.to(device).eval()
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2):
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def end_to_end(model, cfg: Config, store: SkeletonStore, merl_root: str, device, stride: int = 5) -> Dict:
    """Accuracy over every ground-truth end frame of the test videos.

    The denominator comes from the ``.mat`` labels, not from what the pose
    model happened to detect: on the 10 fps sample grid every ``stride``-th
    sample is an end frame, and it counts if its label is one of the actions.
    The window for an end frame is the 30 most recent detections ending at it
    whose neighbours are at most ``max_gap`` source frames apart. No detection
    at the end frame, or fewer than 30 such detections, is an error.
    """
    import cv2

    names = list(cfg.label_names)
    n_cls, window, max_gap = len(names), int(cfg.sequence.window), int(cfg.sequence.max_gap)
    rows_of: Dict[str, np.ndarray] = {}
    order = np.lexsort((store.frames, store.clip_ids))
    for i in order:
        rows_of.setdefault(store.clip_names[int(store.clip_ids[i])], []).append(int(i))  # type: ignore[union-attr]

    test_clips = split_official(discover_clips(merl_root))[2]
    windows, meta, failures = [], [], {"no_detection_at_end": 0, "fewer_than_window": 0}
    for clip in test_clips:
        total = video_frame_count(clip.video_path)
        cap = cv2.VideoCapture(str(clip.video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        step = max(1, int(round(src_fps / max(cfg.sequence.fps, 1))))
        labels = clip.frame_labels(total)
        rows = np.asarray(rows_of.get(clip.clip_id, []), dtype=np.int64)
        frames = store.frames[rows] if len(rows) else np.zeros(0, dtype=np.int64)
        pos_of = {int(f): p for p, f in enumerate(frames)}
        for end in np.arange(0, total, step)[::stride]:
            y = int(labels[end])
            if y >= n_cls:
                continue  # background end frames are outside the denominator
            pos = pos_of.get(int(end))
            if pos is None:
                failures["no_detection_at_end"] += 1
                meta.append((clip.subject, y, False))
                continue
            idx = rows[max(0, pos - window + 1): pos + 1]
            gaps = np.diff(store.frames[idx])
            if len(idx) < window or (gaps <= 0).any() or (gaps > max_gap).any():
                failures["fewer_than_window"] += 1
                meta.append((clip.subject, y, False))
                continue
            windows.append((idx, y))
            meta.append((clip.subject, y, True))

    preds = _predict(model, cfg, store, windows, device)
    subject = np.array([m[0] for m in meta])
    y_true = np.array([m[1] for m in meta])
    scored = np.array([m[2] for m in meta])
    y_pred = np.full(len(meta), n_cls, dtype=np.int64)   # n_cls = no output
    y_pred[scored] = preds
    correct = y_pred == y_true

    cm = confusion_matrix(y_true, y_pred, n_cls + 1)[:n_cls]
    report = classification_report(y_true, y_pred, names + [NO_OUTPUT])
    f1 = [report[n]["f1-score"] for n in names]
    per_subject = {
        str(s): {"end_frames": int((subject == s).sum()), "correct": int(correct[subject == s].sum()),
                 "accuracy": round(float(correct[subject == s].mean()), 4)}
        for s in sorted(set(subject.tolist()))
    }
    return {
        "end_frames": int(len(meta)),
        "end_frames_per_class": np.bincount(y_true, minlength=n_cls).tolist(),
        "scored": int(scored.sum()),
        "failures": failures,
        "coverage": round(float(scored.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "accuracy_when_scored": round(float(correct[scored].mean()), 4) if scored.any() else None,
        "macro_f1": round(float(np.mean(f1)), 4),
        "confusion_matrix_rows_true_cols_pred_plus_no_output": cm.tolist(),
        "per_subject": per_subject,
        "videos": len(test_clips),
        "end_frame_rule": f"10 fps sample index j with j % {stride} == 0",
    }


def cmd_end2end(args) -> int:
    model, cfg, device = _load(args.checkpoint)
    store = SkeletonStore.load(args.store)
    if store.num_keypoints != cfg.num_keypoints:
        raise SystemExit(f"store has {store.num_keypoints} joints, checkpoint expects {cfg.num_keypoints}")
    result = end_to_end(model, cfg, store, args.merl, device)
    result.update({"checkpoint": args.checkpoint, "store": args.store})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    LOGGER.info("%s on %s: end-to-end accuracy %.4f over %d end frames (coverage %.3f, failures %s)",
                args.checkpoint, args.store, result["accuracy"], result["end_frames"],
                result["coverage"], result["failures"])
    return 0


def cmd_int8(args) -> int:
    import torch

    from car.export.quantize import quantize_torch_int8

    model, cfg, _ = _load(args.checkpoint)
    store = SkeletonStore.load(args.test_store)
    windows = _cut_windows(store, cfg)
    y_true = np.array([l for _, l in windows])
    cpu = torch.device("cpu")
    fp32 = _predict(model, cfg, store, windows, cpu)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantized, saved = quantize_torch_int8(model, out_path)
    int8 = _predict(quantized, cfg, store, windows, cpu)
    names = list(cfg.label_names)
    result = {
        "checkpoint": args.checkpoint, "test_store": args.test_store, "windows": len(windows),
        "fp32_accuracy": round(float((fp32 == y_true).mean()), 4),
        "int8_accuracy": round(float((int8 == y_true).mean()), 4),
        "int8_macro_f1": round(float(classification_report(y_true, int8, names)["macro avg"]["f1-score"]), 4),
        "agreement_fp32_int8": round(float((fp32 == int8).mean()), 4),
        "fp32_bytes": Path(args.checkpoint).stat().st_size,
        "int8_bytes": Path(saved).stat().st_size,
        "int8_path": str(saved),
    }
    out_path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    LOGGER.info("%s", result)
    return 0


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def bootstrap_mean(values: np.ndarray, n: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n, len(values)))
    means = values[draws].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _subject_table(run_dirs: Sequence[str], source: str) -> Tuple[List[str], np.ndarray, List[Dict]]:
    """(subjects, accuracy[seed, subject], per-seed headline dicts)."""
    per_seed, heads = [], []
    for d in run_dirs:
        if source == "test":
            s = json.loads((Path(d) / "summary.json").read_text())
            subj = s["test_per_subject"]
            heads.append({"run": d, "seed": s.get("seed"), "accuracy": s["test_accuracy"],
                          "macro_f1": s["test_macro_f1"], "val_accuracy": s["val_accuracy"],
                          "best_epoch": s["best_epoch"], "windows": s["test_windows"]})
        else:
            s = json.loads((Path(d) / source).read_text())
            subj = s["per_subject"]
            heads.append({"run": d, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                          "coverage": s["coverage"], "end_frames": s["end_frames"]})
        per_seed.append(subj)
    subjects = sorted(per_seed[0], key=int)
    for subj in per_seed:
        assert sorted(subj, key=int) == subjects, "runs cover different subjects"
    table = np.array([[subj[k]["accuracy"] for k in subjects] for subj in per_seed])
    return subjects, table, heads


def summarise(run_dirs: Sequence[str], source: str = "test") -> Dict:
    subjects, table, heads = _subject_table(run_dirs, source)
    acc = np.array([h["accuracy"] for h in heads])
    f1 = np.array([h["macro_f1"] for h in heads])
    by_subject = table.mean(0)                      # averaged over seeds, one value per subject
    lo, hi = bootstrap_mean(by_subject)
    return {
        "source": source, "seeds": heads,
        "accuracy_mean": round(float(acc.mean()), 4), "accuracy_std": round(float(acc.std(ddof=1)), 4) if len(acc) > 1 else None,
        "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(float(f1.std(ddof=1)), 4) if len(f1) > 1 else None,
        "subjects": subjects,
        "subject_accuracy_seed_mean": by_subject.round(4).tolist(),
        "subject_mean_accuracy": round(float(by_subject.mean()), 4),
        "subject_bootstrap_ci95": [round(lo, 4), round(hi, 4)],
        "n_subjects": len(subjects), "bootstrap_resamples": 10000,
    }


def cmd_aggregate(args) -> int:
    out = {"a": summarise(args.runs, args.source)}
    if args.versus:
        out["b"] = summarise(args.versus, args.source)
        a = np.array(out["a"]["subject_accuracy_seed_mean"])
        b = np.array(out["b"]["subject_accuracy_seed_mean"])
        assert out["a"]["subjects"] == out["b"]["subjects"], "the two models cover different subjects"
        lo, hi = bootstrap_mean(a - b)
        out["paired_a_minus_b"] = {
            "mean_difference": round(float((a - b).mean()), 4),
            "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            "subjects_where_a_is_higher": int((a > b).sum()), "n_subjects": len(a),
        }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    for key in ("a", "b"):
        if key in out:
            r = out[key]
            LOGGER.info("%s: accuracy %.4f +- %s over seeds %s | macro-F1 %.4f | subject mean %.4f, 95%% CI %s",
                        key, r["accuracy_mean"], r["accuracy_std"], [h["accuracy"] for h in r["seeds"]],
                        r["macro_f1_mean"], r["subject_mean_accuracy"], r["subject_bootstrap_ci95"])
    if "paired_a_minus_b" in out:
        LOGGER.info("paired a - b: %s", out["paired_a_minus_b"])
    return 0


def _paired(a: Dict, b: Dict) -> Optional[Dict]:
    if a is None or b is None or a["subjects"] != b["subjects"]:
        return None
    x = np.array(a["subject_accuracy_seed_mean"]) - np.array(b["subject_accuracy_seed_mean"])
    lo, hi = bootstrap_mean(x)
    return {"mean_difference": round(float(x.mean()), 4), "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            "subjects_where_first_is_higher": int((x > 0).sum()), "n_subjects": len(x)}


def cmd_report(args) -> int:
    root = Path(args.root)
    groups: Dict[str, List[str]] = {}
    for d in sorted(root.iterdir()):
        m = re.match(r"^(e1|e2full|e2isect|e2j17)_(.+)_(stgcn|lstm)_s(\d+)$", d.name)
        if m and (d / "summary.json").exists():
            groups.setdefault("_".join(m.groups()[:3]), []).append(str(d))

    out: Dict = {"groups": {}, "paired": {}}
    for name, runs in groups.items():
        first = json.loads((Path(runs[0]) / "summary.json").read_text())
        g: Dict = {
            "runs": len(runs), "test": summarise(runs, "test"),
            "windows": {k: first[k] for k in ("train_windows", "val_windows", "test_windows",
                                              "train_class_windows", "train_class_windows_before_balancing",
                                              "test_class_windows")},
            "window_stats_test": first.get("window_stats", {}).get("test"),
        }
        if all((Path(r) / "end2end.json").exists() for r in runs):
            g["end2end"] = summarise(runs, "end2end.json")
            e = json.loads((Path(runs[0]) / "end2end.json").read_text())
            g["end2end"].update({k: e[k] for k in ("end_frames", "coverage", "failures")})
        if all((Path(r) / "int8.json").exists() for r in runs):
            q = [json.loads((Path(r) / "int8.json").read_text()) for r in runs]
            g["int8"] = {k: round(float(np.mean([x[k] for x in q])), 4)
                         for k in ("fp32_accuracy", "int8_accuracy", "agreement_fp32_int8")}
            g["int8"].update({"fp32_bytes": q[0]["fp32_bytes"], "int8_bytes": q[0]["int8_bytes"],
                              "per_seed_int8": [x["int8_accuracy"] for x in q]})
        out["groups"][name] = g

    def get(name: str, source: str) -> Optional[Dict]:
        return out["groups"].get(name, {}).get(source)

    def pair(label: str, a: str, b: str, source: str) -> None:
        res = _paired(get(a, source), get(b, source))
        if res is not None:
            out["paired"][label] = {"first": a, "second": b, "source": source, **res}

    for source in ("test", "end2end"):
        pair(f"e1 stgcn - lstm ({source})", "e1_custom_stgcn", "e1_custom_lstm", source)
    for model in ("stgcn", "lstm"):
        for a, b in (("custom", "teacher_x"), ("custom", "coco_n"), ("teacher_x", "coco_n")):
            pair(f"intersection {a} - {b} ({model}, test)", f"e2isect_{a}_{model}", f"e2isect_{b}_{model}", "test")
        full = {"custom": f"e1_custom_{model}", "teacher_x": f"e2full_teacher_x_{model}", "coco_n": f"e2full_coco_n_{model}"}
        for a, b in (("custom", "teacher_x"), ("custom", "coco_n"), ("teacher_x", "coco_n")):
            pair(f"end-to-end {a} - {b} ({model})", full[a], full[b], "end2end")
        pair(f"17 joints - 12 joints, teacher ({model}, test)",
             f"e2j17_teacher_x17_{model}", f"e2full_teacher_x_{model}", "test")

    lines = ["| run group | seeds | test acc (mean ± std) | macro-F1 | subject mean [95% CI] | end-to-end acc | coverage |",
             "|---|---|---|---|---|---|---|"]
    for name, g in out["groups"].items():
        t, e = g["test"], g.get("end2end")
        seeds = " / ".join("%.4f" % h["accuracy"] for h in t["seeds"])
        lines.append(
            f"| {name} | {seeds} "
            f"| {t['accuracy_mean']:.4f} ± {t['accuracy_std'] if t['accuracy_std'] is not None else float('nan'):.4f} "
            f"| {t['macro_f1_mean']:.4f} | {t['subject_mean_accuracy']:.4f} {t['subject_bootstrap_ci95']} "
            f"| {'%.4f ± %.4f' % (e['accuracy_mean'], e['accuracy_std'] or 0) if e else '-'} "
            f"| {e['coverage'] if e else '-'} |")
    lines += ["", "| paired difference | mean | 95% CI | first higher in |", "|---|---|---|---|"]
    for label, r in out["paired"].items():
        lines.append(f"| {label} | {r['mean_difference']:+.4f} | {r['bootstrap_ci95']} "
                     f"| {r['subjects_where_first_is_higher']}/{r['n_subjects']} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="keep the largest box per frame; track id becomes 1")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drop-head-out", default=None, help="also write the 12-joint copy of a 17-joint store")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("intersect", help="keep frames detected by every pose model")
    p.add_argument("--stores", nargs="+", required=True)
    p.add_argument("--outs", nargs="+", required=True)
    p.set_defaults(fn=cmd_intersect)

    p = sub.add_parser("stats", help="window counts, bridged gaps and real time span")
    p.add_argument("--stores", nargs="+", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("end2end", help="accuracy over every ground-truth end frame")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--store", required=True, help="the test subjects' store of the pose model")
    p.add_argument("--merl", default="data/raw/merl")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_end2end)

    p = sub.add_parser("int8", help="dynamic weight-only INT8 copy, scored on the test store")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-store", required=True)
    p.add_argument("--out", required=True, help="path of the INT8 model file")
    p.set_defaults(fn=cmd_int8)

    p = sub.add_parser("aggregate", help="seed mean, subject bootstrap interval, paired difference")
    p.add_argument("--runs", nargs="+", required=True, help="run directories of one model, one per seed")
    p.add_argument("--versus", nargs="+", default=None, help="a second model's run directories")
    p.add_argument("--source", default="test",
                   help="'test' for summary.json, or the name of an end2end json inside each run dir")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_aggregate)

    p = sub.add_parser("report", help="all finished runs in one JSON and one markdown table")
    p.add_argument("--root", default="runs/subject_split/action")
    p.add_argument("--out", default="runs/subject_split/eval/report.json")
    p.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
