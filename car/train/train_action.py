"""Train ST-GCN / LSTM action classifiers on extracted skeletons.

Hyper-parameters follow Table 4.2 of the thesis: 300 epochs, batch 128, Adam,
categorical cross-entropy, dropout 0.25, early stopping. The thesis does not
give a learning rate, schedule or early-stopping patience, so those are the
project's own choices and live in ``car/config.py`` marked ``# CHOICE``.

Reported numbers come from the three-store mode (``--train-store``,
``--val-store``, ``--test-store``): weights are updated on the training store
only, early stopping watches accuracy on the validation store only, and the
test store is scored exactly once, with the best validation checkpoint.
``--store`` with ``--split-by`` cuts one store in two and is kept for smoke
tests; a window-level split leaks and a clip-level one has no test set.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config, load_config, parse_overrides
from ..datasets.sequence import SkeletonStore, SkeletonWindowDataset
from ..models import build_model
from ..utils.metrics import classification_report, confusion_matrix
from ..utils.viz import plot_confusion_matrix, plot_history

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balance_windows(
    windows: List[Tuple[np.ndarray, int]],
    num_classes: int,
    ratio: float,
    seed: int = 42,
) -> List[Tuple[np.ndarray, int]]:
    """Cap each class at ``ratio`` times the rarest class's window count.

    MERL Shopping is imbalanced by a factor of nearly three between its most
    and least common action. Capping the majority classes stops the model from
    learning the prior instead of the motion, and it is the only way to get
    anything like the even class counts the thesis reports in Table 4.3.
    """
    if ratio <= 0:
        return windows

    by_class: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for item in windows:
        by_class.setdefault(item[1], []).append(item)
    if not by_class:
        return windows

    smallest = min(len(v) for v in by_class.values())
    cap = max(1, int(round(smallest * ratio)))
    rng = np.random.default_rng(seed)

    kept: List[Tuple[np.ndarray, int]] = []
    for label in sorted(by_class):
        items = by_class[label]
        if len(items) <= cap:
            kept.extend(items)
            continue
        picks = rng.choice(len(items), size=cap, replace=False)
        kept.extend(items[int(i)] for i in picks)
        LOGGER.info("class %d: %d -> %d windows", label, len(items), cap)

    rng.shuffle(kept)
    return kept


def _cut_windows(store: SkeletonStore, cfg: Config) -> List[Tuple[np.ndarray, int]]:
    """Every window of one store, labelled by its last detection."""
    return store.build_windows(
        window=cfg.sequence.window,
        stride=cfg.sequence.train_stride,
        label_mode="last",
        keep_labels=tuple(range(cfg.num_classes)),
        max_gap=cfg.sequence.max_gap,
    )


def window_gap_stats(store: SkeletonStore, windows: List[Tuple[np.ndarray, int]], cfg: Config) -> Dict:
    """How often a window bridges missed detections, and how long it really is.

    A window is 30 stored detections, not 30 consecutive samples: neighbours
    may be up to ``max_gap`` source frames apart, so a "3 s" window can cover
    more wall-clock time than that.
    """
    if not windows:
        return {"windows": 0}
    diffs = np.diff(store.frames[np.lexsort((store.frames, store.clip_ids))])
    step = int(np.bincount(diffs[(diffs > 0) & (diffs <= cfg.sequence.max_gap)]).argmax())
    src_fps = float(step * cfg.sequence.fps)
    largest = np.array([int(np.diff(store.frames[idx]).max()) for idx, _ in windows])
    span = np.array([int(store.frames[idx[-1]] - store.frames[idx[0]]) for idx, _ in windows])
    span_s = span / src_fps
    return {
        "windows": len(windows),
        "sample_step": step,
        "share_with_gap_over_step": round(float((largest > step).mean()), 4),
        "span_s_nominal": round((cfg.sequence.window - 1) * step / src_fps, 2),
        "span_s_median": round(float(np.median(span_s)), 2),
        "span_s_p95": round(float(np.percentile(span_s, 95)), 2),
        "span_s_max": round(float(span_s.max()), 2),
    }


_SUBJECT_RE = re.compile(r"^(\d+)_")


def _subject_of(clip_name: str) -> int:
    m = _SUBJECT_RE.match(clip_name)
    return int(m.group(1)) if m else -1


def per_subject_accuracy(
    store: SkeletonStore, windows: List[Tuple[np.ndarray, int]],
    preds: np.ndarray, targets: np.ndarray,
) -> Dict[str, Dict]:
    """Accuracy per ``{subject}_{session}`` subject id, the unit the CI resamples."""
    subjects = np.array([_subject_of(store.clip_names[int(store.clip_ids[idx[0]])]) for idx, _ in windows])
    out: Dict[str, Dict] = {}
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        correct = int((preds[mask] == targets[mask]).sum())
        out[str(subject)] = {
            "windows": int(mask.sum()),
            "correct": correct,
            "accuracy": round(correct / max(int(mask.sum()), 1), 4),
        }
    return out


def _split_windows(
    store: SkeletonStore,
    cfg: Config,
    by: str = "clip",
) -> Tuple[List, List]:
    """Split windows into train/val without letting one clip straddle both.

    Windows overlap heavily, so a random window-level split leaks almost
    identical sequences across the boundary and reports accuracy that will not
    survive a new store.
    """
    windows = _cut_windows(store, cfg)
    if not windows:
        raise RuntimeError(
            f"no windows of length {cfg.sequence.window} could be cut. "
            "The store may hold too few consecutive frames per track."
        )

    if by == "window":
        rng = np.random.default_rng(cfg.train.seed)
        order = rng.permutation(len(windows))
        n_val = int(round(len(order) * cfg.train.val_split))
        val_idx = set(order[:n_val].tolist())
        train = [w for i, w in enumerate(windows) if i not in val_idx]
        val = [w for i, w in enumerate(windows) if i in val_idx]
        return train, val

    clip_of = {}
    for idx, _ in windows:
        clip_of[int(store.clip_ids[idx[0]])] = True
    clips = sorted(clip_of)
    rng = np.random.default_rng(cfg.train.seed)
    shuffled = list(clips)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * cfg.train.val_split)))
    val_clips = set(shuffled[:n_val])

    train, val = [], []
    for idx, label in windows:
        bucket = val if int(store.clip_ids[idx[0]]) in val_clips else train
        bucket.append((idx, label))

    if not val or not train:
        LOGGER.warning(
            "clip-level split produced an empty side (%d train / %d val); "
            "falling back to a window-level split",
            len(train), len(val),
        )
        return _split_windows(store, cfg, by="window")
    return train, val


def _run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    amp: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss, total_n, correct = 0.0, 0, 0
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            if training:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            batch = y.size(0)
            total_loss += float(loss.item()) * batch
            total_n += batch
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            preds_all.append(pred.detach().cpu().numpy())
            targets_all.append(y.detach().cpu().numpy())

    n = max(total_n, 1)
    return (
        total_loss / n,
        correct / n,
        np.concatenate(preds_all) if preds_all else np.zeros(0, dtype=np.int64),
        np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.int64),
    )


def train(
    store_path: str,
    model_name: str,
    cfg: Config,
    output_dir: Optional[str] = None,
    split_by: str = "clip",
    augment: bool = True,
    val_store_path: Optional[str] = None,
    test_store_path: Optional[str] = None,
) -> Dict:
    """Train one classifier.

    With ``val_store_path`` (and normally ``test_store_path``) the three
    stores are used as given: ``store_path`` trains, the validation store
    picks the epoch, the test store is scored once at the end. Without them
    ``store_path`` is split by ``split_by``, which is for smoke tests only.
    """
    import torch
    from torch.utils.data import DataLoader

    set_seed(cfg.train.seed)
    out = Path(output_dir or f"{cfg.paths.runs}/action/{model_name}")
    out.mkdir(parents=True, exist_ok=True)

    store = SkeletonStore.load(store_path)
    LOGGER.info("loaded %d skeleton frames from %s", len(store), store_path)

    if val_store_path is not None:
        val_store = SkeletonStore.load(val_store_path)
        LOGGER.info("loaded %d validation frames from %s", len(val_store), val_store_path)
        train_windows = _cut_windows(store, cfg)
        val_windows = _cut_windows(val_store, cfg)
        if not train_windows or not val_windows:
            raise RuntimeError("the training or the validation store yields no windows")
    else:
        LOGGER.warning(
            "splitting one store by %s; this mode has no test set and is not "
            "the protocol reported numbers come from", split_by,
        )
        val_store = store
        train_windows, val_windows = _split_windows(store, cfg, by=split_by)
    unbalanced_counts = np.bincount(
        [label for _, label in train_windows], minlength=cfg.num_classes
    ).tolist()
    # Balance only the training side; a capped validation set would hide how
    # the model behaves on the real class distribution.
    if cfg.train.balance_ratio > 0:
        before = len(train_windows)
        train_windows = _balance_windows(
            train_windows, cfg.num_classes, cfg.train.balance_ratio, cfg.train.seed
        )
        LOGGER.info("class balancing: %d -> %d train windows", before, len(train_windows))
    train_ds = SkeletonWindowDataset(store, cfg, windows=train_windows, augment=augment)
    val_ds = SkeletonWindowDataset(val_store, cfg, windows=val_windows, augment=False)
    LOGGER.info("train: %s", train_ds.describe())
    LOGGER.info("val:   %s", val_ds.describe())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=pin,
    )

    model = build_model(model_name, cfg).to(device)
    LOGGER.info("%s: %,d parameters".replace(",d", "d"), model_name, model.num_parameters())
    LOGGER.info("device: %s", device)

    weight = None
    if cfg.train.class_balanced:
        counts = train_ds.label_counts().astype(np.float32)
        inv = np.where(counts > 0, counts.sum() / np.clip(counts, 1, None), 0.0)
        weight = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        LOGGER.info("class weights: %s", np.round(weight.cpu().numpy(), 3).tolist())

    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.01
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_metric = -np.inf
    best_epoch = -1
    patience = 0
    started = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc, _, _ = _run_epoch(
            model, train_loader, device, criterion, optimizer, amp=cfg.train.amp
        )
        va_loss, va_acc, va_pred, va_true = _run_epoch(
            model, val_loader, device, criterion, None, amp=cfg.train.amp
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        metric = va_acc if cfg.train.early_stop_metric == "val_acc" else -va_loss
        improved = metric > best_metric + 1e-5
        if improved:
            best_metric, best_epoch, patience = metric, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "val_loss": va_loss,
                    "classes": list(cfg.label_names),
                    "num_parameters": model.num_parameters(),
                },
                out / "best.pt",
            )

        if epoch % 5 == 0 or epoch == 1 or improved:
            LOGGER.info(
                "epoch %3d/%d  train %.4f/%.3f  val %.4f/%.3f%s",
                epoch, cfg.train.epochs, tr_loss, tr_acc, va_loss, va_acc,
                "  *" if improved else "",
            )

        if not improved:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                LOGGER.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break

    elapsed = time.time() - started

    # -- final evaluation with the best checkpoint ----------------------
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, val_acc, preds, targets = _run_epoch(model, val_loader, device, criterion)

    names = list(cfg.label_names)
    cm = confusion_matrix(targets, preds, len(names))
    report = classification_report(targets, preds, names)

    plot_history(history, str(out / "history.png"), title=f"{model_name} training")
    plot_confusion_matrix(cm, names, str(out / "confusion_matrix.png"),
                          title=f"Confusion Matrix ({model_name})")
    plot_confusion_matrix(cm, names, str(out / "confusion_matrix_norm.png"),
                          title=f"Confusion Matrix ({model_name}, normalised)", normalize=True)

    summary = {
        "model": model_name,
        "parameters": model.num_parameters(),
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "train_seconds": round(elapsed, 1),
        "val_accuracy": round(float(val_acc), 4),
        "classes": names,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "train_class_windows": train_ds.label_counts().tolist(),
        "train_class_windows_before_balancing": unbalanced_counts,
        "val_class_windows": val_ds.label_counts().tolist(),
        "seed": cfg.train.seed,
        "store": str(store_path),
        "val_store": str(val_store_path) if val_store_path else None,
        "test_store": str(test_store_path) if test_store_path else None,
        "config": cfg.to_dict(),
    }

    if test_store_path is not None:
        # The one and only look at the test store, with the checkpoint the
        # validation store chose.
        test_store = SkeletonStore.load(test_store_path)
        test_windows = _cut_windows(test_store, cfg)
        test_ds = SkeletonWindowDataset(test_store, cfg, windows=test_windows, augment=False)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.num_workers, pin_memory=pin,
        )
        LOGGER.info("test:  %s", test_ds.describe())
        _, test_acc, t_preds, t_targets = _run_epoch(model, test_loader, device, criterion)
        test_cm = confusion_matrix(t_targets, t_preds, len(names))
        test_report = classification_report(t_targets, t_preds, names)
        subjects = per_subject_accuracy(test_store, test_windows, t_preds, t_targets)
        summary.update({
            "val_report": report,
            "val_confusion_matrix": cm.tolist(),
            "test_windows": len(test_windows),
            "test_class_windows": test_ds.label_counts().tolist(),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_report["macro avg"]["f1-score"]), 4),
            "test_report": test_report,
            "test_confusion_matrix": test_cm.tolist(),
            "test_per_subject": subjects,
            "window_stats": {
                "train": window_gap_stats(store, train_windows, cfg),
                "val": window_gap_stats(val_store, val_windows, cfg),
                "test": window_gap_stats(test_store, test_windows, cfg),
            },
        })
        (out / "test_per_subject.json").write_text(json.dumps(subjects, indent=2))
        np.savez_compressed(
            out / "test_predictions.npz",
            y_true=t_targets, y_pred=t_preds,
            clip=np.array([test_store.clip_names[int(test_store.clip_ids[i[0]])] for i, _ in test_windows]),
            end_frame=np.array([int(test_store.frames[i[-1]]) for i, _ in test_windows]),
        )
        plot_confusion_matrix(test_cm, names, str(out / "confusion_matrix.png"),
                              title=f"Test confusion matrix ({model_name})")
        plot_confusion_matrix(test_cm, names, str(out / "confusion_matrix_norm.png"),
                              title=f"Test confusion matrix ({model_name}, normalised)",
                              normalize=True)
        LOGGER.info(
            "test (scored once): accuracy %.4f, macro f1 %.4f over %d windows, %d subjects",
            test_acc, test_report["macro avg"]["f1-score"], len(test_windows), len(subjects),
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    LOGGER.info(
        "done: val accuracy %.4f, macro f1 %.4f, %d params, %.1fs",
        val_acc, report["macro avg"]["f1-score"], model.num_parameters(), elapsed,
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a skeleton action classifier")
    parser.add_argument("--store", default=None,
                        help="single skeleton .npz, split by --split-by (smoke tests only)")
    parser.add_argument("--train-store", default=None, help="training subjects' skeleton .npz")
    parser.add_argument("--val-store", default=None,
                        help="validation subjects' .npz; early stopping watches only this")
    parser.add_argument("--test-store", default=None,
                        help="test subjects' .npz; scored once with the best checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="overrides train.seed")
    parser.add_argument("--model", default="stgcn", choices=["stgcn", "lstm"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--split-by", default="clip", choices=["clip", "window"])
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = parse_overrides(args.set)
    if args.seed is not None:
        overrides["train.seed"] = args.seed
    cfg = load_config(args.config, overrides)
    if args.train_store:
        if not (args.val_store and args.test_store):
            parser.error("--train-store needs --val-store and --test-store")
        if args.store:
            parser.error("give either --store or the three --*-store files, not both")
    elif not args.store:
        parser.error("give --train-store/--val-store/--test-store (or --store for a smoke test)")
    train(
        store_path=args.train_store or args.store,
        model_name=args.model,
        cfg=cfg,
        output_dir=args.out,
        split_by=args.split_by,
        augment=not args.no_augment,
        val_store_path=args.val_store,
        test_store_path=args.test_store,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
