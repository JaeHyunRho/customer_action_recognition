#!/usr/bin/env python3
"""Run the subject-split protocol (E1, E2) as a restartable job queue.

Skeleton extraction is started separately (``runs/subject_split/chain_*.sh``)
because it takes hours; this queue picks each job up as soon as its inputs
exist, runs one job at a time, and skips whatever is already done, so it can be
stopped and started again at any point::

    .venv/bin/python tools/run_subject_split.py            # run until nothing is left
    .venv/bin/python tools/run_subject_split.py --dry-run  # list jobs and their state

Jobs, in priority order
    prepare   one person per frame for every extracted store; 12-joint copy of the teacher
    isect     frames that Custom, COCO nano and the teacher all detected
    e1        Custom skeletons, ST-GCN and LSTM, seeds 42/43/44, then end-to-end and INT8
    e2full    the same training on COCO nano and on teacher skeletons, then end-to-end
    e2isect   the three pose models on the intersection frames
    e2j17     teacher skeletons with all 17 joints (compare with e2full teacher, 12 joints)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/processed/subject_split"
R = ROOT / "runs/subject_split"
PY = sys.executable
SPLITS = ("train", "val", "test")
MODELS = ("stgcn", "lstm")
SEEDS = (42, 43, 44)
# Settings of the existing real-data runs; balance_ratio defaults to 0 and must be given.
TRAIN_SET = ["train.balance_ratio=1.5", "train.epochs=120", "train.early_stop_patience=20"]
TOOL = [PY, "tools/subject_split.py"]


@dataclass
class Job:
    name: str
    stage: str
    needs: List[Path]
    makes: List[Path]
    cmd: List[str]
    log: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.log = self.log or R / "logs" / f"{self.name}.log"

    def done(self) -> bool:
        return all(p.exists() for p in self.makes)

    def ready(self) -> bool:
        return all(p.exists() for p in self.needs)

    def failed(self) -> bool:
        return (R / "failed" / self.name).exists()


def build_jobs() -> List[Job]:
    jobs: List[Job] = []

    # The sidecar .json is written after the .npz is complete, so it is the readiness signal.
    for pose in ("custom", "coco_n"):
        for sp in SPLITS:
            jobs.append(Job(
                f"prep_{pose}_{sp}", "prepare",
                [D / f"{pose}_{sp}_raw.json"], [D / f"{pose}_{sp}.prepare.json"],
                TOOL + ["prepare", "--raw", str(D / f"{pose}_{sp}_raw.npz"), "--out", str(D / f"{pose}_{sp}.npz")],
            ))
    for sp in SPLITS:
        jobs.append(Job(
            f"prep_teacher_{sp}", "prepare",
            [D / f"teacher_x17_{sp}_raw.json"], [D / f"teacher_x17_{sp}.prepare.json"],
            TOOL + ["prepare", "--raw", str(D / f"teacher_x17_{sp}_raw.npz"),
                    "--out", str(D / f"teacher_x17_{sp}.npz"),
                    "--drop-head-out", str(D / f"teacher_x_{sp}.npz")],
        ))
    trio = ("custom", "coco_n", "teacher_x")
    for sp in SPLITS:
        jobs.append(Job(
            f"isect_{sp}", "isect",
            [D / f"custom_{sp}.prepare.json", D / f"coco_n_{sp}.prepare.json", D / f"teacher_x17_{sp}.prepare.json"],
            [D / f"{p}_{sp}_isect.npz" for p in trio] + [R / "eval" / f"isect_{sp}.done"],
            ["bash", "-c",
             " ".join(TOOL + ["intersect", "--stores"] + [str(D / f"{p}_{sp}.npz") for p in trio]
                      + ["--outs"] + [str(D / f"{p}_{sp}_isect.npz") for p in trio])
             + f" && touch {R / 'eval' / f'isect_{sp}.done'}"],
        ))

    def train(stage: str, pose: str, suffix: str = "", extra: List[str] = (), ready_marker=None) -> List[Path]:
        runs = []
        for seed in SEEDS:
            for model in MODELS:
                name = f"{stage}_{pose}_{model}_s{seed}"
                run = R / "action" / name
                stores = [D / f"{pose}_{sp}{suffix}.npz" for sp in SPLITS]
                jobs.append(Job(
                    name, stage,
                    ready_marker or stores, [run / "summary.json"],
                    [PY, "-m", "car.train.train_action", "--train-store", str(stores[0]),
                     "--val-store", str(stores[1]), "--test-store", str(stores[2]),
                     "--model", model, "--seed", str(seed), "--out", str(run),
                     "--set", *TRAIN_SET, *extra],
                ))
                runs.append(run)
        return runs

    def end_to_end(stage: str, pose: str, runs: List[Path]) -> None:
        for run in runs:
            jobs.append(Job(
                f"e2e_{run.name}", stage,
                [run / "summary.json"], [run / "end2end.json"],
                TOOL + ["end2end", "--checkpoint", str(run / "best.pt"),
                        "--store", str(D / f"{pose}_test.npz"), "--out", str(run / "end2end.json")],
            ))

    prepared = lambda pose: [D / f"{pose}_{sp}.prepare.json" for sp in SPLITS]  # noqa: E731
    teacher_ready = prepared("teacher_x17")
    isect_ready = [R / "eval" / f"isect_{sp}.done" for sp in SPLITS]

    e1 = train("e1", "custom", ready_marker=prepared("custom"))
    end_to_end("e1", "custom", e1)
    for run in e1:
        if "_lstm_" in run.name:
            jobs.append(Job(
                f"int8_{run.name}", "e1",
                [run / "summary.json"], [run / "int8.json"],
                TOOL + ["int8", "--checkpoint", str(run / "best.pt"),
                        "--test-store", str(D / "custom_test.npz"), "--out", str(run / "int8.pt")],
            ))

    end_to_end("e2full", "coco_n", train("e2full", "coco_n", ready_marker=prepared("coco_n")))
    end_to_end("e2full", "teacher_x", train("e2full", "teacher_x", ready_marker=teacher_ready))
    for pose in trio:
        train("e2isect", pose, suffix="_isect", ready_marker=isect_ready)
    train("e2j17", "teacher_x17", extra=["pose.head_removal=false"], ready_marker=teacher_ready)
    return jobs


def available_gb() -> float:
    """MemAvailable; the board is shared and its GPU memory is the same pool."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def write_status(jobs: List[Job]) -> None:
    stages = {}
    for j in jobs:
        s = stages.setdefault(j.stage, {"done": [], "failed": [], "pending": []})
        s["done" if j.done() else "failed" if j.failed() else "pending"].append(j.name)
    (R / "queue_status.json").write_text(json.dumps(
        {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
         "stages": {k: {kk: (len(vv) if kk == "done" else vv) for kk, vv in v.items()} for k, v in stages.items()}},
        indent=1))
    for stage, s in stages.items():
        marker = R / f"stage_{stage}.done"
        if not s["pending"] and not marker.exists():
            marker.write_text(json.dumps({"failed": s["failed"]}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-idle-hours", type=float, default=30.0,
                    help="give up if no job becomes runnable for this long")
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="do not start a job while less memory than this is available")
    args = ap.parse_args()
    for sub in ("logs", "failed", "eval", "action"):
        (R / sub).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()

    if args.dry_run:
        for j in jobs:
            state = "done" if j.done() else "failed" if j.failed() else "ready" if j.ready() else "waiting"
            print(f"{state:8s} {j.stage:8s} {j.name}")
        print(len(jobs), "jobs")
        return 0

    attempts, last_progress = {}, time.time()
    while True:
        write_status(jobs)
        pending = [j for j in jobs if not j.done() and not j.failed()]
        if not pending:
            break
        runnable = [j for j in pending if j.ready()]
        if not runnable:
            if time.time() - last_progress > args.max_idle_hours * 3600:
                print("giving up: nothing runnable; waiting on", [j.name for j in pending][:5], flush=True)
                break
            time.sleep(60)
            continue
        if available_gb() < args.min_free_gb:
            time.sleep(60)
            continue
        job = runnable[0]
        started = time.time()
        print(f"{time.strftime('%m-%d %H:%M:%S')} start {job.name}", flush=True)
        with job.log.open("a") as log:
            rc = subprocess.call(job.cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        mins = (time.time() - started) / 60
        last_progress = time.time()
        if rc == 0 and job.done():
            print(f"{time.strftime('%m-%d %H:%M:%S')} done  {job.name} ({mins:.1f} min)", flush=True)
            continue
        attempts[job.name] = attempts.get(job.name, 0) + 1
        print(f"{time.strftime('%m-%d %H:%M:%S')} FAIL  {job.name} rc={rc} attempt {attempts[job.name]}", flush=True)
        if attempts[job.name] >= 2:
            (R / "failed" / job.name).write_text(f"rc={rc}; see {job.log}\n")
    write_status(jobs)
    failed = [j.name for j in jobs if j.failed()]
    print("queue finished; failed:", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
