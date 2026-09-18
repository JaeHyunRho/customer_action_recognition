#!/usr/bin/env bash
# Run recognition on a recorded top-view clip and save every output.
#
#   bash scripts/run_video.sh data/videos/shop.mp4
#   bash scripts/run_video.sh data/videos/shop.mp4 runs/action/lstm/best.pt
set -euo pipefail

cd "$(dirname "$0")/.."

VIDEO="${1:?usage: run_video.sh <video> [action_weights]}"
WEIGHTS="${2:-runs/action/stgcn/best.pt}"
NAME="$(basename "${VIDEO%.*}")"
OUT="runs/demo/$NAME"

# Prefer the project venv when it exists; it is the one with CUDA torch.
PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

mkdir -p "$OUT"
echo "==> interpreter: $PY"
echo "==> video:       $VIDEO"
echo "==> weights:     $WEIGHTS"
echo "==> output:      $OUT"

if [ ! -f "$WEIGHTS" ]; then
    echo "!!  no action checkpoint at $WEIGHTS"
    echo "!!  running pose + tracking only; train a model with:"
    echo "      $PY -m car.train.train_action --store <skeletons.npz> --model stgcn"
    WEIGHT_ARG=()
else
    WEIGHT_ARG=(--weights "$WEIGHTS")
fi

"$PY" tools/run_recognition.py \
    --source "$VIDEO" \
    "${WEIGHT_ARG[@]}" \
    --save-video "$OUT/annotated.mp4" \
    --save-csv "$OUT/actions.csv" \
    --save-summary "$OUT/summary.json" \
    --no-display \
    "${@:3}"

echo
echo "==> wrote:"
ls -lh "$OUT"
