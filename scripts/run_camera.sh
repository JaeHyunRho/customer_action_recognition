#!/usr/bin/env bash
# Run recognition on a live camera.
#
#   bash scripts/run_camera.sh                 # USB webcam, index 0
#   bash scripts/run_camera.sh 1               # USB webcam, index 1
#   bash scripts/run_camera.sh 0 --csi         # Jetson CSI camera
#   bash scripts/run_camera.sh rtsp://...      # IP camera
#
# Extra flags are passed through, e.g. --no-display for a headless box.
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:-0}"
shift || true

WEIGHTS="${ACTION_WEIGHTS:-runs/action/stgcn/best.pt}"
OUT="runs/live/$(date +%Y%m%d_%H%M%S)"

PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

echo "==> interpreter: $PY"
echo "==> source:      $SOURCE"

# Show what the OS actually sees, so a missing camera is obvious immediately.
if [[ "$SOURCE" =~ ^[0-9]+$ ]]; then
    if ls /dev/video* >/dev/null 2>&1; then
        echo "==> video devices: $(ls /dev/video* | tr '\n' ' ')"
    else
        echo "!!  no /dev/video* devices found."
        echo "!!  USB camera: check the cable and \`dmesg | tail\`."
        echo "!!  CSI camera: pass --csi (it needs GStreamer, not V4L2)."
        exit 1
    fi
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "!!  no action checkpoint at $WEIGHTS -- pose + tracking only"
    WEIGHT_ARG=()
else
    WEIGHT_ARG=(--weights "$WEIGHTS")
    echo "==> weights:     $WEIGHTS"
fi

mkdir -p "$OUT"
echo "==> recording to $OUT   (press q or Esc in the window to stop)"

"$PY" tools/run_recognition.py \
    --source "$SOURCE" \
    "${WEIGHT_ARG[@]}" \
    --save-video "$OUT/live.mp4" \
    --save-csv "$OUT/actions.csv" \
    --save-summary "$OUT/summary.json" \
    "$@"

echo
echo "==> wrote:"
ls -lh "$OUT"
