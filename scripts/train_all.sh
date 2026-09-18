#!/usr/bin/env bash
# Train both action models on one skeleton store and compare them,
# reproducing the thesis' Table 4.4.
#
#   bash scripts/train_all.sh data/processed/merl_skeletons.npz
#   EPOCHS=50 bash scripts/train_all.sh data/processed/demo_skeletons.npz
set -euo pipefail

cd "$(dirname "$0")/.."

STORE="${1:?usage: train_all.sh <skeletons.npz>}"
EPOCHS="${EPOCHS:-300}"
CONFIG="${CONFIG:-configs/default.yaml}"

PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

if [ ! -f "$STORE" ]; then
    echo "!!  $STORE not found."
    echo "!!  Build one with tools/extract_skeletons.py, or for a smoke test:"
    echo "      $PY tools/make_demo_skeletons.py --out $STORE"
    exit 1
fi

echo "==> interpreter: $PY"
echo "==> store:       $STORE"
echo "==> epochs:      $EPOCHS"
"$PY" -c "import torch; print('==> cuda:        ' + str(torch.cuda.is_available()))"

for MODEL in stgcn lstm; do
    echo
    echo "=============================================================="
    echo " training $MODEL"
    echo "=============================================================="
    "$PY" -m car.train.train_action \
        --store "$STORE" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --out "runs/action/$MODEL" \
        --set "train.epochs=$EPOCHS"
done

echo
echo "=============================================================="
echo " comparison (thesis Table 4.4)"
echo "=============================================================="
"$PY" -m car.train.evaluate \
    --checkpoint runs/action/stgcn/best.pt runs/action/lstm/best.pt \
    --store "$STORE" \
    --out runs/eval

echo
echo "==> artefacts:"
echo "    runs/action/stgcn/{best.pt,summary.json,confusion_matrix.png,history.png}"
echo "    runs/action/lstm/{best.pt,summary.json,confusion_matrix.png,history.png}"
echo "    runs/eval/comparison.json"
