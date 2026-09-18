#!/usr/bin/env bash
# Set up a CUDA-enabled Python environment for this project on a Jetson.
#
# The stock pip `torch` wheel is built for a different CUDA than JetPack ships,
# so it silently falls back to CPU. This installs NVIDIA's Jetson wheels into a
# virtualenv, leaving the system Python untouched.
#
#   bash scripts/setup_jetson.sh            # JetPack 6 / CUDA 12.6
#   CUDA_TAG=cu128 bash scripts/setup_jetson.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

CUDA_TAG="${CUDA_TAG:-cu126}"
JP_TAG="${JP_TAG:-jp6}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
INDEX="https://pypi.jetson-ai-lab.io/${JP_TAG}/${CUDA_TAG}"

echo "==> project:  $PROJECT_ROOT"
echo "==> venv:     $VENV"
echo "==> index:    $INDEX"

if [ -f /etc/nv_tegra_release ]; then
    echo "==> board:    $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
    head -1 /etc/nv_tegra_release
else
    echo "!!  /etc/nv_tegra_release not found -- this does not look like a Jetson."
    echo "!!  On a desktop, install torch from pytorch.org instead."
    exit 1
fi

# ---------------------------------------------------------------- venv
if [ ! -d "$VENV" ]; then
    echo "==> creating virtualenv (system site-packages, so JetPack's OpenCV and"
    echo "    TensorRT stay available)"
    if python3 -m venv --system-site-packages "$VENV" 2>/dev/null; then
        :
    else
        echo "    python3-venv missing; falling back to virtualenv"
        python3 -m pip install --user --quiet virtualenv
        python3 -m virtualenv --system-site-packages "$VENV"
    fi
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ------------------------------------------------------------- pytorch
# Install from the Jetson index ALONE. Adding PyPI as an extra index lets pip
# pick PyPI's same-version aarch64 wheel, which is a CPU build -- exactly the
# failure this script exists to avoid. Dependencies come separately below.
echo "==> installing torch $TORCH_VERSION and torchvision $TORCHVISION_VERSION"
"$PIP" install --upgrade pip --quiet
"$PIP" install --no-deps --index-url "$INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

echo "==> installing torch's own dependencies from PyPI"
"$PIP" install filelock typing-extensions sympy networkx jinja2 fsspec pillow

# --------------------------------------------------------- project deps
# --no-deps so ultralytics cannot pull a generic torch wheel over the Jetson one.
echo "==> installing ultralytics (without dependencies)"
"$PIP" install --no-deps ultralytics ultralytics-thop

echo "==> installing the rest"
"$PIP" install \
    numpy scipy pandas PyYAML matplotlib tqdm psutil py-cpuinfo \
    polars nvidia-ml-py onnx onnxruntime pytest

# ------------------------------------------------------------- weights
if [ ! -f weights/yolo11n-pose.pt ]; then
    echo "==> downloading YOLOv11n-Pose weights"
    mkdir -p weights
    curl -fsSL -o weights/yolo11n-pose.pt \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
fi

# -------------------------------------------------------------- verify
echo
echo "==> verification"
"$PY" - <<'PYEOF'
import torch
print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device           {torch.cuda.get_device_name(0)}")
    print(f"  capability       {torch.cuda.get_device_capability(0)}")
    x = torch.randn(512, 512, device="cuda")
    print(f"  matmul check     {float((x @ x).sum()):.1f}")
else:
    print("  !! still CPU-only. Check that CUDA_TAG matches your JetPack's CUDA")
    print("     version (see `jetson_release`).")
try:
    import cv2
    print(f"  opencv           {cv2.__version__}")
except ImportError:
    print("  !! cv2 missing -- rerun with --system-site-packages")
PYEOF

cat <<EOF

==> done. Use this interpreter from now on:

      $PY tools/run_recognition.py --source data/videos/your_clip.mp4

    or activate the environment:

      source $VENV/bin/activate

==> for maximum performance also run:

      sudo nvpmodel -m 0 && sudo jetson_clocks
EOF
