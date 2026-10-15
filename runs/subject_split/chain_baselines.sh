#!/usr/bin/env bash
# The two COCO pose models on the same subject splits: stock nano (12 joints after
# drop_head) and the yolo11x teacher (kept at 17 joints; the 12-joint copy is derived).
set -euo pipefail
cd /home/alooh/customer_action_recognition
PY=.venv/bin/python; R=runs/subject_split; D=data/processed/subject_split; L=$R/logs
for s in train val test; do
  [ -f $D/coco_n_${s}_raw.npz ] || $PY tools/extract_skeletons.py --merl data/raw/merl --split $s \
    --out $D/coco_n_${s}_raw.npz --fps 10 --keep-background --set pose.weights=weights/yolo11n-pose.pt \
    > $L/extract_coco_n_$s.log 2>&1
done
for s in val test train; do
  [ -f $D/teacher_x17_${s}_raw.npz ] || $PY tools/extract_skeletons.py --merl data/raw/merl --split $s \
    --out $D/teacher_x17_${s}_raw.npz --fps 10 --keep-background \
    --set pose.weights=weights/yolo11x-pose.pt pose.head_removal=false \
    > $L/extract_teacher_x17_$s.log 2>&1
done
echo "chain_baselines done"
