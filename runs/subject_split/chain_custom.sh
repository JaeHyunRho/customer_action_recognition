#!/usr/bin/env bash
# Custom pose from subjects 1-20 only: pseudo-label -> train -> extract the three subject splits.
set -euo pipefail
cd /home/alooh/customer_action_recognition
PY=.venv/bin/python; R=runs/subject_split; D=data/processed/subject_split; L=$R/logs
W=weights/custom_yolo11n_pose_s1_20.pt

if [ ! -f data/processed/pose_dataset_s1_20/dataset.yaml ]; then
  $PY tools/autolabel_pose.py --videos data/raw/merl/videos --out data/processed/pose_dataset_s1_20 \
    --teacher weights/yolo11x-pose.pt --fps 1 --conf 0.6 --min-keypoints 8 \
    --subjects 1-20 --val-ratio 0.2 --seed 42 > $L/autolabel.log 2>&1
fi
if [ ! -f $W ]; then
  $PY -m car.train.train_pose --data data/processed/pose_dataset_s1_20/dataset.yaml \
    --weights weights/yolo11n-pose.pt --out $R/pose --name custom_yolo11n_pose_s1_20 \
    --set pose_train.imgsz=480 pose_train.batch_size=24 pose_train.epochs=40 pose_train.patience=12 \
    > $L/train_pose.log 2>&1
fi
for s in train val test; do
  [ -f $D/custom_${s}_raw.npz ] || $PY tools/extract_skeletons.py --merl data/raw/merl --split $s \
    --out $D/custom_${s}_raw.npz --fps 10 --keep-background --set pose.weights=$W \
    > $L/extract_custom_$s.log 2>&1
done
echo "chain_custom done"
