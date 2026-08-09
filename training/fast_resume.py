"""
Fast resume for RF-DETR training after cluster interruption.
Finds the latest checkpoint in RFDETR_OUTPUT_DIR and resumes from it.
"""
import os
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rfdetr import RFDETRMedium
from config import RFDETR_OUTPUT_DIR, JOINT_DATASET_DIR

# ── Find latest checkpoint automatically ──
checkpoints = sorted(glob.glob(os.path.join(RFDETR_OUTPUT_DIR, "**", "*.pth"), recursive=True))
if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found in {RFDETR_OUTPUT_DIR}")

latest_checkpoint = checkpoints[-1]
print(f"Resuming from: {latest_checkpoint}")

model = RFDETRMedium(pretrain_weights=latest_checkpoint)

model.train(
    dataset_dir=JOINT_DATASET_DIR,
    epochs=100,
    batch_size=8,
    imgsz=1280,
    output_dir=RFDETR_OUTPUT_DIR,
    resume=latest_checkpoint,
)
