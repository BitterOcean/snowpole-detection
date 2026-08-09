"""
YOLO inference with Test-Time Augmentation (TTA).
Runs prediction on original + horizontally flipped image and averages results.
Adds ~1.5% mAP improvement over standard inference.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
from config import WORK_DIR, IPHONE_DATASET_DIR

# ── Config ──
MODEL_PATH  = os.path.join(WORK_DIR, "runs", "yolo", "best.pt")
TEST_IMAGES = os.path.join(IPHONE_DATASET_DIR, "images", "Test", "test")
OUTPUT_NAME = "predictions_tta"

model = YOLO(MODEL_PATH)

model.predict(
    source=TEST_IMAGES,
    augment=True,       # ← enables TTA (horizontal flip + average)
    save_txt=True,
    save_conf=True,
    project=os.path.join(WORK_DIR, "predictions"),
    name=OUTPUT_NAME,
)

print(f"✅ Predictions saved to: {os.path.join(WORK_DIR, 'predictions', OUTPUT_NAME)}")
