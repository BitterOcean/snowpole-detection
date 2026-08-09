import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
from config import YOLO_YAML_PATH

# ── Config ──
MODEL_TYPE = "yolov9t.pt"   # Options: yolo11n.pt, yolo11s.pt, yolov9t.pt
IMG_SIZE   = 1280            # 1280+ required — poles become invisible at 640px
BATCH_SIZE = 16

model = YOLO(MODEL_TYPE)

model.train(
    data=YOLO_YAML_PATH,
    epochs=100,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,

    # Augmentations — critical for thin small objects
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.3,

    # Hardware
    workers=8,
    cache=True,

    # Training dynamics
    rect=False,       # False = better mosaic augmentation
    cos_lr=True,
    patience=15,
    save_period=5,

    project="runs/yolo",
    name=f"{MODEL_TYPE.replace('.pt','')}_{IMG_SIZE}_batch{BATCH_SIZE}"
)

model.export()
