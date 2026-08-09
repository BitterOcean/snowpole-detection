import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rfdetr import RFDETRMedium
from config import RFDETR_OUTPUT_DIR, RFDETR_PRETRAIN_WEIGHTS, JOINT_DATASET_DIR

os.makedirs(RFDETR_OUTPUT_DIR, exist_ok=True)

model = RFDETRMedium(
    pretrain_weights=RFDETR_PRETRAIN_WEIGHTS  # None = train from scratch
)

model.train(
    dataset_dir=JOINT_DATASET_DIR,
    epochs=100,
    batch_size=8,
    imgsz=1280,
    output_dir=RFDETR_OUTPUT_DIR,
)
