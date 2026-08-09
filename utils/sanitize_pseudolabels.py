"""Remove low-quality pseudo-labels (empty label files or near-zero boxes)."""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WORK_DIR

DATASET_ROOT = os.path.join(WORK_DIR, "yolo_psudo")
MIN_BOX_SIZE = 0.005   # Remove boxes smaller than 0.5% of image dimension

removed = 0
for label_file in glob.glob(os.path.join(DATASET_ROOT, "labels", "**", "*.txt"), recursive=True):
    lines = open(label_file).readlines()
    valid = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        w, h = float(parts[3]), float(parts[4])
        if w > MIN_BOX_SIZE and h > MIN_BOX_SIZE:
            valid.append(line)
    if not valid:
        os.remove(label_file)
        removed += 1
    else:
        with open(label_file, "w") as f:
            f.writelines(valid)

print(f"✅ Sanitization done. Removed {removed} empty/noisy label files.")
