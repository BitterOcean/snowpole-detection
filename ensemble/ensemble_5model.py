"""
5-Model WBF Ensemble with tuned weights.
Combines RF-DETR + multiple YOLO runs for maximum robustness.
"""
import os
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
from config import ENSEMBLE_FOLDERS, WORK_DIR

# ── Config ──
FOLDERS    = ENSEMBLE_FOLDERS               # All 5 prediction folders
WEIGHTS    = [1, 1, 1, 5, 7]               # Tune based on individual model mAP
OUTPUT_DIR = os.path.join(WORK_DIR, "submissions", "ensemble_5model_" + "_".join(map(str, WEIGHTS)))
IOU_THR    = 0.55
CONF_THR   = 0.01


def yolo_to_xyxy(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return [max(0., min(1., v)) for v in [x1, y1, x2, y2]]


def xyxy_to_yolo(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return max(0, x1 + w / 2), max(0, y1 + h / 2), max(0, w), max(0, h)


def read_preds(path):
    boxes, scores, labels = [], [], []
    if os.path.exists(path):
        for line in open(path):
            p = line.strip().split()
            if len(p) < 6:
                continue
            cx, cy, w, h = map(float, p[1:5])
            boxes.append(yolo_to_xyxy(cx, cy, w, h))
            scores.append(float(p[5]))
            labels.append(int(p[0]))
    return boxes, scores, labels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = set()
    for folder in FOLDERS:
        all_files.update(os.path.basename(f) for f in glob.glob(os.path.join(folder, "*.txt")))

    print(f"🚀 Ensembling {len(all_files)} files from {len(FOLDERS)} models | weights={WEIGHTS}")

    for filename in tqdm(sorted(all_files)):
        boxes_list, scores_list, labels_list = [], [], []
        for folder in FOLDERS:
            b, s, l = read_preds(os.path.join(folder, filename))
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(l)

        if any(boxes_list):
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=WEIGHTS, iou_thr=IOU_THR, skip_box_thr=CONF_THR
            )
        else:
            boxes, scores, labels = [], [], []

        with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
            for i in range(len(boxes)):
                cx, cy, w, h = xyxy_to_yolo(*boxes[i])
                f.write(f"{int(labels[i])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {scores[i]:.6f}\n")

    print(f"✅ Done → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
