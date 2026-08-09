"""Convert COCO-format annotations to YOLO format."""
import os, sys, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IPHONE_DATASET_DIR, WORK_DIR

SOURCE_ROOT = IPHONE_DATASET_DIR
YOLO_ROOT   = os.path.join(WORK_DIR, "yolo_dataset")

for split in ["train", "valid", "test"]:
    ann_path = os.path.join(SOURCE_ROOT, split, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        print(f"Skipping {split} — not found")
        continue

    out_img_dir = os.path.join(YOLO_ROOT, "images", split)
    out_lbl_dir = os.path.join(YOLO_ROOT, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(ann_path) as f:
        coco = json.load(f)

    id2info = {img["id"]: img for img in coco["images"]}
    labels  = {img["id"]: [] for img in coco["images"]}

    for ann in coco["annotations"]:
        img  = id2info[ann["image_id"]]
        W, H = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        labels[ann["image_id"]].append(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")

    for img in coco["images"]:
        src = os.path.join(SOURCE_ROOT, split, img["file_name"])
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_img_dir, img["file_name"]))
        with open(os.path.join(out_lbl_dir, os.path.splitext(img["file_name"])[0] + ".txt"), "w") as f:
            f.write("\n".join(labels[img["id"]]))

print(f"✅ Done → {YOLO_ROOT}")
