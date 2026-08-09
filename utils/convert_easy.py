"""Convert YOLO test-split labels to COCO JSON format for evaluation."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IPHONE_DATASET_DIR, WORK_DIR

LABEL_DIR   = os.path.join(IPHONE_DATASET_DIR, "labels", "Test", "test")
IMAGE_DIR   = os.path.join(IPHONE_DATASET_DIR, "images", "Test", "test")
OUTPUT_JSON = os.path.join(WORK_DIR, "dataset", "test", "_annotations.coco.json")

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

coco = {
    "images": [], "annotations": [],
    "categories": [{"id": 0, "name": "snowpole", "supercategory": "none"}]
}

img_id = ann_id = 0
for img_file in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    name = os.path.basename(img_file)
    from PIL import Image
    w, h = Image.open(img_file).size
    coco["images"].append({"id": img_id, "file_name": name, "width": w, "height": h})

    label_file = os.path.join(LABEL_DIR, os.path.splitext(name)[0] + ".txt")
    if os.path.exists(label_file):
        for line in open(label_file):
            p = line.strip().split()
            if len(p) < 5: continue
            cx, cy, bw, bh = map(float, p[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 0,
                "bbox": [x1, y1, bw * w, bh * h],
                "area": bw * w * bh * h, "iscrowd": 0
            })
            ann_id += 1
    img_id += 1

with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f)

print(f"✅ Saved {img_id} images, {ann_id} annotations → {OUTPUT_JSON}")
