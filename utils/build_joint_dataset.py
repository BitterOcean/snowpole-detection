"""Merge iPhone dataset + pseudo-labeled dataset into one joint dataset."""
import os, sys, json, shutil, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IPHONE_DATASET_DIR, PSEUDO_DATASET_DIR, JOINT_DATASET_DIR

os.makedirs(JOINT_DATASET_DIR, exist_ok=True)

for split in ["train", "valid", "test"]:
    out_dir = os.path.join(JOINT_DATASET_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    combined = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "snowpole"}]}
    img_id = ann_id = 0

    for src_root in [IPHONE_DATASET_DIR, PSEUDO_DATASET_DIR]:
        ann_path = os.path.join(src_root, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            coco = json.load(f)

        old2new = {}
        for img in coco["images"]:
            new_name = f"{img_id:08d}_{img['file_name']}"
            src_img  = os.path.join(src_root, split, img["file_name"])
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(out_dir, new_name))
            old2new[img["id"]] = img_id
            combined["images"].append({"id": img_id, "file_name": new_name,
                                       "width": img["width"], "height": img["height"]})
            img_id += 1

        for ann in coco["annotations"]:
            if ann["image_id"] not in old2new:
                continue
            combined["annotations"].append({**ann, "id": ann_id, "image_id": old2new[ann["image_id"]]})
            ann_id += 1

    with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
        json.dump(combined, f)

print(f"✅ Joint dataset saved → {JOINT_DATASET_DIR}")
