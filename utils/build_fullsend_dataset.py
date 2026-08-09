"""
Full Send: merge Train + Valid into one training set for final leaderboard submission.
RF-DETR needs a valid folder, so a small dummy valid set (10 images) is created.
Note: validation metrics will show ~100% accuracy and should be ignored.
"""
import os
import sys
import json
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from globox import AnnotationSet, Annotation
from tqdm import tqdm
from config import IPHONE_DATASET_DIR, WORK_DIR

SOURCE_ROOT = IPHONE_DATASET_DIR
OUTPUT_DIR  = os.path.join(WORK_DIR, "iphone_leaderboard_FULL_SEND")


def load_with_paths(json_path, image_folder):
    temp_set = AnnotationSet.from_coco(file_path=json_path)
    id_map = {}
    try:
        with open(json_path) as f:
            data = json.load(f)
        for img in data["images"]:
            id_map[img["id"]]       = img["file_name"]
            id_map[str(img["id"])] = img["file_name"]
    except:
        pass

    new_set = AnnotationSet()
    for image in temp_set:
        filename  = id_map.get(image.image_id, os.path.basename(str(image.image_id)))
        full_path = os.path.join(image_folder, filename)
        if os.path.exists(full_path):
            new_set.add(Annotation(image_id=full_path, image_size=image.image_size, boxes=image.boxes))
    return new_set


def save_dataset(dataset, split_name):
    target_dir = os.path.join(OUTPUT_DIR, split_name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    clean_set = AnnotationSet()
    for ann in dataset:
        clean_set.add(Annotation(
            image_id=os.path.basename(ann.image_id),
            image_size=ann.image_size,
            boxes=ann.boxes
        ))
    clean_set.save_coco(os.path.join(target_dir, "_annotations.coco.json"), auto_ids=True)

    for image in tqdm(dataset, desc=f"Copying {split_name}"):
        try:
            shutil.copy2(image.image_id, os.path.join(target_dir, os.path.basename(image.image_id)))
        except:
            pass
    print(f"  Saved {len(dataset)} images → {target_dir}")


print("🚀 Building Full Send dataset (Train + Valid merged)...")

train_set = load_with_paths(
    os.path.join(SOURCE_ROOT, "train", "_annotations.coco.json"),
    os.path.join(SOURCE_ROOT, "train")
)
valid_set = load_with_paths(
    os.path.join(SOURCE_ROOT, "valid", "_annotations.coco.json"),
    os.path.join(SOURCE_ROOT, "valid")
)

full_train = AnnotationSet()
for i in train_set: full_train.add(i)
for i in valid_set: full_train.add(i)

print(f"  Merged: {len(train_set)} + {len(valid_set)} = {len(full_train)} images")

# Dummy valid: first 10 images (needed by RF-DETR, metrics are meaningless)
dummy_valid = AnnotationSet()
for idx, img in enumerate(full_train):
    if idx >= 10: break
    dummy_valid.add(img)

save_dataset(full_train,  "train")
save_dataset(dummy_valid, "valid")

# Copy test set as-is
test_src = os.path.join(SOURCE_ROOT, "test")
test_dst = os.path.join(OUTPUT_DIR, "test")
if os.path.exists(test_src):
    shutil.copytree(test_src, test_dst, dirs_exist_ok=True)
    print(f"  Copied test set → {test_dst}")

print(f"\n✅ Full Send dataset ready → {OUTPUT_DIR}")
