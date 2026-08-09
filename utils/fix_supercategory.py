"""Fix missing supercategory field in COCO JSON files."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PSEUDO_DATASET_DIR

json_files = glob.glob(os.path.join(PSEUDO_DATASET_DIR, "**", "_annotations.coco.json"), recursive=True)

for path in json_files:
    with open(path) as f:
        data = json.load(f)
    for cat in data.get("categories", []):
        if "supercategory" not in cat:
            cat["supercategory"] = "none"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Fixed: {path}")

print("✅ Done")
