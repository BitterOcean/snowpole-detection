import json
import os

# Define the files that need fixing
files_to_fix = [
    "dataset_joint_leaderboard/train/_annotations.coco.json",
    "dataset_joint_leaderboard/valid/_annotations.coco.json",
    "dataset_joint_leaderboard/test/_annotations.coco.json"
]

# The dummy info block required by pycocotools
dummy_info = {
    "description": "Snowpole Dataset",
    "url": "",
    "version": "1.0",
    "year": 2025,
    "contributor": "User",
    "date_created": "2025-11-21"
}

print("Checking JSON files for missing 'info' key...")

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping {file_path} (File not found)")
        continue
        
    try:
        # 1. Read existing data
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 2. Check and Inject
        if "info" not in data:
            print(f"🔧 Fixing {file_path}...")
            data["info"] = dummy_info
            
            # 3. Save back to disk
            with open(file_path, 'w') as f:
                json.dump(data, f)
            print(f"✅ Saved {file_path}")
        else:
            print(f"👍 {file_path} already has 'info'.")
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")

print("\nDone! Try running training again.")