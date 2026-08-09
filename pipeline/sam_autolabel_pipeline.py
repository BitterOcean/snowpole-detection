import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import subprocess
import glob
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import (
    WORK_DIR, LOCAL_ROADPOLES_DIR, PSEUDO_DATASET_DIR,
    YOUTUBE_URLS, SAM_TEXT_PROMPT, SAM_CONFIDENCE_THRESHOLD,
    SAM_FPS, SAM_BATCH_SIZE
)
from transformers import Sam3Processor, Sam3Model

# ── Derived paths ──
SPLIT_RATIOS    = {"train": 0.79, "valid": 0.20, "test": 0.01}
RAW_VIDEO_DIR   = os.path.join(WORK_DIR, "raw_videos")
TEMP_FRAME_DIR  = os.path.join(WORK_DIR, "temp_frames")
FINAL_DATASET_DIR = PSEUDO_DATASET_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_dirs():
    if os.path.exists(TEMP_FRAME_DIR):
        shutil.rmtree(TEMP_FRAME_DIR)
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)


def download_video(url):
    output_template = os.path.join(RAW_VIDEO_DIR, "%(id)s.%(ext)s")
    cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
           "--no-playlist", "-o", output_template, url]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    video_id = url.split("v=")[-1].split("&")[0]
    found = glob.glob(os.path.join(RAW_VIDEO_DIR, f"{video_id}.*"))
    return found[0] if found else None


def extract_frames(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_pattern = os.path.join(TEMP_FRAME_DIR, f"{video_name}_%06d.jpg")
    cmd = ["ffmpeg", "-n", "-i", video_path, "-vf", f"fps={SAM_FPS}", "-q:v", "2", output_pattern]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(glob.glob(os.path.join(TEMP_FRAME_DIR, f"{video_name}_*.jpg")))


def collect_local_images(base_path):
    print(f"🔍 Scanning {base_path} for images...")
    found = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                found.append(os.path.join(root, f))
    return found


def save_split(data_list, split_name):
    output_dir = os.path.join(FINAL_DATASET_DIR, split_name)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "_annotations.coco.json")

    coco = {
        "info": {"description": f"Snowpole {split_name}", "year": 2025},
        "images": [], "annotations": [],
        "categories": [{"id": 0, "name": SAM_TEXT_PROMPT}]
    }

    ann_id = 0
    print(f"💾 Saving {len(data_list)} images to '{split_name}'...")

    for img_id, item in enumerate(tqdm(data_list, desc=f"Writing {split_name}")):
        filename = f"{img_id:06d}_{os.path.basename(item['path'])}"
        dst_path = os.path.join(output_dir, filename)
        item['pil_image'].save(dst_path)

        coco["images"].append({
            "id": img_id, "file_name": filename,
            "width": item['width'], "height": item['height']
        })

        for box, score in zip(item['boxes'], item['scores']):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 0,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h), "score": float(score), "iscrowd": 0
            })
            ann_id += 1

    with open(json_path, "w") as f:
        json.dump(coco, f)


def main():
    print("🚀 Starting SAM Auto-Label Pipeline...")
    setup_dirs()

    print(f"Loading SAM 3 on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    all_frames = []

    print("── Phase A: YouTube videos ──")
    for url in YOUTUBE_URLS:
        try:
            v_path = download_video(url)
            if v_path:
                frames = extract_frames(v_path)
                for f in frames:
                    all_frames.append({'path': f, 'needs_rotation': False})
        except Exception as e:
            print(f"  Skipping {url}: {e}")

    print("── Phase B: Local datasets ──")
    if os.path.exists(LOCAL_ROADPOLES_DIR):
        imgs = collect_local_images(LOCAL_ROADPOLES_DIR)
        for f in imgs:
            all_frames.append({'path': f, 'needs_rotation': True})
    else:
        print(f"  ⚠️  LOCAL_ROADPOLES_DIR not found: {LOCAL_ROADPOLES_DIR}")
        print("  Set the SNOWPOLE_LOCAL_DATA environment variable or edit config.py")

    print(f"\n🔥 Running inference on {len(all_frames)} frames...")
    processed_data = []

    for i in tqdm(range(0, len(all_frames), SAM_BATCH_SIZE)):
        batch_items = all_frames[i: i + SAM_BATCH_SIZE]
        pil_images, valid_indices = [], []

        for idx, item in enumerate(batch_items):
            try:
                img = Image.open(item['path']).convert("RGB")
                if item['needs_rotation']:
                    img = img.transpose(Image.ROTATE_270)
                pil_images.append(img)
                valid_indices.append(idx)
            except:
                continue

        if not pil_images:
            continue

        prompts = [SAM_TEXT_PROMPT] * len(pil_images)

        with torch.no_grad():
            inputs = processor(images=pil_images, text=prompts, return_tensors="pt").to(device)
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs, threshold=SAM_CONFIDENCE_THRESHOLD, mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist()
        )

        for j, res in enumerate(results):
            boxes  = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            if len(boxes) == 0:
                continue
            processed_data.append({
                "path": batch_items[j]['path'],
                "pil_image": pil_images[j],
                "width":  pil_images[j].width,
                "height": pil_images[j].height,
                "boxes":  boxes,
                "scores": scores,
            })

    print(f"\n🔀 Shuffling and splitting {len(processed_data)} annotated frames...")
    random.seed(42)
    random.shuffle(processed_data)

    total     = len(processed_data)
    train_end = int(total * SPLIT_RATIOS["train"])
    valid_end = int(total * (SPLIT_RATIOS["train"] + SPLIT_RATIOS["valid"]))

    save_split(processed_data[:train_end],       "train")
    save_split(processed_data[train_end:valid_end], "valid")
    save_split(processed_data[valid_end:],       "test")

    if os.path.exists(TEMP_FRAME_DIR):
        shutil.rmtree(TEMP_FRAME_DIR)

    print(f"\n✅ Done! Dataset saved to: {FINAL_DATASET_DIR}")


if __name__ == "__main__":
    main()
