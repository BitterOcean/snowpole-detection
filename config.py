# =============================================================
# config.py — Central configuration for all scripts
# Edit these paths to match your local setup before running.
# =============================================================

import os

# ── Root directory where all data and outputs will be stored ──
WORK_DIR = os.environ.get("SNOWPOLE_WORK_DIR", "./work")

# ── Dataset paths ──
# Original labeled iPhone/RoadPoles dataset
IPHONE_DATASET_DIR   = os.path.join(WORK_DIR, "dataset")
# Scraped + SAM pseudo-labeled COCO dataset
PSEUDO_DATASET_DIR   = os.path.join(WORK_DIR, "psudodataset")
# Pseudo-labels converted to YOLO format
YOLO_PSEUDO_DIR      = os.path.join(WORK_DIR, "yolo_psudo")
# Joint dataset (iPhone + pseudo) used for final leaderboard training
JOINT_DATASET_DIR    = os.path.join(WORK_DIR, "dataset_joint_leaderboard")
# YOLO .yaml path for training
YOLO_YAML_PATH       = os.path.join(WORK_DIR, "yolo_psudo", "data.yaml")

# ── Local raw data (if you have access to the original pole images) ──
LOCAL_ROADPOLES_DIR  = os.environ.get("SNOWPOLE_LOCAL_DATA", "./local_data/RoadPoles-MSJ")

# ── RF-DETR paths ──
RFDETR_OUTPUT_DIR    = os.path.join(WORK_DIR, "rfout")
# Pretrained/resumed checkpoint (set to None to train from scratch)
RFDETR_PRETRAIN_WEIGHTS = None  # e.g. os.path.join(RFDETR_OUTPUT_DIR, "checkpoint_best_ema.pth")

# ── Ensemble: prediction label folders ──
# Point each to the 'labels/' folder produced by YOLO/RF-DETR inference
ENSEMBLE_FOLDERS = [
    os.path.join(WORK_DIR, "predictions", "rfdetr", "labels"),
    os.path.join(WORK_DIR, "predictions", "yolo_run1", "labels"),
    os.path.join(WORK_DIR, "predictions", "yolo_run2", "labels"),
]

# ── SAM Pipeline settings ──
YOUTUBE_URLS = [
    'https://www.youtube.com/watch?v=_IPZ99KD5AQ',   # fjell driving
    'https://www.youtube.com/watch?v=GxCb0pgO9Ig&t=7270s',  # geiranger
]
SAM_TEXT_PROMPT        = "snowpole"
SAM_CONFIDENCE_THRESHOLD = 0.70
SAM_FPS                = 1
SAM_BATCH_SIZE         = 8
