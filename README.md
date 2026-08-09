# Snow Pole Detection for Autonomous Driving

> **97.6% mAP@50 · 79.5% mAP@50:95 · Ranked #1 on Course Leaderboard**

A data-centric object detection pipeline for snow pole detection in Nordic winter driving conditions, where snow-covered roads make standard lane detection unreliable.

![sample](https://github.com/user-attachments/assets/7c1068e6-9964-4c9b-b1c4-31c84a8e622b)

---

## Overview

Snow poles are physical markers that define road boundaries when lane markings are buried under snow. This project builds a robust detection pipeline combining:

- **SAM 3** auto-labeling to generate ~36,000 pseudo-labeled frames from YouTube winter driving footage
- **Two-stage transfer learning**: pre-train on large noisy dataset → fine-tune on high-quality domain data
- **CNN + Transformer ensemble** (YOLOv9t + YOLO11n + RF-DETR) with Weighted Boxes Fusion

---

## Results

| Model | mAP@50 | mAP@50:95 |
|-------|--------|-----------|
| Baseline (YOLO11n) | 92.0% | 65.0% |
| RF-DETR (Stage 1) | 89.8% | 66.6% |
| RF-DETR (Stage 2) | 95.0% | 74.5% |
| **Final Ensemble (WBF)** | **97.6%** | **79.5%** |

![result](https://github.com/user-attachments/assets/11d30dc7-55a5-48bc-b764-227884ced2f1)

---

## Repository Structure

```
├── pipeline/
│   └── sam_autolabel_pipeline.py   # SAM 3 auto-labeling from YouTube + local data
│
├── training/
│   ├── train_yolo.py               # YOLOv9t / YOLO11n training config
│   ├── train_rfdetr.py             # RF-DETR training config
│   └── fast_resume.py              # Resume interrupted training
│
├── ensemble/
│   ├── ensemble_3model.py          # WBF ensemble: RF-DETR + 2x YOLO (final winner)
│   ├── ensemble_5model.py          # WBF ensemble: 5 models with tuned weights
│   └── rfdetr_submit.py            # RF-DETR standalone leaderboard submission
│
├── utils/
│   ├── coco2yolo.py                # Convert COCO annotations to YOLO format
│   ├── pseudo2yolo.py              # Convert pseudo-labels to YOLO format
│   ├── build_joint_dataset.py      # Merge multiple datasets
│   ├── sanitize_pseudolabels.py    # Filter low-quality pseudo-labels
│   ├── fix_json_info.py            # Fix COCO JSON metadata
│   ├── fix_supercategory.py        # Fix COCO supercategory fields
│   ├── convert_easy.py             # Quick format conversion helper
│   └── check_gpu_ready.py          # Verify GPU setup before training
│
├── requirements.txt
└── .gitignore
```

---

## Pipeline

### Step 1 — Auto-Labeling with SAM 3

```bash
python pipeline/sam_autolabel_pipeline.py
```

This script:
1. Downloads YouTube winter driving videos via `yt-dlp`
2. Extracts frames at 1 FPS using `ffmpeg`
3. Runs SAM 3 with text prompt `"snowpole"` to generate bounding boxes
4. Filters detections below confidence threshold (0.70)
5. Outputs a COCO-formatted dataset (~36,000 pseudo-labeled frames)

### Step 2 — Training

**YOLO models:**
```bash
python training/train_yolo.py
```

**RF-DETR:**
```bash
python training/train_rfdetr.py
```

Key training decisions:
- `imgsz=1280` — required because poles become invisible at 640px
- Two-stage: pre-train on YouTube pseudo-data → fine-tune on iPhone/RoadPoles dataset
- Augmentations: mosaic, mixup, copy-paste (critical for small thin objects)

### Step 3 — Ensemble Inference

```bash
python ensemble/ensemble_3model.py
```

Combines RF-DETR + YOLOv9t + YOLO11n predictions using Weighted Boxes Fusion (WBF) instead of NMS — averaging overlapping boxes rather than suppressing them.

**TTA (Test Time Augmentation):** inference on original + horizontally flipped image, predictions averaged. Adds ~1.5% mAP.

---

## Installation

```bash
git clone https://github.com/BitterOcean/snowpole-detection
cd snowpole-detection
pip install -r requirements.txt
```

Also required (system-level):
```bash
# ffmpeg
sudo apt install ffmpeg   # Linux
brew install ffmpeg       # macOS
```

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Resolution matters | Training at 640px made distant poles invisible; 1280px+ was required |
| Data > hypertuning | SAM pipeline yielded larger gains than model tuning |
| Pseudo-label risk | Too many epochs on pseudo-labels causes teacher-mistake memorization |
| CNN vs Transformer | YOLO: faster, higher recall. RF-DETR: higher precision in complex backgrounds |
| Ensemble diversity | CNN + Transformer combo outperforms same-architecture ensembles |

---

## Compute

Training was performed on IDUN Cluster (A100) and Cybele Lab (RTX 4090).

| Task | Time |
|------|------|
| SAM 3 Pipeline | ~5 GPU hours |
| RF-DETR Training | ~12 GPU hours |
| YOLO Experiments | ~8 GPU hours |
| **Total** | **~25 GPU hours** |

Energy: ~8.75 kWh ≈ equivalent to driving 54 km in a Tesla Model Y.

---

## Latency (RTX 4090)

| Model | Latency |
|-------|---------|
| YOLOv9t | 18.0 ms |
| YOLO11n | 18.5 ms |
| RF-DETR | 36.4 ms |

Parallel inference on edge hardware (e.g. NVIDIA Orin) means total latency equals the slowest model, not the sum.
