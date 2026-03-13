# Snow Pole Detection using YOLOv9t and YOLO11s

TDT17 -- Visual Intelligence Mini Project

![sample](https://github.com/user-attachments/assets/7c1068e6-9964-4c9b-b1c4-31c84a8e622b)

------------------------------------------------------------------------

# 1. Background & Motivation

### The Problem

Autonomous driving (AD) systems rely heavily on **lane markings**.\
In Nordic winters, roads are often **covered in snow**, making standard
lane detection unreliable.

### The Solution

**Snow poles** act as the *ground truth* for road boundaries in winter.\
Accurate detection of these poles is therefore critical for **safe
autonomous driving**.

### Challenges

**Thin Objects** - Snow poles are extremely thin. - At distance they can
appear only **1--2 pixels wide**.

**Real-time Constraint** - Detection must run on **edge hardware** with
**low latency**.

### Goal

Develop a robust **object detection pipeline** that maximizes **mAP
(Mean Average Precision)** while maintaining **real-time performance**.

------------------------------------------------------------------------

# 2. Approach & Strategy

We adopted a **Data-Centric AI approach** rather than only tuning model
hyperparameters.

## 1. Data Engine

The provided dataset (\~1k images) was insufficient for robust
generalization.

We therefore collected additional data by:

-   Scraping **10+ hours of YouTube winter driving footage**
-   Extracting frames to create a large pseudo-dataset.

## 2. Auto-Labeling Pipeline

We used **SAM 3 (Segment Anything Model)** to automatically generate
labels.

This produced approximately:

**\~36,000 pseudo-labeled frames**

## 3. Transfer Learning Hierarchy

**Stage 1 -- Pre-training**

Train on large **noisy YouTube dataset**\
→ provides **general visual knowledge**.

**Stage 2 -- Fine-tuning**

Train on high-quality **iPhone / Roadpoles dataset**\
→ provides **domain specificity**.

## 4. Ensemble Architecture

To increase robustness we combined:

-   **CNN detectors (YOLO)**
-   **Transformer detectors (RF-DETR)**

This architectural diversity improves performance across different
scenarios.

------------------------------------------------------------------------

# 3. Data Analysis (EDA)

![eda](https://github.com/user-attachments/assets/2876024f-7d20-44d8-8f96-2e429034c397)

## Provided Dataset (iPhone)

**Size** \~1,000 labeled images

**Quality** High resolution: 1920 × 1080

**Issue** Dataset splits were **sequential**, causing **data leakage**
where train and test images looked very similar.

## Scraped Dataset (YouTube)

**Size** \~15,000 processed frames

**Environmental Variety**

-   Sunny
-   Overcast
-   Heavy snow
-   Highway vs rural roads

**Labeling** Auto-labeled using **SAM 3** with prompt:

`snowpole`

**Filtering**

Low-confidence detections were removed to avoid **training on incorrect
labels**.

------------------------------------------------------------------------

# 4. Methods & Models

## Architecture 1: YOLOv9t & YOLO11s

### Why YOLOv9t?

YOLOv9 introduces **PGI (Programmable Gradient Information)**.

This mechanism helps preserve **fine-grained visual details**, which is
crucial for detecting **thin snow poles**.

Our experiments showed that YOLOv9 preserved faint pole structures
better than nano-scale models like YOLO11n.

### Training Configuration

Native resolution training:

-   imgsz = 1280
-   imgsz = 1920

Higher resolution was required because poles become **invisible at low
resolution**.

## Architecture 2: RF-DETR (Transformer)

### Why RF-DETR?

CNN detectors focus mainly on **local features**.

Transformers instead use **global attention**, allowing the model to
reason about **scene context**.

### Benefit

RF-DETR can understand that:

> A vertical white line inside a tree is **not** a snow pole.

YOLO detectors sometimes **hallucinate poles in forest backgrounds**,
while transformers reduce such errors.

------------------------------------------------------------------------

# SAM 3 Auto-Labeling Pipeline

We implemented a custom Python pipeline (`pipeline.py`) to scale data
generation.

### Pipeline Steps

1.  Input -- YouTube URL\
2.  Extract -- `ffmpeg` extracts frames at 1 FPS (high quality)\
3.  Rotate -- handle orientation differences\
4.  Label -- SAM3 inference with prompt `"snowpole"`\
5.  Filter -- remove low confidence detections (\<0.30)\
6.  Output -- COCO formatted dataset

This increased dataset size by **\~15× without manual labeling**.

------------------------------------------------------------------------

# Inference Optimization ("Secret Sauce")

To achieve top leaderboard performance we applied two techniques.

## 1. TTA --- Test Time Augmentation

Inference is run on:

-   original image
-   horizontally flipped image

Predictions are averaged.

Effect:

**\~1.5% improvement in mAP**

## 2. WBF --- Weighted Boxes Fusion

Instead of **Non-Max Suppression (NMS)**, WBF:

**averages overlapping boxes**.

### Ensemble Strategy

Final model combines:

-   YOLOv9t (shape expert)
-   YOLO11s (generalist)
-   RF-DETR (context expert)

Result:

More accurate bounding boxes and higher **mAP@50-95**.

------------------------------------------------------------------------

# 5. Real-World Feasibility

## Deployment Strategy

Although ensembles increase compute cost, modern edge AI hardware (e.g.,
NVIDIA Orin) supports **asynchronous parallel execution**.

Reference:

https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/

## Latency Benchmarks (RTX 4090)

| Model   | Latency |
|---------|---------|
| YOLOv9t | 18.0 ms |
| YOLO11s | 18.5 ms |
| RF-DETR | 36.4 ms |

Parallel inference means system latency is determined by the **slowest
model**, not the sum.

------------------------------------------------------------------------

# 6. Results

![result](https://github.com/user-attachments/assets/11d30dc7-55a5-48bc-b764-227884ced2f1)

| Model | mAP@50 | mAP@50:95 | Notes |
|------|--------|-----------|------|
| Baseline (YOLOv11n) | 92.0% | 65.0% | Fast but loose boxes |
| RF-DETR (Stage 1) | 89.8% | 66.6% | Robust but missed domain specifics |
| RF-DETR (Stage 2) | 95.0% | 74.5% | Fine-tuned on iPhone data |
| Final Ensemble (WBF) | **97.6%** | **79.5%** | Rank #1 / #2 |

Key finding: **mAP@50 was easy (\~97%) but mAP@50‑95 required tighter
bounding boxes.**

------------------------------------------------------------------------

# 7. Discussion

### Resolution Matters

Training at **640px** made distant poles invisible.

Training at **1280px and above** was necessary.

### Teacher Effect

SAM-generated labels acted as a **teacher**, improving generalization to
new weather conditions.

### YOLO vs Transformer

YOLO: - Faster - Higher recall on simple cases

RF-DETR: - Slower - Higher precision in complex backgrounds

Combining them solved both weaknesses.

------------------------------------------------------------------------

# 8. Sustainability & Compute

Training was performed on:

-   **IDUN Cluster (A100 GPUs)**
-   **Cybele Lab (RTX 4090)**

### Total Training Time

\~25 GPU hours

Breakdown:

| Task | Time |
|------|------|
| SAM3 Pipeline | ~5 hours |
| RF-DETR Training | ~12 hours |
| YOLO Experiments | ~8 hours |

### Energy Consumption

Average GPU power:

\~350 W

Total energy:

25h × 0.35kW ≈ **8.75 kWh**

### Tesla Metric

Tesla Model Y consumption:

\~16 kWh / 100km

Project energy (8.75 kWh) ≈ **54 km driving distance**.

------------------------------------------------------------------------

# 9. Key Learning Points

1.  **Data Engineering \> Model Tuning**\
    Building the SAM3 pipeline yielded larger gains than hyperparameter
    tuning.

2.  **Pseudo‑Labeling Risks**\
    Too many epochs on pseudo-labels causes memorization of teacher
    mistakes.

3.  **Smart Ensembling**\
    Combining **different architectures (CNN + Transformer)** works
    better than identical models.

4.  **Infrastructure Skills**\
    Handling:

-   slurm queues
-   rsync transfers
-   distributed training
