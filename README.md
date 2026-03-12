# Snow Pole Detection using YOLOv9t and YOLO11n
TDT17 – Visual Intelligence Mini Project

## Project Overview
This project focuses on detecting **snow poles in winter road scenes** using deep learning. Snow poles are commonly used in Norway to mark road boundaries during snowy conditions, and detecting them can help improve **autonomous driving systems in winter environments**.

The task is formulated as an **object detection problem**, where the model predicts **bounding boxes around snow poles** in road images.

The dataset used is **Poles2025**, captured in the Trøndelag region.

---

## Dataset

**Dataset:** Poles2025

**Dataset locations**

IDUN cluster:
```text
/cluster/projects/vc/courses/TDT17/ad/Poles2025
```

Cybele lab:
```text
datasets/TDT17/ad/Poles2025
```

**Dataset characteristics**
- Single object class: **snow pole**
- Labels in **YOLO format**
- Test set labels are hidden for the leaderboard
- Additional road pole data is used for qualitative testing

---

## Implemented Models

In this project, I worked with the following lightweight detection models:

- **YOLOv9t**
- **YOLO11n**

These models were selected because they are suitable for **real-time object detection** and **edge deployment**, which is important for autonomous driving applications.

The project compares compact YOLO variants in terms of detection quality and practical usability.

---

## Project Workflow

The project is divided into four main stages:

1. **Exploratory Data Analysis (EDA)**
2. **Model Training**
3. **Model Evaluation**
4. **Testing on external and qualitative datasets**

---

## Project Structure

```text
project/
│
├── EDA.ipynb
├── Training.ipynb
├── Evaluation.ipynb
│
├── Testing_MSJ.ipynb
├── Testing_roadpoles_v1.ipynb
├── Testing_Road_poles_iPhone.ipynb
│
├── LICENSE
└── README.md
```

---

## Environment Setup

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install ultralytics
pip install torch
pip install torchvision
pip install matplotlib
pip install opencv-python
pip install pandas
pip install seaborn
pip install pyyaml
```

---

## 1. Exploratory Data Analysis

**Notebook:** `EDA.ipynb`

The EDA stage was used to better understand the dataset before training.

### Goals
- Inspect dataset structure
- Visualize annotations
- Analyze bounding box distributions
- Identify dataset challenges relevant to training

### Typical outputs
- Sample images with bounding boxes
- Dataset statistics
- Distribution plots
- Observations about annotation density and image variation

EDA was important because the project expectations require that **model development is guided by exploratory data analysis**.

---

## 2. Model Training

**Notebook:** `Training.ipynb`

The training notebook prepares the dataset and trains compact YOLO detectors for snow pole detection.

### Models used
- **YOLOv9t**
- **YOLO11n**

### Why these models?
- Designed for fast inference
- Small enough for edge-oriented deployment
- Strong baselines for one-class object detection

### Example training settings
```text
Image size: 1280
Epochs: 100
Batch size: 16
Device: CUDA
```

### Training outputs
Training runs are saved under:

```text
runs/snowpole_detection/
```

Each experiment produces files such as:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
training_metadata.json
```

---

## 3. Model Evaluation

**Notebook:** `Evaluation.ipynb`

The models are evaluated using the required object detection metrics:

- **Precision**
- **Recall**
- **mAP@50**
- **mAP@0.5:0.95**

### Metric explanation
- **Precision**: how many predicted poles are correct
- **Recall**: how many real poles are successfully detected
- **mAP@50**: detection quality at IoU threshold 0.5
- **mAP@0.5:0.95**: average performance across multiple IoU thresholds

These metrics provide a balanced view of the model performance.

---

## 4. Testing

The project also includes qualitative testing notebooks:

- `Testing_MSJ.ipynb`
- `Testing_roadpoles_v1.ipynb`
- `Testing_Road_poles_iPhone.ipynb`

These notebooks:
- Load trained model weights
- Run inference on new or external data
- Visualize predicted bounding boxes
- Support qualitative comparison between trained models

This helps evaluate how well the models generalize beyond the main validation split.

---

## Sustainability

Training deep learning models requires computational resources and energy.

For this project, sustainability is considered by estimating:
- Total training time
- Approximate energy usage
- Equivalent driving distance using the same energy in a **Tesla Model Y**

This provides a practical interpretation of the environmental cost of model development.

---

## Discussion

This project demonstrates how lightweight object detectors can be applied to a real-world autonomous driving problem in winter conditions.

Using **YOLOv9t** and **YOLO11n** makes it possible to balance:
- detection performance,
- inference speed,
- and model size.

Because snow pole detection is intended for real-time use on limited hardware, compact models are a suitable design choice.

---

## Key Learnings

Main takeaways from this project:

- Applying deep learning to a real-world computer vision task
- Performing exploratory data analysis for object detection datasets
- Training and evaluating compact YOLO detectors
- Understanding object detection metrics
- Comparing lightweight models for practical deployment
- Reflecting on sustainability in AI experimentation

---

## References

- **Ultralytics YOLO**  
  https://github.com/ultralytics/ultralytics

- **PyTorch**  
  https://pytorch.org/

---

## Author

TDT17 – Visual Intelligence Mini Project  
NTNU
