# Snow Pole Detection using YOLOv8

TDT17 -- Visual Intelligence Mini Project

## Project Overview

This project focuses on detecting **snow poles in winter road scenes**
using deep learning. Snow poles are commonly used in Norway to mark the
road edges during snowy conditions, and detecting them can help improve
**autonomous driving systems in winter environments**.

The task is formulated as an **object detection problem**, where the
model predicts **bounding boxes around snow poles** in images.

The dataset used is **Poles2025**, captured in the Trøndelag region.

------------------------------------------------------------------------

## Dataset

Dataset: **Poles2025**

Dataset locations:

IDUN cluster

    /cluster/projects/vc/courses/TDT17/ad/Poles2025

Cybele lab

    datasets/TDT17/ad/Poles2025

Dataset characteristics:

-   Single object class: **snow pole**
-   Labels in **YOLO format**
-   Test set labels are hidden for the leaderboard

------------------------------------------------------------------------

## Project Workflow

The project is divided into four main stages:

1.  Exploratory Data Analysis (EDA)\
2.  Model Training\
3.  Model Evaluation\
4.  Testing on new images

------------------------------------------------------------------------

## Project Structure

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

------------------------------------------------------------------------

## Environment Setup

Create a virtual environment:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

``` bash
pip install ultralytics
pip install torch
pip install torchvision
pip install matplotlib
pip install opencv-python
pip install pandas
pip install seaborn
```

------------------------------------------------------------------------

## Exploratory Data Analysis

Notebook:

    EDA.ipynb

Purpose of the analysis:

-   Inspect dataset structure
-   Visualize annotations
-   Check bounding box distributions
-   Understand dataset characteristics

------------------------------------------------------------------------

## Model Training

Notebook:

    Training.ipynb

Model used:

**YOLOv8**

Example configuration:

    Model: YOLOv8n
    Epochs: 100
    Batch size: 16
    Image size: 640

Training outputs are saved in:

    runs/detect/train/

Important files:

    best.pt
    last.pt
    results.png

------------------------------------------------------------------------

## Model Evaluation

Notebook:

    Evaluation.ipynb

Metrics used:

-   Precision
-   Recall
-   mAP@50
-   mAP@0.5:0.95

------------------------------------------------------------------------

## Testing

Testing notebooks:

    Testing_MSJ.ipynb
    Testing_roadpoles_v1.ipynb
    Testing_Road_poles_iPhone.ipynb

These notebooks load the trained model and run inference on new images
or videos.

------------------------------------------------------------------------

## Sustainability

Training deep learning models consumes computational resources and
energy.\
In this project we estimate:

-   Training time
-   Energy consumption
-   Equivalent driving distance using a Tesla Model Y

------------------------------------------------------------------------

## References

YOLOv8 -- Ultralytics\
https://github.com/ultralytics/ultralytics

PyTorch\
https://pytorch.org/

------------------------------------------------------------------------

## Author

TDT17 -- Visual Intelligence Mini Project\
NTNU
