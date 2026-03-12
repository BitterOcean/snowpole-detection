# Snow Pole Detection - TDT17 Mini-Project

## 📋 Project Overview

Real-time object detection of snow poles for autonomous driving in winter conditions. This project uses YOLOv8 to detect snow poles from images captured in the Trøndelag region.

**Course**: TDT17 - Visual Intelligence  
**Topic**: Autonomous Driving (AD)  
**Task**: Snow pole detection with bounding boxes  
**Dataset**: Poles2025 (YOLO format)

---

## 🚀 Quick Start Guide for University Computers

### Step 1: Setup Environment

```bash
# Navigate to your working directory
cd ~

# Clone/copy this project
# (If files are already there, skip to next step)

# Navigate to project directory
cd snow_pole_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Data Analysis (5-10 minutes)

```bash
# On IDUN cluster:
python data_analysis.py --data_path /cluster/projects/vc/courses/TDT17/ad/Poles2025

# On Cybele lab:
python data_analysis.py --data_path datasets/TDT17/ad/Poles2025
```

**Output**: 
- `analysis_results/data_analysis.png` - Statistical visualizations
- `analysis_results/sample_annotations.png` - Sample images with labels
- `analysis_results/analysis_report.json` - Detailed statistics

### Step 3: Train Model (1-3 hours)

```bash
# YOLOv8 Nano (fastest, recommended for quick results)
# On IDUN:
python train.py --data_path /cluster/projects/vc/courses/TDT17/ad/Poles2025 \
                --model_size n \
                --epochs 100 \
                --batch_size 16 \
                --device cuda

# On Cybele:
python train.py --data_path datasets/TDT17/ad/Poles2025 \
                --model_size n \
                --epochs 100 \
                --batch_size 16 \
                --device cuda
```

**For 2-person groups** (train a second model):
```bash
# YOLOv8 Small (better accuracy, longer training)
python train.py --data_path <DATA_PATH> \
                --model_size s \
                --epochs 100 \
                --batch_size 16 \
                --device cuda
```

**Output**:
- `runs/yolov8n_YYYYMMDD_HHMMSS/weights/best.pt` - Best model
- `runs/yolov8n_YYYYMMDD_HHMMSS/weights/last.pt` - Last epoch
- Training curves, sustainability metrics, and logs

### Step 4: Evaluate Model (10-15 minutes)

```bash
# Replace <MODEL_PATH> with your trained model path
# Example: runs/yolov8n_20241119_140523/weights/best.pt

# On IDUN:
python evaluate.py --model_path <MODEL_PATH> \
                   --data_path /cluster/projects/vc/courses/TDT17/ad/Poles2025 \
                   --conf_threshold 0.25

# On Cybele:
python evaluate.py --model_path <MODEL_PATH> \
                   --data_path datasets/TDT17/ad/Poles2025 \
                   --conf_threshold 0.25
```

**Output**:
- Validation metrics (Precision, Recall, mAP@50, mAP@0.5:0.95)
- Test predictions (for leaderboard)
- Prediction visualizations
- Confidence distribution analysis

---

## 📊 Expected Results

### YOLOv8 Nano (n)
- **Training time**: ~1-2 hours on RTX 4090
- **Expected mAP@50**: 0.75-0.85
- **Expected mAP@0.5:0.95**: 0.50-0.65
- **Model size**: ~6 MB
- **Inference speed**: ~2-3 ms/image

### YOLOv8 Small (s)
- **Training time**: ~2-3 hours on RTX 4090
- **Expected mAP@50**: 0.80-0.90
- **Expected mAP@0.5:0.95**: 0.55-0.70
- **Model size**: ~22 MB
- **Inference speed**: ~4-5 ms/image

---

## 🗂️ Project Structure

```
snow_pole_detection/
├── data_analysis.py          # Exploratory data analysis
├── train.py                  # Model training
├── evaluate.py               # Model evaluation
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
├── analysis_results/         # EDA outputs
│   ├── data_analysis.png
│   ├── sample_annotations.png
│   └── analysis_report.json
│
└── runs/                     # Training outputs
    └── yolov8n_YYYYMMDD_HHMMSS/
        ├── weights/
        │   ├── best.pt       # Best model
        │   └── last.pt       # Last epoch
        ├── results.csv       # Training metrics
        ├── training_info.json
        ├── sustainability_metrics.json
        └── evaluation/       # Evaluation outputs
            ├── validation_metrics.json
            ├── prediction_samples.png
            └── test_predictions/
```

---

## 📈 Performance Metrics

The project calculates the following metrics as required:

1. **Precision**: Proportion of true positives among all positive predictions
2. **Recall**: Proportion of true positives among all actual positives  
3. **mAP@50**: Mean Average Precision at IoU threshold 0.5
4. **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95

---

## 🌱 Sustainability Analysis

The training script automatically calculates:
- Total training time
- Estimated energy consumption (kWh)
- Equivalent distance in Tesla Model Y
- Percentage of Trondheim-Oslo trip (490 km)

Example output:
```
Training Time: 1.5 hours
Estimated Energy Used: 0.540 kWh
Tesla Model Y Comparison:
  Distance possible: 38.0 km
  Percentage of trip to Oslo: 7.8%
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --data_path <DATA_PATH> --batch_size 8
```

### Slow Training on CPU
```bash
# Reduce epochs for testing
python train.py --data_path <DATA_PATH> --epochs 50 --device cpu
```

### IDUN Queue Times
- Submit job during off-peak hours (evenings/weekends)
- Use Cybele lab as alternative
- Start training early to account for delays

---

## 📝 Presentation Structure

Your video presentation (12 min solo / 14 min group) should cover:

1. **Background / Motivation** (2 min)
   - Winter driving challenges
   - Role of snow poles in autonomous driving
   - Project objectives

2. **Data Analysis** (2 min)
   - Dataset statistics
   - Key findings from EDA
   - Challenges identified

3. **Approach / Strategy** (1 min)
   - Why YOLO was chosen
   - Model selection rationale
   - Training strategy

4. **Methods / Models** (2 min)
   - YOLOv8 architecture overview
   - Model configuration
   - Training parameters

5. **Results** (3 min)
   - Training curves
   - Validation metrics
   - Visual examples
   - Comparison (if 2 models)

6. **Discussion** (1 min)
   - What worked well
   - Limitations
   - Potential improvements

7. **Sustainability** (1 min)
   - Energy consumption
   - Tesla Model Y equivalence

8. **Key Learning Points** (1 min)
   - Technical insights
   - Challenges overcome

---

## 📚 References

- **YOLOv8**: Ultralytics YOLO - https://github.com/ultralytics/ultralytics
- **PyTorch**: https://pytorch.org/
- **Autonomous Driving in Winter**: Relevant papers on snow/winter detection

---

## 👥 Group Work (if applicable)

If working in a group of 2, clearly document:
- Who implemented which parts
- Division of analysis tasks
- Collaboration on presentation

---

## ✅ Checklist Before Submission

- [ ] Data analysis completed and visualizations saved
- [ ] Model(s) trained successfully
- [ ] Evaluation metrics calculated
- [ ] Test predictions generated
- [ ] Sustainability metrics documented
- [ ] Video presentation recorded (12/14 min)
- [ ] Code uploaded to GitHub or kept locally
- [ ] All outputs saved and organized

---

## 📧 Contact

For questions about the project:
- Course instructor: Frank Lindseth (frankl@ntnu.no)
- Check course materials on Blackboard
- Use course discussion forum

---

## 🎯 Tips for Success

1. **Start early** - Training takes time, especially with queue delays
2. **Test on small subset first** - Verify code works before full training
3. **Save intermediate results** - Don't lose work due to crashes
4. **Document everything** - Take notes for your presentation
5. **Use visualization** - Good plots make better presentations
6. **Compare results** - Try different confidence thresholds
7. **Backup your work** - Save to multiple locations

---

**Good luck with your project! 🎉**
