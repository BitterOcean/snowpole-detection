#!/bin/bash

###############################################################################
# Snow Pole Detection - Complete Project Pipeline
# TDT17 Mini-Project
#
# This script runs the complete project from start to finish:
# 1. Data Analysis
# 2. Model Training
# 3. Model Evaluation
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "           SNOW POLE DETECTION - COMPLETE PROJECT PIPELINE"
echo "                        TDT17 Mini-Project"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if data path is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Data path not provided${NC}"
    echo ""
    echo "Usage:"
    echo "  On IDUN:   bash run_complete_project.sh /cluster/projects/vc/courses/TDT17/ad/Poles2025"
    echo "  On Cybele: bash run_complete_project.sh datasets/TDT17/ad/Poles2025"
    echo ""
    exit 1
fi

DATA_PATH=$1
MODEL_SIZE=${2:-n}  # Default to nano if not specified
EPOCHS=${3:-100}    # Default to 100 epochs
BATCH_SIZE=${4:-16} # Default batch size

echo -e "${GREEN}Configuration:${NC}"
echo "  Data Path: $DATA_PATH"
echo "  Model Size: YOLOv8-$MODEL_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}Error: Data path does not exist: $DATA_PATH${NC}"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Please create one first:${NC}"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import torch; import ultralytics; import cv2; import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Required packages not installed${NC}"
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

###############################################################################
# STEP 1: DATA ANALYSIS
###############################################################################

echo -e "${BLUE}"
echo "───────────────────────────────────────────────────────────────────────────────"
echo "STEP 1/3: DATA ANALYSIS"
echo "───────────────────────────────────────────────────────────────────────────────"
echo -e "${NC}"

if [ -d "analysis_results" ] && [ -f "analysis_results/analysis_report.json" ]; then
    echo -e "${YELLOW}Analysis results already exist. Skip? (y/n)${NC}"
    read -r skip_analysis
    if [ "$skip_analysis" != "y" ]; then
        python data_analysis.py --data_path "$DATA_PATH"
    else
        echo -e "${GREEN}Skipping data analysis${NC}"
    fi
else
    python data_analysis.py --data_path "$DATA_PATH"
fi

echo -e "${GREEN}✓ Data analysis complete${NC}"
echo ""

###############################################################################
# STEP 2: MODEL TRAINING
###############################################################################

echo -e "${BLUE}"
echo "───────────────────────────────────────────────────────────────────────────────"
echo "STEP 2/3: MODEL TRAINING"
echo "───────────────────────────────────────────────────────────────────────────────"
echo -e "${NC}"

# Check if training should be skipped
LATEST_RUN=$(find runs -type d -name "yolov8${MODEL_SIZE}_*" 2>/dev/null | sort -r | head -n 1)
if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/weights/best.pt" ]; then
    echo -e "${YELLOW}Found existing trained model: $LATEST_RUN${NC}"
    echo -e "${YELLOW}Skip training and use existing model? (y/n)${NC}"
    read -r skip_training
    if [ "$skip_training" = "y" ]; then
        echo -e "${GREEN}Using existing model${NC}"
        MODEL_PATH="$LATEST_RUN/weights/best.pt"
    else
        echo -e "${YELLOW}Starting training...${NC}"
        python train.py \
            --data_path "$DATA_PATH" \
            --model_size "$MODEL_SIZE" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --device cuda
        
        # Get the newly created model
        MODEL_PATH=$(find runs -type f -name "best.pt" | sort -r | head -n 1)
    fi
else
    echo -e "${YELLOW}Starting training...${NC}"
    python train.py \
        --data_path "$DATA_PATH" \
        --model_size "$MODEL_SIZE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --device cuda
    
    # Get the newly created model
    MODEL_PATH=$(find runs -type f -name "best.pt" | sort -r | head -n 1)
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model training failed or model not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Model training complete${NC}"
echo -e "${GREEN}  Model saved at: $MODEL_PATH${NC}"
echo ""

###############################################################################
# STEP 3: MODEL EVALUATION
###############################################################################

echo -e "${BLUE}"
echo "───────────────────────────────────────────────────────────────────────────────"
echo "STEP 3/3: MODEL EVALUATION"
echo "───────────────────────────────────────────────────────────────────────────────"
echo -e "${NC}"

python evaluate.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --conf_threshold 0.25

echo -e "${GREEN}✓ Model evaluation complete${NC}"
echo ""

###############################################################################
# SUMMARY
###############################################################################

echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                          PROJECT COMPLETE! 🎉"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

echo -e "${GREEN}All outputs saved:${NC}"
echo ""
echo "1. Data Analysis:"
echo "   └── analysis_results/"
echo "       ├── data_analysis.png"
echo "       ├── sample_annotations.png"
echo "       └── analysis_report.json"
echo ""
echo "2. Model Training:"
echo "   └── $(dirname $(dirname $MODEL_PATH))/"
echo "       ├── weights/best.pt"
echo "       ├── results.csv"
echo "       ├── training_info.json"
echo "       └── sustainability_metrics.json"
echo ""
echo "3. Model Evaluation:"
echo "   └── $(dirname $MODEL_PATH)/../evaluation/"
echo "       ├── validation_metrics.json"
echo "       ├── prediction_samples.png"
echo "       └── test_predictions/"
echo ""

# Display final metrics if available
EVAL_METRICS="$(dirname $MODEL_PATH)/../evaluation/validation_metrics.json"
if [ -f "$EVAL_METRICS" ]; then
    echo -e "${YELLOW}Final Performance Metrics:${NC}"
    python3 << EOF
import json
with open('$EVAL_METRICS', 'r') as f:
    metrics = json.load(f)
print(f"  Precision:    {metrics['precision']:.4f}")
print(f"  Recall:       {metrics['recall']:.4f}")
print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
EOF
    echo ""
fi

# Display sustainability metrics
SUSTAINABILITY="$(dirname $(dirname $MODEL_PATH))/sustainability_metrics.json"
if [ -f "$SUSTAINABILITY" ]; then
    echo -e "${YELLOW}Sustainability Metrics:${NC}"
    python3 << EOF
import json
with open('$SUSTAINABILITY', 'r') as f:
    sus = json.load(f)
print(f"  Training Time: {sus['training_time_hours']:.2f} hours")
print(f"  Energy Used: {sus['energy_consumed_kwh']:.3f} kWh")
print(f"  Tesla Model Y Range: {sus['tesla_model_y']['distance_possible_km']:.1f} km")
print(f"  % of Trondheim-Oslo: {sus['tesla_model_y']['percentage_of_trip']:.1f}%")
EOF
    echo ""
fi

echo -e "${GREEN}Next Steps:${NC}"
echo "  1. Review the analysis results in analysis_results/"
echo "  2. Check training curves and metrics in runs/"
echo "  3. Examine evaluation results"
echo "  4. Prepare your video presentation (12/14 minutes)"
echo "  5. Upload to Teams before Nov 29th, 23:59 PM"
echo ""
echo -e "${BLUE}Good luck with your presentation! 🚀${NC}"
echo ""
