#!/bin/bash

#SBATCH --job-name=snow_pole_detection
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output_%j.log
#SBATCH --error=slurm_error_%j.log

###############################################################################
# SLURM Job Script for IDUN Cluster
# Snow Pole Detection - TDT17 Mini-Project
#
# Usage:
#   sbatch run_idun.sh
#
# Monitor job:
#   squeue -u $USER
#   tail -f slurm_output_JOBID.log
###############################################################################

# Print job information
echo "========================================================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "========================================================================="
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

echo "Modules loaded:"
module list
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi
echo ""

# Dataset path on IDUN
DATA_PATH="/cluster/projects/vc/courses/TDT17/ad/Poles2025"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pandas matplotlib seaborn Pillow pyyaml tqdm scikit-learn plotly albumentations

echo ""
echo "Python environment:"
which python
python --version
echo ""

# Check PyTorch and CUDA
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
EOF
echo ""

# Run data analysis
echo "========================================================================="
echo "STEP 1: DATA ANALYSIS"
echo "========================================================================="
python data_analysis.py --data_path "$DATA_PATH"
echo ""

# Run training
echo "========================================================================="
echo "STEP 2: MODEL TRAINING"
echo "========================================================================="
python train.py \
    --data_path "$DATA_PATH" \
    --model_size n \
    --epochs 100 \
    --batch_size 16 \
    --device cuda \
    --workers 8

# Find the trained model
MODEL_PATH=$(find runs -type f -name "best.pt" | sort -r | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model not found!"
    exit 1
fi

echo ""
echo "Model saved at: $MODEL_PATH"
echo ""

# Run evaluation
echo "========================================================================="
echo "STEP 3: MODEL EVALUATION"
echo "========================================================================="
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --conf_threshold 0.25

echo ""
echo "========================================================================="
echo "Job completed: $(date)"
echo "========================================================================="
