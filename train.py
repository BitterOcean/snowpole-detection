#!/usr/bin/env python3
"""
Snow Pole Detection - Training Script
TDT17 Mini-Project
"""

import os
import yaml
import torch
import time
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class SnowPoleTrainer:
    def __init__(self, data_path, model_size='n', epochs=100, imgsz=640, batch_size=16):
        """
        Initialize trainer
        
        Args:
            data_path: Path to Poles2025 dataset
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size for training
        """
        self.data_path = Path(data_path)
        self.model_size = model_size
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        
        # Create output directory
        self.output_dir = Path('runs') / f'yolov8{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.training_time = 0
        
    def create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        data_yaml = {
            'path': str(self.data_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 1,  # number of classes
            'names': ['snow_pole']
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✓ Created data.yaml at {yaml_path}")
        return yaml_path
    
    def train(self, device='cuda', workers=8, patience=20, save_period=10):
        """
        Train YOLO model
        
        Args:
            device: Device to train on ('cuda' or 'cpu')
            workers: Number of worker threads
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
        """
        print("\n" + "="*80)
        print(f"TRAINING YOLOv8{self.model_size.upper()} MODEL")
        print("="*80)
        
        # Check GPU availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠ CUDA not available, switching to CPU")
            device = 'cpu'
        else:
            print(f"✓ Using device: {device}")
            if device == 'cuda':
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create data.yaml
        data_yaml = self.create_data_yaml()
        
        # Load model
        model_name = f'yolov8{self.model_size}.pt'
        print(f"\n✓ Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Training parameters
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Workers: {workers}")
        print(f"  Patience: {patience}")
        print(f"  Device: {device}")
        
        # Start training
        print("\n" + "─"*80)
        print("Starting training...")
        print("─"*80 + "\n")
        
        start_time = time.time()
        
        results = model.train(
            data=str(data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=device,
            workers=workers,
            patience=patience,
            save_period=save_period,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            plots=True,
            save=True,
            save_json=True,
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=True,  # Single class detection
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            split='val',
            save_period=save_period,
        )
        
        end_time = time.time()
        self.training_time = end_time - start_time
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Training time: {self.training_time/3600:.2f} hours ({self.training_time/60:.2f} minutes)")
        print(f"Model saved to: {self.output_dir}")
        
        # Save training info
        self.save_training_info(results)
        
        return results
    
    def save_training_info(self, results):
        """Save training information and metrics"""
        info = {
            'model_size': self.model_size,
            'epochs': self.epochs,
            'image_size': self.imgsz,
            'batch_size': self.batch_size,
            'training_time_seconds': self.training_time,
            'training_time_minutes': self.training_time / 60,
            'training_time_hours': self.training_time / 3600,
            'output_dir': str(self.output_dir),
            'best_weights': str(self.output_dir / 'weights' / 'best.pt'),
            'last_weights': str(self.output_dir / 'weights' / 'last.pt'),
        }
        
        info_path = self.output_dir / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"\n✓ Training info saved to {info_path}")
        
        # Calculate energy consumption
        self.calculate_sustainability_metrics()
    
    def calculate_sustainability_metrics(self):
        """Calculate sustainability metrics"""
        print("\n" + "="*80)
        print("SUSTAINABILITY ANALYSIS")
        print("="*80)
        
        # Assumptions for GPU power consumption
        # RTX 4090: ~450W TDP
        # Assume 80% average utilization during training
        gpu_power_watts = 450 * 0.8  # 360W average
        
        # Calculate energy consumption
        energy_kwh = (gpu_power_watts * self.training_time) / (1000 * 3600)
        
        # Tesla Model Y: ~75 kWh battery, ~530 km range (WLTP)
        # Energy efficiency: ~14.2 kWh/100km
        tesla_efficiency = 14.2 / 100  # kWh per km
        distance_possible = energy_kwh / tesla_efficiency
        
        # Distance from Trondheim to Oslo: ~490 km
        trondheim_oslo_distance = 490
        percentage_to_oslo = (distance_possible / trondheim_oslo_distance) * 100
        
        print(f"\nTraining Time: {self.training_time/3600:.2f} hours")
        print(f"Estimated GPU Power Consumption: {gpu_power_watts:.0f}W")
        print(f"Estimated Energy Used: {energy_kwh:.3f} kWh")
        print(f"\nTesla Model Y Comparison:")
        print(f"  Distance possible with same energy: {distance_possible:.1f} km")
        print(f"  Trondheim to Oslo distance: {trondheim_oslo_distance} km")
        print(f"  Percentage of trip: {percentage_to_oslo:.1f}%")
        
        if percentage_to_oslo >= 100:
            print(f"  ✓ You could drive to Oslo and back {percentage_to_oslo/100:.1f} times!")
        else:
            print(f"  ✓ You could drive {percentage_to_oslo:.1f}% of the way to Oslo")
        
        # Save sustainability metrics
        sustainability = {
            'training_time_hours': self.training_time / 3600,
            'estimated_gpu_power_watts': gpu_power_watts,
            'energy_consumed_kwh': energy_kwh,
            'tesla_model_y': {
                'efficiency_kwh_per_km': tesla_efficiency,
                'distance_possible_km': distance_possible,
                'trondheim_oslo_distance_km': trondheim_oslo_distance,
                'percentage_of_trip': percentage_to_oslo
            }
        }
        
        sustainability_path = self.output_dir / 'sustainability_metrics.json'
        with open(sustainability_path, 'w') as f:
            json.dump(sustainability, f, indent=4)
        
        print(f"\n✓ Sustainability metrics saved to {sustainability_path}")
    
    def visualize_training_results(self):
        """Visualize training results"""
        results_csv = self.output_dir / 'results.csv'
        
        if not results_csv.exists():
            print("⚠ Results CSV not found, skipping visualization")
            return
        
        import pandas as pd
        
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
        axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # mAP
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Validation mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR Group 0')
        axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR Group 1')
        axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR Group 2')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {self.output_dir / 'training_curves.png'}")
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Snow Pole Detection Model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to Poles2025 dataset')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SnowPoleTrainer(
        data_path=args.data_path,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size
    )
    
    # Train model
    results = trainer.train(device=args.device, workers=args.workers)
    
    # Visualize results
    trainer.visualize_training_results()
    
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80)
    print(f"\nBest model: {trainer.output_dir / 'weights' / 'best.pt'}")
    print(f"Results directory: {trainer.output_dir}")
