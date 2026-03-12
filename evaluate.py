#!/usr/bin/env python3
"""
Snow Pole Detection - Evaluation Script
TDT17 Mini-Project
"""

import os
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

class ModelEvaluator:
    def __init__(self, model_path, data_path):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model weights
            data_path: Path to Poles2025 dataset
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = YOLO(str(model_path))
        
        # Create output directory
        self.output_dir = self.model_path.parent.parent / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def evaluate_on_validation(self, imgsz=640, conf_threshold=0.25, iou_threshold=0.45):
        """
        Evaluate model on validation set
        
        Args:
            imgsz: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        print("\n" + "="*80)
        print("EVALUATING MODEL ON VALIDATION SET")
        print("="*80)
        
        # Create data.yaml path
        data_yaml = self.model_path.parent.parent / 'data.yaml'
        
        if not data_yaml.exists():
            # Create temporary data.yaml
            import yaml
            data_config = {
                'path': str(self.data_path.absolute()),
                'train': 'train/images',
                'val': 'valid/images',
                'test': 'test/images',
                'nc': 1,
                'names': ['snow_pole']
            }
            with open(data_yaml, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
        
        # Run validation
        print(f"\nRunning validation with:")
        print(f"  Image size: {imgsz}")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  IoU threshold: {iou_threshold}")
        
        results = self.model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            split='val',
            save_json=True,
            save_hybrid=False,
            plots=True,
            verbose=True,
            project=str(self.output_dir.parent),
            name='evaluation',
            exist_ok=True
        )
        
        # Extract metrics
        metrics = {
            'precision': float(results.box.p[0]) if hasattr(results.box, 'p') else 0.0,
            'recall': float(results.box.r[0]) if hasattr(results.box, 'r') else 0.0,
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        
        # Save metrics
        metrics_path = self.output_dir / 'validation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✓ Metrics saved to {metrics_path}")
        
        return metrics
    
    def evaluate_on_test(self, imgsz=640, conf_threshold=0.25):
        """
        Run inference on test set and save predictions
        
        Args:
            imgsz: Input image size
            conf_threshold: Confidence threshold
        """
        print("\n" + "="*80)
        print("RUNNING INFERENCE ON TEST SET")
        print("="*80)
        
        test_images_path = self.data_path / 'test' / 'images'
        
        if not test_images_path.exists():
            print(f"⚠ Test images not found at {test_images_path}")
            return
        
        test_images = list(test_images_path.glob('*.jpg')) + list(test_images_path.glob('*.png'))
        print(f"\nFound {len(test_images)} test images")
        
        # Create output directory for predictions
        predictions_dir = self.output_dir / 'test_predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        # Run predictions
        all_predictions = []
        
        for img_path in tqdm(test_images, desc="Processing test images"):
            results = self.model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf_threshold,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )
            
            # Extract predictions
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    all_predictions.append({
                        'image': img_path.name,
                        'class': cls,
                        'confidence': conf,
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    })
        
        # Save predictions
        predictions_path = predictions_dir / 'predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(all_predictions, f, indent=4)
        
        print(f"\n✓ Predictions saved to {predictions_path}")
        print(f"Total predictions: {len(all_predictions)}")
        
        # Create predictions CSV for leaderboard
        df = pd.DataFrame(all_predictions)
        if len(df) > 0:
            csv_path = predictions_dir / 'predictions.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Predictions CSV saved to {csv_path}")
        
        return all_predictions
    
    def visualize_predictions(self, num_samples=12, conf_threshold=0.25):
        """
        Visualize model predictions on validation images
        
        Args:
            num_samples: Number of samples to visualize
            conf_threshold: Confidence threshold
        """
        print("\n" + "="*80)
        print("VISUALIZING PREDICTIONS")
        print("="*80)
        
        val_images_path = self.data_path / 'valid' / 'images'
        val_images = list(val_images_path.glob('*.jpg'))[:num_samples]
        
        rows = (num_samples + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
        axes = axes.ravel() if num_samples > 1 else [axes]
        
        for idx, img_path in enumerate(val_images):
            # Run prediction
            results = self.model.predict(
                source=str(img_path),
                conf=conf_threshold,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )
            
            # Read and draw on image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            num_detections = 0
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f'Snow Pole {conf:.2f}'
                    cv2.putText(img, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    num_detections += 1
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'{img_path.name}\n{num_detections} poles detected')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(val_images), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        viz_path = self.output_dir / 'prediction_samples.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {viz_path}")
        plt.close()
    
    def analyze_confidence_distribution(self, conf_threshold=0.25):
        """Analyze confidence score distribution"""
        print("\n" + "="*80)
        print("ANALYZING CONFIDENCE DISTRIBUTION")
        print("="*80)
        
        val_images_path = self.data_path / 'valid' / 'images'
        val_images = list(val_images_path.glob('*.jpg'))
        
        all_confidences = []
        
        for img_path in tqdm(val_images, desc="Processing validation images"):
            results = self.model.predict(
                source=str(img_path),
                conf=0.001,  # Very low threshold to get all detections
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    conf = float(boxes[i].conf[0])
                    all_confidences.append(conf)
        
        if len(all_confidences) == 0:
            print("⚠ No detections found")
            return
        
        # Plot distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(all_confidences, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(conf_threshold, color='r', linestyle='--', 
                       label=f'Threshold: {conf_threshold}')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_conf = np.sort(all_confidences)
        cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        axes[1].plot(sorted_conf, cumulative)
        axes[1].axvline(conf_threshold, color='r', linestyle='--',
                       label=f'Threshold: {conf_threshold}')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Distribution plot saved to {dist_path}")
        plt.close()
        
        # Statistics
        print(f"\nConfidence Statistics:")
        print(f"  Total detections: {len(all_confidences)}")
        print(f"  Mean: {np.mean(all_confidences):.4f}")
        print(f"  Median: {np.median(all_confidences):.4f}")
        print(f"  Std: {np.std(all_confidences):.4f}")
        print(f"  Min: {np.min(all_confidences):.4f}")
        print(f"  Max: {np.max(all_confidences):.4f}")
        print(f"  Above threshold ({conf_threshold}): {sum(c >= conf_threshold for c in all_confidences)}")
    
    def run_full_evaluation(self, imgsz=640, conf_threshold=0.25):
        """Run complete evaluation pipeline"""
        print("\n" + "="*80)
        print("RUNNING FULL EVALUATION PIPELINE")
        print("="*80)
        
        # 1. Evaluate on validation set
        val_metrics = self.evaluate_on_validation(imgsz=imgsz, conf_threshold=conf_threshold)
        
        # 2. Run inference on test set
        test_predictions = self.evaluate_on_test(imgsz=imgsz, conf_threshold=conf_threshold)
        
        # 3. Visualize predictions
        self.visualize_predictions(conf_threshold=conf_threshold)
        
        # 4. Analyze confidence distribution
        self.analyze_confidence_distribution(conf_threshold=conf_threshold)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE! 🎉")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        
        return val_metrics, test_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Snow Pole Detection Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights (best.pt)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to Poles2025 dataset')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    # Run evaluation
    evaluator.run_full_evaluation(
        imgsz=args.imgsz,
        conf_threshold=args.conf_threshold
    )
