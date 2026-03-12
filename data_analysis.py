#!/usr/bin/env python3
"""
Snow Pole Detection - Exploratory Data Analysis
TDT17 Mini-Project
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import defaultdict
import json

class DataAnalyzer:
    def __init__(self, data_path):
        """
        Initialize data analyzer
        
        Args:
            data_path: Path to Poles2025 dataset
        """
        self.data_path = Path(data_path)
        self.train_images = self.data_path / 'train' / 'images'
        self.train_labels = self.data_path / 'train' / 'labels'
        self.val_images = self.data_path / 'valid' / 'images'
        self.val_labels = self.data_path / 'valid' / 'labels'
        
        self.stats = defaultdict(list)
        
    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        boxes = []
        if not os.path.exists(label_path):
            return boxes
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    boxes.append({
                        'class_id': int(class_id),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        return boxes
    
    def analyze_dataset(self, split='train'):
        """Analyze a dataset split"""
        if split == 'train':
            images_path = self.train_images
            labels_path = self.train_labels
        else:
            images_path = self.val_images
            labels_path = self.val_labels
        
        print(f"\n{'='*60}")
        print(f"Analyzing {split} dataset...")
        print(f"{'='*60}")
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        split_stats = {
            'num_images': 0,
            'num_boxes': 0,
            'boxes_per_image': [],
            'box_widths': [],
            'box_heights': [],
            'box_areas': [],
            'box_aspect_ratios': [],
            'image_widths': [],
            'image_heights': [],
            'images_without_boxes': 0,
            'box_positions_x': [],
            'box_positions_y': []
        }
        
        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            split_stats['image_widths'].append(w)
            split_stats['image_heights'].append(h)
            split_stats['num_images'] += 1
            
            # Read labels
            label_file = labels_path / f"{img_file.stem}.txt"
            boxes = self.parse_yolo_label(label_file)
            
            if len(boxes) == 0:
                split_stats['images_without_boxes'] += 1
                split_stats['boxes_per_image'].append(0)
                continue
            
            split_stats['boxes_per_image'].append(len(boxes))
            split_stats['num_boxes'] += len(boxes)
            
            for box in boxes:
                # Convert normalized to pixel coordinates
                box_w = box['width'] * w
                box_h = box['height'] * h
                
                split_stats['box_widths'].append(box_w)
                split_stats['box_heights'].append(box_h)
                split_stats['box_areas'].append(box_w * box_h)
                split_stats['box_aspect_ratios'].append(box_w / box_h if box_h > 0 else 0)
                split_stats['box_positions_x'].append(box['x_center'])
                split_stats['box_positions_y'].append(box['y_center'])
        
        return split_stats
    
    def print_statistics(self, stats, split_name):
        """Print dataset statistics"""
        print(f"\n{split_name.upper()} Dataset Statistics:")
        print(f"{'─'*60}")
        print(f"Total Images: {stats['num_images']}")
        print(f"Total Boxes: {stats['num_boxes']}")
        print(f"Images without boxes: {stats['images_without_boxes']}")
        print(f"Images with boxes: {stats['num_images'] - stats['images_without_boxes']}")
        
        if stats['boxes_per_image']:
            print(f"\nBoxes per Image:")
            print(f"  Mean: {np.mean(stats['boxes_per_image']):.2f}")
            print(f"  Median: {np.median(stats['boxes_per_image']):.2f}")
            print(f"  Max: {max(stats['boxes_per_image'])}")
            print(f"  Min: {min(stats['boxes_per_image'])}")
        
        if stats['box_widths']:
            print(f"\nBox Dimensions (pixels):")
            print(f"  Width  - Mean: {np.mean(stats['box_widths']):.2f}, Std: {np.std(stats['box_widths']):.2f}")
            print(f"  Height - Mean: {np.mean(stats['box_heights']):.2f}, Std: {np.std(stats['box_heights']):.2f}")
            print(f"  Area   - Mean: {np.mean(stats['box_areas']):.2f}, Std: {np.std(stats['box_areas']):.2f}")
            print(f"  Aspect Ratio - Mean: {np.mean(stats['box_aspect_ratios']):.2f}")
        
        if stats['image_widths']:
            print(f"\nImage Dimensions:")
            print(f"  Width  - Mean: {np.mean(stats['image_widths']):.2f}")
            print(f"  Height - Mean: {np.mean(stats['image_heights']):.2f}")
    
    def visualize_statistics(self, train_stats, val_stats, output_dir='analysis_results'):
        """Create visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Boxes per image distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist(train_stats['boxes_per_image'], bins=30, alpha=0.7, label='Train', edgecolor='black')
        axes[0, 0].hist(val_stats['boxes_per_image'], bins=30, alpha=0.7, label='Validation', edgecolor='black')
        axes[0, 0].set_xlabel('Number of Boxes per Image')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Boxes per Image')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box size distribution
        axes[0, 1].scatter(train_stats['box_widths'], train_stats['box_heights'], 
                          alpha=0.5, s=10, label='Train')
        axes[0, 1].scatter(val_stats['box_widths'], val_stats['box_heights'], 
                          alpha=0.5, s=10, label='Validation')
        axes[0, 1].set_xlabel('Box Width (pixels)')
        axes[0, 1].set_ylabel('Box Height (pixels)')
        axes[0, 1].set_title('Box Dimensions Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box aspect ratio
        axes[1, 0].hist(train_stats['box_aspect_ratios'], bins=30, alpha=0.7, 
                       label='Train', edgecolor='black')
        axes[1, 0].hist(val_stats['box_aspect_ratios'], bins=30, alpha=0.7, 
                       label='Validation', edgecolor='black')
        axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Box Aspect Ratio Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Box position heatmap
        axes[1, 1].hist2d(train_stats['box_positions_x'], train_stats['box_positions_y'], 
                         bins=50, cmap='hot')
        axes[1, 1].set_xlabel('X Position (normalized)')
        axes[1, 1].set_ylabel('Y Position (normalized)')
        axes[1, 1].set_title('Box Position Heatmap (Train)')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/data_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {output_dir}/data_analysis.png")
        plt.close()
        
        # 5. Box area distribution
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(train_stats['box_areas'], bins=50, alpha=0.7, label='Train', edgecolor='black')
        ax.hist(val_stats['box_areas'], bins=50, alpha=0.7, label='Validation', edgecolor='black')
        ax.set_xlabel('Box Area (pixels²)')
        ax.set_ylabel('Frequency')
        ax.set_title('Box Area Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/box_area_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Box area visualization saved to {output_dir}/box_area_distribution.png")
        plt.close()
    
    def visualize_samples(self, num_samples=6, output_dir='analysis_results'):
        """Visualize sample images with annotations"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = list(self.train_images.glob('*.jpg'))[:num_samples]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, img_file in enumerate(image_files):
            # Read image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Read labels
            label_file = self.train_labels / f"{img_file.stem}.txt"
            boxes = self.parse_yolo_label(label_file)
            
            # Draw boxes
            for box in boxes:
                x_center = box['x_center'] * w
                y_center = box['y_center'] * h
                box_w = box['width'] * w
                box_h = box['height'] * h
                
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, 'Snow Pole', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'{img_file.name}\n{len(boxes)} poles detected')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_annotations.png', dpi=300, bbox_inches='tight')
        print(f"✓ Sample visualizations saved to {output_dir}/sample_annotations.png")
        plt.close()
    
    def save_analysis_report(self, train_stats, val_stats, output_dir='analysis_results'):
        """Save analysis report as JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'train': {
                'num_images': train_stats['num_images'],
                'num_boxes': train_stats['num_boxes'],
                'images_without_boxes': train_stats['images_without_boxes'],
                'avg_boxes_per_image': float(np.mean(train_stats['boxes_per_image'])),
                'avg_box_width': float(np.mean(train_stats['box_widths'])) if train_stats['box_widths'] else 0,
                'avg_box_height': float(np.mean(train_stats['box_heights'])) if train_stats['box_heights'] else 0,
                'avg_box_area': float(np.mean(train_stats['box_areas'])) if train_stats['box_areas'] else 0,
            },
            'validation': {
                'num_images': val_stats['num_images'],
                'num_boxes': val_stats['num_boxes'],
                'images_without_boxes': val_stats['images_without_boxes'],
                'avg_boxes_per_image': float(np.mean(val_stats['boxes_per_image'])),
                'avg_box_width': float(np.mean(val_stats['box_widths'])) if val_stats['box_widths'] else 0,
                'avg_box_height': float(np.mean(val_stats['box_heights'])) if val_stats['box_heights'] else 0,
                'avg_box_area': float(np.mean(val_stats['box_areas'])) if val_stats['box_areas'] else 0,
            }
        }
        
        with open(f'{output_dir}/analysis_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n✓ Analysis report saved to {output_dir}/analysis_report.json")
        
    def run_full_analysis(self):
        """Run complete data analysis"""
        print("\n" + "="*60)
        print("SNOW POLE DETECTION - DATA ANALYSIS")
        print("="*60)
        
        # Analyze train and validation sets
        train_stats = self.analyze_dataset('train')
        val_stats = self.analyze_dataset('validation')
        
        # Print statistics
        self.print_statistics(train_stats, 'train')
        self.print_statistics(val_stats, 'validation')
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.visualize_statistics(train_stats, val_stats)
        self.visualize_samples()
        
        # Save report
        self.save_analysis_report(train_stats, val_stats)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nKey Findings:")
        print(f"  • Total training images: {train_stats['num_images']}")
        print(f"  • Total validation images: {val_stats['num_images']}")
        print(f"  • Average poles per image (train): {np.mean(train_stats['boxes_per_image']):.2f}")
        print(f"  • Average poles per image (val): {np.mean(val_stats['boxes_per_image']):.2f}")
        
        return train_stats, val_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Snow Pole Detection Dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to Poles2025 dataset')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(args.data_path)
    analyzer.run_full_analysis()
