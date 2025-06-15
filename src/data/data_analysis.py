"""Comprehensive data analysis for plant disease detection."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import json

class DataAnalyzer:
    @staticmethod
    def analyze_dataset_comprehensive(dataset, title="Dataset Analysis"):
        """
        Comprehensive dataset analysis including class balance and augmentation effects.
        """
        # Class distribution analysis
        if hasattr(dataset, 'targets'):
            class_counts = Counter(dataset.targets)
        else:
            class_counts = Counter([dataset[i][1] for i in range(len(dataset))])
        
        classes = dataset.classes
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Class distribution
        axes[0, 0].bar(range(len(classes)), [class_counts[i] for i in range(len(classes))])
        axes[0, 0].set_xticks(range(len(classes)))
        axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 0].set_title(f'{title} - Class Distribution')
        axes[0, 0].set_ylabel('Number of Images')
        
        # 2. Class imbalance visualization
        counts = list(class_counts.values())
        axes[0, 1].hist(counts, bins=10, alpha=0.7)
        axes[0, 1].set_title('Class Size Distribution')
        axes[0, 1].set_xlabel('Number of Images per Class')
        axes[0, 1].set_ylabel('Number of Classes')
        
        # 3. Sample augmented images
        dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
        images, labels = next(iter(dataloader))
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        for i in range(min(4, len(images))):
            row = i // 2
            col = (i % 2) + 1 if i < 2 else (i % 2) + 1
            
            img = images[i] * std + mean
            img = torch.clamp(img, 0, 1)
            
            if i < 2:
                axes[0, 2].imshow(img.permute(1, 2, 0)) if i == 0 else axes[1, 0].imshow(img.permute(1, 2, 0))
                (axes[0, 2] if i == 0 else axes[1, 0]).set_title(f'Augmented: {classes[labels[i]]}')
                (axes[0, 2] if i == 0 else axes[1, 0]).axis('off')
            else:
                axes[1, i-1].imshow(img.permute(1, 2, 0))
                axes[1, i-1].set_title(f'Augmented: {classes[labels[i]]}')
                axes[1, i-1].axis('off')
        
        # 4. Dataset statistics
        stats_text = f"""
        Total Images: {len(dataset)}
        Number of Classes: {len(classes)}
        Min samples per class: {min(counts)}
        Max samples per class: {max(counts)}
        Mean samples per class: {np.mean(counts):.1f}
        Std samples per class: {np.std(counts):.1f}
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 2].set_title('Dataset Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_images': len(dataset),
            'num_classes': len(classes),
            'class_distribution': dict(class_counts),
            'classes': classes,
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': np.mean(counts),
            'std_samples': np.std(counts)
        }
