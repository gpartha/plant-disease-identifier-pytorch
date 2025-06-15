"""Aggressive data augmentation for small plant disease datasets."""

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class PlantDiseaseAugmentation:
    def __init__(self, config):
        self.config = config
    
    def get_train_transforms(self):
        """
        Aggressive augmentation strategy for small datasets.
        Based on research showing effectiveness for plant disease detection[2][3].
        """
        if self.config.AUGMENTATION_STRENGTH == "aggressive":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(self.config.IMAGE_SIZE[0], scale=(0.6, 1.0)),
                
                # Geometric transformations - preserve disease features
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=45),
                transforms.RandomAffine(
                    degrees=25, translate=(0.15, 0.15), 
                    scale=(0.8, 1.3), shear=15
                ),
                
                # Color augmentations - simulate field conditions
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, 
                    saturation=0.5, hue=0.3
                ),
                transforms.RandomGrayscale(p=0.1),
                
                # Advanced augmentations
                transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Cutout-style augmentation for robustness
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.25))
            ])
        else:
            return self.get_standard_transforms()
    
    def get_val_transforms(self):
        """Standard validation transforms."""
        return transforms.Compose([
            transforms.Resize(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def visualize_augmentations(self, dataset, save_path=None):
        """Visualize augmentation effects to ensure disease features are preserved."""
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        images, labels = next(iter(dataloader))
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(8, len(images))):
            img = images[i] * std + mean
            img = torch.clamp(img, 0, 1)
            
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(f'Augmented: {dataset.classes[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
