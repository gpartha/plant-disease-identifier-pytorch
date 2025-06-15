"""Data preparation with DVC integration for small datasets."""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import json

logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, config):
        self.config = config
    
    def split_dataset_stratified(self, source_dir, output_dir):
        """
        Split single directory into train/val/test with stratification.
        Optimized for small datasets based on search results[3].
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_path / split).mkdir(parents=True, exist_ok=True)
        
        # Get all class directories
        classes = [d.name for d in source_path.iterdir() if d.is_dir()]
        
        split_stats = {
            'total_images': 0,
            'classes': len(classes),
            'class_distribution': {}
        }
        
        for class_name in classes:
            class_path = source_path / class_name
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))

            if len(images) < 3:
                logger.warning(f"Class {class_name} has only {len(images)} images - may cause issues")
                continue
            
            # Stratified split ensuring each split has samples from all classes
            train_imgs, temp_imgs = train_test_split(
                images, test_size=(1-self.config.TRAIN_RATIO), 
                random_state=42, shuffle=True
            )
            val_imgs, test_imgs = train_test_split(
                temp_imgs, 
                test_size=(self.config.TEST_RATIO/(self.config.VAL_RATIO + self.config.TEST_RATIO)),
                random_state=42
            )
            
            # Create class directories and copy files
            for img_list, split in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
                split_class_dir = output_path / split / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img in img_list:
                    shutil.copy2(img, split_class_dir / img.name)
            
            split_stats['class_distribution'][class_name] = {
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs),
                'total': len(images)
            }
            split_stats['total_images'] += len(images)
            
            logger.info(f"Class {class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
        
        # Save split statistics for DVC tracking
        with open(output_path / 'split_stats.json', 'w') as f:
            json.dump(split_stats, f, indent=2)
        
        return split_stats
