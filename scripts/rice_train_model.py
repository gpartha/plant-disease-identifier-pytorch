"""Main training script for training rice plant disease detection using Intel XPU."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import logging
import mlflow

from config.config import Config
from src.data.data_preparation import DataPreparator
from src.data.data_augmentation import PlantDiseaseAugmentation
from src.data.data_analysis import DataAnalyzer
from src.models.model_architecture import PlantDiseaseModel
from src.training.trainer import IntelXPUTrainer

def main():
    # Set the plant name for MLflow tracking (set as class variable!)
    Config.PLANT_NAME = "Rice"
    Config.MODEL_NAME_TOSAVE = "RiceDiseaseDetector"
    logger.info(f"Plant name set to: {Config.PLANT_NAME}")

    # Setup
    config = Config()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Verify Intel XPU[1]
    config.verify_xpu()
    
    # Data preparation
    source_dir = "data/raw/rice_images"  # Your single directory
    output_dir = "data/processed/rice"

    # Now get the experiment name and set it in MLflow
    config.MLFLOW_EXPERIMENT_NAME = config.get_experiment_name()
    logger.info(f"MLflow experiment name set to: {config.MLFLOW_EXPERIMENT_NAME}")

    preparator = DataPreparator(config)
    split_stats = preparator.split_dataset_stratified(source_dir, output_dir)
    logger.info(f"Dataset split completed: {split_stats}")
    
    # Data loading with augmentation
    augmentation = PlantDiseaseAugmentation(config)
    
    train_dataset = datasets.ImageFolder(
        root=Path(output_dir) / 'train',
        transform=augmentation.get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=Path(output_dir) / 'val',
        transform=augmentation.get_val_transforms()
    )
    
    # Data analysis
    analyzer = DataAnalyzer()
    train_analysis = analyzer.analyze_dataset_comprehensive(train_dataset, "Training Dataset")
    logger.info(f"Dataset analysis: {train_analysis}")
    
    # Visualize augmentations
    augmentation.visualize_augmentations(train_dataset, 'augmentation_samples.png')
    
    # Data loaders optimized for Intel XPU
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model creation
    model = PlantDiseaseModel(
        num_classes=len(train_dataset.classes),
        config=config
    )
    
    logger.info(f"Model created with {len(train_dataset.classes)} classes: {train_dataset.classes}")
    
    # Training
    trainer = IntelXPUTrainer(model, train_loader, val_loader, config)
    training_results = trainer.train(train_dataset.classes)
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation accuracy: {training_results['best_val_acc']:.2f}%")

if __name__ == "__main__":
    main()
