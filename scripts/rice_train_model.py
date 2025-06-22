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

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set the plant name for MLflow tracking (set as class variable!)
    Config.PLANT_NAME = "Rice"
    Config.MODEL_NAME_TOSAVE = "RiceDiseaseDetector"
    logger.info(f"Plant name set to: {Config.PLANT_NAME}")

    # Setup
    config = Config()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verify Intel XPU[1]
    config.verify_xpu()
    
    # Data preparation
    source_dir = "data/raw/rice_images"  # Your single directory
    output_dir = "data/processed/rice"

    # Now get the experiment name and set it in MLflow
    config.MLFLOW_EXPERIMENT_NAME = config.get_experiment_name()
    logger.info(f"MLflow experiment name set to: {config.MLFLOW_EXPERIMENT_NAME}")

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    if config.DEVICE == "cpu":
        mlflow.start_run(run_name=f"{config.PLANT_NAME}_Training_Run_CPU")
    else:
        mlflow.start_run(run_name=f"{config.PLANT_NAME}_Training_Run_GPU")


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
    
    # Log total images and class distribution BEFORE augmentation
    total_images_before = len(train_dataset)
    class_counts_before = {cls: 0 for cls in train_dataset.classes}
    for _, label in train_dataset:
        class_counts_before[train_dataset.classes[label]] += 1
    mlflow.log_param("total_images_before_augmentation", total_images_before)
    mlflow.log_dict(class_counts_before, "class_distribution_before_augmentation.json")


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

    # Log total images and class distribution AFTER augmentation (approximate, since augmentations are random)
    # We'll sample one batch and count the class distribution
    aug_class_counts = {cls: 0 for cls in train_dataset.classes}
    inputs, labels = next(iter(train_loader))
    for label in labels:
        aug_class_counts[train_dataset.classes[label]] += 1

    mlflow.log_param("total_images_after_augmentation_sample_batch", len(labels))
    mlflow.log_dict(aug_class_counts, "class_distribution_after_augmentation_sample_batch.json")
    
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
