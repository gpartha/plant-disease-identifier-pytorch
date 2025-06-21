"""
Intel XPU optimized trainer with comprehensive MLflow integration.
Based on Intel XPU optimization patterns[1][7].
"""

import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch as ipex
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class IntelXPUTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Verify and move to Intel XPU[1]
        config.verify_xpu()
        self.device = config.DEVICE
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Intel XPU optimization - critical for performance[1][7]
        self.model, self.optimizer = ipex.optimize(
            self.model, 
            optimizer=self.optimizer, 
            dtype=torch.float32
        )
        
        # Learning rate scheduler for small datasets
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.5, min_lr=1e-7
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self):
        """Train for one epoch with Intel XPU optimization."""
        self.model.train()
        # Move model parameters to Intel XPU[1]
        self.model.to(self.device)
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            # Move data to XPU[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)  # Ensure labels are on the same device
            
            if batch_idx == 0:
                # Debug: print device info for first batch
                logger.info(f"[DEBUG] Model device: {next(self.model.parameters()).device}")
                logger.info(f"[DEBUG] Inputs device: {inputs.device}")
                logger.info(f"[DEBUG] Labels device: {labels.device}")
            
            self.optimizer.zero_grad()
            
            # Mixed precision training for Intel XPU performance
            if self.config.USE_MIXED_PRECISION:
                with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)  # Ensure labels are on the same device
                
                if batch_idx == 0:
                    # Debug: print device info for first batch
                    logger.info(f"[DEBUG] Model device: {next(self.model.parameters()).device}")
                    logger.info(f"[DEBUG] Inputs device: {inputs.device}")
                    logger.info(f"[DEBUG] Labels device: {labels.device}")
                
                if self.config.USE_MIXED_PRECISION:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, class_names):
        """
        Train with comprehensive MLflow logging[6][16].
        """
        # mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
        # Remove mlflow.start_run() here; assume run is started in main script
        
        # Log hyperparameters[16]
        mlflow.log_params({
            "model_architecture": "MobileNetV3Large_Intel_XPU",
            "device": self.device,
            "batch_size": self.config.BATCH_SIZE,
            "learning_rate": self.config.LEARNING_RATE,
            "num_epochs": self.config.NUM_EPOCHS,
            "num_classes": len(class_names),
            "mixed_precision": self.config.USE_MIXED_PRECISION,
            "weight_decay": self.config.WEIGHT_DECAY,
            "augmentation_strength": self.config.AUGMENTATION_STRENGTH,
            "framework": "PyTorch_Intel_XPU"
        })
        
        # Log model architecture[16]
        model_summary = self.model.get_model_summary()
        with open("model_summary.txt", "w") as f:
            f.write(str(model_summary))
        mlflow.log_artifact("model_summary.txt")
        
        # Start training timer
        total_start = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')

            # start time for the epoch
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            # Time taken for the epoch
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1} time: {epoch_time:.2f} seconds")
            mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)
            
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Log metrics to MLflow[16]
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            logger.info(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                # Save checkpoint with comprehensive metadata
                model_path = self.config.MODELS_DIR / f"{self.config.PLANT_NAME}_best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'class_names': class_names,
                    'config': self.config.__dict__,
                    'model_summary': str(model_summary)
                }, model_path)
                
                # Log model to MLflow[6]
                mlflow.pytorch.log_model(
                    self.model, 
                    self.config.MODEL_NAME_TOSAVE,
                    registered_model_name=self.config.MODEL_NAME_TOSAVE
                )
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.PATIENCE:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        # Total training time
        total_time = time.time() - total_start
        logger.info(f"Total training time: {total_time:.2f} seconds")
        mlflow.log_metric("total_training_time_sec", total_time)
        
        # Log final metrics and training curves
        mlflow.log_metric("best_val_accuracy", self.best_val_acc)
        
        # Save and log training curves
        self._save_training_curves()
        
        logger.info(f'Training completed. Best accuracy: {self.best_val_acc:.2f}%')
        
        return {
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }
    
    def _save_training_curves(self):
        """Save and log training curves to MLflow."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('training_curves.png')
        plt.close()
