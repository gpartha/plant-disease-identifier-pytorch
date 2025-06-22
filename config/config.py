"""Intel XPU optimized configuration for plant disease detection."""

from pathlib import Path
import torch

class Config:
    # Intel XPU specific settings
    DEVICE = "xpu:0" if torch.xpu.is_available() else "cpu"
    #DEVICE = "cpu" # Running on CPU to see if it works
    USE_MIXED_PRECISION = True
    
    # Model configuration - based on search results showing MobileNetV3 effectiveness[4][5]
    MODEL_NAME = "MobileNetV3Large"
    IMAGE_SIZE = (224, 224)
    PRETRAINED = True
    
    # Training parameters optimized for small datasets[3]
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 15
    WEIGHT_DECAY = 0.01
    
    # Data augmentation for limited data[2]
    #AUGMENTATION_STRENGTH = "aggressive"
    AUGMENTATION_STRENGTH = "None"
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # MLflow settings[6]
    # Set to None and will be set dynamically in training script   
    PLANT_NAME = None
    MLFLOW_EXPERIMENT_NAME = "Plant_Disease_Intel_XPU_Default"
    MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
    
    # MLflow model registry/model name
    MODEL_NAME_TOSAVE = "PlantDiseaseDetector_Intel_XPU"
    
    @classmethod
    def get_experiment_name(cls):
        plant = cls.PLANT_NAME if cls.PLANT_NAME else "Plant"
        return f"{plant}_Disease_Intel_XPU"

    @classmethod
    def verify_xpu(cls):
        """Verify Intel XPU availability."""
        if not torch.xpu.is_available():
            raise RuntimeError("Intel XPU not available!")
        print(f"Using Intel XPU: {torch.xpu.get_device_name()}")
