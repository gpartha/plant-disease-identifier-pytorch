"""Intel XPU optimized inference for WhatsApp deployment."""

import torch
import torch.nn as nn
from PIL import Image
import io
from typing import List, Dict
import intel_extension_for_pytorch as ipex
import json

from src.models.model_architecture import PlantDiseaseModel
from src.data.data_augmentation import PlantDiseaseAugmentation

class PlantDiseasePredictor:
    def __init__(self, model_path, config):
        self.config = config
        self.device = config.DEVICE
        
        # Setup transforms
        augmentation = PlantDiseaseAugmentation(config)
        self.transform = augmentation.get_val_transforms()
        
        # Load model
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load trained model optimized for Intel XPU inference."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # Initialize model
        self.model = PlantDiseaseModel(
            num_classes=len(self.class_names),
            config=self.config
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Optimize for Intel XPU inference[1]
        self.model = ipex.optimize(self.model, dtype=torch.float32)
        self.model.eval()
        
        print(f"Model loaded successfully. Classes: {self.class_names}")
    
    def predict_from_bytes(self, image_bytes, top_k=3):
        """
        Predict from image bytes for WhatsApp integration.
        Optimized for Intel XPU performance.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.config.USE_MIXED_PRECISION:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model(input_tensor)
                else:
                    outputs = self.model(input_tensor)
                
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
            
            results = []
            for i in range(len(top_probs)):
                results.append({
                    'disease': self.class_names[top_indices[i]],
                    'confidence': top_probs[i].item() * 100
                })
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def predict_image_path(self, image_path, top_k=3):
        """Predict from image file path."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.predict_from_bytes(image_bytes, top_k)
    
    def batch_predict(self, image_paths, top_k=3):
        """Batch prediction for multiple images."""
        results = []
        for image_path in image_paths:
            result = self.predict_image_path(image_path, top_k)
            results.append(result)
        return results
