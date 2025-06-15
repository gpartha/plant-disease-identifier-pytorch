"""
MobileNetV3Large optimized for plant disease detection on Intel XPU.
Based on research showing 99.42% accuracy for plant disease detection[5].
"""

from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, config):
        super(PlantDiseaseModel, self).__init__()
        self.num_classes = num_classes
        self.config = config
        
        # Load MobileNetV3Large backbone - proven effective[4][5]
        self.backbone = models.mobilenet_v3_large(pretrained=config.PRETRAINED)
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier optimized for small datasets and edge deployment
        self.classifier = nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))), # Capture global features
            ("flatten", nn.Flatten()),                 # Flatten features for linear layers
            ("dropout1", nn.Dropout(0.5)),             # Regularization to prevent overfitting
            ("linear1", nn.Linear(960, 512)),          # Reduce dimensionality
            ("batchnorm1", nn.BatchNorm1d(512)),       # Help stabilize learning and accuracy
            ("relu1", nn.ReLU(inplace=True)),          # ReLU activation for non-linearity, learning complex patterns, preventing vanishing gradients
            ("dropout2", nn.Dropout(0.4)),             # Further dropout to reduce overfitting
            ("linear2", nn.Linear(512, 256)),          # Further reduce dimensionality
            ("batchnorm2", nn.BatchNorm1d(256)),       # Help stabilize learning and accuracy
            ("relu2", nn.ReLU(inplace=True)),          # ReLU activation for non-linearity, learning complex patterns, preventing vanishing gradients
            ("dropout3", nn.Dropout(0.3)),             # Further dropout to reduce overfitting
            ("linear3", nn.Linear(256, num_classes))   # Final output layer for classification
        ]))
        
        # Move the entire model to the correct device after all modifications
        self.to(config.DEVICE)
        
        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Ensure input is on the same device as the model
        x = x.to(next(self.parameters()).device)
        features = self.backbone.features(x)
        output = self.classifier(features)
        return output
    
    def get_model_summary(self, input_size=(1, 3, 224, 224)):
        """Get detailed model summary for logging."""
        return summary(self, input_size=input_size, verbose=0)
    
    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
