from torch import nn
from torchvision import models

def create_model(model_name: str, num_classes: int) -> nn.Module:
    """Create ResNet model with custom classifier"""
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
    elif model_name == "resnet101":
        model = models.resnet101(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model