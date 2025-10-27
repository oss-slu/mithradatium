from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from dataclasses import dataclass, field
from typing import Tuple, List
import json

def load_resnet18(model_path: str | None):
    """
    Load a ResNet-18 model with optional checkpoint.
    
    Args:
        model_path: Path to checkpoint file, or None for random init.
        
    Returns:
        Tuple of (model, feature_module).
    """
    model = models.resnet18(weights=None)

    # expose the penultimate layer (avgpool -> flatten) for features
    feature_module = model.avgpool

    # try to load a checkpoint if provided
    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location="cpu")
        # allow partial load to avoid shape mismatches early on
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[loader] loaded ckpt; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[loader] checkpoint not found at '{model_path}'. Using randomly initialized model (ok for pipeline tests).")

    model.eval()
    return model, feature_module

def get_feature_module(model):
    """
    Returns the penultimate feature module for a given model architecture.
    
    Args:
        model: PyTorch model instance.
        
    Returns:
        The feature extraction module (e.g., model.avgpool for ResNet).
        
    Raises:
        NotImplementedError: If architecture is not supported.
    """
    arch = model.__class__.__name__
    if arch == 'ResNet':
        return model.avgpool
    # Example for future extension:
    # elif arch == 'VGG':
    #     return model.classifier[0]
    else:
        raise NotImplementedError(f"Feature module not defined for architecture: {arch}")
    
def build_model(arch: str = "resnet18", num_classes: int = 10):
    """
    Build a model with the specified architecture.
    
    Args:
        arch: Architecture name (currently only "resnet18" supported).
        num_classes: Number of output classes.
        
    Returns:
        Tuple of (model, feature_module).
    """
    if arch.lower() == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None)
    elif arch == "resnet34":
        from torchvision.models import resnet34
        m = resnet34(weights=None)
    else:
        raise NotImplementedError(f"Architecture '{arch}' not yet supported")
        
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m, get_feature_module(m)
 

def load_weights(model, ckpt_path: str):
    """
    Load model weights from a checkpoint file.
    
    Args:
        model: PyTorch model instance.
        ckpt_path: Path to checkpoint file.
        
    Returns:
        Model with loaded weights.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] load_weights: missing={missing}, unexpected={unexpected}")
    return model