from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

def load_resnet18(model_path: str | None):
    model = models.resnet18(weights=None)

    # QUICKFIX Resize the FC layer for CIFAR-10 (10 output classes)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 10)

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
    For ResNet-18, returns model.avgpool.
    """
    arch = model.__class__.__name__
    if arch == 'ResNet':
        return model.avgpool
    else:
        raise NotImplementedError(f"Feature module not defined for architecture: {arch}")
