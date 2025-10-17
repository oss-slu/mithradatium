from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _unwrap_state_dict(ckpt):
    sd = ckpt
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("state_dict"), dict):
            sd = ckpt["state_dict"]
        elif isinstance(ckpt.get("model"), dict):
            sd = ckpt["model"]

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def load_resnet18(model_path: str | None):
    model = models.resnet18(weights=None)

    in_feats = model.fc.in_features
    if getattr(model.fc, "out_features", 1000) != 10:
        model.fc = nn.Linear(in_feats, 10)

    feature_module = model.avgpool

    if model_path and Path(model_path).exists():
        ckpt = _safe_torch_load(model_path)
        sd = _unwrap_state_dict(ckpt)

        fc_w, fc_b = sd.get("fc.weight"), sd.get("fc.bias")
        if fc_w is not None and fc_w.shape[0] != 10:
            sd.pop("fc.weight", None)
            sd.pop("fc.bias", None)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[loader] loaded ckpt; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[loader] checkpoint not found at '{model_path}'. Using randomly initialized model (ok for pipeline tests).")

    model.eval()
    return model, feature_module

def get_feature_module(model):
    """
    Returns the penultimate feature module for a given model architecture.
    For ResNet-18, returns model.avgpool.
    Extend this function for other architectures as needed.
    """
    arch = model.__class__.__name__
    if arch == 'ResNet':
        return model.avgpool
    else:
        raise NotImplementedError(f"Feature module not defined for architecture: {arch}")