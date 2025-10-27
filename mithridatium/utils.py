# mithridatium/utils.py
"""
Utility functions for data loading, preprocessing, and model configuration.
"""
from pathlib import Path
import torch
from torchvision import datasets, transforms
from dataclasses import dataclass, field
from typing import Tuple, List
import json


@dataclass
class PreprocessConfig:
    """Configuration for input preprocessing."""
    input_size: Tuple[int, int] = (32, 32)  # (H, W)
    channels_first: bool = True              # True = NCHW, False = NHWC
    value_range: Tuple[float, float] = (0.0, 1.0)
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)  # (R, G, B)
    std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)   # (R, G, B)
    normalize: bool = True
    ops: List[str] = field(default_factory=list)  # e.g., ["resize:32"]


def load_preprocess_config(model_path: str) -> PreprocessConfig:
    """
    Load preprocessing config from model's JSON sidecar file.
    
    Args:
        model_path: Path to the model checkpoint file.
        
    Returns:
        PreprocessConfig with loaded or default values.
    """
    card_path = Path(model_path).with_suffix(".json")
    if not card_path.exists():
        print(f"[warn] No model sidecar found at {card_path}, using CIFAR-10 defaults")
        return PreprocessConfig()
    
    data = json.loads(card_path.read_text())
    pp = data.get("preprocess", {})
    return PreprocessConfig(
        input_size=tuple(pp.get("input_size", (32, 32))),
        channels_first=pp.get("channels_first", True),
        value_range=tuple(pp.get("value_range", (0.0, 1.0))),
        mean=tuple(pp["mean"]),
        std=tuple(pp["std"]),
        normalize=pp.get("normalize", True),
        ops=list(pp.get("ops", [])),
    )


def dataloader_for(model_path: str, dataset: str, split: str, batch_size: int = 256):
    """
    Create a dataloader for the specified dataset.
    
    Args:
        model_path: Path to model (used to load preprocessing config).
        dataset: Dataset name (currently only "cifar10" supported).
        split: "train" or "test".
        batch_size: Batch size for the dataloader.
        
    Returns:
        torch.utils.data.DataLoader for the specified dataset.
    """
    # Load preprocessing config
    try:
        config = load_preprocess_config(model_path)
        mean, std = config.mean, config.std
    except Exception as e:
        print(f"[warn] Failed to load preprocess config: {e}")
        # Fallback to CIFAR-10 defaults
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if dataset.lower() != "cifar10":
        raise NotImplementedError(f"Only CIFAR-10 supported for now, got: {dataset}")
    
    ds = datasets.CIFAR10(
        root="data",
        train=(split == "train"),
        download=True,
        transform=tfm
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2
    )