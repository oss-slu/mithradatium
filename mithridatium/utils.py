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
from typing import Tuple, List

class PreprocessConfig:
    """Configuration for input preprocessing."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (3, 32, 32),   # (C, H, W)
        channels_first: bool = True,              # True = NCHW, False = NHWC
        value_range: Tuple[float, float] = (0.0, 1.0),
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),  # (R, G, B)
        std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),   # (R, G, B)
        normalize: bool = True,
        ops: List[str] = None,                     # e.g., ["resize:32"]
        dataset: str = "Unlisted"
    ):
        self.input_size = input_size
        self.channels_first = channels_first
        self.value_range = value_range
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.ops = ops if ops is not None else []
        self.dataset = dataset

    # ======== Getters ========
    def get_input_size(self):
        return self.input_size

    def get_channels_first(self):
        return self.channels_first

    def get_value_range(self):
        return self.value_range

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_normalize(self):
        return self.normalize

    def get_ops(self):
        return self.ops
    
    def get_dataset(self):
        return self.dataset

    # ======== Setters ========
    def set_input_size(self, input_size: Tuple[int, int]):
        self.input_size = input_size

    def set_channels_first(self, channels_first: bool):
        self.channels_first = channels_first

    def set_value_range(self, value_range: Tuple[float, float]):
        self.value_range = value_range

    def set_mean(self, mean: Tuple[float, float, float]):
        self.mean = mean

    def set_std(self, std: Tuple[float, float, float]):
        self.std = std

    def set_normalize(self, normalize: bool):
        self.normalize = normalize

    def set_ops(self, ops: List[str]):
        self.ops = ops

    def set_dataset(self, dataset):
        self.dataset = dataset
    



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