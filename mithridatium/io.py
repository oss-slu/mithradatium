from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class PreprocessConfig:
    channels_first: bool = True             # True = NCHW, False = NHWC
    value_range: Tuple[float, float] = (0.0, 1.0)
    input_size: Tuple[int, int] = (32, 32)  # (H, W)
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)  # (R, G, B)
    std:  Tuple[float, float, float] = (0.2023, 0.1994, 0.2010) # (R, G, B)
    normalize: bool = True
    ops: List[str] = field(default_factory=list)  # e.g., ["resize:32"]

def load_preprocess_config(model_path: str) -> PreprocessConfig:
    print(f"[dummy] load_preprocess_config({model_path}) -> CIFAR-10 defaults")
    return PreprocessConfig()
	

	# Notes: True → NCHW (batch, channels, height, width) — common in PyTorch. False → NHWC
