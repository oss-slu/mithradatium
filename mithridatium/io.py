from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class PreprocessConfig:
    input_size: Tuple[int, int] = (32, 32)
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float]  = (0.2023, 0.1994, 0.2010)
    normalize: bool = True
    ops: List[str] = field(default_factory=list) 

def load_preprocess_config(model_path: str) -> PreprocessConfig:
    print(f"[dummy] load_preprocess_config({model_path}) -> CIFAR-10 defaults")
    return PreprocessConfig()