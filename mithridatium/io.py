from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PreprocessConfig:
	input_size: Tuple[int, int]      # (H, W)
	channels_first: bool             # True = NCHW, False = NHWC
	value_range: Tuple[float, float] # e.g., (0.0, 1.0)
	mean: Tuple[float, float, float] # (R, G, B)
	std:  Tuple[float, float, float] # (R, G, B)
	ops:  List[str]                  # e.g., ["resize:32"]

	# Notes: True → NCHW (batch, channels, height, width) — common in PyTorch. False → NHWC
