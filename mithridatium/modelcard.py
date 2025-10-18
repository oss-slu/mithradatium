import json
from pathlib import Path
from .io import PreprocessConfig

def load_preprocess_config(model_path: str) -> PreprocessConfig:
	card_path = Path(model_path).with_suffix(".json")
	data = json.loads(card_path.read_text())
	pp = data["preprocess"]
	return PreprocessConfig(
		input_size=tuple(pp["input_size"]),
		channels_first=pp.get("channels_first", True),
		value_range=tuple(pp.get("value_range", (0.0, 1.0))),
		mean=tuple(pp["mean"]),
		std=tuple(pp["std"]),
		ops=list(pp.get("ops", [])),
	)
