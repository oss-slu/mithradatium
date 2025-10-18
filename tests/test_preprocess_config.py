import os
import pytest
from mithridatium.modelcard import load_preprocess_config

def test_load_preprocess_config():
	model_path = os.path.join(os.path.dirname(__file__), '../models/resnet18_bd.pth') # Load the sidecar
	config = load_preprocess_config(model_path)
	assert config.input_size == (32, 32) # check if the sidecar has expected values
	assert config.channels_first is True
	assert config.value_range == (0.0, 1.0)
	assert config.mean == (0.4914, 0.4822, 0.4465)
	assert config.std == (0.2023, 0.1994, 0.2010)
	assert config.ops == ["resize:32"]