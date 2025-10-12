# mithridatium/data.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .io import PreprocessConfig

def build_transform(pp: PreprocessConfig):
    t = []

    # Optional ops list (e.g., ["resize:32", "center_crop:32"])
    ops = getattr(pp, "ops", []) or []

    if ops:
        for op in ops:
            if op.startswith("resize:"):
                size = int(op.split(":")[1])
                t.append(transforms.Resize(size))
            elif op.startswith("center_crop:"):
                size = int(op.split(":")[1])
                t.append(transforms.CenterCrop(size))
    else:
        # Fallback to config-driven pipeline
        if getattr(pp, "input_size", None):
            h, w = (pp.input_size if isinstance(pp.input_size, tuple)
                    else (pp.input_size, pp.input_size))
            # Only add a resize if not already  (32,32)
            if (h, w) != (32, 32):
                t.append(transforms.Resize((h, w)))

    # Always tensor; normalize if requested
    t.append(transforms.ToTensor())
    if getattr(pp, "normalize", True):
        t.append(transforms.Normalize(pp.mean, pp.std))

    return transforms.Compose(t)

def build_dataloader(dataset_name: str, split: str, pp: PreprocessConfig,
                     batch_size: int = 128, root: str = "./data", num_workers: int = 2):
    tf = build_transform(pp)
    if dataset_name.lower() == "cifar10":
        ds = datasets.CIFAR10(root=root, train=(split == "train"), download=True, transform=tf)
    else:
        raise NotImplementedError(f"Dataset not supported: {dataset_name}")
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)