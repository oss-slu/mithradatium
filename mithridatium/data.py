# mithridatium/data.py
import torch
from torchvision import datasets, transforms

def dataloader_for(model_path: str, dataset: str, split: str, batch_size: int = 256):
    # TEMP: hardcoded CIFAR-10; replace with PreprocessConfig next sprint
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010]),
    ])
    if dataset.lower() != "cifar10":
        raise NotImplementedError("Only CIFAR-10 for now")
    ds = datasets.CIFAR10(root="data", train=(split=="train"), download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), num_workers=2)