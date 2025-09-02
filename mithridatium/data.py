# mithridatium/data.py
import torch
from torchvision import datasets, transforms

def get_cifar10_loader(batch_size: int = 128):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader
