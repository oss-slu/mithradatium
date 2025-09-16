import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import argparse
import random

def get_device(device_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, test_loader, device, criterion):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

def main(args):

    device = get_device(args.device)

    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())

    use_pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=2, pin_memory=use_pin, generator=g)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)

    
    model = resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    epochs = args.epochs

    print("Training with the following parameters:\n", 
        f"Epochs = {args.epochs}\n",
        f"Train Batch Size = {args.train_batch_size}\n",
        f"Evaluation Batch Size = {args.eval_batch_size}\n",
        f"Learning Rate = {args.lr}\n",
        f"Seed = {args.seed}\n",
        f"Output Path = {args.output_path}\n",
        f"Device = {args.device}\n")
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f}")

    torch.save(model.state_dict(), args.output_path)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="# of epochs to iterate through", type=int, default=60)
    parser.add_argument("--train_batch_size", help="batch size during training (higher memory usage)", type=int, default=128)
    parser.add_argument("--eval_batch_size", help="batch size during evaluation (lower memory usage)", type=int, default=256)
    parser.add_argument("--lr", help="learning rate for optimizer", default=0.1, type=float)
    parser.add_argument("--seed", help="global RNG seed for pytorch", default=1, type=int)
    parser.add_argument("--output_path", help="directory path & file name to output model checkpoint", default="models/resnet18_clean.pth", type=str)
    parser.add_argument("--device", help="cuda device #, default is 0", default=0, type=int)
    args = parser.parse_args()
    main(args)