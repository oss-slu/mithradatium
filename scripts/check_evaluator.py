import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import mithridatium.evaluator as evaluator
import mithridatium.loader as loader

def main():
    parser = argparse.ArgumentParser()
    '''
    .venv/bin/python -m scripts.check_evaluator --model models/resnet18_poison.pth
    '''
    parser.add_argument("--model", type=str, default="models/resnet18_bd.pth", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    args = parser.parse_args()

    # Load model from checkpoint
    model, feature_module = loader.load_resnet18(args.model)

    # Prepare CIFAR-10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Extract embeddings
    embs, labels = evaluator.extract_embeddings(model, test_loader, feature_module)
    print(f"Embeddings shape: {embs.shape}")

    # Evaluate accuracy
    loss, accy = evaluator.evaluate(model, test_loader)
    print(f"Test accuracy: {accy*100:.2f}% | Test loss: {loss:.4f}")

if __name__ == "__main__":
    main()
