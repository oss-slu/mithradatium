import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import mithridatium.evaluator as evaluator
import mithridatium.loader as loader

def main():
    # Load a pretrained or random resnet18
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.eval()

    # Prepare CIFAR10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    feature_module = loader.get_feature_module(model)

    embs, labels = evaluator.extract_embeddings(model, test_loader, feature_module)
    print(f"Embeddings shape: {embs.shape}")

    loss, accy = evaluator.evaluate(model, test_loader)
    print(f"Test accuracy: {accy*100:.2f}% | Test loss: {loss:.4f}")

if __name__ == "__main__":
    main()
