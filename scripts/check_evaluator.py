# scripts/check_evaluator.py
from mithridatium.loader import load_resnet18
from mithridatium.data import get_cifar10_loader
from mithridatium.evaluator import extract_embeddings

def main():
    model, feat = load_resnet18("models/resnet18.pth")  # fine if missing
    loader = get_cifar10_loader(batch_size=64)          # downloads CIFAR-10 once
    embs, labels = extract_embeddings(model, loader, feat)
    print("Embeddings shape:", embs.shape)  # expect ~ [10000, 512] for ResNet-18
    print("Labels shape:", labels.shape)    # expect [10000]

if __name__ == "__main__":
    main()
