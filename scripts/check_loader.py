# scripts/check_loader.py
from mithridatium.loader import load_resnet18

def main():
    model, feature_layer = load_resnet18("models/resnet18.pth")
    print("Model type:", type(model))
    print("Feature layer:", feature_layer)

if __name__ == "__main__":
    main()
