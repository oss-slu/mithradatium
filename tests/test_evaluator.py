import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import mithridatium.evaluator as evaluator
import mithridatium.loader as loader
import unittest

class TestEvaluator(unittest.TestCase):
    def test_extract_embeddings_and_evaluate(self):
        # Use a small subset of CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        indices = list(range(512))
        subset = torch.utils.data.Subset(testset, indices)
        loader_ = DataLoader(subset, batch_size=128, shuffle=False)

        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.eval()

        feature_module = loader.get_feature_module(model)
        embs, labels = evaluator.extract_embeddings(model, loader_, feature_module)
        print(f"Embeddings shape: {embs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"First 5 labels: {labels[:5].tolist()}")
        loss, accy = evaluator.evaluate(model, loader_)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accy*100:.2f}%")
        self.assertTrue(embs.shape[0] > 0)
        self.assertTrue(labels.shape[0] > 0)
        self.assertTrue(loss >= 0)
        self.assertTrue(accy >= 0)

if __name__ == "__main__":
    unittest.main()
