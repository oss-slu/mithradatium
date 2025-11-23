import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mithridatium.defenses.strip import strip_scores

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 4) # 10 features, 4 classes

    def forward(self, x):
        # Flatten input if needed, but here we assume input is already flat or we flatten it
        # For this test, let's assume input is (N, 10)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)

def test_strip_scores():
    print("Testing strip_scores...")
    
    # Setup
    torch.manual_seed(42)
    model = MockModel()
    
    # Create dummy data: 100 samples, 10 features each
    data = torch.randn(100, 10)
    labels = torch.randint(0, 4, (100,))
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Test execution
    try:
        results = strip_scores(model, dataloader, num_bases=5, num_perturbations=10)
        
        entropies = results.get("entropies")
        
        print(f"Entropies: {entropies}")
        
        assert isinstance(entropies, list), "Entropies should be a list"
        assert len(entropies) == 5, f"Expected 5 entropies, got {len(entropies)}"
        assert all(isinstance(e, float) for e in entropies), "All entropies should be floats"
        
        print("strip_scores test passed!")
        
    except Exception as e:
        print(f"strip_scores test failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_strip_scores()
