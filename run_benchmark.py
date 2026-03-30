import time
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        # Create a tensor of shape (3, 224, 224) simulating an image
        self.data = [torch.randn(3, 224, 224) for _ in range(size)]
        self.labels = [torch.randint(0, 10, (1,)).item() for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def benchmark_loader(pin_memory):
    dataset = DummyDataset(size=5000)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark with pin_memory={pin_memory} on {device}")

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    return duration

if __name__ == "__main__":
    if torch.cuda.is_available():
        # Warmup
        _ = torch.randn(10).cuda()

    duration_without = benchmark_loader(pin_memory=False)
    duration_with = benchmark_loader(pin_memory=True)

    if duration_with < duration_without:
        improvement = (duration_without - duration_with) / duration_without * 100
        print(f"Improvement: {improvement:.2f}% faster with pin_memory=True")
    else:
        print("No improvement or slower with pin_memory=True (might be due to CPU-only env or system variance)")
