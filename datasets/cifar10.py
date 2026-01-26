from torch.utils.data import Dataset
from torchvision import datasets, transforms
from utils.registry import DATASETS

@DATASETS.register
class CIFAR10Dataset(Dataset):
    def __init__(self, split="train", batch_size=64, num_workers=4):
        # 注意：这里我们只是示例，实际可能要从配置中读取更多参数
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=(split == "train"),
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]