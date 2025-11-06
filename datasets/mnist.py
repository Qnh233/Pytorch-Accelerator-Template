from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, split="train", data_dir="./data/mnist"):
        # 注意：这里我们只是示例，实际可能要从配置中读取更多参数
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        self.dataset = datasets.MNIST(
            root=data_dir,
            train=(split == "train"),
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 确保返回的是张量格式
        return self.dataset[idx]

