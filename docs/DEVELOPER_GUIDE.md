# Developer Guide

## Registry System

This project uses a **Registry** system to manage Models, Datasets, Losses, and Callbacks. This allows for modularity and easy extension without modifying core code.

### How it works

The registry maps string names (from configuration files) to Python classes.
We have 4 global registries defined in `utils/registry.py`:
- `MODELS`
- `DATASETS`
- `LOSSES`
- `CALLBACKS`

### Adding a New Model

1. Create a new python file in `models/` (e.g., `models/my_model.py`).
2. Import `MODELS` registry.
3. Decorate your class with `@MODELS.register`.

```python
import torch.nn as nn
from utils.registry import MODELS

@MODELS.register
class MyCustomModel(nn.Module):
    def __init__(self, param1=10):
        super().__init__()
        # ...

    def forward(self, x):
        # ...
        return x
```

4. Use it in your config:
```yaml
model:
  name: "MyCustomModel"
  params:
    param1: 20
```

### Adding a New Dataset

1. Create a new file in `datasets/` (e.g., `datasets/my_dataset.py`).
2. Import `DATASETS` registry.
3. Decorate your class with `@DATASETS.register`.

```python
from torch.utils.data import Dataset
from utils.registry import DATASETS

@DATASETS.register
class MyDataset(Dataset):
    def __init__(self, split="train", my_path="/tmp"):
        # ...
        pass
```

### Auto-Discovery

You **do not** need to manually import your new files in `__init__.py`.
The system automatically scans the `models/` and `datasets/` directories and registers any decorated classes found in `.py` files.

### Standard Components

- **Losses**: Standard PyTorch losses (CrossEntropyLoss, MSELoss, etc.) are pre-registered. You can use them by name.
- **Callbacks**: Several callbacks are available in `training/callbacks.py`.

## Pre-flight Checks (Sanity Checks)

The trainer performs several checks before starting:
1. **Data Leakage**: Checks if validation samples exist in training set (requires `get_sample_id` method on dataset).
2. **Visual Check**: Saves a batch of augmented images to `output_dir/sanity_check/`.
3. **Configuration**: Warns about suboptimal settings (e.g., `num_workers=0`).
