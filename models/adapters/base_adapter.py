import torch
import torch.nn as nn
from utils.registry import ADAPTERS

@ADAPTERS.register
class ConcatAdapter(nn.Module):
    """
    Adapter that extracts specified keys from input dict and concatenates them.
    If input is a Tensor, it passes it through (assuming it's already processed).
    """
    def __init__(self, keys, dim=1):
        super().__init__()
        self.keys = keys
        self.dim = dim

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch

        if not isinstance(batch, dict):
            raise TypeError(f"ConcatAdapter expects dict or Tensor input, got {type(batch)}")

        tensors = []
        for key in self.keys:
            if key not in batch:
                raise KeyError(f"Key '{key}' not found in input batch. Available keys: {list(batch.keys())}")
            tensors.append(batch[key])

        if not tensors:
            raise ValueError("No keys specified for ConcatAdapter")

        # Concatenate
        return torch.cat(tensors, dim=self.dim)
