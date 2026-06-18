import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TokenDataset(Dataset):
    """Memory-mapped token dataset from a pre-tokenized .bin file."""

    def __init__(self, path: str | Path, context_length: int):
        self.context_length = context_length
        data = np.memmap(path, dtype=np.uint16, mode="r")
        self.data = torch.from_numpy(data.astype(np.int64))

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        return chunk[:-1], chunk[1:]


def make_dataloader(path: str | Path, context_length: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TokenDataset(path, context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
