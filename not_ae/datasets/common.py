from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class TrainDataset(Dataset):
    def __init__(self, dataset: Dataset, length: int):
        self.dataset = dataset
        assert hasattr(self.dataset, "__len__")
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]


class IgnoreLabelDataset(Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


class FakeDataset(Dataset):
    def __init__(
        self,
        dataset: Sequence,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.dataset = dataset
        self.norm_transform = T.Normalize(mean, std)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = torch.from_numpy(self.dataset[index]).float()
        item = self.norm_transform(item)
        return item
