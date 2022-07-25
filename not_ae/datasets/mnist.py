from pathlib import Path
from typing import Callable, Optional

import torch
from torchvision import transforms as T
from torchvision.datasets import MNIST

from not_ae.utils.general import DATA_DIR, REGISTRY
from not_ae.utils.transform import NormalizeInverse


@REGISTRY.dataset.register()
class MNISTDataset(MNIST):
    def __init__(
        self,
        root=Path(DATA_DIR, "mnist"),
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split: str = "train",
    ) -> None:
        download = not Path(DATA_DIR, "mnist").exists()
        super().__init__(root, split == "train", transform, target_transform, download)
        self.mean, self.std = [0.5], [0.5]
        self.norm_transform = T.Normalize(self.mean, self.std)
        self.transform = T.Compose([T.Resize(32), T.ToTensor(), self.norm_transform])
        self.inverse_transform = NormalizeInverse(self.mean, self.std)

    def __getitem__(self, index: int) -> torch.Tensor:
        out = super().__getitem__(index)
        return out[0]
