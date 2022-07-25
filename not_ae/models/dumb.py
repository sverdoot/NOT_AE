import torch
from torch import nn

from not_ae.utils.general import REGISTRY


@REGISTRY.model.register()
class Dumb_D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device)
