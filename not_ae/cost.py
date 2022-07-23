import torch
from torch import nn
from torch.nn import functional as F

from not_ae.utils.general import REGISTRY
from not_ae.utils.perceptual_networks import SimpleExtractor


@REGISTRY.cost.register()
class L2Cost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, real_batch: torch.Tensor, fake_batch: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(
            real_batch.reshape(real_batch.shape[0], -1),
            fake_batch.reshape(fake_batch.shape[0], -1),
        )


@REGISTRY.cost.register()
class L1Cost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, real_batch: torch.Tensor, fake_batch: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(
            real_batch.reshape(real_batch.shape[0], -1),
            fake_batch.reshape(fake_batch.shape[0], -1),
        )


@REGISTRY.cost.register()
class PerceptualCost(nn.Module):
    def __init__(self, backbone="vgg11", layer: int = 5):
        self.backbone = SimpleExtractor(backbone, layer, frozen=True, sigmoid_out=True)

    def forward(
        self, real_batch: torch.Tensor, fake_batch: torch.Tensor
    ) -> torch.Tensor:
        real = self.backbone(real_batch).view(real_batch.shape[0], -1)
        fake = self.backbone(real_batch).view(fake_batch.shape[0], -1)
        return F.mse_loss(real, fake)
