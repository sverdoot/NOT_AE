import torch

from not_ae.utils.general import REGISTRY


@REGISTRY.cost.register()
def L2Cost(real_batch: torch.Tensor, fake_batch: torch.Tensor) -> torch.Tensor:
    return torch.norm(real_batch - fake_batch, dim=-1, p=2) ** 2


@REGISTRY.cost.register()
def L1Cost(real_batch: torch.Tensor, fake_batch: torch.Tensor) -> torch.Tensor:
    return torch.norm(real_batch - fake_batch, dim=-1, p=1)


# TODO
