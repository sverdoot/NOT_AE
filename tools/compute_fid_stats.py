from pathlib import Path
from typing import Optional, Tuple, Union

import fire
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3

from not_ae.utils.general import REGISTRY
from not_ae.utils.metrics import get_activation_statistics


def compute_fid_stats(
    dataset_name: str,
    save_path: Optional[Union[str, Path]] = None,
    batch_size: int = 100,
    dp: bool = True,
    device="cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = REGISTRY.dataset.create(dataset_name, split="test")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    if dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    mu, sigma, _ = get_activation_statistics(
        dataset, model, dims=2048, batch_size=batch_size
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(save_path.as_posix(), mu=mu, sigma=sigma)
    return mu, sigma


if __name__ == "__main__":
    fire.Fire(compute_fid_stats)
