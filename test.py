import json
import logging
from pathlib import Path
from typing import Dict, Union

import fire
import lpips
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from ruamel import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from not_ae.datasets.common import FakeDataset
from not_ae.utils.general import REGISTRY
from not_ae.utils.metrics import get_activation_statistics


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def compute_fid(
    ae: nn.Module, dataset: Dataset, stats_path: Union[str, Path], batch_size: int = 100
) -> float:
    data_stat = np.load(Path(stats_path))

    device = next(ae.parameters()).device
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()

    mu, sigma = get_activation_statistics(
        dataset,
        inception,
        2048,
        batch_size,
        num_workers=1,
    )[:2]

    score = calculate_frechet_distance(
        data_stat["mu"],
        data_stat["sigma"],
        mu,
        sigma,
    )

    return score


@torch.no_grad()
def compute_lpips(ae: nn.Module, dataloader, rec_dataloader) -> float:
    device = next(ae.parameters()).device
    lpips_alex = lpips.LPIPS(net="alex", eval_mode=True).to(device)
    lpips_value = 0
    for batch_real, batch_fake in zip(dataloader, rec_dataloader):
        lpips_value += lpips_alex(
            batch_real.to(device), batch_fake.to(device)
        ).sum().item() / len(rec_dataloader.dataset)

    return lpips_value


def test(
    config_path: Union[Path, str], ae_ckpt_path: Union[str, Path], split: str = "test"
):
    config: Dict = yaml.safe_load(Path(config_path).open("r"))

    print(yaml.safe_dump(config))

    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    test_dataset = REGISTRY.dataset.create(
        config["dataset"]["name"], **config["dataset"]["params"], split=split
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])

    ae = REGISTRY.model.create(
        config["model"]["ae"]["name"], **config["model"]["ae"]["params"]
    )
    ae = ae.to(config["device"])
    ae.load_state_dict(
        torch.load(
            ae_ckpt_path, map_location=config["device"]
        ),  # ["model_state_dict"],
        strict=True,
    )

    ae.inverse_transform = test_dataset.inverse_transform

    images = []
    for batch in test_dataloader:
        rec_batch = ae(batch.to(config["device"]))
        images.append(ae.inverse_transform(rec_batch).detach().cpu().numpy())
    images = np.concatenate(images)
    rec_test_dataset = FakeDataset(images, (0, 0, 0), (1, 1, 1))
    fid = compute_fid(ae, rec_test_dataset, config[f"fid_{split}_stat_path"])

    rec_test_dataset = FakeDataset(images, test_dataset.mean, test_dataset.std)
    rec_test_dataloader = DataLoader(rec_test_dataset, batch_size=config["batch_size"])
    lpips_value = compute_lpips(ae, test_dataloader, rec_test_dataloader)

    result = dict(fid=fid, lpips=lpips_value)
    json.dump(result, Path(config["save_dir"], f"{split}_resut.json").open("w"))


if __name__ == "__main__":
    fire.Fire()
