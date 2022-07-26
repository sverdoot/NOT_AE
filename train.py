import logging
from pathlib import Path
from typing import Dict, Optional, Union

import fire
import torch
from ruamel import yaml
from torch import nn
from torch.utils.data import DataLoader

from not_ae.trainer import Trainer
from not_ae.utils.general import REGISTRY, random_seed


FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def train(config_path: Union[Path, str], seed: Optional[int] = None):
    config: Dict = yaml.safe_load(Path(config_path).open("r"))
    if seed is not None:
        random_seed(seed)

    print(yaml.safe_dump(config))

    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    train_dataset = REGISTRY.dataset.create(
        config["dataset"]["name"], **config["dataset"]["params"], split="train"
    )
    val_dataset = REGISTRY.dataset.create(
        config["dataset"]["name"], **config["dataset"]["params"], split="val"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    ae = REGISTRY.model.create(
        config["model"]["ae"]["name"], **config["model"]["ae"]["params"]
    )
    ae = ae.to(config["device"])
    potential = REGISTRY.model.create(
        config["model"]["potential"]["name"], **config["model"]["potential"]["params"]
    )
    potential = potential.to(config["device"])

    if config["data_parallel"] and config["device"] != "cpu":
        ae = nn.DataParallel(ae)
        potential = nn.DataParallel(potential)
    ae.inverse_transform = train_dataset.inverse_transform

    ae_opt = torch.optim.Adam(ae.parameters(), **config["model"]["ae"]["opt_params"])
    potential_opt = torch.optim.Adam(
        potential.parameters(), **config["model"]["ae"]["opt_params"]
    )

    cost = REGISTRY.model.create(config["cost"]["name"]).to(config["device"])

    callbacks = []
    for callback in config["callbacks"]:
        if "ae" in callback["params"]:
            callback["params"]["ae"] = ae
        if "potential" in callback["params"]:
            callback["params"]["potential"] = potential
        if "test_dataset" in callback["params"]:
            callback["params"]["test_dataset"] = val_dataset
        callback = REGISTRY.model.create(callback["name"], **callback["params"])
        callbacks.append(callback)

    trainer = Trainer(
        ae,
        potential,
        ae_opt,
        potential_opt,
        cost,
        train_dataloader,
        val_dataloader,
        callbacks,
        **config["train_params"],
    )
    trainer.train(config["n_epoch"])


if __name__ == "__main__":
    fire.Fire()
