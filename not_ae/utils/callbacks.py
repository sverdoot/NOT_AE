import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid

from .general import REGISTRY


class Callback(ABC):
    cnt: int = 0

    @abstractmethod
    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        raise NotImplementedError

    def reset(self):
        self.cnt = 0


@REGISTRY.callback.register()
class WandbCallback(Callback):
    def __init__(
        self,
        *,
        invoke_every: int = 1,
        init_params: Optional[Dict] = None,
        keys: Optional[List[str]] = None,
    ):
        self.init_params = init_params if init_params else {}
        import wandb

        self.wandb = wandb
        wandb.init(**self.init_params)

        self.invoke_every = invoke_every
        self.keys = keys

        self.img_transform = transforms.Resize(
            128, interpolation=transforms.InterpolationMode.NEAREST
        )

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        step = info.get("step", self.cnt)
        if step % self.invoke_every == 0:
            wandb = self.wandb
            if not self.keys:
                self.keys = info.keys()
            log = dict()
            for key in self.keys:
                if key not in info:
                    continue
                if isinstance(info[key], np.ndarray):
                    log[key] = wandb.Image(
                        make_grid(
                            self.img_transform(
                                torch.clip(torch.from_numpy(info[key][:25]), 0, 1)
                            ),
                            nrow=5,
                        ),
                        caption=key,
                    )
                else:
                    log[key] = info[key]
            log["step"] = step
            wandb.log(log)
        self.cnt += 1
        return 1

    def reset(self):
        super().reset()
        self.wandb.init(**self.init_params)


@REGISTRY.callback.register()
class LogCallback(Callback):
    def __init__(
        self,
        save_dir: Union[Path, str],
        keys: List[str],
        *,
        invoke_every: int = 1,
        resume=False,
    ):
        self.save_dir = Path(save_dir)
        self.invoke_every = invoke_every
        self.keys = keys

        self.save_paths = []
        for key in keys:
            path = Path(save_dir, f"{key}.txt")
            if not resume:
                path.open("w")
            self.save_paths.append(path)

    @torch.no_grad()
    def invoke(
        self,
        info: Dict[str, Union[float, np.ndarray]],
    ):
        step = info.get("step", self.cnt)
        if step % self.invoke_every == 0:
            for save_path, key in zip(self.save_paths, self.keys):
                if key not in info:
                    continue
                if isinstance(info[key], np.ndarray):
                    save_dir = Path(self.save_dir, key)
                    save_dir.mkdir(exist_ok=True)
                    step = info.get("step", 0)
                    save_path = Path(save_dir, f"{key}_{step}.png")
                    self.plot(info[key], save_path)
                    save_path = Path(save_dir, f"{key}_{step}.pdf")
                    self.plot(info[key], save_path)
                else:
                    with save_path.open("ab") as f:
                        np.savetxt(f, [info[key]], delimiter=" ", newline=" ")
        self.cnt += 1
        return 1

    def reset(self):
        super().reset()
        for save_path in self.save_paths:
            save_path.open("ab").write(b"\n")

    @staticmethod
    def plot(imgs, save_path):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            axs[0, i].imshow(img)
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[0, i].axis("off")
        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")


@REGISTRY.callback.register()
class TrainLogCallback(Callback):
    def __init__(self, invoke_every: int = 1) -> None:
        self.invoke_every = invoke_every

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        step = info.get("step", self.cnt)
        if step % self.invoke_every == 0:
            loss_ae = info["loss_ae"]
            loss_potential = info["loss_potential"]

            logger = logging.getLogger("train")
            logger.info(
                f"\nIteration: [{step}/{info['total']}], \
                Loss AE: {loss_ae:.3f}, Loss Potential: {loss_potential:.3f}"
            )

        self.cnt += 1
        return 1


@REGISTRY.callback.register()
class CheckpointCallback(Callback):
    def __init__(
        self, ae, potential, save_dir: Union[str, Path], *, invoke_every: int = 1
    ) -> None:
        self.invoke_every = invoke_every
        self.ae = ae
        self.potential = potential
        self.save_dir = Path(save_dir, "checkpoints")
        self.save_dir.mkdir(exist_ok=True)

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        step = info.get("step", self.cnt)
        if step % self.invoke_every == 0:
            if self.ae.dp:
                torch.save(
                    self.ae.gen.module.state_dict(),
                    Path(self.save_dir, f"ae_{step}.pth"),
                )
                torch.save(
                    self.potential.module.state_dict(),
                    Path(self.save_dir, f"potential_{step}.pth"),
                )
            else:
                torch.save(self.ae.state_dict(), Path(self.save_dir, f"ae_{step}.pth"))
                torch.save(
                    self.potential.state_dict(),
                    Path(self.save_dir, f"potential_{step}.pth"),
                )

        self.cnt += 1
        return 1
