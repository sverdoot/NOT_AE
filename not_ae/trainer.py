from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
import torch  # noqa: F401
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class Trainer:
    ae: nn.Module
    potential: nn.Model
    ae_opt: Optimizer
    potential_opt: Optimizer
    cost: Any
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    callbacks: Iterable = ()
    start_iter: int = 1
    eval_every: int = 1000
    n_ae: int = 1
    n_potential: int = 1
    grad_acc_steps: int = 1

    def __post_init__(self):
        self.device = next(self.ae.parameters()).device

    @staticmethod
    def compute_grad_norm(model):
        total_norm = 0
        parameters = [
            p for p in model.parameters() if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def step(self, real_batch) -> Tuple[float, ...]:
        self.ae_opt.zero_grad()
        loss_ae_num = 0
        grad_norm_ae_num = 0
        for step_id in range(1, self.n_ae * self.grad_acc_steps + 1):
            fake_batch = self.ae(real_batch)
            score_fake = self.potential(fake_batch).squeeze()

            loss_ae = (self.cost(real_batch, fake_batch) - score_fake).mean()
            loss_ae /= self.grad_acc_steps

            loss_ae.backward()
            loss_ae_num += loss_ae.item()
            grad_norm_ae_num += self.compute_grad_norm(self.ae) / self.grad_acc_steps

            if step_id % self.grad_acc_steps == 0:
                self.ae_opt.step()
                self.ae_opt.zero_grad()

        fake_batch = self.ae(real_batch)

        self.potential_opt.zero_grad()
        loss_potential_num = 0
        grad_norm_potential_num = 0
        for step_id in range(1, self.n_potential * self.grad_acc_steps + 1):
            score_fake = self.potential(fake_batch).squeeze()
            score_real = self.potential(real_batch).squeeze()

            loss_potential = (score_fake - score_real).mean()
            loss_potential /= self.grad_acc_steps
            loss_potential.backward()
            loss_potential_num += loss_potential.item()
            grad_norm_potential_num += (
                self.compute_grad_norm(self.potential) / self.grad_acc_steps
            )
            if step_id % self.grad_acc_steps == 0:
                self.potential_opt.step()
                self.potential_opt.zero_grad()

        return (
            loss_ae_num,
            loss_potential_num,
            grad_norm_ae_num,
            grad_norm_potential_num,
        )

    def train(self):
        self.ae.train()
        self.potential.train()
        loss_ae, loss_potential = [], []
        for batch_id, batch in tqdm(
            enumerate(self.train_dataloader, 1), total=len(self.train_dataloader)
        ):
            if batch_id < self.start_iter:
                continue
            batch = batch.to(self.device)
            l_ae, l_potential, grad_norm_ae, grad_norm_potential = self.step(batch)
            loss_ae = loss_ae[-self.eval_every + 1 :] + [l_ae]
            loss_potential = loss_potential[-self.eval_every + 1 :] + [l_potential]

            info = dict(
                step=batch_id,
                total=len(self.train_dataloader),
                loss_ae=np.mean(loss_ae),
                loss_potential=np.mean(loss_potential),
                grad_norm_ae=grad_norm_ae,
                grad_norm_potential=grad_norm_potential,
            )

            if batch_id % self.eval_every == 0:
                self.ae.eval()
                imgs = []
                for batch_id, batch in enumerate(self.test_dataloader, 1):
                    with torch.no_grad():
                        fake_batch = self.ae(batch.to(self.device))
                    imgs.append(
                        self.ae.inverse_transform(fake_batch).detach().cpu().numpy()
                    )
                imgs = np.concatenate(imgs, axis=0)

                info.update(
                    imgs=imgs,
                )
                self.ae.train()

            for callback in self.callbacks:
                callback.invoke(info)
