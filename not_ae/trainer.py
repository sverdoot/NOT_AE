from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


@dataclass
class Trainer:
    ae: nn.Module
    potential: nn.Module
    ae_opt: Optimizer
    potential_opt: Optimizer
    cost: Any
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    callbacks: Iterable = ()
    start_iter: int = 1
    eval_every: int = 10
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
        fake_batch = self.ae(real_batch).detach()

        self.potential_opt.zero_grad()
        loss_potential_num = 0
        grad_norm_potential_num = 0
        for step_id in range(1, self.n_potential * self.grad_acc_steps + 1):
            score_fake = self.potential(fake_batch)
            score_real = self.potential(real_batch)

            loss_potential = (score_fake - score_real).mean(0)
            loss_potential /= self.grad_acc_steps
            loss_potential.backward()
            loss_potential_num += loss_potential.item()
            # grad_norm_potential_num += (
            #     self.compute_grad_norm(self.potential) / self.grad_acc_steps
            # )
            if step_id % self.grad_acc_steps == 0:
                self.potential_opt.step()
                self.potential_opt.zero_grad()

        self.ae_opt.zero_grad()
        loss_ae_num = 0
        grad_norm_ae_num = 0
        for step_id in range(1, self.n_ae * self.grad_acc_steps + 1):
            fake_batch = self.ae(real_batch)
            score_fake = self.potential(fake_batch)

            loss_ae = (
                self.cost(real_batch, fake_batch) - score_fake + score_real.detach()
            ).mean(0)
            loss_ae /= self.grad_acc_steps
            loss_ae.backward()
            loss_ae_num += loss_ae.item()
            # grad_norm_ae_num += self.compute_grad_norm(self.ae) / self.grad_acc_steps

            if step_id % self.grad_acc_steps == 0:
                self.ae_opt.step()
                self.ae_opt.zero_grad()

        return (
            loss_ae_num,
            loss_potential_num,
            grad_norm_ae_num,
            grad_norm_potential_num,
            fake_batch,
        )

    def train(self, n_epoch: int) -> None:
        self.ae.train()
        self.potential.train()
        loss_ae, loss_potential = [], []
        for epoch_id in trange(1, n_epoch + 1):
            if epoch_id < self.start_iter:
                continue
            for batch_id, batch in tqdm(
                enumerate(self.train_dataloader, 1),
                initial=1,
                total=len(self.train_dataloader),
            ):
                batch = batch.to(self.device)
                l_ae, l_potential, grad_norm_ae, grad_norm_potential, recs = self.step(
                    batch
                )
                loss_ae = loss_ae[-self.eval_every + 1 :] + [l_ae]
                loss_potential = loss_potential[-self.eval_every + 1 :] + [l_potential]

                info = dict(
                    batch_id=batch_id,
                    total=len(self.train_dataloader),
                    loss_ae=np.mean(loss_ae),
                    loss_potential=np.mean(loss_potential),
                    grad_norm_ae=grad_norm_ae,
                    grad_norm_potential=grad_norm_potential,
                    origs=self.ae.inverse_transform(batch).detach().cpu().numpy(),
                    imgs=self.ae.inverse_transform(recs).detach().cpu().numpy(),
                )

                for callback in self.callbacks:
                    callback.invoke(info)

            info = dict(epoch_id=epoch_id, total=n_epoch)
            if epoch_id % self.eval_every == 0:
                self.ae.eval()
                imgs = []
                for batch_id, batch in enumerate(self.test_dataloader, 1):
                    with torch.no_grad():
                        fake_batch = self.ae(batch.to(self.device))
                    imgs.append(
                        self.ae.inverse_transform(fake_batch).detach().cpu().numpy()
                    )
                imgs = np.concatenate(imgs, axis=0)
                print(imgs.shape)
                info.update(
                    imgs=imgs,
                )
                self.ae.train()

            for callback in self.callbacks:
                callback.invoke(info)
