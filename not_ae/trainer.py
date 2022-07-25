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
        self.train_iter = iter(self.train_dataloader)

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

    def sample_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            return next(self.train_iter)

    def step(self, epoch_id: int, step_id: int) -> Tuple[float, ...]:
        loss_potential_num = 0
        grad_norm_potential_num = 0
        loss_ae_num = 0
        grad_norm_ae_num = 0

        # if step_id <= 100 or epoch_id > 1:
        # self.ae.train(); self.potential.eval()
        for _ in range(1, self.n_ae + 1):
            batch = self.sample_batch().to(self.device)
            rec_batch = self.ae(batch)
            score_rec = self.potential(rec_batch)
            score_real = self.potential(batch).detach()

            if False:  # step_id <= 10 and epoch_id == 1:
                loss_ae = self.cost(batch, rec_batch).mean()
            else:
                loss_ae = self.cost(batch, rec_batch).mean(0) + torch.clip(
                    score_real.detach() - score_rec, min=0.0
                ).mean(0)
            # print(self.cost(batch, rec_batch).mean(0), score_real.mean(0), score_rec.mean(0))
            loss_ae /= self.grad_acc_steps
            self.ae_opt.zero_grad()
            loss_ae.backward()
            loss_ae_num += loss_ae.item()
            grad_norm_ae_num += self.compute_grad_norm(self.ae) / self.grad_acc_steps
            # print(grad_norm_ae_num)
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 10.0)
            self.ae_opt.step()

        # self.ae.eval(); self.potential.train()
        # for _ in range(1, self.n_potential + 1):
        loss_potential = torch.zeros(1)
        # while loss_potential.item() >= 0:
        # while loss_potential_num >= 0:
        batch_p = self.sample_batch().to(self.device)
        batch_q = self.sample_batch().to(self.device)
        rec_batch = self.ae(batch_p).detach()
        score_rec = self.potential(rec_batch)
        score_real = self.potential(batch_q)

        # loss_potential = torch.clip(score_rec.mean(0) - score_real.mean(0), max=10.0)
        loss_potential = score_rec.mean(0) - score_real.mean(0)
        loss_potential /= self.grad_acc_steps
        self.potential_opt.zero_grad()
        loss_potential.backward()
        loss_potential_num += loss_potential.item()
        grad_norm_potential_num += self.compute_grad_norm(self.potential)
        # print(grad_norm_potential_num, loss_potential.item(), score_real.mean(0).item())
        torch.nn.utils.clip_grad_norm_(self.potential.parameters(), 10.0)
        self.potential_opt.step()

        return (
            loss_ae_num,
            loss_potential_num,
            grad_norm_ae_num,
            grad_norm_potential_num,
            batch_p,
            rec_batch,
        )

    def train(self, n_epoch: int) -> None:
        self.ae.train()
        self.potential.train()
        loss_ae, loss_potential = [], []
        self.ord = 0
        for epoch_id in trange(1, n_epoch + 1):
            if epoch_id < self.start_iter:
                continue

            for step_id in trange(1000):
                (
                    l_ae,
                    l_potential,
                    grad_norm_ae,
                    grad_norm_potential,
                    batch,
                    rec_batch,
                ) = self.step(epoch_id, step_id)

                loss_ae = loss_ae[-self.eval_every + 1 :] + [l_ae]
                loss_potential = loss_potential[-self.eval_every + 1 :] + [l_potential]

                info = dict(
                    batch_id=step_id,
                    total=len(self.train_dataloader),
                    loss_ae=loss_ae[-1],  # np.mean(loss_ae),
                    loss_potential=loss_potential[-1],  # np.mean(loss_potential),
                    grad_norm_ae=grad_norm_ae,
                    grad_norm_potential=grad_norm_potential,
                    origs=self.ae.inverse_transform(batch).detach().cpu().numpy(),
                    imgs=self.ae.inverse_transform(rec_batch).detach().cpu().numpy(),
                )

                for callback in self.callbacks:
                    callback.invoke(info)

            info = dict(epoch_id=epoch_id, total=n_epoch)
            if epoch_id % self.eval_every == 0:
                self.ae.eval()
                imgs = []
                for _, batch in enumerate(self.test_dataloader, 1):
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
