#!/usr/bin/env bash

python train.py train configs/train_celeba_l2_cost.yml

python train.py train configs/train_celeba_l2_ae.yml

python train.py train configs/train_celeba_l1_cost.yml

python train.py train configs/train_celeba_l1_ae.yml

# python train.py train train_celeba_perceptual_cost.yml


# python train.py train train_artbench_l2_cost.yml

# python train.py train train_artbench_l1_cost.yml

# python train.py train train_artbench_perceptual_cost.yml
