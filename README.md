# Neural Optimal Transport Autoencoders

## Installation

Create environment and set dependencies:
```zsh
conda create -n not_ae python=3.8
```

```zsh
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create false

conda activate not_ae
conda install tensorflow-gpu==2.4.1 # for TF FID computation
poetry install
```
Make bash scripts runable 

```zsh
chmod +x -R scripts/*.sh
```

## Prepare

```zsh
python tools/compute_fid_stats.py CelebADataset stats/celeba_fid_stats_test.npz
```

## Usage

train:

```zsh
python train.py train configs/train_{celeba / artbench}_{l1 / l2 / perceptual}_cost.yml
```

## TODO

* add logging inside epoch
* add lpips metric and callback
* add ```test.py```


```
@article{korotin2022neural,
  title={Neural optimal transport},
  author={Korotin, Alexander and Selikhanovych, Daniil and Burnaev, Evgeny},
  journal={arXiv preprint arXiv:2201.12220},
  year={2022}
}
```