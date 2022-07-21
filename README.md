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
conda install tensorflow-gpu==2.4.1
poetry install
```
Make bash scripts runable 

```zsh
chmod +x -R scripts/*.sh
```

```
@article{korotin2022neural,
  title={Neural optimal transport},
  author={Korotin, Alexander and Selikhanovych, Daniil and Burnaev, Evgeny},
  journal={arXiv preprint arXiv:2201.12220},
  year={2022}
}
```