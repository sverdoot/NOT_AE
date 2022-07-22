import inspect
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


ROOT_DIR = get_project_root()
CONFIGS_DIR = Path(ROOT_DIR, "configs")
DATA_DIR = Path(ROOT_DIR, "data")
CHECKPOINTS_DIR = Path(ROOT_DIR, "checkpoints")


class BaseRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Any:
        def inner_wrapper(wrapped_class: Any) -> Any:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        model = cls.registry[name]
        if inspect.isfunction(model):
            return model
        else:
            return model(**kwargs)


class REGISTRY:
    callback = BaseRegistry()
    model = BaseRegistry()
    dataset = BaseRegistry()
    cost = BaseRegistry()


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
