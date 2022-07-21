from pathlib import Path
from typing import Any, Optional


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
        model = model(**kwargs)
        return model


class REGISTRY:
    callback = BaseRegistry()
    model = BaseRegistry()
    cost = BaseRegistry()
