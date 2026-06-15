"""
Dataset registry. Decoupled from filesystem layout — caller passes a `root`
path, loaders find files inside it.
"""

from typing import Callable, Dict, List, Optional

from .common import Example


LoaderFn = Callable[..., List[Example]]

_REGISTRY: Dict[str, LoaderFn] = {}


def register_loader(name: str, fn: LoaderFn) -> None:
    _REGISTRY[name] = fn


def list_datasets() -> List[str]:
    return sorted(_REGISTRY)


def load(name: str, **kwargs) -> List[Example]:
    if name not in _REGISTRY:
        raise KeyError(
            f"No loader '{name}'. Registered: {list_datasets()}. "
            "Use bluedot.datasets.register_loader(...) to add one."
        )
    return _REGISTRY[name](**kwargs)
