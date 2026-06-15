from .common import Example, to_prompts, to_labels
from .registry import (
    register_loader,
    list_datasets,
    load,
    LoaderFn,
)
from . import polyrefuse  # registers loaders on import
from . import json_files  # registers generic JSON loaders

__all__ = [
    "Example",
    "to_prompts",
    "to_labels",
    "register_loader",
    "list_datasets",
    "load",
    "LoaderFn",
]
