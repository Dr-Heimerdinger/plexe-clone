import json
import pkgutil
from functools import lru_cache
from typing import List

import pooch

from ..base import Dataset

dataset_registry = {}


def register_dataset(
    name: str,
    cls: Dataset,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`Dataset` subclass with the given name.

    Args:
        name: The name of the dataset.
        cls: The class of the dataset.
        args: The arguments to instantiate the dataset.
        kwargs: The keyword arguments to instantiate the dataset.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{pooch.os_cache('relbench')}/{name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    dataset_registry[name] = (cls, args, kwargs)


def get_dataset_names() -> List[str]:
    r"""Return a list of names of the registered datasets."""
    return list(dataset_registry.keys())

@lru_cache(maxsize=None)
def get_dataset(name: str, download=True) -> Dataset:
    r"""Return a dataset object by name.

    Args:
        name: The name of the dataset.
        download: If True, download the dataset from the RelBench server.

    Returns:
        Dataset: The dataset object.

    If `download` is True, the database comprising the dataset will be
    downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the database will be processed from the
    raw files of the original source.

    Once the database is cached, either because of download or processing from
    raw files, the cache will be used. `download=True` will verify that the
    cached database matches the RelBench version even in this case.
    """


    cls, args, kwargs = dataset_registry[name]

    dataset = cls(*args, **kwargs)
    return dataset
