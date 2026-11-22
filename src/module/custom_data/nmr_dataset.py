from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from torch_geometric.data import InMemoryDataset


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_train_data(path: str | Path) -> InMemoryDataset:
    data_list = load_pickle(path)
    dataset = InMemoryDataset()
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset


def load_train_test_data(
    train_path: str | Path,
    test_path_mo: str | Path,
    test_path_di: str | Path,
    test_path_tri: str | Path,
) -> tuple[InMemoryDataset, Any, Any, Any]:
    return (
        load_train_data(train_path),
        load_pickle(test_path_mo),
        load_pickle(test_path_di),
        load_pickle(test_path_tri),
    )

