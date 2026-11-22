from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["Trainer", "build_model_kwargs", "run_reproduction", "run_training"]


def __getattr__(name: str) -> Any:
    if name in {"Trainer", "build_model_kwargs", "run_training"}:
        module = import_module("module.custom_tasks.nmr_shift")
        return getattr(module, name)
    if name == "run_reproduction":
        module = import_module("module.custom_tasks.tenfold")
        return getattr(module, name)
    raise AttributeError(f"module 'module.custom_tasks' has no attribute {name!r}")
