# Copyright (c) Meta Platforms, Inc. and affiliates.

from importlib import import_module

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
]


def __getattr__(name):
    if name == "SAM3DBodyEstimator":
        return import_module("sam_3d_body.sam_3d_body_estimator").SAM3DBodyEstimator
    if name in {"load_sam_3d_body", "load_sam_3d_body_hf"}:
        module = import_module("sam_3d_body.build_models")
        return getattr(module, name)
    raise AttributeError(f"module 'sam_3d_body' has no attribute {name!r}")
