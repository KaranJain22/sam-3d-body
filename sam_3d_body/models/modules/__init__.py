# Copyright (c) Meta Platforms, Inc. and affiliates.

from importlib import import_module

from .misc import to_2tuple, to_3tuple, to_4tuple, to_ntuple

__all__ = [
    "aa_to_rotmat",
    "cam_crop_to_full",
    "focal_length_normalization",
    "get_focalLength_from_fieldOfView",
    "get_intrinsic_matrix",
    "inverse_perspective_projection",
    "log_depth",
    "perspective_projection",
    "rot6d_to_rotmat",
    "transform_points",
    "undo_focal_length_normalization",
    "undo_log_depth",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "to_ntuple",
]


def __getattr__(name):
    if name in {
        "aa_to_rotmat",
        "cam_crop_to_full",
        "focal_length_normalization",
        "get_focalLength_from_fieldOfView",
        "get_intrinsic_matrix",
        "inverse_perspective_projection",
        "log_depth",
        "perspective_projection",
        "rot6d_to_rotmat",
        "transform_points",
        "undo_focal_length_normalization",
        "undo_log_depth",
    }:
        module = import_module("sam_3d_body.models.modules.geometry_utils")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
