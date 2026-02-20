# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from yacs.config import CfgNode as CN


def _get_default_cfg() -> CN:
    """Return default config values for optional conditioning features."""
    cfg = CN(new_allowed=True)

    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.CONDITIONING = CN(new_allowed=True)
    cfg.MODEL.CONDITIONING.EPS = 1e-12
    cfg.MODEL.CONDITIONING.TAU = 1e-6
    cfg.MODEL.CONDITIONING.PERCENTILE_BOUNDS = [0.1, 0.9]
    cfg.MODEL.CONDITIONING.SPECTRAL_EIGENSOLVER = CN(new_allowed=True)
    cfg.MODEL.CONDITIONING.SPECTRAL_EIGENSOLVER.ENABLE = False
    cfg.MODEL.CONDITIONING.SPECTRAL_EIGENSOLVER.METHOD = "lanczos"
    cfg.MODEL.CONDITIONING.SPECTRAL_EIGENSOLVER.NUM_EIGENVALUES = 16
    cfg.MODEL.CONDITIONING.SPECTRAL_EIGENSOLVER.MAX_ITERS = 100
    cfg.MODEL.CONDITIONING.SPECTRAL_EIGENSOLVER.TOL = 1e-6

    cfg.MODEL.PERCEPTUAL_WEIGHT = CN(new_allowed=True)
    cfg.MODEL.PERCEPTUAL_WEIGHT.ENABLE = False
    cfg.MODEL.PERCEPTUAL_WEIGHT.NORMALIZE = False
    cfg.MODEL.PERCEPTUAL_WEIGHT.MIN_VALUE = 0.0

    cfg.MODEL.ADAPTIVE_ROUTER = CN(new_allowed=True)
    cfg.MODEL.ADAPTIVE_ROUTER.ENABLE = False
    cfg.MODEL.ADAPTIVE_ROUTER.LOW_KAPPA_THRESHOLD = 0.6
    cfg.MODEL.ADAPTIVE_ROUTER.HIGH_KAPPA_THRESHOLD = 1.0
    cfg.MODEL.ADAPTIVE_ROUTER.LOW_KAPPA_INFERENCE_TYPE = "body"
    cfg.MODEL.ADAPTIVE_ROUTER.MID_KAPPA_INFERENCE_TYPE = "full"
    cfg.MODEL.ADAPTIVE_ROUTER.HIGH_KAPPA_INFERENCE_TYPE = "full"

    cfg.EVAL = CN(new_allowed=True)
    cfg.EVAL.CONDITIONING_BUCKETS = []

    return cfg


# OmegaConf support for variable interpolation (e.g. ${paths.MODEL}/weights.ckpt)
# Skips interpolations it can't resolve, which lets us load Hydra .yamls without a Hydra runtime
def resolve_omegaconf_to_dict(conf):
    """
    Recursively convert an OmegaConf object to a dictionary, resolving interpolations
    where possible and leaving unsupported ones as-is.
    """
    if isinstance(conf, DictConfig):
        result = {}
        for k, v in conf.items():
            try:
                result[k] = resolve_omegaconf_to_dict(v)
            except InterpolationResolutionError:
                # Convert unresolved OmegaConf objects to containers without resolving
                result[k] = OmegaConf.to_container(v, resolve=False)
        return result
    elif isinstance(conf, ListConfig):
        result = []
        for item in conf:
            try:
                result.append(resolve_omegaconf_to_dict(item))
            except InterpolationResolutionError:
                # Convert unresolved OmegaConf objects to containers without resolving
                result.append(OmegaConf.to_container(item, resolve=False))
        return result
    else:
        # Base case: conf is a primitive value or an interpolation
        if OmegaConf.is_config(conf):
            try:
                return OmegaConf.to_container(conf, resolve=True)
            except InterpolationResolutionError:
                # Convert unresolved OmegaConf objects to containers without resolving
                return OmegaConf.to_container(conf, resolve=False)
        else:
            # conf is a primitive value
            return conf


def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}


def get_config(config_file: str) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    cfg = _get_default_cfg()

    # Resolve config with OmegaConf
    conf = OmegaConf.load(config_file)
    conf_dict = resolve_omegaconf_to_dict(conf)
    conf_cfg = CN(conf_dict)

    # Merge resolved config with the defaults.
    cfg.merge_from_other_cfg(conf_cfg)

    cfg.freeze()
    return cfg
