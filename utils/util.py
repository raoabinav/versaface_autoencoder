# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for VersaFace Autoencoder.

This module provides configuration loading utilities used throughout the project.
"""

import os
import json5


def _load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary.
    
    Args:
        config_fn: Path to configuration file (JSON or JSON5 format)
        lowercase: Whether to convert all keys to lowercase
        
    Returns:
        Dictionary containing configuration values
    """
    with open(config_fn, "r") as f:
        data = f.read()
    config_ = json5.loads(data)
    
    # Handle base_config inheritance
    if "base_config" in config_:
        try:
            p_config_path = os.path.join(os.getenv("WORK_DIR", ""), config_["base_config"])
        except:
            p_config_path = config_["base_config"]
        p_config_ = _load_config(p_config_path)
        config_ = _override_config(p_config_, config_)
    
    if lowercase:
        config_ = _get_lowercase_keys_config(config_)
    
    return config_


def _override_config(base_config, new_config):
    """Recursively update base_config with values from new_config."""
    for k, v in new_config.items():
        if isinstance(v, dict):
            if k not in base_config:
                base_config[k] = {}
            base_config[k] = _override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def _get_lowercase_keys_config(cfg):
    """Convert all keys in config to lowercase."""
    updated_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            v = _get_lowercase_keys_config(v)
        updated_cfg[k.lower()] = v
    return updated_cfg


def load_config(config_fn, lowercase=False):
    """Load configuration file into a JsonHParams object.
    
    This is the primary function for loading model configurations.
    
    Args:
        config_fn: Path to configuration file (JSON or JSON5 format)
        lowercase: Whether to convert all keys to lowercase
        
    Returns:
        JsonHParams object containing configuration values
        
    Example:
        >>> cfg = load_config("models/tts/metis/config/base.json")
        >>> print(cfg.model.semantic_codec.codebook_size)
        8192
    """
    config_ = _load_config(config_fn, lowercase=lowercase)
    cfg = JsonHParams(**config_)
    return cfg


class JsonHParams:
    """Hyperparameter container with attribute-style access.
    
    Allows accessing config values as attributes:
        cfg.model.hidden_size instead of cfg['model']['hidden_size']
    
    Nested dictionaries are automatically converted to JsonHParams.
    """
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
