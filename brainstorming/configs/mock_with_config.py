# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mock implementations with inner Config classes for config_dataclasses.py.

These show the pattern where each class has an inner Config class.
In a real project, you would modify your own classes to add these Config inner classes.
"""

from dataclasses import dataclass
from typing import Any, Type

import torch
import torch.nn as nn


# =============================================================================
# Mocks with Inner Config Classes
# =============================================================================


class TokenizerWithConfig:
    """Custom tokenizer with inner Config class."""

    @dataclass
    class Config:
        path: str

        def build(self) -> "TokenizerWithConfig":
            return TokenizerWithConfig(self)

    def __init__(self, config: Config):
        self.config = config
        self.path = config.path


class MultiHeadAttentionWithConfig(nn.Module):
    """Attention module with inner Config class."""

    @dataclass
    class Config:
        num_heads: int

        def build(self) -> "MultiHeadAttentionWithConfig":
            return MultiHeadAttentionWithConfig(self)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads


class LlamaModelWithConfig(nn.Module):
    """Model with inner Config class that contains nested config."""

    @dataclass
    class Config:
        attn_config: MultiHeadAttentionWithConfig.Config

        def build(self) -> "LlamaModelWithConfig":
            return LlamaModelWithConfig(self)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attn = config.attn_config.build()

    def parameters(self):
        """Mock parameters for demo."""
        return iter([torch.zeros(10, 10)])


# =============================================================================
# Generic Config Wrappers for External Libraries
# =============================================================================


@dataclass
class ComponentConfig:
    """Generic wrapper for any component class."""

    component_cls: Type
    kwargs: dict[str, Any]

    def build(self, *args, **runtime_kwargs):
        merged_kwargs = {**self.kwargs, **runtime_kwargs}
        return self.component_cls(*args, **merged_kwargs)
