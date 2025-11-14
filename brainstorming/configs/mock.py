# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mock implementations for demonstration purposes.

In a real project, these would be actual imports from:
- torchtune.models.llama3 import llama3_tokenizer
- torchtune.models.llama3_2 import llama3_2_1b
- torchtune.modules.attention import MultiHeadAttention
- torchtune.training import FullModelHFCheckpointer
- etc.
"""

import torch
import torch.nn as nn
import torch.optim


# =============================================================================
# Mock Functions and Classes (Standard)
# =============================================================================


llama3_tokenizer = lambda path, **kwargs: type("Tokenizer", (), {})()


def llama3_2_1b(attn_config=None):
    """Mock Llama model."""

    class LlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            if isinstance(attn_config, nn.Module):
                self.attn = attn_config
            else:
                # partial
                self.attn = attn_config()
            self.dummy = nn.Parameter(torch.zeros(10, 10))

        def parameters(self, recurse=True):
            return super().parameters(recurse)

    return LlamaModel()


alpaca_cleaned_dataset = lambda tokenizer, packed=False, split="train": type(
    "Dataset", (), {}
)()


class MultiHeadAttention(nn.Module):
    """Mock MultiHeadAttention module."""

    def __init__(self, num_heads=32, **kwargs):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x):
        return x


class FullModelHFCheckpointer:
    """Mock checkpoint loader."""

    __slots__ = ("checkpoint_dir",)

    def __init__(self, checkpoint_dir, **kwargs):
        self.checkpoint_dir = checkpoint_dir


LinearCrossEntropyLoss = type("LinearCrossEntropyLoss", (), {})

DiskLogger = type("DiskLogger", (), {})

setup_torch_profiler = lambda enabled, output_dir: None
