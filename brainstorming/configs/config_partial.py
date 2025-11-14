# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config using functools.partial approach (stdlib lazy instantiation).


Pros:
- Stdlib only (no dependencies)
- Lazy instantiation

Cons:
- Confusing if an object is a partial or not
- No validation
"""

from dataclasses import dataclass
from functools import partial

import torch.optim

from mock import llama3_2_1b, llama3_tokenizer, MultiHeadAttention


@dataclass
class DataArgs:
    """Plain dataclass for non-instantiated config block (PATTERN 4)."""

    batch_size: int = 4
    shuffle: bool = True


def llama3_2_1b_full():
    output_dir = "/tmp/torchtune/llama3_2_1B/full"

    return {
        "output_dir": output_dir,
        # PATTERN 1: Simple Component Instantiation
        "tokenizer": partial(
            llama3_tokenizer,
            path="/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model",
        ),
        # PATTERN 2: Component with Nested Instantiation
        "model": partial(
            llama3_2_1b,
            attn_config=partial(MultiHeadAttention, num_heads=32),
        ),
        # PATTERN 3: Component Needing Runtime Args (Partial)
        "optimizer": partial(
            torch.optim.AdamW,
            lr=2e-5,
        ),
        # PATTERN 4: Non-Instantiated Config Block (Plain Data)
        "data_args": DataArgs(
            batch_size=4,
            shuffle=True,
        ),
        # PATTERN 5: Plain Top-Level Hyperparameters
        "epochs": 1,
        "gradient_accumulation_steps": 8,
    }


if __name__ == "__main__":
    # =========================================================================
    # Scenario 1: Basic Instantiation
    # =========================================================================
    cfg = llama3_2_1b_full()

    # PATTERN 1: Simple Component Instantiation
    tokenizer = cfg["tokenizer"]()

    # PATTERN 2: Component with Nested Instantiation
    model = cfg["model"]()

    # PATTERN 3: Component Needing Runtime Args (Partial)
    optimizer = cfg["optimizer"](params=model.parameters())

    # =========================================================================
    # Scenario 2: Override Config Values
    # =========================================================================
    cfg2 = llama3_2_1b_full()

    # PATTERN 1: Simple Component Instantiation
    tokenizer2 = cfg2["tokenizer"](path="/new/path")

    # PATTERN 2: Component with Nested Instantiation
    inner_partial = cfg2["model"].keywords["attn_config"]
    inner_partial.keywords["num_heads"] = 64
    model2 = cfg2["model"]()

    # PATTERN 3: Component Needing Runtime Args (Partial)
    optimizer2 = cfg2["optimizer"](params=model2.parameters(), lr=1e-4)

    # =========================================================================
    # Scenario 3: Config Composition
    # =========================================================================
    def llama3_2_1b_large_lr():
        """Variant with larger learning rate and more attention heads."""
        base = llama3_2_1b_full()
        base["optimizer"].keywords["lr"] = 1e-3
        base["model"].keywords["attn_config"].keywords["num_heads"] = 64
        return base

    cfg_variant = llama3_2_1b_large_lr()
    model_variant = cfg_variant["model"]()
    optimizer_variant = cfg_variant["optimizer"](params=model_variant.parameters())
