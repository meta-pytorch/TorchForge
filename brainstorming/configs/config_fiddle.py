# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config using Fiddle approach (Google's lazy instantiation).

Fiddle provides dict-style config with lazy instantiation.

Pros:
- Easy to override before building
- Clear separation: config definition vs instantiation

Cons:
- External dependency (pip install fiddle-config)
- Learning curve
- Less common outside Google
"""

import fiddle as fdl
import torch.optim

from mock import llama3_2_1b, llama3_tokenizer, MultiHeadAttention


# ======================================================================
# Config Factory Function
# ======================================================================


def llama3_2_1b_full():
    output_dir = "/tmp/torchtune/llama3_2_1B/full"
    batch_size = 4

    # Start with empty dict and build step by step
    cfg = {}
    cfg["output_dir"] = output_dir

    # PATTERN 1: Simple Component Instantiation
    cfg["tokenizer"] = fdl.Config(
        llama3_tokenizer,
        path="/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model",
    )

    # PATTERN 2: Component with Nested Instantiation
    cfg["model"] = fdl.Config(
        llama3_2_1b,
        attn_config=fdl.Config(MultiHeadAttention, num_heads=32),
    )

    # PATTERN 3: Component Needing Runtime Args (Partial)
    cfg["optimizer"] = fdl.Partial(
        torch.optim.AdamW,
        lr=2e-5,
    )

    # PATTERN 4: Non-Instantiated Config Block (Plain Data)
    cfg["data_args"] = {
        "batch_size": batch_size,
        "shuffle": True,
    }

    # PATTERN 5: Plain Top-Level Hyperparameters
    cfg["epochs"] = 1
    cfg["gradient_accumulation_steps"] = 8

    return cfg


if __name__ == "__main__":

    # =========================================================================
    # Scenario 1: Basic Instantiation
    # =========================================================================
    cfg = llama3_2_1b_full()

    # PATTERN 1: Simple Component Instantiation
    tokenizer = fdl.build(cfg["tokenizer"])

    # PATTERN 2: Component with Nested Instantiation
    model = fdl.build(cfg["model"])

    # PATTERN 3: Component Needing Runtime Args (Partial)
    optimizer_partial = fdl.build(cfg["optimizer"])
    optimizer = optimizer_partial(params=model.parameters())

    # =========================================================================
    # Scenario 2: Override Before Build
    # =========================================================================
    cfg2 = llama3_2_1b_full()

    # PATTERN 1: Simple Component Instantiation
    cfg2["tokenizer"].path = "/new/path"

    # PATTERN 2: Component with Nested Instantiation
    cfg2["model"].attn_config.num_heads = 64

    # PATTERN 3: Component Needing Runtime Args (Partial)
    cfg2["optimizer"].lr = 1e-4

    model2 = fdl.build(cfg2["model"])
    optimizer_partial2 = fdl.build(cfg2["optimizer"])
    optimizer2 = optimizer_partial2(params=model2.parameters())

    assert cfg2["model"].attn_config.num_heads == 64

    # =========================================================================
    # Scenario 3: Config Composition (Base + Variant)
    # =========================================================================
    def llama3_2_1b_large_lr():
        """Variant with larger learning rate."""
        base = llama3_2_1b_full()
        base["optimizer"].lr = 1e-3
        base["model"].attn_config.num_heads = 64
        return base

    cfg_variant = llama3_2_1b_large_lr()
    model_variant = fdl.build(cfg_variant["model"])
    optimizer_partial_variant = fdl.build(cfg_variant["optimizer"])
    optimizer_variant = optimizer_partial_variant(params=model_variant.parameters())
