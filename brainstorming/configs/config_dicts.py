# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config using Plain Python Dicts.

Pros:
- Extremely simple
- No dependencies
- Easy to understand
- Flexible

Cons:
- No type hints (cfg["batch_szie"] typo won't be caught)
- No validation (cfg["batch_size"] = "invalid" won't error)
- Very loose, users can pass anything
"""

import torch.optim

from mock import llama3_2_1b, llama3_tokenizer, MultiHeadAttention


def llama3_2_1b_full():
    output_dir = "/tmp/torchtune/llama3_2_1B/full"
    batch_size = 4

    return {
        "output_dir": output_dir,
        # PATTERN 1: Simple Component Instantiation
        "tokenizer": {
            "cls": llama3_tokenizer,
            "kwargs": {
                "path": "/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model",
            },
        },
        # PATTERN 2: Component with Nested Instantiation
        "model": {
            "cls": llama3_2_1b,
            "kwargs": {
                "attn_config": {
                    "cls": MultiHeadAttention,
                    "kwargs": {
                        "num_heads": 32,
                    },
                }
            },
        },
        # PATTERN 3: Component Needing Runtime Args (Partial)
        "optimizer": {
            "cls": torch.optim.AdamW,
            "kwargs": {
                "lr": 2e-5,
            },
        },
        # PATTERN 4: Non-Instantiated Config Block (Plain Data)
        "data_args": {
            "batch_size": batch_size,
            "shuffle": True,
        },
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
    tokenizer = cfg["tokenizer"]["cls"](**cfg["tokenizer"]["kwargs"])

    # PATTERN 2: Component with Nested Instantiation
    attn_config = cfg["model"]["kwargs"]["attn_config"]["cls"](
        **cfg["model"]["kwargs"]["attn_config"]["kwargs"]
    )
    model = cfg["model"]["cls"](attn_config=attn_config)

    # PATTERN 3: Component Needing Runtime Args (Partial)
    optimizer = cfg["optimizer"]["cls"](
        model.parameters(), **cfg["optimizer"]["kwargs"]
    )

    # =========================================================================
    # Scenario 2: Override Config Values
    # =========================================================================
    cfg2 = llama3_2_1b_full()

    # PATTERN 1: Simple Component Instantiation
    cfg2["tokenizer"]["kwargs"]["path"] = "/new/tokenizer"

    # PATTERN 2: Component with Nested Instantiation
    cfg2["model"]["kwargs"]["attn_config"]["kwargs"]["num_heads"] = 64

    # PATTERN 3: Component Needing Runtime Args (Partial)
    cfg2["optimizer"]["kwargs"]["lr"] = 1e-4

    model2 = cfg2["model"]["cls"](
        attn_config=cfg2["model"]["kwargs"]["attn_config"]["cls"](
            **cfg2["model"]["kwargs"]["attn_config"]["kwargs"]
        )
    )
    optimizer2 = cfg2["optimizer"]["cls"](
        model2.parameters(), **cfg2["optimizer"]["kwargs"]
    )

    # =========================================================================
    # Scenario 3: Config Composition
    # =========================================================================
    def llama3_2_1b_large_lr():
        """Variant with larger learning rate."""
        base = llama3_2_1b_full()
        base["optimizer"]["kwargs"]["lr"] = 1e-3
        base["model"]["kwargs"]["attn_config"]["kwargs"]["num_heads"] = 64
        return base

    cfg_variant = llama3_2_1b_large_lr()
    attn_config_variant = cfg_variant["model"]["kwargs"]["attn_config"]["cls"](
        **cfg_variant["model"]["kwargs"]["attn_config"]["kwargs"]
    )
    model_variant = cfg_variant["model"]["cls"](attn_config=attn_config_variant)
    optimizer_variant = cfg_variant["optimizer"]["cls"](
        model_variant.parameters(), **cfg_variant["optimizer"]["kwargs"]
    )
