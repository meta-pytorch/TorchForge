# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataclass config with inner Config classes.

Pros:
- Type safety for instantiated configs

Cons:
- Requires modifying target classes (not feasible for external libraries - Need to use wrapper)
- Boilerplate (every class needs Config + __init__)

eg:

```
class TokenizerWithConfig:
    @dataclass
    class Config:
        path: str

        def build(self) -> "TokenizerWithConfig":
            return TokenizerWithConfig(self)

    def __init__(self, config: Config):
        self.config = config
        self.path = config.path
```
"""

from dataclasses import dataclass

import torch

from mock_with_config import (
    ComponentConfig,
    LlamaModelWithConfig,
    MultiHeadAttentionWithConfig,
    TokenizerWithConfig,
)


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
        "tokenizer": TokenizerWithConfig.Config(
            path="/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model",
        ),
        # PATTERN 2: Component with Nested Instantiation
        "model": LlamaModelWithConfig.Config(
            attn_config=MultiHeadAttentionWithConfig.Config(
                num_heads=32,
            )
        ),
        # PATTERN 3: Component Needing Runtime Args (Partial)
        "optimizer": ComponentConfig(
            component_cls=torch.optim.AdamW,
            kwargs={"lr": 2e-5},
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
    tokenizer = cfg["tokenizer"].build()

    # PATTERN 2: Component with Nested Instantiation
    model = cfg["model"].build()

    # PATTERN 3: Component Needing Runtime Args (Partial)
    optimizer = cfg["optimizer"].build(model.parameters())

    # =========================================================================
    # Scenario 2: Override Config Values
    # =========================================================================
    cfg2 = llama3_2_1b_full()

    # PATTERN 1: Simple Component Instantiation
    cfg2["tokenizer"].path = "/new/path"

    # PATTERN 2: Component with Nested Instantiation
    cfg2["model"].attn_config.num_heads = 64

    # PATTERN 3: Component Needing Runtime Args (Partial)
    cfg2["optimizer"].kwargs["lr"] = 1e-4

    model2 = cfg2["model"].build()
    optimizer2 = cfg2["optimizer"].build(model2.parameters())

    # =========================================================================
    # Scenario 3: Config Composition
    # =========================================================================
    def llama3_2_1b_large_lr():
        """Variant with larger learning rate and different model config."""
        base = llama3_2_1b_full()
        # Overrides
        base["optimizer"].kwargs["lr"] = 1e-3
        base["model"].attn_config.num_heads = 64
        return base

    cfg_variant = llama3_2_1b_large_lr()
    model_variant = cfg_variant["model"].build()
    optimizer_variant = cfg_variant["optimizer"].build(model_variant.parameters())
    assert optimizer_variant.param_groups[0]["lr"] == 1e-3
