# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config using Hydra (YAML-based lazy instantiation).

Pros:
- YAML syntax (human-readable)
- Native composition of yamls
- Lazy instantiation via hydra.utils.instantiate
- Command-line override for free (--optimizer.lr=1e-4)

Cons:
- External dependency (pip install hydra-core)
- yaml is not .py
"""

import os

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def load_config():
    """Load baseline.yaml using Hydra."""
    config_dir = os.path.abspath(os.path.dirname(__file__))

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="baseline")

    return cfg


if __name__ == "__main__":
    # =========================================================================
    # Scenario 1: Basic Instantiation
    # =========================================================================
    cfg = load_config()

    # PATTERN 1: Simple Component Instantiation
    tokenizer = instantiate(cfg.tokenizer)

    # PATTERN 2: Component with Nested Instantiation
    model = instantiate(cfg.model)

    # PATTERN 3: Component Needing Runtime Args (Partial)
    optimizer_partial = instantiate(cfg.optimizer)
    optimizer = optimizer_partial(params=model.parameters())

    # =========================================================================
    # Scenario 2: Override Config Values
    # =========================================================================
    cfg2 = load_config()

    # PATTERN 1: Simple Component Instantiation
    cfg2.tokenizer.path = "/new/path"
    tokenizer2 = instantiate(cfg2.tokenizer)

    # PATTERN 2: Component with Nested Instantiation
    cfg2.model.attn_config.num_heads = 64
    model2 = instantiate(cfg2.model)

    # PATTERN 3: Component Needing Runtime Args (Partial)
    cfg2.optimizer.lr = 1e-4
    optimizer_partial2 = instantiate(cfg2.optimizer)
    optimizer2 = optimizer_partial2(params=model2.parameters())

    # =========================================================================
    # Scenario 3: Config Composition (Base + Variant)
    # =========================================================================
    # Load variant config that uses defaults to inherit from baseline
    config_dir = os.path.abspath(os.path.dirname(__file__))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_variant = compose(config_name="variant/baseline_different_bsz")

    # Verify the variant has inherited from baseline and overridden batch_size
    assert cfg_variant.data_args.batch_size == 32

    # Can instantiate components from the variant config
    model_variant = instantiate(cfg_variant.model)
