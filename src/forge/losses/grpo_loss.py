# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class SimpleGRPOLoss(nn.Module):
    """Simplified GRPO Loss for simplified single step updates
    Inspired by the Hugging Face TRL implementation:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624.
    
    Args:
        beta: The KL divergence coefficient for the loss.
    
    Forward args:
        logprobs: Log probabilities from the current policy being trained
        behavior_logprobs: Log probabilities from the policy that generated the responses (behavior policy)
        ref_logprobs: Log probabilities from the reference model (for KL regularization)
        advantages: Computed advantages for each token
        padding_mask: Mask indicating valid tokens (1) vs padding (0)
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logprobs, behavior_logprobs, ref_logprobs, advantages, padding_mask):
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        per_token_policy_loss = torch.exp(logprobs - behavior_logprobs.detach()) * advantages
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()
        return loss
