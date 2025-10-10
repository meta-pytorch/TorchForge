# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from forge.data_models.loss_metrics import LossMetrics


class SimpleGRPOLoss(nn.Module):
    """Simplified GRPO Loss for simplified single step updates
    Inspired by the Hugging Face TRL implementation:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624.
    """

    def __init__(self, beta: float = 0.1) -> tuple[torch.Tensor, LossMetrics]:
        super().__init__()
        self.beta = beta

    def forward(
        self, logprobs, ref_logprobs, advantages, padding_mask
    ) -> tuple[torch.tensor, LossMetrics]:
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

        per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()

        kl_padded = kl * padding_mask
        # We want to save consistency with Reduce.MEAN in the trainer.
        # Unfortunately, just kl * padding_mask is not working due to the different factor in the denominator
        kl_padded = kl.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1.0)

        return loss, LossMetrics(
            kl_divergence=kl_padded, policy_entropy=torch.tensor(0)
        )
