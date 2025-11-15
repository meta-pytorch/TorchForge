# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Type definitions for the Forge API."""

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import torch


# Loss function signature: takes model outputs (as dict) and batch, returns scalar loss
# The dict will typically contain logits, but may include other keys depending on use case.
LossFn: TypeAlias = Callable[[dict[str, Any], "TextTrainBatch"], torch.Tensor]


@dataclass
class TextTrainBatch:
    """A batch of text training data for forward_backward.

    This dataclass defines the standard format for text training batches across all
    Forge text trainers.

    Attributes:
        input_ids: Input token IDs. Shape: [batch_size, seq_len]
        target_ids: Target token IDs for loss computation. Shape: [batch_size, seq_len]
        target_mask: Mask indicating which tokens to compute loss on.
            Shape: [batch_size, seq_len]. Values are 0 (ignore) or 1 (compute loss).
            If None, computes loss on all tokens.
        target_weights: Per-token weights for loss computation.
            Shape: [batch_size, seq_len]. Used for importance weighting, such as
            advantages in RL (GRPO, PPO) or custom loss weighting schemes.
            If None, all tokens have weight 1.0.

    Example:
        >>> batch = TextTrainBatch(
        >>>     input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        >>>     target_ids=torch.tensor([[2, 3, 4, 5, 6]]),
        >>>     target_mask=torch.tensor([[0, 0, 1, 1, 1]]),  # Only predict last 3 tokens
        >>>     target_weights=torch.tensor([[0, 0, 1.0, 0.8, 1.2]]),  # Weight by advantage
        >>> )
        >>> result = await trainer.forward_backward(batch)
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor | None = None
    target_weights: torch.Tensor | None = None


@dataclass
class ForwardBackwardResult:
    """Result from a forward_backward pass.

    Attributes:
        loss: Loss value computed for the batch
        metrics: Additional metrics computed during training (e.g., perplexity,
            accuracy, KL divergence). May be empty if no additional metrics are tracked.
            Values can be scalars, tensors, or other structured data depending on the loss.

    Example:
        >>> result = await trainer.forward_backward(batch)
        >>> print(f"Loss: {result.loss:.4f}")
        >>> if result.metrics:
        >>>     print(f"Metrics: {result.metrics}")
    """

    loss: float
    metrics: dict[str, Any]


@dataclass
class OptimStepResult:
    """Result from an optimizer step.

    Attributes:
        step: Training step number after this optimizer step
        learning_rate: Current learning rate used for this step
        accumulated_microbatches: Number of forward_backward calls that were
            accumulated before this optimizer step. Useful for tracking gradient
            accumulation behavior.

    Example:
        >>> result = await trainer.optim_step()
        >>> print(f"Step {result.step}, LR={result.learning_rate:.2e}")
        >>> print(f"Accumulated {result.accumulated_microbatches} batches")
    """

    step: int
    learning_rate: float
    accumulated_microbatches: int


@dataclass
class ParallelismInfo:
    """Parallelism configuration for distributed training.

    Attributes:
        dp_degree: Data parallel degree (number of data parallel replicas)
        tp_degree: Tensor parallel degree (model sharding across devices)
        pp_degree: Pipeline parallel degree (model sharding across pipeline stages)
        world_size: Total number of processes in the distributed training job
        dp_rank: Current data parallel rank (0 to dp_degree-1)
        tp_rank: Current tensor parallel rank (0 to tp_degree-1)
        device: Device identifier (e.g., "cuda:0", "cuda:1")

    Example:
        >>> info = await trainer.get_info()
        >>> p = info.parallelism
        >>> print(f"DP={p.dp_degree}, TP={p.tp_degree}, PP={p.pp_degree}")
        >>> print(f"Running on {p.device} (DP rank {p.dp_rank})")
    """

    dp_degree: int
    tp_degree: int
    pp_degree: int
    world_size: int
    dp_rank: int
    tp_rank: int
    device: str


@dataclass
class TrainerInfo:
    """Static trainer and model metadata.

    This contains information about the trainer configuration and model architecture
    that doesn't change during training.

    Attributes:
        model_name: Name or path of the model being trained
        step: Current training step
        model_config: Model architecture configuration. Common keys include:
            - vocab_size: int - Size of the vocabulary
            - hidden_size: int - Hidden dimension size
            - num_layers: int - Number of transformer layers
            - num_attention_heads: int - Number of attention heads
            - max_seq_len: int - Maximum sequence length
        parallelism: Parallelism configuration for distributed training

    Example:
        >>> info = await trainer.get_info()
        >>> print(f"Training {info.model_name} at step {info.step}")
        >>> print(f"Vocab size: {info.model_config['vocab_size']}")
        >>> print(f"DP={info.parallelism.dp_degree}, TP={info.parallelism.tp_degree}")
        >>> print(f"Device: {info.parallelism.device}")
    """

    model_name: str
    step: int
    model_config: dict[str, Any]
    parallelism: ParallelismInfo


@dataclass
class TrainerStatus:
    """Runtime status of the trainer.

    This contains dynamic information about the trainer's current state that
    changes during training.

    Attributes:
        step: Current training step
        accumulated_microbatches: Number of batches accumulated since the last
            optim_step. Will be 0 if gradients were just applied/cleared.

    Example:
        >>> status = await trainer.get_status()
        >>> print(f"Current step: {status.step}")
        >>> if status.accumulated_microbatches > 0:
        >>>     print(f"Warning: {status.accumulated_microbatches} batches "
        >>>           f"accumulated without optimizer step")
    """

    step: int
    accumulated_microbatches: int
