# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer protocol for Forge.

This module defines the unified training interface that all trainer
implementations must conform to.

"""

from typing import Protocol, runtime_checkable

import torch

from forge.api.types import (
    ForwardBackwardResult,
    LossFn,
    OptimStepResult,
    TextTrainBatch,
    TrainerInfo,
    TrainerStatus,
)


@runtime_checkable
class Trainer(Protocol):
    """Protocol defining the standard interface for all Forge trainers.

    Trainer implementations are expected to accept a default loss function at
    initialization time. This loss function is used when loss_fn is not
    provided to forward_backward(). The default loss should follow the
    LossFn signature.
    """

    async def forward_backward(
        self, batch: TextTrainBatch, loss_fn: LossFn | None = None
    ) -> ForwardBackwardResult:
        """Execute forward pass and backward pass for one batch of data.

        Basic usage - single batch per optimizer step:
            >>> batch = TextTrainBatch(
            >>>     input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            >>>     target_ids=torch.tensor([[2, 3, 4, 5, 6]]),
            >>> )
            >>> result = await trainer.forward_backward(batch)
            >>> await trainer.optim_step()  # Apply gradients

        To accumulate gradients over multiple batches before optimizer step:
            >>> await trainer.forward_backward(batch1)  # Accumulates
            >>> await trainer.forward_backward(batch2)  # Accumulates another batch
            >>> await trainer.optim_step()  # Apply all accumulated gradients

        Custom loss function for specific batches:
            >>> def custom_loss(outputs: dict[str, Any], batch: TextTrainBatch) -> torch.Tensor:
            >>>     # Custom loss computation (e.g., PPO clip, DPO, cut cross entropy, etc.)
            >>>     logits = outputs["logits"]
            >>>     # ... compute loss from logits, or use other outputs like hidden_states
            >>>     return loss
            >>>
            >>> result = await trainer.forward_backward(batch, loss_fn=custom_loss)

        Args:
            batch: TextTrainBatch containing input_ids, target_ids, and optional
                target_mask/target_weights. See forge.api.types.TextTrainBatch for details.
            loss_fn: Optional custom loss function. If None, uses the loss function
                configured at trainer creation. Signature: (outputs, batch) -> loss
                where outputs is a dict with at least "logits" key.
                Useful for mixed training objectives or experimentation.

        Returns:
            ForwardBackwardResult containing loss and metrics

        Note:
            The default loss function is configured at trainer creation time via the
            `loss` parameter. The `loss_fn` parameter here allows per-batch override.
            All loss functions should accept (outputs: dict[str, Any], batch: TextTrainBatch)
            where outputs contains at minimum a "logits" key.
        """
        ...

    async def optim_step(self) -> OptimStepResult:
        """Apply optimizer step using accumulated gradients, then clear gradients.

        This method:
        1. Applies accumulated gradients via the optimizer
        2. Steps the learning rate scheduler
        3. Clears all gradients (zero_grad)
        4. Increments the training step counter
        5. May trigger automatic checkpointing (implementation-dependent)

        Gradients must have been accumulated via forward_backward() calls before
        calling this method.

        Returns:
            OptimStepResult containing step number, learning rate, and accumulated batch count

        Example:
            >>> # Accumulate over 4 batches
            >>> for batch in batches[:4]:
            >>>     await trainer.forward_backward(batch)
            >>> result = await trainer.optim_step()
            >>> print(f"Step {result.step}, LR {result.learning_rate:.2e}")
            >>> print(f"Accumulated {result.accumulated_microbatches} batches")
        """
        ...

    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying them.

        Use this when you need to discard accumulated gradients without performing
        an optimizer step. Common scenarios:
        - Exception during gradient accumulation
        - Skipping a training step due to some condition
        - Recovering from OOM or other errors

        This is equivalent to calling optimizer.zero_grad() and resetting internal
        accumulation counters.

        Example - Error recovery:
            >>> try:
            >>>     for batch in batches:
            >>>         await trainer.forward_backward(batch)
            >>>     await trainer.optim_step()
            >>> except torch.cuda.OutOfMemoryError:
            >>>     await trainer.clear_gradients()  # Discard partial gradients
            >>>     # Retry with smaller batches

        Example - Conditional skip:
            >>> await trainer.forward_backward(batch)
            >>> if should_skip_step():
            >>>     await trainer.clear_gradients()  # Don't apply these gradients
            >>> else:
            >>>     await trainer.optim_step()
        """
        ...

    async def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass only, without backward pass (for evaluation/inference).

        This method executes the model's forward pass without computing gradients.
        Useful for:
        - Evaluation on validation/test data
        - Getting model predictions/logits
        - Debugging model outputs

        Args:
            inputs: Dictionary containing model inputs. Typically includes:
                - input_ids: torch.Tensor [batch_size, seq_len]
                Other keys depend on the model architecture.

        Returns:
            Model output logits. Shape: [batch_size, seq_len, vocab_size]

        Note:
            This runs in torch.no_grad() context - no gradients are computed.

        Example:
            >>> eval_batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
            >>> logits = await trainer.forward(eval_batch)  # [1, 4, vocab_size]
            >>> predictions = logits.argmax(dim=-1)  # [1, 4]
        """
        ...

    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        """Save trainer state or weights to persistent storage.

        By default, saves complete training state (model weights, optimizer state,
        learning rate scheduler state, and step counter). Set weights_only=True to
        save only model weights for inference/deployment.

        Args:
            name: Optional checkpoint name/identifier. If None, uses the current
                step number (e.g., "step-1000" or "weights-step-1000").
            path: Optional base directory or URI where checkpoint should be saved.
                If None, uses the default checkpoint directory configured at trainer
                creation. Supports different backends via URI schemes:
                - `/local/path` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3
            weights_only: If True, saves only model weights (lighter, for inference).
                If False (default), saves full training state including optimizer.

        Location resolution:
            - Both provided: path/name (e.g., "/checkpoints" + "best" = "/checkpoints/best")
            - Only path: use path directly
            - Only name: default_dir/name
            - Neither: default_dir/step-{current_step}

        Returns:
            Full path/URI where checkpoint was saved

        Example:
            >>> # Save full training state (default)
            >>> path = await trainer.save(name="checkpoint-1000")
            >>> print(f"Saved to {path}")  # => "Saved to /default/checkpoint-1000"
            >>>
            >>> # Save weights only for inference
            >>> path = await trainer.save(name="policy-v1", weights_only=True)
            >>>
            >>> # Save to TorchStore
            >>> path = await trainer.save(name="best", path="ts://checkpoints")
            >>> # => "ts://checkpoints/best"
        """
        ...

    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint.

        Restores training state from a checkpoint. Automatically handles both
        full checkpoints and weights-only checkpoints.

        Args:
            path: Optional path or URI to the checkpoint to load. If None, loads
                the most recent checkpoint from the default directory. Can be:
                - `/local/path/checkpoint` - local filesystem
                - `ts://key` - TorchStore
                - `s3://bucket/key` - S3

        Returns:
            Path/URI that was loaded

        Example:
            >>> # Load latest checkpoint from default location
            >>> path = await trainer.load()
            >>> print(f"Loaded from {path}")
            >>>
            >>> # Load specific checkpoint by path
            >>> path = await trainer.load("/checkpoints/step-5000")
            >>>
            >>> # Load from TorchStore
            >>> path = await trainer.load("ts://checkpoint-key")
        """
        ...

    async def get_info(self) -> TrainerInfo:
        """Get static trainer and model metadata.

        Returns information about the trainer configuration and model architecture
        that doesn't change during training.

        Returns:
            TrainerInfo containing model name, step, model_config, and parallelism settings

        Example:
            >>> info = await trainer.get_info()
            >>> print(f"Training {info.model_name} at step {info.step}")
            >>> print(f"Vocab size: {info.model_config['vocab_size']}")
            >>> print(f"DP={info.parallelism.dp_degree}, TP={info.parallelism.tp_degree}")
            >>> print(f"Device: {info.parallelism.device}")
        """
        ...

    async def get_status(self) -> TrainerStatus:
        """Get current runtime status of the trainer.

        Returns dynamic information about the trainer's current state that changes
        during training.

        Returns:
            TrainerStatus containing current step and accumulated batch count

        Example:
            >>> status = await trainer.get_status()
            >>> print(f"Current step: {status.step}")
            >>> if status.accumulated_microbatches > 0:
            >>>     print(f"Warning: {status.accumulated_microbatches} "
            >>>           f"batches accumulated without optimizer step")
        """
        ...

    async def get_tokenizer(self):
        """Get the tokenizer associated with this model.

        Returns the tokenizer used for encoding/decoding text with this model.
        Useful for preprocessing inputs or decoding model outputs.

        Returns:
            PreTrainedTokenizer: The HuggingFace tokenizer for this model

        Example:
            >>> tokenizer = await trainer.get_tokenizer()
            >>> tokens = tokenizer.encode("Hello world")
            >>> text = tokenizer.decode([1, 2, 3, 4])
        """
        ...
