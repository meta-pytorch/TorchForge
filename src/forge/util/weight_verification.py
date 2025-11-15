# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for verifying model weight updates during training."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class WeightSnapshot:
    """Snapshot of model weights at a specific point in time."""

    params: dict[str, torch.Tensor]
    version: int | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_model(
        cls, model: nn.Module, version: int | None = None, device: str = "cpu"
    ) -> "WeightSnapshot":
        """Create a snapshot of model parameters.

        Args:
            model: PyTorch model to snapshot
            version: Optional version identifier
            device: Device to store snapshot tensors (default: cpu)

        Returns:
            WeightSnapshot containing detached copies of all parameters
        """
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.detach().to(device).clone()

        return cls(params=params, version=version)


@dataclass
class WeightVerificationResult:
    """Result of weight verification check."""

    weights_changed: bool
    num_params_checked: int
    num_params_changed: int
    num_params_unchanged: int
    num_params_skipped: int
    changed_params: list[str]
    unchanged_params: list[str]
    skipped_params: list[str]
    max_delta: float | None = None
    mean_delta: float | None = None

    def __str__(self) -> str:
        status = "✅ CHANGED" if self.weights_changed else "⚠️ UNCHANGED"
        max_delta = f"{self.max_delta:.6e}" if self.max_delta is not None else "N/A"
        mean_delta = f"{self.mean_delta:.6e}" if self.mean_delta is not None else "N/A"

        return (
            f"Weight Verification {status}:\n"
            f"  Checked: {self.num_params_checked}\n"
            f"  Changed: {self.num_params_changed}\n"
            f"  Unchanged: {self.num_params_unchanged}\n"
            f"  Skipped: {self.num_params_skipped}\n"
            f"  Max delta: {max_delta}\n"
            f"  Mean delta: {mean_delta}"
        )


def verify_weights_changed(
    prev_snapshot: WeightSnapshot,
    current_model: nn.Module,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    skip_non_float: bool = True,
    verbose: bool = False,
) -> WeightVerificationResult:
    """Verify that model weights have changed compared to a previous snapshot.

    This is a more robust verification than simple parameter hashing, as it:
    - Checks each parameter individually
    - Uses proper floating point comparison (torch.allclose)
    - Provides detailed information about which parameters changed
    - Computes statistics about the magnitude of changes

    Args:
        prev_snapshot: Previous weight snapshot to compare against
        current_model: Current model to check
        atol: Absolute tolerance for considering weights unchanged
        rtol: Relative tolerance for considering weights unchanged
        skip_non_float: Whether to skip non-floating point parameters
        verbose: Whether to log detailed information

    Returns:
        WeightVerificationResult with detailed information about changes
    """
    changed_params = []
    unchanged_params = []
    skipped_params = []
    deltas = []

    for name, param in current_model.named_parameters():
        if skip_non_float and not torch.is_floating_point(param):
            skipped_params.append(name)
            if verbose:
                logger.info(f"Skipping non-float param: {name}")
            continue

        if name not in prev_snapshot.params:
            logger.warning(f"Parameter {name} not found in previous snapshot")
            skipped_params.append(name)
            continue

        prev_param = prev_snapshot.params[name]
        curr_param = param.detach().cpu()

        # Check if parameters are close (i.e., unchanged)
        is_close = torch.allclose(prev_param, curr_param, atol=atol, rtol=rtol)

        if is_close:
            unchanged_params.append(name)
        else:
            changed_params.append(name)
            # Compute delta for statistics
            delta = (curr_param - prev_param).abs().max().item()
            deltas.append(delta)

            if verbose:
                logger.info(
                    f"Parameter {name} changed - max delta: {delta:.6e}, "
                    f"mean delta: {(curr_param - prev_param).abs().mean().item():.6e}"
                )

    # Compute statistics
    max_delta = max(deltas) if deltas else 0
    mean_delta = sum(deltas) / len(deltas) if deltas else 0

    result = WeightVerificationResult(
        weights_changed=len(changed_params) > 0,
        num_params_checked=len(changed_params) + len(unchanged_params),
        num_params_changed=len(changed_params),
        num_params_unchanged=len(unchanged_params),
        num_params_skipped=len(skipped_params),
        changed_params=changed_params,
        unchanged_params=unchanged_params,
        skipped_params=skipped_params,
        max_delta=max_delta,
        mean_delta=mean_delta,
    )

    logger.info(str(result))

    return result


def verify_weights_all_zeros(
    current_model: nn.Module,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    skip_non_float: bool = True,
    verbose: bool = False,
) -> tuple[bool, list[str], list[str]]:
    """Verify that all model parameters are zero.

    Args:
        current_model: Model to check
        atol: Absolute tolerance
        rtol: Relative tolerance
        skip_non_float: Whether to skip non-floating point parameters
        verbose: Whether to log detailed information

    Returns:
        Tuple of (all_zeros, zero_params, non_zero_params)
    """
    zero_params = []
    non_zero_params = []

    for name, param in current_model.named_parameters():
        if skip_non_float and not torch.is_floating_point(param):
            if verbose:
                logger.info(f"Skipping non-float param: {name}")
            continue

        param_cpu = param.detach().cpu()
        is_zero = torch.allclose(
            torch.zeros_like(param_cpu), param_cpu, atol=atol, rtol=rtol
        )

        if is_zero:
            zero_params.append(name)
        else:
            non_zero_params.append(name)
            if verbose:
                logger.info(
                    f"Parameter {name} is not zero - "
                    f"max: {param_cpu.abs().max().item():.6e}, "
                    f"mean: {param_cpu.abs().mean().item():.6e}"
                )

    all_zeros = len(non_zero_params) == 0

    logger.info(
        f"Zero check: {'✅ PASS' if all_zeros else '⚠️ FAIL'} - "
        f"{len(zero_params)} zero, {len(non_zero_params)} non-zero"
    )

    return all_zeros, zero_params, non_zero_params
