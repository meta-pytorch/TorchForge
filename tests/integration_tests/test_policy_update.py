# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import shutil
from pathlib import Path

import monarch
import pytest
import pytest_asyncio

import torch
import torchstore as ts
from forge.actors.generator import Generator

from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner

from forge.controller.service.service import uuid
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import resolve_hf_hub_paths
from forge.util.weight_verification import (
    verify_weights_all_zeros,
    verify_weights_changed,
    WeightSnapshot,
)
from monarch.actor import endpoint

from omegaconf import DictConfig, OmegaConf

# Workaround for monarch mesh shutdown exit code during teardown
# Without this, proc_mesh.stop will raise exit code 1 after test completes
monarch.actor.unhandled_fault_hook = lambda failure: None


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

from huggingface_hub import snapshot_download

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Run tests:

TORCHSTORE_RDMA_ENABLED=0  \
PYTHONPATH=. pytest -s tests/integration_tests/test_policy_update.py::TestWeightSync::test_sanity_check \
    --config tests/integration_tests/fixtures/qwen3_1_7b_tp.yaml

"""

# Temp directory won't work for multi-node because NFS does not cover the tmp path
TEST_DCP_DIR = "test_dcp_tmp"


class MockTitanTrainer(TitanTrainer):
    @endpoint
    async def zero_out_model_states(self):
        """This simply sets all model weights to zero."""
        for model_part in self.engine.model_parts:
            sd = model_part.state_dict()
            for k in sd.keys():
                if not torch.is_floating_point(sd[k]):
                    logger.info(
                        f"[MockTitanTrainer] zero_out_model_states(): skipping non-float param {k}"
                    )
                    continue
                sd[k] *= 0.0


def _load_config(config_path: str) -> DictConfig:
    cfg = None
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        pytest.fail(f"Failed to load config file {config_path}: {e}")

    assert isinstance(cfg, DictConfig)

    cfg = resolve_hf_hub_paths(cfg)
    return cfg


def _test_validate_params_unchanged(
    prev_params, curr_model, logger
) -> Exception | None:
    """Validate that current parameters are the same as prev_params.

    Uses the new weight_verification utility for robust checking.
    """
    prev_snapshot = WeightSnapshot(params=prev_params, version=None)
    result = verify_weights_changed(
        prev_snapshot, curr_model, atol=1e-3, rtol=1e-2, verbose=False
    )

    logger.info(
        f"Validation: {result.num_params_checked} params checked, "
        f"{result.num_params_changed} changed, {result.num_params_unchanged} unchanged"
    )

    # We EXPECT no changes for this validation
    if result.weights_changed:
        error_msg = (
            f"Weights unexpectedly changed! {result.num_params_changed} params changed "
            f"(max_delta={result.max_delta:.6e}). Changed params: {result.changed_params[:5]}"
        )
        logger.error(error_msg)
        return AssertionError(error_msg)


def _test_validate_params_all_zeros(
    prev_params, curr_model, logger
) -> Exception | None:
    """Validate all parameters are set to zero."""
    _ = prev_params  # Unused

    all_zeros, zero_params, non_zero_params = verify_weights_all_zeros(
        curr_model, atol=1e-4, rtol=1e-3, verbose=False
    )

    logger.info(
        f"Zero validation: {len(zero_params)} zero params, {len(non_zero_params)} non-zero params"
    )

    if not all_zeros:
        error_msg = (
            f"Not all params are zero! {len(non_zero_params)} non-zero params found. "
            f"First few non-zero: {non_zero_params[:5]}"
        )
        logger.error(error_msg)
        return AssertionError(error_msg)

    return None


@pytest_asyncio.fixture(autouse=True)
async def _setup_and_teardown(request):
    # ---- setup ---- #
    config_path = request.config.getoption("--config", default=None)
    if not config_path:
        pytest.skip(
            "No config file provided. Use --config <path> to specify a YAML config file"
        )

    use_dcp_override = request.config.getoption("--use_dcp")
    cfg = _load_config(config_path=config_path)

    trainer_proc_size = cfg.actors.trainer.procs
    policy_tp_size = cfg.policy.engine_args.tensor_parallel_size

    if policy_tp_size != cfg.services.policy.procs:
        pytest.fail(
            f"Expect policy proc = {cfg.services.policy.procs} to be equal to tensor parallel size = {policy_tp_size}"
        )

    model_card = cfg.model
    logger.info(f"Running sanity check with config: {config_path}")
    logger.info(f"Model name: {model_card}")
    logger.info(f"Trainer proc size: {trainer_proc_size}")
    logger.info(f"Policy tensor parallel size: {policy_tp_size}")

    logger.info("Downloading model checkpoint from HuggingFace Hub")
    cached_dir = snapshot_download(repo_id=model_card)
    logger.info("Finished downloading model checkpoint from HuggingFace Hub")

    services_policy_cfg = cfg.services.policy
    services_policy_cfg.num_replicas = 1

    trainer_cfg = cfg.trainer
    trainer_cfg.dcp_path = TEST_DCP_DIR
    trainer_cfg.checkpoint = {
        "enable": True,
        "folder": "/tmp/saved_checkpoints",
        "initial_load_path": cached_dir,
        "initial_load_in_hf": True,
    }

    if use_dcp_override is not None:
        trainer_cfg["use_dcp"] = use_dcp_override
        logger.info(f"`trainer.use_dcp` is overridden to {use_dcp_override}")

    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    await ts.initialize(strategy=ts.ControllerStorageVolumes())

    policy, titan_trainer = await asyncio.gather(
        *[
            Generator.options(**services_policy_cfg).as_service(**cfg.policy),
            MockTitanTrainer.options(**cfg.actors.trainer).as_actor(**trainer_cfg),
        ]
    )

    yield policy, titan_trainer

    # ---- teardown ---- #
    logger.info("Shutting down services and cleaning up DCP directory..")

    # Call cleanup to destroy process group before shutdown
    # This prevents TCPStore connection errors from NCCL heartbeat threads
    await titan_trainer.cleanup.call()

    # Shutdown sequentially to avoid race conditions
    await policy.shutdown()
    await TitanTrainer.shutdown(titan_trainer)
    await ts.shutdown()

    # Cleanup DCP directory
    path = Path(TEST_DCP_DIR)
    if not path.exists() or not path.is_dir():
        return
    try:
        shutil.rmtree(path)
        logger.info(f"Successfully removed {TEST_DCP_DIR}")
    except Exception as e:
        logger.error(f"Failed to remove {TEST_DCP_DIR}: {e}")


class TestWeightSync:
    """Tests for weight sync between trainer and policy."""

    @pytest.mark.asyncio
    @requires_cuda
    async def test_sanity_check(self, _setup_and_teardown):
        """
        Sanity check for weight sync sharding between TitanTrainer and Policy for a given model config.

        The check performs the following steps:
        - Initialize trainer and push weights v0 (original huggingface ckpt)
        - Step the trainer, setting all weights to zero and push weights v1
        - Load weights v0 and check the policy has all zero weights
        - Load weights v1 and check the policy has all the weights back

        """

        policy, titan_trainer = _setup_and_teardown

        v0 = uuid.uuid4().int
        v1 = v0 + 1

        await titan_trainer.push_weights.call(policy_version=v0)
        # Setting everything to zero
        await titan_trainer.zero_out_model_states.call()
        await titan_trainer.push_weights.call(policy_version=v1)
        await policy.save_model_params.fanout()

        # Sanity check that before update all the tests pass
        all_errs = await policy.validate_model_params.fanout(
            _test_validate_params_unchanged
        )
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        await policy.update_weights.fanout(version=v1)
        all_errs = await policy.validate_model_params.fanout(
            _test_validate_params_all_zeros
        )
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        # Reloading v0, getting back original weights
        await policy.update_weights.fanout(version=v0)
        all_errs = await policy.validate_model_params.fanout(
            _test_validate_params_unchanged
        )
        for errs in all_errs:
            for _, e in errs.items():
                assert not e, f"Validation failed with exception: {e}"

        logger.info("✅ Weight sharding sanity check passed!")
