# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.sft.main --config apps/sft/llama3_8b.yaml
"""

import asyncio

import logging
import math
import os
import sys
from functools import partial
from typing import Any

import torch

import torchtitan.experiments.forge.train_spec as forge_train_spec
from forge.controller import ForgeActor
from forge.data.collate import collate_packed
from forge.data.datasets.packed import PackedDataset, TextPacker
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.util.config import parse

from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

# stubs for now
Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ForgeSFTRecipe(ForgeActor, ForgeEngine):
    job_config: ForgeJobConfig
    train_spec: forge_train_spec.ForgeTrainSpec
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: LossFunction
    optimizer: OptimizersContainer
    lr_scheduler: LRSchedulersContainer
    checkpointer: Checkpointer
    tokenizer: Tokenizer
    train_dataloader: Dataloader
    val_dataloader: Dataloader
    metric_logger: MetricLogger
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, config: DictConfig):
        job_config = ForgeJobConfig().to_dict()
        # Hack to deal with literal types from titan
        job_config = OmegaConf.merge(job_config, config)

        self.current_step = 0
        self.num_training_steps = job_config.training.steps
        self.metric_logger = None  # TODO: fix this
        self.gradient_accumulation_steps = 1  # Example value, adjust as needed
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())

        # Evaluation settings from validation config
        validation_config = job_config.get("validation", {})
        self.validation_enabled = validation_config.get("enabled", False)

        if self.validation_enabled:
            self.eval_interval = validation_config.get("eval_interval")
            self.eval_steps = validation_config.get("eval_steps")

            if self.eval_interval is None:
                raise ValueError(
                    "validation.eval_interval is required when validation.enabled is true"
                )
            if self.eval_steps is None:
                raise ValueError(
                    "validation.eval_steps is required when validation.enabled is true"
                )
        else:
            self.eval_interval = None
            self.eval_steps = None

        self._init_dist()
        super().__init__(job_config)

    def _init_dist(self):
        """Initializes torch distributed.

        torchrun normally hands this, but we need to do it ourselves
        in monarch for now.

        We should consider putting this into ForgeActor, but having this
        be explicit for now.

        """
        env = {
            "RANK": str(self._rank),
            "LOCAL_RANK": str(self._rank),
            "LOCAL_WORLD_SIZE": str(self._size),
            "GROUP_RANK": str(self._size),
            "GROUP_WORLD_SIZE": str(self._size),
            "ROLE_RANK": str(self._rank),
            "ROLE_WORLD_SIZE": str(self._size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self._size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)
        logger.info("env: {}".format(env))

    @endpoint
    async def setup(self):
        # Setup training data from config
        dataset_config = self.job_config.get("dataset")

        self.train_dataloader = self.setup_data(
            dataset_path=dataset_config.get("path"),
            dataset_split=dataset_config.get("split"),
        )

        # Setup validation data from config
        dataset_val_config = self.job_config.get("dataset_val", {})
        self.val_dataloader = self.setup_data(
            dataset_path=dataset_val_config.get("path", dataset_config.get("path")),
            dataset_split=dataset_val_config.get("split", dataset_config.get("split")),
        )

        # Load checkpoint if resuming
        self.checkpointer.load(step=self.current_step)

    def setup_data(self, dataset_path: str, dataset_split: str):
        """Setup data with configurable dataset path and split."""
        if not dataset_path or not dataset_split:
            raise ValueError(
                f"dataset.path and dataset.split are required in YAML config. Got path={dataset_path}, split={dataset_split}"
            )
        print(os.path.join(self.job_config.model.hf_assets_path, "tokenizer.json"))
        tokenizer = HuggingFaceModelTokenizer(
            tokenizer_json_path=os.path.join(
                self.job_config.model.hf_assets_path, "tokenizer.json"
            ),
            tokenizer_config_json_path=os.path.join(
                self.job_config.model.hf_assets_path, "tokenizer_config.json"
            ),
            generation_config_path=os.path.join(
                self.job_config.model.hf_assets_path, "generation_config.json"
            ),
        )

        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            path=dataset_path,
            split=dataset_split,
        )
        packer = TextPacker(padding_idx=0)
        dataset = PackedDataset(
            dataset=dataset,
            packer=packer,
            target_tokens_per_pack=self.job_config.training.seq_len,  # TODO: get this from model
        )
        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=self.job_config.training.local_batch_size,
            collate_fn=partial(
                collate_packed, mask_fn=packer.create_block_mask, device=self.device
            ),
        )

        return dataloader

    def forward_backward(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["tokens"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    def forward_only(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass only for evaluation (no backward)."""
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs = input_dict["tokens"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward only
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward only
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                del pred

        return loss

    def train_step(self, batch) -> None:
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)

        logger.info(f"{self.current_step} / {self.num_training_steps}|Loss: {loss}")
        self.optimizers.step()
        self.lr_schedulers.step()

    def _extract_epoch_from_batch(self, batch: dict) -> int | None:
        """Extract epoch number from batch metrics."""
        if "metrics" in batch:
            for metric in batch["metrics"]:
                if (
                    hasattr(metric, "metric_name")
                    and metric.metric_name == "num_epochs"
                ):
                    return metric.value
        return None

    async def evaluate(self) -> dict[str, float]:
        """Run evaluation with async all_reduce for cross-rank epoch synchronization."""
        logger.info("=" * 50)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 50)

        for model_part in self.model_parts:
            model_part.eval()

        val_dataloader = iter(self.val_dataloader)
        total_loss, num_batches, starting_epoch = 0.0, 0, None

        # Prefetch first batch
        try:
            next_batch = next(val_dataloader)
        except StopIteration:
            logger.warning("Validation dataloader is empty")
            return {"val_loss": 0.0, "val_batches": 0}

        should_break, pending_work, epoch_tensor = False, None, None

        with torch.no_grad():
            while True:
                # Wait for previous async all_reduce to complete
                if pending_work is not None:
                    pending_work.wait()
                    should_break = (
                        epoch_tensor.item() > 0 if epoch_tensor is not None else False
                    )
                    pending_work = None

                if should_break:
                    logger.info(
                        "Epoch completed across all ranks - stopping evaluation"
                    )
                    break

                if self.eval_steps > 0 and num_batches >= self.eval_steps:
                    logger.info(f"Reached eval_steps cap of {self.eval_steps}")
                    break

                batch = next_batch

                # Track starting epoch
                current_epoch = self._extract_epoch_from_batch(batch)
                if current_epoch is not None and starting_epoch is None:
                    starting_epoch = current_epoch

                # Prefetch next batch and start async epoch check
                try:
                    next_batch = next(val_dataloader)
                    next_epoch = self._extract_epoch_from_batch(next_batch)

                    if next_epoch is not None and starting_epoch is not None:
                        epoch_increment = next_epoch - starting_epoch
                        if torch.distributed.is_initialized():
                            epoch_tensor = torch.tensor(
                                [epoch_increment], dtype=torch.long, device=self.device
                            )
                            pending_work = torch.distributed.all_reduce(
                                epoch_tensor,
                                op=torch.distributed.ReduceOp.MAX,
                                async_op=True,
                            )
                        else:
                            should_break = epoch_increment > 0
                except StopIteration:
                    should_break = True

                # Process current batch (overlaps with async all_reduce)
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                labels = batch.pop("labels")
                loss = self.forward_only(batch, labels)
                total_loss += loss.item()
                num_batches += 1

                if num_batches % 10 == 0:
                    logger.info(f"  Eval batch {num_batches} | Loss: {loss.item():.4f}")

        for model_part in self.model_parts:
            model_part.train()

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(
            f"EVALUATION COMPLETE | Val Loss: {avg_loss:.4f} | Batches: {num_batches}"
        )
        logger.info("=" * 50)

        return {"val_loss": avg_loss, "val_batches": num_batches}

    @endpoint
    async def train(self) -> None:
        dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()
        # TODO: tqdm is broken in Monarch actors
        # self.pbar = tqdm(initial=self.current_step, total=self.num_training_steps)

        while self.current_step < self.num_training_steps:
            batch = next(dataloader)
            # Move tensors to the appropriate device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)  # TODO: hardcoded for now
            self.train_step(batch)
            self.current_step += 1

            # Run evaluation periodically if enabled
            if self.validation_enabled and self.current_step % self.eval_interval == 0:
                eval_metrics = await self.evaluate()
                logger.info(f"Step {self.current_step} | Eval metrics: {eval_metrics}")

            # Save checkpoints
            self.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metric_logger:
            self.metric_logger.close()

    def __repr__(self) -> str:
        return "Trainer"


async def run(cfg: DictConfig) -> None:
    logging.info("Spawing recipe...")
    process_cfg = cfg.pop("processes")
    recipe = await ForgeSFTRecipe.options(**process_cfg).as_actor(cfg)

    logging.info("Created recipe, running setup.")
    await recipe.setup.call()

    logging.info("Recipe has been setup. Training now.")
    await recipe.train.call()

    logging.info("Done training. Clean up")
    await recipe.cleanup.call()
    await recipe.mesh.stop()
    logging.info("All done!")


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())
