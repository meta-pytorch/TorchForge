# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import shutil

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torch.distributed.checkpoint as dcp
import torchstore as ts

from monarch.actor import current_rank, current_size, endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

from forge.actors._torchstore_utils import (
    DcpHandle,
    get_dcp_whole_state_dict_key,
    get_param_key,
)

from forge.controller import ForgeActor
from forge.data.utils import batch_to_device

from forge.data_models.loss_metrics import LossMetrics
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def cleanup_old_weight_versions(
    state_dict_key: str,
    delim: str,
    current_policy_version: int,
) -> None:
    """Delete old weight versions, keeping only current and N-1 versions.

    TODO - issues/194: provide a more robust way to handle eviction.

    Args:
        state_dict_key: The base key for state dict storage
        delim: The delimiter used between key and version
        current_policy_version: The current policy version to keep
    """
    if current_policy_version <= 1:
        return  # No cleanup needed for versions 0 or 1

    prefix = f"{state_dict_key}{delim}"
    current_weights = f"{prefix}{current_policy_version}"
    previous_weights = f"{prefix}{current_policy_version - 1}"

    # Find all weight directories that match our pattern
    parent_dir = os.path.dirname(prefix) or "."
    if os.path.exists(parent_dir):
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if (
                item.startswith(os.path.basename(prefix))
                and item != os.path.basename(current_weights)
                and item != os.path.basename(previous_weights)
                and os.path.isdir(item_path)
            ):
                try:
                    shutil.rmtree(item_path, ignore_errors=True)
                    logger.debug(f"Removed old weights at {item_path}")
                except OSError as e:
                    logger.debug(f"Error deleting {item_path}: {e}")


@dataclass
class RLTrainer(ForgeActor):
    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"
    use_dcp: bool = True
    dcp_path: str = "forge_dcp_tmp"

    def __post_init__(self):
        """Initializes config types and env variables.

        torchrun normally hands env variables, but we need to do it ourselves
        in monarch for now.

        """
        super().__init__()

        if self.use_dcp:
            # DCP specific optimization
            torch.serialization.set_crc32_options(False)

        # Instantiate dict fields
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 1  # fragile contract.
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        self.rank = current_rank().rank
        self.size = math.prod(current_size().values())

        env = {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.size),
            "GROUP_RANK": str(self.size),
            "GROUP_WORLD_SIZE": str(self.size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        # TODO: update ForgeEngine to not use ForgeJobConfig
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "state_dict_key",
            "use_dcp",
            "dcp_path",
        }:
            engine_config.pop(key)  # Not part of job config
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.step)
        self.engine.optimizers.zero_grad()

    def forward_backward(
        self, inputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> Tensor | LossMetrics:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        # if getattr(self.engine.model_args, "use_flex_attn", False):
        #     cp_mesh = (
        #         parallel_dims.world_mesh["cp"] if parallel_dims.cp_enabled else None
        #     )
        #     init_attention_mask(
        #         inputs, self.engine.tokenizer.base_tokenizer.eos_id, cp_mesh
        #     )

        # optional_context_parallel_ctx = (
        #     dist_utils.create_context_parallel_ctx(
        #         cp_mesh=parallel_dims.world_mesh["cp"],
        #         cp_buffers=[inputs, targets] + [m.freqs_cis for m in model_parts],
        #         cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        #         cp_no_restore_buffers={inputs, targets},
        #         cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
        #     )
        #     if parallel_dims.cp_enabled
        #     else None
        # )
        optional_context_parallel_ctx = None

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
            # TODO implement PP
            # # Pipeline Parallel forward / backward inside step() call
            # with self.train_context(optional_context_parallel_ctx):
            #     targets, losses = (
            #         (labels, []) if self.pp_has_last_stage else (None, None)
            #     )
            #     if self.pp_has_first_stage:
            #         self.pp_schedule.step(
            #             inputs, target=targets, losses=losses, input_batch=inputs
            #         )
            #     else:
            #         self.pp_schedule.step(
            #             target=targets, losses=losses, input_batch=inputs
            #         )
            #
            # # accumulate losses across pipeline microbatches
            # # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            # loss = (
            #     torch.mean(torch.stack(losses)).to(self.device)
            #     if self.pp_has_last_stage
            #     else torch.tensor([-1.0], device=self.device)
            # )
        else:
            # Non-PP forward / backward
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    logits = model_parts[0](**inputs)
                    loss, loss_metrics = self.loss(logits, **targets)
                # need to free to before bwd to avoid peaking memory
                del logits
                loss.backward()

        return loss, loss_metrics

    @endpoint
    async def train_step(
        self, inputs: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
    ) -> float:

        # Log timesteps
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        self.engine.gc_handler.run(self.step)
        local_inputs = inputs[self.engine.dp_rank]
        local_targets = targets[self.engine.dp_rank]
        batch_to_device(local_inputs, self.engine.device)
        batch_to_device(local_targets, self.engine.device)
        # compute policy logprobs
        # TODO implement gradient accumulation
        # with GradientAccumulation(
        #     self.gradient_accumulation_steps,
        #     self.model,
        #     self.data_parallel_size,
        # ) as grad_acc:
        loss, loss_metrics = self.forward_backward(local_inputs, local_targets)
        torch.distributed.all_reduce(loss)
        t.step("forward_backward")

        # Get learning rate from scheduler
        current_lr = (
            self.engine.lr_schedulers.get_last_lr()[0]
            if hasattr(self.engine.lr_schedulers, "get_last_lr")
            else 0.001
        )
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()
        t.step("optimizer_step")

        # Record training metrics
        # TODO: delete item() to avoid cpu-gpu sync
        loss = loss.detach().cpu().item()
        record_metric("rl_trainer/count_training_steps", 1, Reduce.SUM)
        record_metric("rl_trainer/avg_grpo_loss", loss, Reduce.MEAN)

        if loss_metrics.kl_divergence.sum().item() != 0:
            record_metric(
                "rl_trainer/step/avg_kl_divergence",
                loss_metrics.kl_divergence,
                Reduce.MEAN,
            )
            record_metric(
                "rl_trainer/step/std_kl_divergence",
                loss_metrics.kl_divergence,
                Reduce.STD,
            )

        if loss_metrics.policy_entropy.sum().item() != 0:
            record_metric(
                "rl_trainer/step/avg_policy_entropy",
                loss_metrics.policy_entropy,
                Reduce.MEAN,
            )

        self.step += 1
        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()
        return loss

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to torchstore in HF format."""
        t = Tracer("rl_trainer_perf/push_weights", timer="gpu", track_memory=True)
        t.start()
        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        t.step("flatten_state_dict")
        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        t.step("to_hf")
        if self.use_dcp:
            key = get_dcp_whole_state_dict_key(policy_version)
            dcp_id = f"{self.dcp_path}/{key}"
            storage_writer = torch.distributed.checkpoint.FileSystemWriter(
                dcp_id, single_file_per_rank=False, thread_count=8
            )
            metadata = dcp.save(storage_writer=storage_writer, state_dict=hf_state_dict)
            dcp_handle = DcpHandle(
                checkpoint_id=dcp_id,
                metadata=metadata,
                param_names=hf_state_dict.keys(),
            )
            await ts.put(key, dcp_handle)
            t.step("dcp_save")
        else:
            for name, param in hf_state_dict.items():
                key = get_param_key(policy_version, name)
                await ts.put(key, param)
            t.step("ts_save")
        t.stop()
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()


def _shard_and_concat(sources: list[torch.Tensor], dim: int, tp: int) -> torch.Tensor:
    """Shard and concatenate tensors along a given dimension.

    Args:
        source (list[torch.Tensor]): List of tensors to shard and concatenate.
        dim (int): Dimension along which to shard and concatenate.
        tp (int): Number of tensor parallel groups.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    sharded_sources = []
    for source in sources:
        sharded_sources.append(torch.chunk(source, tp, dim=dim))

    combined_shards = []
    for shard_idx in range(tp):
        combined = torch.cat([s[shard_idx] for s in sharded_sources], dim=dim)
        combined_shards.append(combined)
    return torch.cat(combined_shards, dim=dim)


def _qwen3_hf_to_vllm(
    sd: dict[str, torch.Tensor], num_layers: int, vllm_tp: int
) -> dict[str, torch.Tensor]:
    """Convert transformers state dict to vLLM format. Specifically, this fuses
    QKV projection and MLP gate_up_proj layers.

    Args:
        sd (dict): State dict from HF model.
        num_layers (int): Number of layers in the model.

    Returns:
        dict: State dict in vLLM format.
    """
    load_sd = {}

    def unwrap(t):
        """Unwrap a DTensor to a Tensor."""
        return t.full_tensor() if isinstance(t, torch.distributed.tensor.DTensor) else t

    for key in sd.keys():
        sd[key] = unwrap(sd[key]).cpu()

    # Copy over directly mapped keys
    for k in sd:
        if any(
            x in k
            for x in [
                "down_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "o_proj",
                "norm.weight",
                "embed_tokens.weight",
                "lm_head.weight",
            ]
        ):
            load_sd[k] = sd[k]

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        # QKV fusion
        q = sd[prefix + "self_attn.q_proj.weight"]
        k = sd[prefix + "self_attn.k_proj.weight"]
        v = sd[prefix + "self_attn.v_proj.weight"]

        load_sd[prefix + "self_attn.qkv_proj.weight"] = _shard_and_concat(
            [q, k, v], dim=0, tp=vllm_tp
        )

        # Untested: QKV fusion - handle bias if present
        q_bias_key = prefix + "self_attn.q_proj.bias"
        k_bias_key = prefix + "self_attn.k_proj.bias"
        v_bias_key = prefix + "self_attn.v_proj.bias"

        if all(key in sd for key in [q_bias_key, k_bias_key, v_bias_key]):
            q_bias = sd[q_bias_key]
            k_bias = sd[k_bias_key]
            v_bias = sd[v_bias_key]
            load_sd[prefix + "self_attn.qkv_proj.bias"] = _shard_and_concat(
                [q_bias, k_bias, v_bias], dim=0, tp=vllm_tp
            )

        # MLP gate_up_proj fusion
        gate = sd[prefix + "mlp.gate_proj.weight"]
        up = sd[prefix + "mlp.up_proj.weight"]
        load_sd[prefix + "mlp.gate_up_proj.weight"] = _shard_and_concat(
            [gate, up], dim=0, tp=vllm_tp
        )

        # Untested: MLP gate_up_proj fusion - handle bias if present
        gate_bias_key = prefix + "mlp.gate_proj.bias"
        up_bias_key = prefix + "mlp.up_proj.bias"

        if all(key in sd for key in [gate_bias_key, up_bias_key]):
            gate_bias = sd[gate_bias_key]
            up_bias = sd[up_bias_key]
            # Same sharding has to happen here
            load_sd[prefix + "mlp.gate_up_proj.bias"] = _shard_and_concat(
                [gate_bias, up_bias], dim=0, tp=vllm_tp
            )

    return load_sd
