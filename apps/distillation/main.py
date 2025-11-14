# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.distillation.main --config apps/distillation/qwen3_distillation.yaml

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchstore as ts
from datasets import load_dataset
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import RLTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from omegaconf import DictConfig
from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # Processed data
    completion: Completion | None = None
    teacher_logprobs: torch.Tensor | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


# Represents the group (G) of episodes
Group = list[Episode]

# Represents the Student Model to generate data and train
StudentGenerator = Generator


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches into a single batch of inputs and targets for distillation.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        teacher_logprobs = [e.teacher_logprobs for e in batch]
        teacher_logprobs = torch.stack(teacher_logprobs).squeeze()  # [b x s]

        pad_id = batch[0].pad_id
        mask = response != pad_id

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "teacher_logprobs": teacher_logprobs,
            "padding_mask": mask,
        }
        inputs.append(input)
        targets.append(target)
    return inputs, targets


def distillation_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Online distillation loss using importance sampling with negative KL as reward.

    Formulation:
        reward = -reverse_kl = -(sampled_logprobs - teacher_logprobs)
        loss = -E[importance_weight * reward]

    where importance_weight = exp(logprobs - logprobs.detach()) and the KL term
    is detached (no backprop through it).

    This is similar to GRPO's policy gradient but without the KL penalty term,
    treating the negative KL as a reward signal.

    Args:
        logits: Student model logits [batch_size, seq_len, vocab_size]
        response: Response token ids [batch_size, seq_len]
        teacher_logprobs: Teacher log probabilities [batch_size, seq_len]
        padding_mask: Mask for valid (non-padding) tokens [batch_size, seq_len]

    Returns:
        Importance sampling loss with negative KL as reward
    """
    student_logprobs: torch.Tensor = compute_logprobs(logits, response)

    # Compute reward as negative reverse KL (detached)
    # reverse_kl = sampled_logprobs - teacher_logprobs
    # reward = -reverse_kl (higher reward when student matches teacher better)
    reverse_kl = student_logprobs.detach() - teacher_logprobs
    reward = -reverse_kl

    # Importance sampling loss (like GRPO policy gradient, but without KL penalty)
    # importance_weight = exp(current_logprobs - sampled_logprobs)
    per_token_policy_loss = (
        torch.exp(student_logprobs - student_logprobs.detach()) * reward
    )

    # Negative sign because we want to maximize reward
    per_token_loss = -per_token_policy_loss

    # Average over valid (non-padded) tokens
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()

    return loss


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "openai/gsm8k"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        self._epoch = 0

        def gsm8k_transform(sample):
            system_prompt = """
            Put all your scratchpad work between <think> and </think> tags.
            Your final answer should be between <answer> and </answer> tags otherwise it will not be scored.
            """
            request: str = sample["question"]
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            target: str = sample["answer"]
            formatted_target = target.split("#### ")[1]
            return {"request": formatted_request, "target": formatted_target}

        self._base_dataset = load_dataset(
            self.path, self.revision, split=self.data_split, streaming=self.streaming
        )
        self._base_dataset = self._base_dataset.map(gsm8k_transform)
        self._base_dataset = self._base_dataset.shuffle()
        self._iterator = iter(self._base_dataset)

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)

            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)
            record_metric(
                "dataset/sample/avg_sample_len",
                len(sample["request"]),
                Reduce.MEAN,
            )
            record_metric("dataset/sample/current_epoch", self._epoch, Reduce.MAX)

            return sample
        except StopIteration:
            # Restart iterator for next epoch with reshuffling
            self._epoch += 1
            print(
                f"Dataset epoch {self._epoch - 1} completed. Starting epoch {self._epoch}"
            )
            self._base_dataset.set_epoch(self._epoch)
            self._iterator = iter(self._base_dataset)
            return next(self._iterator)

    @endpoint
    async def pad_token(self):
        return self._tokenizer.pad_token_id


async def drop_weights(version: int):
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    # TODO: once we have something like `get_meta()` in torchstore, we can just
    # query the type of the object instead of relying on keys.
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


async def main(cfg: DictConfig):
    """Main online distillation training loop with rollout and training processes."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup services ---- #

    (
        dataloader,
        student_generator,
        trainer,
        replay_buffer,
        teacher_model,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        StudentGenerator.options(**cfg.services.student_generator).as_service(
            **cfg.student_generator
        ),
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=distillation_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ReferenceModel.options(**cfg.services.teacher_model).as_service(
            **cfg.teacher_model
        ),
    )

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()
    # Here we spawn a torchstore storage volume per trainer process.
    # We initialize after service initialization because torchstore currently
    # requires access to the underlying proc meshes in the local rank strategy.
    # We should be able to hide this in the future.
    # TODO: support multiple host meshes
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Core distillation loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.call_one()
        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()
            sample = await dataloader.sample.call_one()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            t.step("data_loading")

            prompt, target = sample["request"], sample["target"]
            responses: list[Completion] = await student_generator.generate.route(prompt)
            t.step("student_generation")

            # Construct episodes with teacher logprobs
            episodes = []
            input_ids = torch.ones(
                (group_size, max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )
            for i, response in enumerate(responses):
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=target,
                    completion=response,
                )
                episodes.append(episode)

                # Build input_ids for teacher logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

            t.step("episode_construction")

            # Get teacher logprobs on student-generated completions
            teacher_logprobs = await teacher_model.forward.route(
                input_ids, max_req_tokens, return_logprobs=True
            )
            t.step("teacher_model_calculate_logprobs")

            for i, episode in enumerate(episodes):
                episode.teacher_logprobs = teacher_logprobs[i]
                await replay_buffer.add.call_one(episode)

            del teacher_logprobs, input_ids

            rollout_count += 1
            record_metric(
                "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
            )
            t.stop()

    async def continuous_training():
        training_step = 0
        restart_tracer = True  # Flag to control when to restart tracer

        while max_steps == -1 or training_step < max_steps:
            # Restart tracer when needed (initial start or after completing a training step)
            # Otherwise, we cannot measure time waiting for buffer
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                t.step("waiting_for_buffer")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                await student_generator.update_weights.fanout(training_step)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                # Flush metrics every training step to WandB
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting online distillation with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down... (this may take a few seconds)")
        shutdown_event.set()

        try:
            # Give rollouts up to 5s to finish naturally
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for rollouts; forcing cancellation...")
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        training_task.cancel()

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
