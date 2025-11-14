# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

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
from forge.actors.trainer import TitanTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data.rewards import MathReward, ThinkingReward
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
    behavior_logprobs: torch.Tensor | None = None  # Logprobs from the policy that generated the response
    ref_logprobs: torch.Tensor | None = None  # Logprobs from the reference model
    reward: float | None = None
    advantage: float | None = None

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
    
    @property
    def behavior_logprobs_tensor(self) -> torch.Tensor:
        """Get behavior logprobs padded to response_len."""
        tensor: torch.Tensor = self.behavior_logprobs
        if tensor.shape[0] < self.response_len:  # right pad with zeros (will be masked)
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=0.0)
        return tensor


# Represents the group (G) of episodes in GRPO
Group = list[Episode]

# Represents the Policy Model to collect data from
Policy = Generator


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collates a list of batches into a single batch of inputs and targets.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        behavior_logprobs = [e.behavior_logprobs_tensor for e in batch]
        behavior_logprobs = torch.stack(behavior_logprobs).squeeze()  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs).squeeze()  # [b x s]

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]

        pad_id = batch[0].pad_id
        mask = response != pad_id

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "behavior_logprobs": behavior_logprobs,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "padding_mask": mask,
        }
        inputs.append(input)
        targets.append(target)
    return inputs, targets


# Note: This is also available in losses.grpo_loss via `SimpleGRPOLoss`
_grpo_logger = logging.getLogger(__name__)

def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    # Sanity check: behavior_logprobs should not be None
    assert behavior_logprobs is not None, "behavior_logprobs is None!"
    
    logprobs: torch.Tensor = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - behavior_logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
    return loss


@dataclass
class RewardActor(ForgeActor):

    reward_functions: list[Callable]

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        total_rewards = 0.0
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, response, target)
            total_rewards += reward

            # Get a name for the reward function (works for classes, functions, lambdas)
            reward_fn_name = getattr(
                reward_fn, "__name__", reward_fn.__class__.__name__
            )
            # per function reward
            record_metric(
                f"reward/evaluate_response/sum_{reward_fn_name}_reward",
                reward,
                Reduce.SUM,
            )
            record_metric(
                f"reward/evaluate_response/avg_{reward_fn_name}_reward",
                reward,
                Reduce.MEAN,
            )
            record_metric(
                f"reward/evaluate_response/std_{reward_fn_name}_reward",
                reward,
                Reduce.STD,
            )

            record_metric(
                "reward/evaluate_response/avg_total_reward",
                reward,
                Reduce.MEAN,
            )

            record_metric(
                f"reward/evaluate_response/count_{reward_fn_name}_calls",
                1,
                Reduce.SUM,
            )

        avg_reward = total_rewards / len(self.reward_functions)
        return avg_reward


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def compute(self, group: Group) -> list[float]:
        # TODO: add batch processing
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "openai/gsm8k"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    async def setup(self):
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
        # Use pad_token_id if available, otherwise use eos_token_id
        # Llama models don't have a pad token by default
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        else:
            return self._tokenizer.eos_token_id


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
    """Main GRPO training loop with rollout and training processes."""
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
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
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

    # ---- Core RL loops ---- #
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
            responses: list[Completion] = await policy.generate.route(prompt)
            t.step("policy_generation")

            # Construct episodes and calculate rewards
            episodes = []
            input_ids = torch.ones(
                (group_size, max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )
            for i, response in enumerate(responses):
                # Extract behavior policy logprobs from the completion
                behavior_logprobs = response.logprobs
                if behavior_logprobs is None:
                    raise ValueError(
                        "Behavior logprobs are not available in the completion. "
                        "Please enable logprobs in the sampling_params configuration."
                    )
                
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=target,
                    completion=response,
                    behavior_logprobs=behavior_logprobs,
                )
                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=prompt, response=response.text, target=target
                )
                episodes.append(episode)

                # Build input_ids for reference logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

            t.step("reward_evaluation")

            ref_logprobs = await ref_model.forward.route(
                input_ids, max_req_tokens, return_logprobs=True
            )
            t.step("reference_model_calculate_logprobs")

            for i, episode in enumerate(episodes):
                episode.ref_logprobs = ref_logprobs[i]
            del ref_logprobs, input_ids

            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

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

                await policy.update_weights.fanout(training_step)
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
        f"Starting GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
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
