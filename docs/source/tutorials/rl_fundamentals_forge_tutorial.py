"""
RL Fundamentals Using Forge Terminology
========================================

**Author:** `Your Name <https://github.com/yourusername>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * Core RL components in Forge (Dataset, Policy, Reward Model, etc.)
       * How RL concepts map to distributed Forge services
       * Building scalable RL training loops with fault tolerance
       * Resource management and independent scaling patterns

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.0.0+
       * GPU access recommended
       * Basic understanding of reinforcement learning
       * Familiarity with async/await in Python

This tutorial teaches RL fundamentals using Forge's exact terminology and architecture.
We'll start with a simple math tutoring example to understand how traditional RL concepts
map to Forge's distributed service model.

"""

########################################################################
# Core RL Components in Forge
# ----------------------------
#
# Let's start with a simple math tutoring example to understand RL concepts
# with the exact names Forge uses. Think of it as teaching an AI student:
#
# - **Dataset**: Provides questions (like "What is 2+2?")
# - **Policy**: The AI student being trained (generates answers like "The answer is 4")
# - **Reward Model**: The teacher that evaluates answer quality (gives scores like 0.95)
# - **Reference Model**: Copy of original student (prevents drift from baseline)
# - **Replay Buffer**: Notebook that stores experiences (question + answer + score)
# - **Trainer**: The tutor that improves the student based on experiences
#
# Here's how these components interact in a conceptual RL step:

import asyncio
from typing import Any, Dict, Optional

import torch


def conceptual_rl_step():
    """
    Conceptual example showing the RL learning flow.
    See apps/grpo/main.py for actual GRPO implementation.
    """
    # 1. Get a math problem
    question = "What is 2+2?"  # dataset.sample()

    # 2. Student generates answer
    answer = "The answer is 4"  # policy.generate(question)

    # 3. Teacher grades it
    score = 0.95  # reward_model.evaluate(question, answer)

    # 4. Compare to original student
    baseline = 0.85  # reference_model.compute_logprobs(question, answer)

    # 5. Store the experience
    experience = {
        "question": question,
        "answer": answer,
        "score": score,
        "baseline": baseline,
    }
    # replay_buffer.add(experience)

    # 6. When enough experiences collected, improve student
    # trainer.train_step(batch)  # Student gets better!

    return experience


example_experience = conceptual_rl_step()
print("Example RL experience:", example_experience)

########################################################################
# From Concepts to Forge Services
# --------------------------------
#
# Here's the key insight: **Each RL component becomes a Forge service**.
# The toy example above maps directly to Forge's distributed architecture:
#
# * Dataset → DatasetActor
# * Policy → Policy
# * Reward Model → RewardActor
# * Reference Model → ReferenceModel
# * Replay Buffer → ReplayBuffer
# * Trainer → RLTrainer
#
# Let's see how the conceptual example translates to actual Forge service calls:


async def forge_rl_step(services: Dict[str, Any], step: int) -> Optional[float]:
    """
    RL step using actual Forge service APIs.
    This shows the same logic as conceptual_rl_step but with real service calls.
    """
    # 1. Get a math problem - Using actual DatasetActor API
    sample = await services["dataloader"].sample.call_one()
    prompt, target = sample["request"], sample["target"]

    # 2. Student generates answer - Using actual Policy API
    responses = await services["policy"].generate.route(prompt=prompt)
    answer = responses[0].text

    # 3. Teacher grades it - Using actual RewardActor API
    score = await services["reward_actor"].evaluate_response.route(
        prompt=prompt, response=answer, target=target
    )

    # 4. Compare to baseline - Using actual ReferenceModel API
    # Note: ReferenceModel.forward requires input_ids, max_req_tokens, return_logprobs
    input_ids = torch.cat([responses[0].prompt_ids, responses[0].token_ids])
    ref_logprobs = await services["ref_model"].forward.route(
        input_ids.unsqueeze(0), max_req_tokens=512, return_logprobs=True
    )

    # 5. Store experience - Using actual Episode structure from apps/grpo/main.py
    episode = create_episode_from_response(responses[0], score, ref_logprobs, step)
    await services["replay_buffer"].add.call_one(episode)

    # 6. Improve student - Using actual trainer pattern
    batch = await services["replay_buffer"].sample.call_one(curr_policy_version=step)
    if batch is not None:
        inputs, targets = batch  # GRPO returns (inputs, targets) tuple
        loss = await services["trainer"].train_step.call(inputs, targets)

        # 7. Policy synchronization - Using actual weight update pattern
        await services["trainer"].push_weights.call(step + 1)
        await services["policy"].update_weights.fanout(step + 1)

        return loss

    return None


def create_episode_from_response(response, score, ref_logprobs, step):
    """Helper function to create episode from response data"""
    return {
        "response": response,
        "score": score,
        "ref_logprobs": ref_logprobs,
        "step": step,
    }


########################################################################
# Setting Up Forge Services
# --------------------------
#
# Here's how to initialize the complete RL system with proper resource allocation.
# Each service can scale independently based on its computational needs:


async def setup_forge_rl_system():
    """
    Complete setup of Forge RL services with proper resource allocation.
    This example uses Qwen 3.1-1.7B model for demonstration.
    """
    # Note: In actual Forge environment, imports would be:
    # from forge.actors.policy import Policy
    # from forge.actors.replay_buffer import ReplayBuffer
    # from forge.actors.reference_model import ReferenceModel
    # from forge.actors.trainer import RLTrainer
    # from apps.grpo.main import DatasetActor, RewardActor, ComputeAdvantages
    # from forge.data.rewards import MathReward, ThinkingReward

    model = "Qwen/Qwen3-1.7B"
    group_size = 1

    # Initialize all services with appropriate resource allocation
    services = await asyncio.gather(
        # Dataset actor (CPU intensive for I/O)
        create_dataset_actor(model),
        # Policy service (GPU for inference)
        create_policy_service(model, group_size),
        # Trainer actor (GPU for training)
        create_trainer_actor(model),
        # Replay buffer (CPU for memory management)
        create_replay_buffer_actor(),
        # Advantage computation (CPU)
        create_advantages_actor(),
        # Reference model (GPU for baseline)
        create_reference_model_actor(model),
        # Reward actor (CPU/small GPU for evaluation)
        create_reward_actor(),
    )

    service_names = [
        "dataloader",
        "policy",
        "trainer",
        "replay_buffer",
        "compute_advantages",
        "ref_model",
        "reward_actor",
    ]

    return dict(zip(service_names, services))


# Service creation functions (would use actual Forge APIs)
async def create_dataset_actor(model):
    """DatasetActor for loading training data"""
    return {
        "name": "DatasetActor",
        "config": {
            "path": "openai/gsm8k",
            "revision": "main",
            "data_split": "train",
            "streaming": True,
            "model": model,
        },
        "resources": "CPU",
        "sample": lambda: {
            "call_one": lambda: {"request": "What is 2+2?", "target": "4"}
        },
    }


async def create_policy_service(model, group_size):
    """Policy service for text generation"""
    return {
        "name": "Policy",
        "config": {
            "engine_config": {
                "model": model,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "enforce_eager": False,
            },
            "sampling_config": {
                "n": group_size,
                "max_tokens": 16,
                "temperature": 1.0,
                "top_p": 1.0,
            },
        },
        "resources": "GPU",
        "generate": lambda: {"route": lambda prompt: [MockResponse()]},
    }


async def create_trainer_actor(model):
    """RLTrainer for policy optimization"""
    return {
        "name": "RLTrainer",
        "config": {
            "model": {
                "name": "qwen3",
                "flavor": "1.7B",
                "hf_assets_path": f"hf://{model}",
            },
            "optimizer": {"name": "AdamW", "lr": 1e-5},
            "training": {"local_batch_size": 2, "seq_len": 2048},
        },
        "resources": "GPU",
        "train_step": lambda: {"call": lambda inputs, targets: 0.5},
    }


async def create_replay_buffer_actor():
    """ReplayBuffer for experience storage"""
    return {
        "name": "ReplayBuffer",
        "config": {"batch_size": 2, "max_policy_age": 1, "dp_size": 1},
        "resources": "CPU",
        "add": lambda: {"call_one": lambda episode: None},
        "sample": lambda: {"call_one": lambda curr_policy_version: ([], [])},
    }


async def create_advantages_actor():
    """ComputeAdvantages for advantage estimation"""
    return {"name": "ComputeAdvantages", "resources": "CPU"}


async def create_reference_model_actor(model):
    """ReferenceModel for baseline computation"""
    return {
        "name": "ReferenceModel",
        "config": {
            "model": {
                "name": "qwen3",
                "flavor": "1.7B",
                "hf_assets_path": f"hf://{model}",
            },
            "training": {"dtype": "bfloat16"},
        },
        "resources": "GPU",
        "forward": lambda: {
            "route": lambda input_ids, max_req_tokens, return_logprobs: torch.tensor(
                [0.1, 0.2]
            )
        },
    }


async def create_reward_actor():
    """RewardActor for response evaluation"""
    return {
        "name": "RewardActor",
        "config": {"reward_functions": ["MathReward", "ThinkingReward"]},
        "resources": "CPU",
        "evaluate_response": lambda: {"route": lambda prompt, response, target: 0.95},
    }


class MockResponse:
    """Mock response object for demonstration"""

    def __init__(self):
        self.text = "The answer is 4"
        self.prompt_ids = torch.tensor([1, 2, 3])
        self.token_ids = torch.tensor([4, 5, 6])


# Demonstrate the setup
print("Setting up Forge RL system...")
# services = await setup_forge_rl_system()  # Would work in async context
print("Forge services configured with independent scaling capabilities")

########################################################################
# Why Forge Architecture Matters
# -------------------------------
#
# Traditional ML infrastructure fails for RL because each component has
# different resource needs, scaling patterns, and failure modes:


def show_infrastructure_challenges():
    """
    Demonstrate why traditional monolithic RL fails and how Forge solves it.
    """
    print("=== Infrastructure Challenges ===\n")

    print("Problem 1: Different Resource Needs")
    resource_requirements = {
        "Policy (Student AI)": {
            "generates": "'The answer is 4'",
            "needs": "Large GPU memory",
            "scaling": "Multiple replicas for speed",
        },
        "Reward Model (Teacher)": {
            "scores": "answers: 0.95",
            "needs": "Moderate compute",
            "scaling": "CPU or small GPU",
        },
        "Trainer (Tutor)": {
            "improves": "student weights",
            "needs": "Massive GPU compute",
            "scaling": "Distributed training",
        },
        "Dataset (Question Bank)": {
            "provides": "'What is 2+2?'",
            "needs": "CPU intensive I/O",
            "scaling": "High memory bandwidth",
        },
    }

    for component, reqs in resource_requirements.items():
        print(f"{component}:")
        for key, value in reqs.items():
            print(f"  {key}: {value}")
        print()

    print("Problem 2: Coordination Complexity")
    print("Unlike supervised learning with independent batches,")
    print("RL requires complex coordination between components:")
    print("- Policy waits idle while reward model works")
    print("- Training waits for single episode (batch size = 1)")
    print("- Everything stops if any component fails")
    print()

    print("=== Forge Solutions ===\n")

    print("✅ Automatic Resource Management")
    print("- Routing to least loaded replica")
    print("- GPU memory management")
    print("- Batch optimization")
    print("- Failure recovery")
    print("- Auto-scaling based on demand")
    print()

    print("✅ Independent Scaling")
    print("- Policy: num_replicas=8 for high inference demand")
    print("- RewardActor: num_replicas=16 for parallel evaluation")
    print("- Trainer: Multiple actors for distributed training")
    print()

    print("✅ Fault Tolerance")
    print("- Automatic routing to healthy replicas")
    print("- Background replica respawn")
    print("- Graceful degradation")
    print("- System continues during component failures")


show_infrastructure_challenges()

########################################################################
# Production Scaling Example
# ---------------------------
#
# Here's how you would scale the system for production workloads:


def demonstrate_production_scaling():
    """
    Show how Forge services scale independently for production.
    """
    print("=== Production Scaling Configuration ===\n")

    scaling_config = {
        "Policy Service": {
            "replicas": 8,
            "reason": "High inference demand from multiple training runs",
            "resources": "GPU-heavy instances",
        },
        "RewardActor Service": {
            "replicas": 16,
            "reason": "Parallel evaluation of many responses",
            "resources": "CPU/small GPU instances",
        },
        "Trainer Actor": {
            "replicas": 4,
            "reason": "Distributed training across multiple nodes",
            "resources": "Large GPU clusters",
        },
        "Dataset Actor": {
            "replicas": 2,
            "reason": "I/O intensive data loading",
            "resources": "High-bandwidth CPU instances",
        },
        "ReplayBuffer Actor": {
            "replicas": 1,
            "reason": "Centralized experience storage",
            "resources": "High-memory instances",
        },
    }

    for service, config in scaling_config.items():
        print(f"{service}:")
        print(f"  Replicas: {config['replicas']}")
        print(f"  Reason: {config['reason']}")
        print(f"  Resources: {config['resources']}")
        print()

    print("Key Benefits:")
    print("- Each service scales based on its bottlenecks")
    print("- Resource utilization is optimized")
    print("- Costs are minimized (no idle GPUs)")
    print("- System maintains performance under load")


demonstrate_production_scaling()

########################################################################
# Complete RL Training Loop
# --------------------------
#
# Here's a complete example showing multiple RL training steps:


async def complete_rl_training_example(num_steps: int = 5):
    """
    Complete RL training loop using Forge services.
    """
    print(f"=== Running {num_steps} RL Training Steps ===\n")

    # Setup services (mock for demonstration)
    services = {
        "dataloader": await create_dataset_actor("Qwen/Qwen3-1.7B"),
        "policy": await create_policy_service("Qwen/Qwen3-1.7B", 1),
        "trainer": await create_trainer_actor("Qwen/Qwen3-1.7B"),
        "replay_buffer": await create_replay_buffer_actor(),
        "ref_model": await create_reference_model_actor("Qwen/Qwen3-1.7B"),
        "reward_actor": await create_reward_actor(),
    }

    losses = []

    for step in range(num_steps):
        print(f"Step {step + 1}:")

        # Simulate the RL step (would use actual forge_rl_step in practice)
        sample = await services["dataloader"]["sample"]()["call_one"]()
        print(f"  Question: {sample['request']}")
        print(f"  Target: {sample['target']}")

        # Generate response
        responses = await services["policy"]["generate"]()["route"](sample["request"])
        print(f"  Generated: {responses[0].text}")

        # Get reward
        score = await services["reward_actor"]["evaluate_response"]()["route"](
            sample["request"], responses[0].text, sample["target"]
        )
        print(f"  Reward: {score}")

        # Simulate training (every few steps when buffer has enough data)
        if step >= 2:  # Start training after accumulating some experience
            loss = await services["trainer"]["train_step"]()["call"]([], [])
            losses.append(loss)
            print(f"  Training Loss: {loss:.4f}")

        print()

    print(f"Training completed! Average loss: {sum(losses)/len(losses):.4f}")
    return losses


# Run the example (would work in async context)
print("Complete RL training example:")
print("(In real usage, run: await complete_rl_training_example(5))")

########################################################################
# Conclusion
# ----------
#
# This tutorial demonstrated how RL fundamentals map to Forge's distributed
# service architecture. Key takeaways:
#
# 1. **Service Mapping**: Each RL component (Dataset, Policy, Reward, etc.)
#    becomes an independent, scalable Forge service
#
# 2. **Resource Optimization**: Services scale independently based on their
#    computational needs (GPU for inference/training, CPU for data/rewards)
#
# 3. **Fault Tolerance**: Individual service failures don't stop the entire
#    training pipeline - Forge handles routing and recovery automatically
#
# 4. **Simple Interface**: Complex distributed systems are hidden behind
#    simple async function calls
#
# The same RL logic that works conceptually scales to production workloads
# without infrastructure code - Forge handles distribution, scaling, and
# fault tolerance automatically.
#
# Further Reading
# ---------------
#
# * `Forge Architecture Documentation <#>`_
# * `GRPO Implementation (apps/grpo/main.py) <#>`_
# * `Forge Service APIs <#>`_
# * `Production RL Scaling Guide <#>`_

