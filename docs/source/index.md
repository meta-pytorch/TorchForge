# TorchForge Documentation

**TorchForge** is a PyTorch-native library for RL post-training and agentic development. Built on the principle that **researchers should write algorithms, not infrastructure**.

```{note}
**Experimental Status:** TorchForge is currently in early development. Expect bugs, incomplete features, and API changes. Please file issues on [GitHub](https://github.com/meta-pytorch/forge) for bug reports and feature requests.
```

## Why TorchForge?

Reinforcement Learning has become essential to frontier AI - from instruction following and reasoning to complex research capabilities. But infrastructure complexity often dominates the actual research.

TorchForge lets you **express RL algorithms as naturally as pseudocode**, while powerful infrastructure handles distribution, fault tolerance, and optimization underneath.

### Core Design Principles

- **Algorithms, Not Infrastructure**: Write your RL logic without distributed systems code
- **Any Degree of Asynchrony**: From fully synchronous PPO to fully async off-policy training
- **Composable Components**: Mix and match proven frameworks (vLLM, TorchTitan) with custom logic
- **Built on Solid Foundations**: Leverages Monarch's single-controller model for simplified distributed programming

## Foundation: The Technology Stack

TorchForge is built on carefully selected, battle-tested components:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} **Monarch**
:link: https://meta-pytorch.org/monarch

Single-controller distributed programming framework that orchestrates clusters like you'd program a single machine. Provides actor meshes, fault tolerance, and RDMA-based data transfers.

**Why it matters:** Eliminates SPMD complexity, making distributed RL tractable
:::

:::{grid-item-card} **vLLM**
:link: https://docs.vllm.ai

High-throughput, memory-efficient inference engine with PagedAttention and continuous batching.

**Why it matters:** Handles policy generation efficiently at scale
:::

:::{grid-item-card} **TorchTitan**
:link: https://github.com/pytorch/torchtitan

Meta's production-grade LLM training platform with FSDP, pipeline parallelism, and tensor parallelism.

**Why it matters:** Battle-tested training infrastructure proven at scale
:::

:::{grid-item-card} **TorchStore**
:link: https://github.com/meta-pytorch/torchstore

Distributed, in-memory key-value store for PyTorch tensors built on Monarch, optimized for weight synchronization with automatic DTensor resharding.

**Why it matters:** Solves the weight transfer bottleneck in async RL
:::

::::

## What You Can Build

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} Supervised Fine-Tuning
Adapt foundation models to specific tasks using labeled data with efficient multi-GPU training.
:::

:::{grid-item-card} GRPO Training
Train models with Generalized Reward Policy Optimization for aligning with human preferences.
:::

:::{grid-item-card} Asynchronous RL
Continuous rollout generation with non-blocking training for maximum throughput.
:::

:::{grid-item-card} Code Execution
Safe, sandboxed code execution environments for RL on coding tasks (RLVR).
:::

:::{grid-item-card} Tool Integration
Extensible environment system for agents that interact with tools and APIs.
:::

:::{grid-item-card} Custom Workflows
Build your own components and compose them naturally with existing infrastructure.
:::

::::

## Requirements at a Glance

Before diving in, check out {doc}`getting_started` and ensure your system meets the requirements.

## Writing RL Code

With TorchForge, your RL logic looks like pseudocode:

```python
async def generate_episode(dataloader, policy, reward, replay_buffer):
    # Sample a prompt
    prompt, target = await dataloader.sample.route()

    # Generate response
    response = await policy.generate.route(prompt)

    # Score the response
    reward_value = await reward.evaluate_response.route(
        prompt=prompt,
        response=response.text,
        target=target
    )

    # Store for training
    await replay_buffer.add.route(
        Episode(prompt_ids=response.prompt_ids,
                response_ids=response.token_ids,
                reward=reward_value)
    )
```

No retry logic, no resource management, no synchronization code - just your algorithm.

## Documentation Paths

Choose your learning path:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting_started
:link-type: doc

Installation, prerequisites, verification, and your first training run.

**Time to first run: ~15 minutes**
:::

:::{grid-item-card} 💻 Tutorials
:link: tutorials
:link-type: doc

Step-by-step guides and practical examples for training with TorchForge.

**For hands-on development**
:::

:::{grid-item-card} 📖 API Reference
:link: api
:link-type: doc

Complete API documentation for customization and extension.

**For deep integration**
:::

::::

## Validation & Partnerships

TorchForge has been validated in real-world deployments:

- **Stanford Collaboration**: Integration with the Weaver weak verifier project, training models that hill-climb on challenging reasoning benchmarks (MATH, GPQA)
- **CoreWeave**: Large-scale training on 512 H100 GPU clusters with smooth, efficient performance
- **Scale**: Tested across hundreds of GPUs with continuous rollouts and asynchronous training

## Community

- **GitHub**: [meta-pytorch/forge](https://github.com/meta-pytorch/forge)
- **Issues**: [Report bugs and request features](https://github.com/meta-pytorch/forge/issues)
- **Contributing**: [CONTRIBUTING.md](https://github.com/meta-pytorch/forge/blob/main/CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](https://github.com/meta-pytorch/forge/blob/main/CODE_OF_CONDUCT.md)

```{tip}
Before starting significant work, signal your intention in the issue tracker to coordinate with maintainers.
```
* **Post-Training Focus**: Specializes in techniques
  like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO)
* **PyTorch Integration**: Built natively on PyTorch with
  dependencies on [PyTorch nightly](https://pytorch.org/get-started/locally/),
  [Monarch](https://meta-pytorch.org/monarch), [vLLM](https://docs.vllm.ai/en/latest/),
  and [TorchTitan](https://github.com/pytorch/torchtitan).
* **Multi-GPU Support**: Designed for distributed training
  with minimum 3 GPU requirement for GRPO training
* **Model Support**: Includes pre-configured setups for popular models
  like Llama3 8B and Qwen3.1 7B

```{toctree}
:maxdepth: 2
:caption: Documentation

getting_started
tutorials
api
```

## Indices

* {ref}`genindex` - Index of all documented objects
* {ref}`modindex` - Python module index

---

**License**: BSD 3-Clause | **GitHub**: [meta-pytorch/forge](https://github.com/meta-pytorch/forge)
