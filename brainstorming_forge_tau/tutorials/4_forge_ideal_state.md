
** WORK IN PROGRESS -- NEEDS CHANGES / CLEANUP / DETAILS **

# Part 4.0: What a Multi-Turn Tool Calling with Forge + vLLM + OpenEnv would look like

For tool calling, we extend Forge's GRPO pattern to handle **multi-turn interactions** where:
- One task → multiple LLM generations + tool executions → one Episode
- Episode contains **concatenated tokens** from all turns
- Training and replay buffer logic remains unchanged

**Key Principle:** Multi-turn only changes the **rollout phase**. Training stays the same.

---

## Setup: Services + Multi-Environment Support

Notice that an Env in OpenEnv is a **tool execution environment**. It doesn't know about tasks. It only knows about tools.
Other Envs may have more responsabilities, such as holding history conversation and providing the data.

```python
# 1. Setup services (same as single-turn, plus environments)
policy = Generator(...)
trainer = TitanTrainer(...)
replay_buffer = ReplayBuffer(...)
ref_model = ReferenceModel(...)

# Dataloader provides tasks (prompts + metadata)
dataloader = DataLoader(Tau2BenchDataset(...))

# Task-based routing
# Different environments = different tools, max_turns, rewards
env_map = {
    "websearch": WebSearchEnv.from_docker_image("tau2bench/websearch:latest"),
    "coding": CodingEnv.from_docker_image("tau2bench/coding:latest"),
    "airline": AirlineEnv.from_docker_image("tau2bench/airline:latest"),
}

# Environment-specific configuration
max_turns_config = {
    "websearch": 10,
    "coding": 15,
    "airline": 8,
}
```

**References:**
- Verifiers: `verifiers/envs/env_group.py`
- Tinker: `tinker-cookbook/distillation/datasets.py:45-83`

---

## Rollout Loop: Multi-Turn with Environment Routing

```python
# 2. Rollout loop (continuous_rollouts with multi-turn)
async def continuous_rollouts():
    while True:
        # Sample task from dataloader
        task = await dataloader.sample.call_one()
        # task.prompt: "Book a flight from SF to NYC on March 15th"
        # task.task_type: "websearch" | "coding" | "airline"
        # task.metadata: Additional task-specific info

        # Route to correct environment based on task type
        env_client = env_map[task.task_type]
        max_turns = max_turns_config[task.task_type]

        # Reset environment to get tools (env doesn't know the task)
        # Reference: OpenEnv/src/core/http_env_client.py:142-154
        env_state = env_client.reset()
        tool_schemas = env_state.observation.tools  # Available tools for this env

        # Generate G samples for this task
        # TODO: Investigate parallelizing with asyncio.gather() instead of sequential
        episodes = []
        for _ in range(group_size):  # G samples per task
            episode = await play_task(
                policy=policy,
                task_prompt=task.prompt,  # From dataloader
                tool_schemas=tool_schemas,  # From environment
                env=env_client,
                max_turns=max_turns
            )
            episodes.append(episode)

        # Add to replay buffer (same as single-turn)
        for episode in episodes:
            await replay_buffer.add.call_one(episode)
```

**Critical insight:** Dataset provides tasks, environment provides tools. They are separate.

---

## Multi-Turn Rollout: play_task()

This replaces the single `policy.generate()` call in single-turn GRPO.

```python
# Reference: OpenEnv/src/core/client_types.py (StepResult)
from openenv.core.client_types import StepResult
from openenv.core.env_server import ToolCallAction

async def play_task(
    policy: Generator,
    task_prompt: str,  # From dataloader
    tool_schemas: list[dict],  # From env.reset()
    env: OpenEnvClient,
    max_turns: int = 10
) -> Episode:
    """
    Play one task to completion, return single Episode.

    Args:
        policy: Generator actor for LLM generation
        task_prompt: Task from dataloader (e.g., "Book flight SF->NYC")
        tool_schemas: Available tools from env.reset()
        env: Environment client for tool execution
        max_turns: Maximum conversation turns

    Returns:
        Episode with all turns concatenated
    """

    # Initialize conversation with task
    # System prompt handled by tokenizer.apply_chat_template() with tools=
    # Or dataset can provide task.system_prompt if needed
    messages = [{"role": "user", "content": task_prompt}]

    # Storage: concatenate all turns into single sequence
    all_tokens = []
    all_logprobs = []
    response_mask = []  # 1=train on LLM output, 0=skip tool results
    metadata = {}  # Track episode stats

    done = False
    turn = 0

    while not done and turn < max_turns:
        # 1. Format prompt with conversation history + tools
        # Tokenizer injects system prompt with tool definitions when tools= is passed
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,  # From env.reset()
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Generate response
        response = await policy.generate.route(prompt, n=1)

        # 3. Parse tool call from response
        # Using Tinker pattern: XML tags <tool_call>...</tool_call>
        # Alternative: vLLM native parsing with tool_call_parser="hermes" (see Appendix)
        tool_calls = parse_tool_calls(response.text)  # Returns list of tool calls

        if tool_calls:
            # Tool execution path
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": tool_calls  # Structured tool call data
            })

            # Collect LLM output tokens - TRAIN on these
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            # Execute tools (parallel if multiple calls)
            # TODO: Confirm environment can handle parallel requests
            try:
                tool_tasks = [
                    env.execute_tool(tc["name"], tc["args"])
                    for tc in tool_calls
                ]
                tool_results = await asyncio.gather(*tool_tasks)
            except Exception as e:
                # Handle tool execution errors
                tool_results = [{"content": f"Error: {str(e)}"}]

            # Add tool results to messages and tokens
            for tool_result in tool_results:
                tool_content = tool_result.content

                # Truncate long tool responses to avoid context overflow
                tool_tokens = tokenizer.encode(tool_content, add_special_tokens=False)
                tool_tokens = truncate(tool_tokens, max_length=256)
                # TODO: Decide where truncate() lives (env vs rollout loop vs utility)
                tool_content = tokenizer.decode(tool_tokens)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_content
                })

                # Collect tool result tokens - DON'T TRAIN on these
                all_tokens.extend(tool_tokens)
                all_logprobs.extend([0.0] * len(tool_tokens))
                response_mask.extend([0] * len(tool_tokens))

            # Check if environment signals done
            done = tool_results[-1].get("done", False) if tool_results else False

        else:
            # Final answer (no tool call)
            messages.append({
                "role": "assistant",
                "content": response.text
            })

            # Collect final response tokens - TRAIN on these
            all_tokens.extend(response.token_ids)
            all_logprobs.extend(response.logprobs)
            response_mask.extend([1] * len(response.token_ids))

            done = True

        turn += 1

    # Populate episode metadata
    metadata = {
        "num_turns": turn,
        "truncated": turn >= max_turns,
        # other stats...
    }

    # Get final reward from environment
    final_reward = env.get_reward(messages) #TODO: confirm messages as input

    # Create Episode
    # TODO: this abstraction will have to change. It was created for single-turn.
    completion = Completion(
        prompt_ids=None,  # Not stored (can reconstruct from messages)
        token_ids=torch.tensor(all_tokens),
        logprobs=torch.tensor(all_logprobs),
        text=tokenizer.decode(all_tokens),
        generator_version=0
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        request_len=0,  # Varies per turn, not fixed
        response_len=len(all_tokens),
        target=None,  # Tau2Bench doesn't expose ground truth during training
        completion=completion,
        response_mask=torch.tensor(response_mask),  # NEW: Mask for training
        ref_logprobs=None,  # Computed later by ref_model
        reward=final_reward,
        advantage=None,  # Computed later with group
        metadata=metadata  # NEW: Episode statistics
    )

    return episode
```
## Training Loop

Stays the same, but we add `response_mask`

```python
# Reference: apps/grpo/main.py

# 3. Training loop (minimal changes - just add response_mask)
async def continuous_training():
    while True:
        # Sample batch from replay buffer
        batch = await replay_buffer.sample(batch_size)

        # Get reference logprobs
        ref_logprobs = await ref_model.forward.route(
            prompt_ids=batch["prompt_ids"],
            response_ids=batch["response_ids"]
        )

        # Compute advantages (group-relative)
        advantages = compute_group_advantages(batch["rewards"])

        # Train on batch with response mask
        await trainer.train_step(
            inputs=batch["prompt_ids"],
            targets=batch["response_ids"],
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            response_mask=batch["response_mask"],  # NEW: Mask tool results
        )

        # Update policy weights
        version = await trainer.push_weights()
        await policy.update_weights(version)
```
