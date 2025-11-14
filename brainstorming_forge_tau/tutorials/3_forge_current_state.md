# Part 3: How Forge Currently Works

## 3.1 Current Forge GRPO Flow (GSM8K Example)

Forge currently implements GRPO (Group Relative Policy Optimization) for single-turn tasks like math problems.

**Architecture:**
```python
# apps/grpo/main.py

# 1. Setup services (distributed actors via Monarch)
policy = Generator(...)              # vLLM-based generation
trainer = TitanTrainer(...)          # Training service
replay_buffer = ReplayBuffer(...)    # Store episodes
ref_model = ReferenceModel(...)      # Reference for KL
reward_actor = RewardActor(...)      # Score responses

# 2. Rollout loop (continuous_rollouts)
async def continuous_rollouts():
    while True:
        # Sample prompt from dataset
        sample = await dataloader.sample.call_one()
        prompt, target = sample["prompt"], sample["target"]

        # Generate G responses (group)
        responses = await policy.generate.route(
            prompt,
            n=group_size  # e.g., 8 responses
        )

        # Score each response
        episodes = []
        for response in responses:
            episode = Episode(...)
            episode.reward = await reward_actor.evaluate_response.route(
                prompt=prompt,
                response=response.text,
                target=target
            )
            episodes.append(episode)

        # Get reference logprobs
        ref_logprobs = await ref_model.forward.route(...)

        # Compute advantages (group-relative)
        advantages = compute_advantages(episodes)

        # Add to replay buffer
        for episode in episodes:
            await replay_buffer.add.call_one(episode)

# 3. Training loop (continuous_training)
async def continuous_training():
    while True:
        batch = await replay_buffer.sample(batch_size)

        # Train on batch
        await trainer.train_step(
            inputs=batch["inputs"],
            targets=batch["targets"],
            advantages=batch["advantages"]
        )

        # Update policy weights
        version = await trainer.push_weights()
        await policy.update_weights(version)
```

**Key features:**
- **Async distributed**: Actors communicate via Monarch
- **Parallel rollouts**: Multiple `continuous_rollouts()` tasks
- **Decoupled**: Rollout and training loops run independently
- **Replay buffer**: Stores episodes for training

## 3.2 What Forge is Missing for Tool Calling

**Current GSM8K flow:**
```
Sample prompt → Generate response → Score → Train
```

**Needed for tool calling:**
```
Sample task → Multi-turn loop → Train
              ↓
              Generate → Parse → Execute tool → Update state → Repeat -> Score
```

**Missing components:**

### 1. Multi-turn Loop
**Current**: Single `policy.generate.route(prompt)`
**Needed**: Loop with multiple generation calls

```python
# Need to add:
while not done:
    response = await policy.generate.route(prompt)
    if has_tool_call(response):
        tool_result = execute_tool(...)
        # Continue loop
    else:
        done = True
```

### 2. Tool Call Detection & Parsing
**Current**: No parsing
**Needed**: Extract tool calls from model output

```python
# Need to add:
def parse_tool_call(response_text):
    if "<function_call>" in response_text:
        # Parse JSON
        return tool_call
    return None
```

### 3. Message History Management
**Current**: Single prompt
**Needed**: Accumulate multi-turn conversation

```python
# Need to add:
messages = [
    {"role": "user", "content": task},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": result},
    # ... more turns
]
```

### 4. Tool Execution
**Current**: No tool support
**Needed**: Environment to execute tools

```python
# Need to add:
env = Environment(task=task)
result = env.step(tool_call)
```

### 5. Response Masking
**Current**: Naively split between prompt/answer and train on the answer. This
 would train on all tokens, including tool calls.
**Needed**: Mask to ignore tool results in the loss function

```python
# Need to add:
response_mask = [
    1, 1, 1,  # LLM output - TRAIN
    0, 0, 0,  # Tool result - IGNORE
    1, 1, 1,  # LLM output - TRAIN
]
```

### 6. Episode Structure
**Current** (from `apps/grpo/main.py:44-74`):
```python
@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # Processed data
    completion: Completion | None = None  # Contains prompt_ids, token_ids, logprobs
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None
```

**Multi turn**:

**References**:
**Tinker** `tinker-cookbook/tinker_cookbook/rl/types.py`,
**VERL** `verl/experimental/agent_loop/tool_agent_loop.py`,
**TRL** `trl/examples/scripts/openenv/catch.py`
**NeMo-RL** `RL/nemo_rl/experience/rollouts.py`

- Store all turns (transition) in single Episode (trajectory)
- Concatenate turns during rollout or when converting to training data
- Build response_mask to exclude tool results from training

**Tinker's approach** (`tinker-cookbook/tinker_cookbook/rl/types.py`):
```python
Observation: TypeAlias = tinker.ModelInput

@dataclass
class Transition:
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)

@dataclass(frozen=True)
class Trajectory:
    transitions: list[Transition]
    final_ob: Observation

@dataclass
class TrajectoryGroup:
    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

    def get_total_rewards(self) -> list[float]:
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]
```

### 7. Prompt Formatting with Tools
**Current**: Simple prompt.
**Needed**: Our tokenizer jinja template already supports tools, but need to investigate how to use it
and write `format_tool_schemas`

```python
# Need to add:
system_prompt = f"""
You have access to these tools:

{format_tool_schemas(tools)}

Call tools using this format:
<function_call>{{"name": "tool_name", "args": {{}}}}</function_call>
"""
```

### 8. Reward Computation
**Current** (from `apps/grpo/main.py:385-398`): Immediate reward from `RewardActor`
```python
# For each response in the group
for i, response in enumerate(responses):
    episode.reward = await reward_actor.evaluate_response.route(
        prompt=prompt,
        response=response.text,
        target=target
    )
    # reward_actor compares response to target immediately
```

**Needed for multi-turn**: Sparse reward from environment after episode completes, i.e. the input to the reward calculator is the **full trajectory**.

```python
for i, response in enumerate(responses):
    ...

# add this
final_reward = sum(previous_rewards_if_any) + env.get_rewards(responses)
# or just:
final_reward = env.get_rewards(responses)
```




---

**Summary Table:**

| Component | GSM8K (Current) | Tool Calling (Needed) |
|-----------|----------------|----------------------|
| **Loop** | Single generate | Multi-turn while loop |
| **Tools** | None | Parse & execute |
| **Reward** | Per-response | Sparse at end |
| **Loss** | All tokens | Masked (exclude tool results) |
| **Episode** | Single turn | multi-turn |
