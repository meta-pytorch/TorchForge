# Part 2: The Fundamentals

## 2.1 What is Tool Calling?

Tool calling allows the LLM to invoke functions instead of just generating text.

**Example:**
```python
# Without tools:
User: "What's the weather in NYC?"
Model: "I don't have access to real-time weather data."

# With tools:
User: "What's the weather in NYC?"
Model: <tool_call>get_weather(city="NYC")</tool_call>
Tool: {"temperature": 72, "conditions": "sunny"}
Model: "It's 72°F and sunny in NYC."
```

## 2.2 How Tool Calling Works

**Core concept:** Models are trained to output special formats (tokens or text tags), then we parse them to extract structured tool calls.

**Two parsing approaches exist in practice:**

### Token-Based Parsing (vLLM Native)
Some models use **special token IDs** (e.g., token 12971 = `<|python_tag|>`). vLLM can parse these directly:

```yaml
# vLLM config
enable_auto_tool_choice: true
tool_call_parser: "hermes"  # Model-specific: "mistral", "llama", "internlm"
```

### Text-Based Parsing (Manual)
Most libraries parse text tags with regex (seen in Tinker, TRL, Verifiers):

```python
# Example from tinker-cookbook/tinker_cookbook/renderers.py
def parse_response(self, response_tokens):
    text = self.tokenizer.decode(response_tokens)
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        return Message(role="assistant", tool_calls=[json.loads(match.group(1))])
    return Message(role="assistant", content=text)
```

**Reference:** [Tinker renderers.py](../../tinker-cookbook/tinker_cookbook/renderers.py)

**NOTE**: Every model has its own format. We shouldn't use arbitrary tags with arbitrary models.

## 2.3 What is Multi-turn?

Multi-turn = multiple back-and-forth exchanges in a single episode.

**Single-turn:**
```
User: "What's 2+2?"
Model: "4"
[Done]
```

**Multi-turn:**
```
User: "What's 2+2?"
Model: "4"
User: "What's 4+2?"
Model: "6"
User: "What's 6+2?"
Model: "8"
[Done]
```

For tool calling, multi-turn enables:
1. Call tool
2. Get result
3. Use result to decide next action
4. Repeat until task complete

## 2.4 Multi-turn Loop: A Simple Python Example

```python
# Conceptual multi-turn loop
env = create_env(task="Book a flight to NYC")
messages = [{"role": "user", "content": "Book me a flight to NYC"}]
done = False

while not done:
    # 1. Build prompt from message history
    prompt = build_prompt(messages)

    # 2. Generate response
    # On first iteration it calls the tool and gets the results
    # On following iterations it acts based on the result
    # repeat until model says it is done
    # Another option is to have another LLM here acting as an user.
    response = model.generate(prompt)

    # 3. Check if tool call
    if has_tool_call(response):
        # Parse and execute tool
        tool_call = parse_tool_call(response)
        tool_result = env.execute_tool(tool_call)

        # Add to history
        messages.append({"role": "assistant", "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": tool_result})
    else:
        # Final answer
        messages.append({"role": "assistant", "content": response})
        done = True

# Get final reward
reward = env.get_reward()
```

Key points:
- **Loop** until done
- **Accumulate** messages (conversation history)
- **Tools** execute via environment
- **Reward** computed at end (sparse)

## 2.5 What is an Environment?

An **environment** manages:
1. **Tool execution**: Runs tools, returns results
2. **State management**: Tracks what's been done
3. **Reward computation**: Scores the episode

**Standard API** (gym-like):

```python
# Initialize
env = Environment(task=task_data)
state = env.reset()  # Returns initial state/observation

# Step
result = env.step(action)  # Execute tool or message
# result contains:
#   - observation: New state (tool result, env feedback)
#   - reward: Immediate reward (often 0.0 for intermediate steps)
#   - done: Is episode complete?
#   - info: Extra metadata

# Final reward
if result.done:
    final_reward = result.reward
```

**Relationship to tools:**
- Environment **owns** the tools
- `env.step(tool_call)` executes the tool
- Returns tool result as observation
- Updates internal state (databases, etc.)

## 2.6 Message Format (OpenAI Standard)

Take the example:
```
"Assistant: I'll search for flights and check the weather for you. <tool_call>
{"name": "search_flights", "arguments": {"destination": "NYC"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "NYC"}}
</tool_call>"
```

**After parsing, this becomes the structured message** with separate `content` and `tool_calls` fields. Most libraries use OpenAI's chat format:

```python
messages = [
    # System message (optional)
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools..."
    },

    # User message
    {
        "role": "user",
        "content": "Book me a flight to NYC and check the weather there"
    },

    # Assistant message (with content AND tool calls in ONE message)
    {
        "role": "assistant",
        "content": "I'll search for flights and check the weather for you.",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "search_flights",
                    "arguments": '{"destination": "NYC"}'
                }
            },
            {
                "id": "call_124",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}'
                }
            }
        ]
    },

    # Tool results (one per tool call)
    {
        "role": "tool",
        "content": '[{"flight": "AA100", "price": "$200"}]',
        "tool_call_id": "call_123"
    },
    {
        "role": "tool",
        "content": '{"temperature": 72, "conditions": "sunny"}',
        "tool_call_id": "call_124"
    }
]
```

**Key fields:**
- `role`: "system", "user", "assistant", or "tool"
- `content`: Text content
- `tool_calls`: List of tool invocations (assistant only)
- `tool_call_id`: Links tool result to invocation

**Chat template** converts messages to model input:
```python
# Using tokenizer
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)
# Returns formatted string ready for model
```
