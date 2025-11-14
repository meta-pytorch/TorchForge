# Part 1: Tau2Bench Overview - What Are We Building For?

## 1.1 What is Tau2Bench?

**Reference**: `tau2-bench/README.md`, `tau2-bench/src/tau2/evaluator/evaluator.py`

Tau2Bench is a benchmark for evaluating conversational agents in customer service scenarios. It tests whether your RL-trained model can:
- Follow domain policies correctly
- Use tools appropriately (search databases, update records, etc.)
- Communicate effectively with users

Example task: "Create a task called 'Important Meeting' for user_1 with description 'Quarterly planning' and deadline tomorrow."

The agent must call `create_task(user_id="user_1", title="Important Meeting", ...)` with the right parameters, then confirm to the user.

## 1.2 Tau2 Modes

**Reference**: `tau2-bench/src/tau2/orchestrator.py:67-174`

**Solo Mode** (Recommended for training):
- Agent works alone on tickets/tasks
- No user interaction
- Simpler, deterministic
- Use this for initial training

**Normal Mode**:
- Agent + User Simulator (LLM playing customer)
- More realistic but harder

## 1.3 Tau2 Task Structure

**Reference**: Task files at `tau2-bench/data/tau2/domains/{domain}/tasks.json`, data model at `tau2-bench/src/tau2/data_model/tasks.py`

Tasks are defined in JSON format:

```json
{
  "id": "create_task_1",
  "ticket": "User wants to create a task titled 'Important Meeting' for user_1",
  "evaluation_criteria": {
    "actions": [
      {
        "action_id": "create_1",
        "name": "create_task",
        "arguments": {
          "user_id": "user_1",
          "title": "Important Meeting"
        }
      }
    ],
    "reward_basis": ["ACTION", "COMMUNICATE"]
  }
}
```

Key fields:
- `ticket`: Initial task description
- `evaluation_criteria.actions`: Expected tool calls
- `reward_basis`: What to score (ACTION, ENV, COMMUNICATE, NL_ASSERTIONS)

**NOTE ON EVAL**: In this case, evaluation is checking if the tool was called. In other cases, it may be having another LLM verify if the task was completed correctly.

## 1.4 Tau2 Available Tools (Mock Domain)

```python
# Mock domain tools for demonstration
tools = [
    {
        "name": "create_task",
        "description": "Create a new task",
        "parameters": {
            "user_id": "string",
            "title": "string",
            "description": "string (optional)",
            "deadline": "string (optional)"
        }
    },
    {
        "name": "update_task",
        "description": "Update an existing task",
        "parameters": {
            "task_id": "string",
            "status": "string (pending|completed|cancelled)"
        }
    },
    {
        "name": "done",
        "description": "Signal task completion",
        "parameters": {}
    }
]
```

**Production Domains**: Tau2Bench includes three main production domains with domain-specific tools, policies, and databases:
- **Airline**: Flight booking, modifications, cancellations (`tau2-bench/src/tau2/domains/airline/`)
- **Retail**: Product orders, returns, exchanges (`tau2-bench/src/tau2/domains/retail/`)
- **Telecom**: Technical support, bill payments, line management (`tau2-bench/src/tau2/domains/telecom/`)

## 1.5 Example Multi-turn Interaction on Tau2

**Solo Mode Example:**

```
Turn 1:
Agent: Let me create that task for you.
       create_task(user_id="user_1", title="Important Meeting",
                   description="Quarterly planning", deadline="2024-01-16")
Env:   Task created with ID: task_123

Turn 2:
Agent: Task created successfully. Is there anything else you need?
       done()
Env:   Episode complete.
```

**Note**: `done()` signals episode end. In Normal Mode, users can also end with keywords like "bye", "thanks" (see `tau2-bench/src/tau2/orchestrator.py:171-174` for stop conditions)

## 1.6 How Tau2 Scores Episodes

**Reference**: Evaluation logic in `tau2-bench/src/tau2/evaluator/evaluator.py`, metrics in `tau2-bench/src/tau2/metrics/agent_metrics.py`

Tau2Bench computes rewards based on multiple criteria:

**1. ACTION Score** (0.0 or 1.0):
- Did agent call the right tools?
- With the right arguments (or subset via `compare_args`)?
- Order doesn't matter

**2. ENV Score** (0.0 or 1.0):
- Is environment state correct?
- Database checks (e.g., task_id="task_2" has status="pending")

**3. COMMUNICATE Score** (0.0 or 1.0):
- Did agent communicate required information to user?

**4. NL_ASSERTIONS Score** (0.0 or 1.0):
- LLM-based evaluation of conversation quality (experimental)

**Final Reward:**
```python
final_reward = ACTION_score * ENV_score * COMMUNICATE_score * NL_ASSERTIONS_score
```

**CRITICAL**: Episode must end with either:
- `AGENT_STOP`: Agent calls `done()` tool
- `USER_STOP`: User says stop keywords

Otherwise: `reward = 0.0` regardless of actions!

**Sparse Rewards**: You only get the final reward at episode end. Intermediate tool calls get `reward=0.0`.

---

## 1.7 Tau2Bench Production Domains

Tau2Bench includes three production-ready customer service domains. Each domain has its own policy, tools, database, and evaluation tasks.

### Airline Domain

**Location**: `tau2-bench/data/tau2/domains/airline/`
- **Tasks**: 50 tasks in `tasks.json`
- **Policy**: `policy.md`
- **Code**: `tau2-bench/src/tau2/domains/airline/tools.py`

**What agents do**: Book, modify, and cancel flight reservations, handle refunds and compensation, manage baggage and travel insurance.

**Example tasks**:
- Cancellation policy testing (refuse invalid cancellations)
- Membership verification for baggage allowance
- Compensation fraud detection
- Complex modifications (multiple changes at once)
- Multi-reservation management

**Available tools**:
- `get_user_details()`, `get_reservation_details()`
- `search_flights()`, `book_flight()`, `modify_flight()`, `cancel_reservation()`
- `add_baggage()`, `get_compensation()`
- `transfer_to_human_agents()`

**Key policy rules**:
- Basic economy flights cannot be modified after booking
- Cancellations only allowed if: within 24hrs of booking, airline cancelled, business flight, or insurance covers reason
- Max 24 hours confirmation required before database-modifying actions
- Travel insurance: $30/passenger, enables full refund for covered reasons

**Rewards**: DB checks, ENV_ASSERTION, ACTION-based evaluation

### Retail Domain

**Location**: `tau2-bench/data/tau2/domains/retail/`
- **Tasks**: 114 tasks in `tasks.json`
- **Policy**: `policy.md`
- **Code**: `tau2-bench/src/tau2/domains/retail/tools.py`

**What agents do**: Help customers return/exchange delivered orders, cancel/modify pending orders, manage payment methods and addresses, provide product information.

**Example tasks**:
- Multi-item exchanges with specific options
- Conditional exchanges (fallback options if unavailable)
- Product information queries + multiple returns
- Pending order modifications (change color, material, etc.)
- Cross-order refunds (complex refunds across multiple orders)
- Selective returns (specific items from orders)
- Address modifications for pending orders

**Available tools**:
- `find_user_id_by_name_zip()`, `find_user_id_by_email()`
- `get_order_details()`, `get_product_details()`
- `cancel_pending_order()`, `modify_pending_order_items()`
- `return_delivered_order_items()`, `exchange_delivered_order_items()`
- `modify_pending_order_payment()`, `modify_user_default_address()`
- `transfer_to_human_agents()`

**Key policy rules**:
- User authentication required via email OR name+zip before any action
- Pending orders can only be cancelled/modified once
- Delivered orders can be returned or exchanged
- Product IDs ≠ Item IDs (must distinguish between catalog and specific variants)
- One order modification max - collect all changes before calling tool
- Product variants: Different options (color, size, material) = different item_ids
- Refunds: Gift card refunds immediate, others 5-7 business days

**Rewards**: DB checks, ACTION-based, COMMUNICATE evaluation

### Telecom Domain

**Location**: `tau2-bench/data/tau2/domains/telecom/`
- **Tasks**: 2,285 tasks in `tasks.json` (many auto-generated variants)
- **Policy**: `main_policy.md`
- **Code**: `tau2-bench/src/tau2/domains/telecom/tools.py` (agent) and `user_tools.py` (simulator)

**What agents do**: Provide technical support for mobile devices and connectivity issues, handle overdue bill payments, manage line suspensions, help with data refueling and plan changes.

**Example task categories**:
- **Mobile data issues** (~1000+ tasks): Roaming problems, data mode issues, network preference problems, VPN connectivity, airplane mode interference, data usage exceeded, multiple combined issues
- **MMS issues**: MMS sending failures with various device states
- **Service issues**: Line suspension problems, network outages, connection problems

**Example task IDs**:
- `[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]` - User abroad with roaming disabled
- `[mobile_data_issue]data_usage_exceeded[PERSONA:Easy]` - User exceeded data limit
- `[mobile_data_issue]airplane_mode_on|data_saver_mode_on[PERSONA:Easy]` - Multiple issues combined

**Available agent tools**:
- `get_customer_by_phone()`, `get_customer_by_id()`, `get_customer_by_name()`
- `get_line()`, `get_line_by_phone()`, `get_bill()`, `get_bills_by_customer()`
- `send_payment_request()`, `make_payment()`
- `refuel_data()` (max 2GB), `change_plan()`
- `suspend_line()`, `resume_line()`
- `transfer_to_human_agents()`

**Unique user tools** (simulates user controlling device):
- `set_user_location()`, `toggle_roaming()`, `toggle_airplane_mode()`, `toggle_mobile_data()`
- `toggle_data_saver_mode()`, `set_network_preference()`, `toggle_vpn()`, `toggle_eSIM()`
- `perform_speed_test()`, `get_status_bar()`, `can_send_mms()`

**Key policy rules**:
- Try to resolve before escalating to human agents
- Overdue bills: Check status → send payment request → customer checks request → make payment
- Line suspension: Only lift after all overdue bills paid (cannot lift for expired contracts)
- Data refueling: Max 2GB per refuel, price varies by plan
- Customer lookup: By phone, ID, or name+DOB
- Bill status types: Draft, Issued, Paid, Overdue, Awaiting Payment, Disputed
- Line status types: Active, Suspended, Pending Activation, Closed

**Rewards**: ENV_ASSERTION (checks device state), ACTION (correct tool calls), COMMUNICATE

**Example telecom evaluation**:
```json
{
  "actions": [{"name": "toggle_roaming", "requestor": "user"}],
  "env_assertions": [
    {"func_name": "assert_mobile_data_status", "expected_status": true},
    {"func_name": "assert_internet_speed", "expected_desc": "excellent"}
  ],
  "reward_basis": ["ENV_ASSERTION"]
}
```

Success = Agent correctly diagnoses problem + user performs correct fix + environment reaches target state

---

## 1.8 Key Tau2Bench References

**Task definitions**:
- Mock domain: `tau2-bench/data/tau2/domains/mock/tasks.json`
- Airline: `tau2-bench/data/tau2/domains/airline/tasks.json` (50 tasks)
- Retail: `tau2-bench/data/tau2/domains/retail/tasks.json` (114 tasks)
- Telecom: `tau2-bench/data/tau2/domains/telecom/tasks.json` (2,285 tasks)

**Policies**:
- Airline: `tau2-bench/data/tau2/domains/airline/policy.md`
- Retail: `tau2-bench/data/tau2/domains/retail/policy.md`
- Telecom: `tau2-bench/data/tau2/domains/telecom/main_policy.md`

**Tool implementations**:
- Airline tools: `tau2-bench/src/tau2/domains/airline/tools.py`
- Retail tools: `tau2-bench/src/tau2/domains/retail/tools.py`
- Telecom agent tools: `tau2-bench/src/tau2/domains/telecom/tools.py`
- Telecom user tools: `tau2-bench/src/tau2/domains/telecom/user_tools.py`

**Evaluation code**:
- Main evaluator: `tau2-bench/src/tau2/evaluator/evaluator.py`
- Metrics (pass^k): `tau2-bench/src/tau2/metrics/agent_metrics.py`
- Orchestrator (runs episodes): `tau2-bench/src/tau2/orchestrator.py`

**Data models**:
- Task structure: `tau2-bench/src/tau2/data_model/tasks.py`
- Airline models: `tau2-bench/src/tau2/domains/airline/data_model.py`
- Retail models: `tau2-bench/src/tau2/domains/retail/data_model.py`
- Telecom models: `tau2-bench/src/tau2/domains/telecom/data_model.py`

---
