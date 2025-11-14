# GRPO Bug Fix Summary

## Critical Bugs Found and Fixed

### Bug 1: Behavior Policy Logprobs Never Stored
**Problem:** The GRPO algorithm requires logprobs from the **behavior policy** (the policy that generated the responses), but these were never being stored or used.

**Impact:** The policy gradient had no signal - it was computing `exp(logprobs - logprobs) = exp(0) = 1`, making the importance sampling ratio always 1, which means the model wasn't actually learning from the policy gradient.

**Fixes Applied:**
1. Added `behavior_logprobs` field to `Episode` dataclass
2. Extracted behavior_logprobs from `completion.logprobs` when creating episodes
3. Added `behavior_logprobs_tensor` property to handle padding
4. Updated `collate()` function to include behavior_logprobs in batches
5. Updated loss functions to use behavior_logprobs instead of `logprobs.detach()`

### Bug 2: Logprobs Not Enabled in vLLM
**Problem:** The sampling_params configuration didn't request logprobs from vLLM, so `completion.logprobs` was always `None`.

**Fix Applied:**
- Added `logprobs: 1` to `sampling_params` in `qwen3_1_7b.yaml`

### Bug 3: Incorrect Importance Sampling Ratio
**Problem:** Both loss functions used:
```python
per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
```

This is always `exp(0) = 1`, providing no learning signal!

**Correct Formula:**
```python
per_token_policy_loss = torch.exp(logprobs - behavior_logprobs.detach()) * advantages
```

**Files Modified:**
- `apps/grpo/main.py` - `simple_grpo_loss()` function
- `src/forge/losses/grpo_loss.py` - `SimpleGRPOLoss.forward()` method

## GRPO Algorithm Explanation

GRPO (Group Relative Policy Optimization) uses **three** sets of logprobs:

1. **Current Policy Logprobs** (`logprobs`): From the model being trained
   - Used to compute gradients
   - Computed on-the-fly during training

2. **Behavior Policy Logprobs** (`behavior_logprobs`): From the policy that generated the responses
   - Used for importance sampling: `ratio = exp(current - behavior)`
   - Must be stored when responses are generated
   - In "off-by-n" setting, this is the policy from n steps ago

3. **Reference Policy Logprobs** (`ref_logprobs`): From a frozen reference model
   - Used for KL regularization to prevent the policy from diverging too much
   - Computed from a frozen copy of the initial model

## Testing the Fix

To verify the fix works:

```bash
# Train with learning rate 1e-5 (should learn)
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml trainer.optimizer.lr=1e-5

# Train with learning rate 0 (should NOT learn - flat rewards)
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml trainer.optimizer.lr=0
```

**Expected behavior after fix:**
- With lr=1e-5: Rewards should improve over time
- With lr=0: Rewards should stay flat (no learning)
- The two runs should have DIFFERENT reward trajectories

**Before the fix:**
- Both runs had identical reward patterns (no actual learning happening)

## Files Changed

1. `apps/grpo/main.py`
   - Updated `Episode` dataclass
   - Updated `collate()` function  
   - Updated `simple_grpo_loss()` function
   - Added behavior_logprobs extraction in rollout loop
   - Added `behavior_logprobs_tensor` property

2. `src/forge/losses/grpo_loss.py`
   - Updated `SimpleGRPOLoss.forward()` signature and implementation
   - Added documentation explaining the three logprob types

3. `apps/grpo/qwen3_1_7b.yaml`
   - Added `logprobs: 1` to sampling_params

