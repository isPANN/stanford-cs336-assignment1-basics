# Problem 009: Softmax

**Difficulty:** Easy  
**Topic:** Numerical stability / Reductions  
**Points:** 1  
**Implement in:** `cs336_basics/nn_utils.py`  
**Wire via:** `tests/adapters.py::run_softmax`

---

## Description

Normalize a tensor along dimension \(i\) into a probability simplex. Attention and the LM head both need this. Numerically: subtract the max along `dim` before `exp` (handout §3.4.4). Introduced next to attention; tests live in `test_nn_utils.py`.

---

## Signature

```python
def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor: ...
```

Adapter:

```python
def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
```

---

## Input / Output

| Param | Type | Meaning |
|-------|------|---------|
| `in_features` | any shape | Logits / scores |
| `dim` | `int` | Axis to normalize |
| **output** | same shape | \(\sum_{\text{dim}} = 1\) |

\[
\mathrm{softmax}(v)_i = \frac{\exp(v_i - \max v)}{\sum_j \exp(v_j - \max v)}
\]

---

## Constraints

- Subtract **max along `dim`**, keep dims for broadcast
- Match `F.softmax` at `atol=1e-5`
- Overflow test: `softmax(x + 100, dim=-1)` equals `softmax(x, dim=-1)`
- Do not call `torch.nn.functional.softmax` as the implementation (defeats the assignment)

---

## Examples

### Example — Test

3×5 tensor vs `F.softmax(..., dim=-1)`, then the same tensor plus 100.

---

## Rules / Invariants

1. Shape preserved
2. Non-negative; sums to 1 on `dim`
3. Shift-invariant: `softmax(v+c) = softmax(v)`

---

## Sub-problems

Pipeline: `max → subtract → exp → sum → divide`

### Sub-problem A — stable softmax

**Tools / docs**

| What | Reference |
|------|-----------|
| Max | `in_features.max(dim=dim, keepdim=True).values` |
| Handout | §3.4.4 eq. (10) + “subtract the largest entry” |

**Input:** tensor, `dim`

**Output:** same shape

**Goal:** Stable softmax.

**Checkpoint:** `uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Large logits (`+100`) | Same as unshifted |
| `dim` not last | Reduce that axis only |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q
```

---

## Complexity / Performance Targets

Fully vectorized reductions.

---

## Debug Checklist

- [ ] `keepdim=True` on max and sum
- [ ] Raw `exp` without max → NaNs on overflow test

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_nn_utils.py::test_softmax_matches_pytorch` | Judge |
| Handout §3.4.4 | Spec |

---

## Wiring reminder

`run_softmax` calls your function with the given `dim`.
