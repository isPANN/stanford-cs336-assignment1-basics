# Problem 018: Cosine LR Schedule with Warmup

**Difficulty:** Easy  
**Topic:** Scheduling  
**Points:** 1  
**Implement in:** `cs336_basics/optimizer.py`  
**Wire via:** `tests/adapters.py::run_get_lr_cosine_schedule`

---

## Description

LLaMA-style cosine annealing (handout §4.4). Pure function of step \(t\):

- **Warm-up** \(t < T_w\): \(\alpha_t = (t / T_w)\, \alpha_{\max}\)
- **Cosine** \(T_w \le t \le T_c\):  
  \(\alpha_t = \alpha_{\min} + \tfrac12\bigl(1+\cos(\frac{t-T_w}{T_c-T_w}\pi)\bigr)(\alpha_{\max}-\alpha_{\min})\)
- **Post** \(t > T_c\): \(\alpha_t = \alpha_{\min}\)

---

## Signature

```python
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
```

(`it` is \(t\); `warmup_iters` is \(T_w\); `cosine_cycle_iters` is \(T_c\).)

---

## Input / Output

| Param | Test value | Role |
|-------|------------|------|
| `it` | `0..24` | Current step |
| `max_learning_rate` | `1` | \(\alpha_{\max}\) |
| `min_learning_rate` | `0.1` | \(\alpha_{\min}\) |
| `warmup_iters` | `7` | \(T_w\) |
| `cosine_cycle_iters` | `21` | \(T_c\) |
| **return** | float | \(\alpha_t\) |

Expected sequence is hardcoded in `test_get_lr_cosine_schedule` (length 25). Notable points: `it=0 → 0`, `it=7 → 1.0`, `it=21 → 0.1`, `it>21 → 0.1`.

---

## Constraints

- \(t=0\) during warmup gives 0 (not \(\alpha_{\max}\))
- At \(t=T_w\) use the **cosine** branch (since warmup is `t < T_w`), which equals \(\alpha_{\max}\)
- At \(t=T_c\) cosine argument is \(\pi\), so \(\alpha_{\min}\)
- Do not decay to 0 unless \(\alpha_{\min}=0\)

---

## Examples

See `expected_lrs` in `tests/test_optimizer.py`.

---

## Rules / Invariants

1. Three mutually exclusive branches
2. Continuous at \(T_w\) and \(T_c\) for these formulas

---

## Sub-problems

Pipeline: `if t < Tw → elif t <= Tc → else`

### Sub-problem A — implement the three pieces

**Tools / docs**

| What | Reference |
|------|-----------|
| Cosine | `math.cos` |
| Handout | §4.4 |

**Input:** `it`, four schedule constants

**Output:** `float`

**Goal:** Match the 25 reference values.

**Checkpoint:** `uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| `it == 0` | 0 |
| `it == Tw` | \(\alpha_{\max}\) |
| `it == Tc` | \(\alpha_{\min}\) |
| `Tw == 0` | no warmup (not in unit test) |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -q
uv run pytest tests/test_optimizer.py -q
```

---

## Complexity / Performance Targets

O(1) arithmetic.

---

## Debug Checklist

- [ ] Warmup `t/Tw` vs `(t+1)/Tw`
- [ ] Inclusive cosine end
- [ ] `cos` argument missing \(\pi\)

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_optimizer.py::test_get_lr_cosine_schedule` | Exact expected list |
| Handout §4.4 | Spec |

---

## Wiring reminder

`run_get_lr_cosine_schedule` is a thin wrap of your function.
