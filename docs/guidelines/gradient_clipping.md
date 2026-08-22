# Problem 016: Gradient Clipping

**Difficulty:** Easy  
**Topic:** In-place grads  
**Points:** 1  
**Implement in:** `cs336_basics/nn_utils.py`  
**Wire via:** `tests/adapters.py::run_gradient_clipping`

---

## Description

If the **global** \(\ell_2\) norm of all parameter gradients exceeds \(M\), scale every `.grad` by \(M / (\|g\|_2 + \varepsilon)\) with \(\varepsilon = 10^{-6}\) (PyTorch default). Modify grads **in place**. Skip parameters with `grad is None` (frozen / unused).

---

## Signature

```python
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
```

Adapter:

```python
def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
```

---

## Input / Output

| Param | Meaning |
|-------|---------|
| `parameters` | Iterable of `nn.Parameter` (some may lack `.grad`) |
| `max_l2_norm` | \(M > 0\) |
| **return** | `None` — side effect on `.grad` |

Total norm: \(\|g\|_2 = \sqrt{\sum_p \|\texttt{p.grad}\|_2^2}\) over params that have grads.

---

## Constraints

- In-place on `.grad`; do not replace the Parameter
- Match `torch.nn.utils.clip_grad.clip_grad_norm_` (`atol=1e-5`)
- Test freezes the last of 6 tensors (`requires_grad_(False)`) — that param has no grad
- If \(\|g\|_2 \le M\), leave grads unchanged

---

## Examples

`max_norm = 1e-2`; 6 random `(5,5)` parameters; one frozen; after `loss = cat(params).sum().backward()`.

---

## Rules / Invariants

1. Combined norm, not per-tensor clip
2. Frozen params ignored
3. \(\varepsilon = 10^{-6}\) in the scale when clipping

---

## Sub-problems

Pipeline: `filter grads → total L2 → maybe scale in-place`

### Sub-problem A — global norm

**Tools / docs**

| What | Reference |
|------|-----------|
| Per-tensor | `p.grad.detach().norm(2)` |
| Handout | §4.5 |

**Input:** parameters

**Output:** scalar \(\|g\|_2\)

**Goal:** One number over all grads.

**Checkpoint:** ignoring `None` grads; length of grad list matches PyTorch

### Sub-problem B — scale

**Tools / docs**

| What | Reference |
|------|-----------|
| In-place | `p.grad.mul_(clip_coef)` |
| Clip coef | `M / (total_norm + 1e-6)` when `total_norm > M` |

**Input:** parameters, `M`

**Output:** mutated `.grad`

**Goal:** Match `clip_grad_norm_`.

**Checkpoint:** `uv run pytest tests/test_nn_utils.py::test_gradient_clipping -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| `total_norm <= M` | No change |
| Some `grad is None` | Skip |
| Returning a new tensor | Blocker — optimizer still sees old `.grad` |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_nn_utils.py::test_gradient_clipping -q
```

Full nn utils:

```bash
uv run pytest tests/test_nn_utils.py -q
```

---

## Complexity / Performance Targets

One pass to compute norm, one pass to scale.

---

## Debug Checklist

- [ ] Per-parameter clip instead of global
- [ ] Forgot \(\varepsilon\)
- [ ] Used `clip_grad_norm_` as the implementation — defeats the problem (quality bar)

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_nn_utils.py::test_gradient_clipping` | Judge |
| Handout §4.5 | Spec |

---

## Wiring reminder

`run_gradient_clipping` calls your in-place function and returns nothing.
