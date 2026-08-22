# Problem 015: Cross-Entropy Loss

**Difficulty:** Easy  
**Topic:** Numerical stability / Loss  
**Points:** 1  
**Implement in:** `cs336_basics/nn_utils.py`  
**Wire via:** `tests/adapters.py::run_cross_entropy`

---

## Description

Mean token NLL (handout §4.1): \(\ell_i = -\log \mathrm{softmax}(o_i)[x_{i+1}]\), averaged over the batch. Cancel \(\log\circ\exp\) via log-sum-exp; subtract max for stability. Do **not** implement as `log(softmax(o))` (less stable).

---

## Signature

```python
def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
```

Adapter docstring shows 2D logits; the test flattens `(batch, seq, vocab)` with `.view(-1, vocab)` first. Handle extra leading dims if you want, but the judge always passes 2D.

---

## Input / Output

| Param | Shape | Meaning |
|-------|-------|---------|
| `inputs` | `(N, vocab_size)` | Logits \(o_i\) |
| `targets` | `(N,)` | Class indices in `[0, vocab)` |
| **output** | scalar | Mean \(\frac{1}{N}\sum_i \ell_i\) |

Stable form: \(\ell_i = -o_i[t_i] + \log\sum_a \exp(o_i[a] - \max o_i) + \max o_i\) (or equivalent).

---

## Constraints

- Average over examples (match `F.cross_entropy`)
- Overflow: `1000 * inputs` still matches PyTorch (`atol=1e-4`)
- Targets are class indices, not one-hot
- Do not use `F.cross_entropy` as the implementation

---

## Examples

Test tensors: logits `(2, 4, 5)` flattened to `(8, 5)`, targets `(2, 4)` flattened to `(8,)`.

---

## Rules / Invariants

1. Scalar return (0-dim tensor)
2. Shift-invariant in the same way as softmax
3. Mean reduction, not sum

---

## Sub-problems

Pipeline: `max-subtract → logsumexp → gather target logit → mean negative`

### Sub-problem A — stable NLL

**Tools / docs**

| What | Reference |
|------|-----------|
| Gather | `inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)` |
| Logsumexp | `torch.logsumexp` after max-subtract, or fused |
| Handout | §4.1 “Cancel out log and exp” |

**Input:** logits, targets

**Output:** scalar mean

**Goal:** Match `F.cross_entropy`.

**Checkpoint:** `uv run pytest tests/test_nn_utils.py::test_cross_entropy -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Huge logits | Finite; matches PyTorch |
| `N=1` | Still a mean (the single loss) |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_nn_utils.py::test_cross_entropy -q
```

---

## Complexity / Performance Targets

Vectorized over `N` and `vocab`.

---

## Debug Checklist

- [ ] Sum instead of mean
- [ ] Softmax then log (overflow)
- [ ] Target dim not gathered on last axis

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_nn_utils.py::test_cross_entropy` | Judge |
| Handout §4.1 | Spec |

---

## Wiring reminder

`run_cross_entropy` calls your function on the already-flattened tensors the test provides.
