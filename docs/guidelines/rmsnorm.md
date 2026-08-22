# Problem 005: RMSNorm

**Difficulty:** Easy  
**Topic:** Tensor ops / Normalization  
**Points:** 1  
**Implement in:** `cs336_basics/model.py`  
**Wire via:** `tests/adapters.py::run_rmsnorm`

---

## Description

Root-mean-square layer norm used in every pre-norm Transformer sub-layer and after the last block. Rescale activations by RMS, then multiply by a learned gain \(g \in \mathbb{R}^{d_\text{model}}\). No mean subtraction (unlike LayerNorm).

---

## Signature

```python
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Adapter:

```python
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
```

---

## Input / Output

### Input (`forward`)

| Param | Type | Meaning |
|-------|------|---------|
| `x` | `(..., d_model)` | Activations; last dim normalized |

### Output

| Return | Type | Same shape as `x` |
|--------|------|-------------------|
| | | \(\mathrm{RMSNorm}(a_i) = a_i / \mathrm{RMS}(a) \cdot g_i\) |

Handout (eq. 4):

\[
\mathrm{RMS}(a) = \sqrt{\frac{1}{d_\text{model}}\sum_i a_i^2 + \varepsilon}
\]

---

## Constraints

- Gain \(g\) is `nn.Parameter` of shape `(d_model,)`, init to ones
- Upcast `x` to `float32` before squaring; downcast result to original dtype
- \(\varepsilon\) default `1e-5`; tests pass `eps=1e-5`
- RMS uses **mean of squares then + eps**, then sqrt — not `mean(x^2 + eps)` vs `mean(x^2)+eps` confusion: handout puts \(\varepsilon\) **inside** the square root with the mean of squares
- Vectorized over batch/seq
- Adapter: construct → load gain → `forward`

---

## Examples

### Example 1 — Handout snippet

```
in_dtype = x.dtype
x = x.to(torch.float32)
# RMSNorm
return result.to(in_dtype)
```

### Example 2 — Test

Fixture `layers.1.ln1.weight`; `in_embeddings` `(4, 12, 64)`; snapshot atol `1e-4`.

---

## Rules / Invariants

1. Output shape == input shape
2. Each position's last dim is independently normalized
3. Gain broadcasts as `(d_model,)` over `(..., d_model)`

---

## Sub-problems

Pipeline: `upcast → rms over last dim → x / rms * g → downcast`

### Sub-problem A — `__init__`

**Tools / docs**

| What | Reference |
|------|-----------|
| Gain | `nn.Parameter(torch.ones(d_model, ...))` |
| Handout | §3.4.1 |

**Input:** `d_model`, `eps`

**Output:** module with `weight` `(d_model,)`

**Goal:** Store affine gain.

**Checkpoint:** `assert m.weight.shape == (d_model,)`

### Sub-problem B — `forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| Mean of squares | `x.pow(2).mean(dim=-1, keepdim=True)` |
| rsqrt | `torch.rsqrt(ms + eps)` |
| Handout | §3.4.1 eq. (4) |

**Input:** `x: (..., d_model)`

**Output:** same shape

**Goal:** RMS rescale + gain.

**Checkpoint:** `uv run pytest tests/test_model.py::test_rmsnorm -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Mixed precision input | Compute in fp32, return original dtype |
| Extra leading dims | Normalize only last dim |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_rmsnorm -q
```

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| RMS | Python over `d_model` | `mean` / `rsqrt` on last dim |

---

## Debug Checklist

- [ ] \(\varepsilon\) inside the sqrt with mean(\(a^2\))
- [ ] fp32 upcast
- [ ] Gain multiply **after** normalize
- [ ] Not `nn.LayerNorm` (that subtracts mean)

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_rmsnorm` | Judge |
| Handout §3.4.1 | Spec |

---

## Wiring reminder

`tests/adapters.py::run_rmsnorm` constructs `RMSNorm`, loads `weights` as the gain, returns `forward`.
