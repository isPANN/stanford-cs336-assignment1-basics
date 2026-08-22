# Problem 007: SwiGLU Feed-Forward

**Difficulty:** Easy  
**Topic:** Tensor ops / FFN  
**Points:** 2  
**Implement in:** `cs336_basics/model.py`  
**Wire via:** `tests/adapters.py::run_swiglu`

---

## Description

Position-wise FFN of the Transformer block (handout eq. 7):

\[
\mathrm{FFN}(x) = W_2\bigl(\mathrm{SiLU}(W_1 x) \odot W_3 x\bigr)
\]

with \(W_1,W_3 \in \mathbb{R}^{d_\text{ff}\times d_\text{model}}\), \(W_2 \in \mathbb{R}^{d_\text{model}\times d_\text{ff}}\). No biases. Canonical \(d_\text{ff} \approx \tfrac{8}{3}d_\text{model}\), rounded to a multiple of 64 (tests pass an explicit `d_ff=128` for `d_model=64`).

---

## Signature

Recommended:

```python
class SwiGLU(nn.Module):  # or PositionwiseFeedForward
    def __init__(self, d_model: int, d_ff: int, ...): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Adapter:

```python
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
```

---

## Input / Output

| Param | Shape | Role |
|-------|-------|------|
| `in_features` | `(..., d_model)` | Block residual stream |
| `w1_weight` | `(d_ff, d_model)` | Gate / SiLU branch |
| `w3_weight` | `(d_ff, d_model)` | Value branch |
| `w2_weight` | `(d_model, d_ff)` | Down-project |
| **output** | `(..., d_model)` | Same as input |

---

## Constraints

- Use your `Linear` + `silu`; **not** `nn.Linear`
- Elementwise product after SiLU on the \(W_1\) branch only
- Adapter may `load_state_dict` if keys match (`w1.weight`, …) or assign `.weight.data`
- Snapshot atol `1e-5`

---

## Examples

### Example 1 — Handout

\(W_2(\mathrm{SiLU}(W_1 x) \odot W_3 x)\).

### Example 2 — Test

Fixture `layers.0.ffn.w{1,2,3}.weight` + `in_embeddings` `(4, 12, 64)` → snapshot `test_swiglu`.

---

## Rules / Invariants

1. Output last dim is `d_model`, not `d_ff`
2. \(W_1 x\) and \(W_3 x\) share the `d_ff` axis for \(\odot\)
3. Position-wise: same FFN at every sequence position (batched matmul)

---

## Sub-problems

Pipeline: `W1x → SiLU → ⊙ W3x → W2`

### Sub-problem A — three linears

**Tools / docs**

| What | Reference |
|------|-----------|
| Your `Linear` | `docs/guidelines/linear.md` |
| Shapes | adapter docstring |

**Input:** `d_model`, `d_ff`  

**Output:** modules `w1`, `w2`, `w3`

**Goal:** Parameterize the three maps.

**Checkpoint:** `w1.weight.shape == (d_ff, d_model)` and `w2.weight.shape == (d_model, d_ff)`

### Sub-problem B — `forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| SiLU | `docs/guidelines/silu.md` |
| Handout | §3.4.2 eq. (7) |

**Input:** `x: (..., d_model)`  

**Output:** `(..., d_model)`

**Goal:** SwiGLU.

**Checkpoint:** `uv run pytest tests/test_model.py::test_swiglu -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Wrong multiply order | Snapshot fail — \(W_2\) is **last** |
| SiLU on both branches | WA |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_swiglu -q
```

---

## Complexity / Performance Targets

Three batched matmuls + elementwise; no loop over `seq`.

---

## Debug Checklist

- [ ] SiLU only on \(W_1 x\)
- [ ] \(\odot\) then \(W_2\), not \(W_2\) then \(\odot\)
- [ ] Adapter actually copies all three weights

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_swiglu` | Judge |
| Handout §3.4.2 | Spec |

---

## Wiring reminder

`run_swiglu` constructs the FFN, loads `w1/w2/w3`, returns `forward(in_features)`.
