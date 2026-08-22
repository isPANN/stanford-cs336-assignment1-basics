# Problem 011: Causal Multi-Head Self-Attention (no RoPE)

**Difficulty:** Medium  
**Topic:** Attention / Rearrange  
**Points:** 5  
**Implement in:** `cs336_basics/attention.py`  
**Wire via:** `tests/adapters.py::run_multihead_self_attention`

---

## Description

Causal MHA (handout eq. 12–14): project \(x\) to Q,K,V with **three** matrices of shape `(d_model, d_model)`, split heads, run SDPA with a causal mask, concat heads, \(W_O\).

\[
d_k = d_v = d_\text{model} / h
\]

This adapter **must not** apply RoPE (`test_multihead_self_attention`). The RoPE variant is the next problem.

Weight layout (adapter docstring): `q_proj.weight` is `torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)` — heads stacked on the **output** axis of \(W_Q\).

---

## Signature

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ...): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Adapter:

```python
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
```

---

## Input / Output

| Param | Shape | Meaning |
|-------|-------|---------|
| `in_features` | `(..., seq, d_model)` | Residual stream |
| `*_proj_weight` | `(d_model, d_model)` | Packed all-head projections |
| **output** | `(..., seq, d_model)` | |

Tests: `d_model=64`, `n_heads=4` so `d_k=16`; `in_embeddings` `(4, 12, 64)`.

---

## Constraints

- Exactly three QKV matmuls (plus output); not `num_heads` separate `Linear`s
- Causal: query \(i\) attends to keys \(j \le i\) only (`torch.triu` or index compare)
- Reuse `scaled_dot_product_attention`
- Rearrange heads as a batch dim so SDPA stays vectorized
- Snapshot atol `1e-5`
- Do **not** use `nn.MultiheadAttention`

---

## Examples

### Example — Test

Fixture `layers.0.attn.{q,k,v,output}_proj.weight`. Snapshot `test_multihead_self_attention`.

---

## Rules / Invariants

1. `d_model % num_heads == 0`
2. Causal: no attention to future positions
3. QKV computed as `x @ W^T` with your `Linear` (weight `(d_model, d_model)`)

---

## Sub-problems

Pipeline:

```
x → Q,K,V  (3 matmuls)
  → split heads (..., h, seq, d_k)
  → causal SDPA
  → concat heads → W_O
```

### Sub-problem A — projections + split

**Tools / docs**

| What | Reference |
|------|-----------|
| Rearrange | `einops.rearrange(q, "... s (h d) -> ... h s d", h=num_heads)` |
| Handout | §3.4.5 “three matrix multiplies” |

**Input:** `x`, four weight matrices

**Output:** Q,K,V as `(..., h, seq, d_k)`

**Goal:** Packed multi-head projections.

**Checkpoint:** `q.shape[-1] == d_model // num_heads`

### Sub-problem B — causal mask + SDPA + output

**Tools / docs**

| What | Reference |
|------|-----------|
| Causal | `torch.arange(seq)[:, None] >= torch.arange(seq)[None, :]` → True = attend |
| SDPA | `docs/guidelines/sdpa.md` |
| Merge heads | `rearrange(..., "... h s d -> ... s (h d)")` |

**Input:** Q,K,V

**Output:** `(..., seq, d_model)`

**Goal:** Causal MHA without RoPE.

**Checkpoint:** `uv run pytest tests/test_model.py::test_multihead_self_attention -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| `seq=1` | Mask is `[[True]]` |
| Extra batch dims | Heads inserted next to seq, not flattening batch |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_multihead_self_attention -q
```

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| Heads | loop `h` | rearrange + one SDPA |
| QKV | 3H linears | 3 linears |

---

## Debug Checklist

- [ ] Causal polarity (lower-triangular True)
- [ ] Split on the **projected** `d_model` axis
- [ ] Output proj applied **after** concat
- [ ] RoPE accidentally applied → this test fails

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_multihead_self_attention` | Judge |
| Handout §3.4.5 | Spec |

---

## Wiring reminder

`run_multihead_self_attention` constructs MHA, loads the four weights, `forward(in_features)`.
