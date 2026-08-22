# Problem 013: Pre-Norm Transformer Block

**Difficulty:** Medium  
**Topic:** Residual / Composition  
**Points:** 3  
**Implement in:** `cs336_basics/transformer.py` (or `model.py`)  
**Wire via:** `tests/adapters.py::run_transformer_block`

---

## Description

One pre-norm block (Figure 2 / eqs. 15 and the FFN half):

\[
y = x + \mathrm{MHA}_{\text{RoPE}}(\mathrm{RMSNorm}(x))
\]
\[
z = y + \mathrm{SwiGLU}(\mathrm{RMSNorm}(y))
\]

Clean residual stream (no norm on the skip). Uses RoPE MHA.

---

## Signature

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        ...
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

Adapter loads a **stripped** state dict (keys without `layers.0.` prefix). See `run_transformer_block` docstring for exact keys:

`attn.q_proj.weight`, `attn.k_proj.weight`, `attn.v_proj.weight`, `attn.output_proj.weight`, `ln1.weight`, `ffn.w1.weight`, `ffn.w2.weight`, `ffn.w3.weight`, `ln2.weight`.

---

## Input / Output

| Param | Shape |
|-------|-------|
| `in_features` | `(batch, seq, d_model)` |
| **output** | same |

Tests: `d_model=64`, `n_heads=4`, `d_ff=128`, `max_seq_len=n_keys=16`, `theta=10000`.

---

## Constraints

- Pre-norm, not post-norm
- Residuals **add**, they do not replace
- RoPE on
- `load_state_dict` key names must match the fixture (or remap in the adapter only)
- Snapshot atol `1e-4`

---

## Examples

Test copies `layers.0.*` keys, strips the prefix, calls `run_transformer_block`.

---

## Rules / Invariants

1. `ln1` before attention; `ln2` before FFN
2. Two residual adds
3. Positions default to `arange(seq)` if the adapter does not pass them

---

## Sub-problems

Pipeline: `x → ln1 → MHA+RoPE → +x → ln2 → SwiGLU → +`

### Sub-problem A — submodules

**Tools / docs**

| What | Reference |
|------|-----------|
| RMSNorm, MHA+RoPE, SwiGLU | previous guidelines |
| Handout | §3.4, §3.5 eq. (15) |

**Input:** constructor hyperparameters

**Output:** `ln1`, `attn`, `ln2`, `ffn`

**Goal:** Wire the four pieces.

**Checkpoint:** parameter names include `ln1.weight`, `attn.q_proj.weight`, `ffn.w1.weight`

### Sub-problem B — `forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| Pre-norm residual | \(x + f(\mathrm{Norm}(x))\) |

**Input:** `x`

**Output:** `x` after both sub-layers

**Goal:** One Transformer block.

**Checkpoint:** `uv run pytest tests/test_model.py::test_transformer_block -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Post-norm by mistake | Snapshot fail |
| Missing residual | Snapshot fail |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_transformer_block -q
```

---

## Complexity / Performance Targets

No Python over `seq` or `heads`.

---

## Debug Checklist

- [ ] Key remap: adapter dict uses `attn.q_proj.weight` not `layers.0.attn...`
- [ ] Your `Linear` parameter name is `weight` so nested `load_state_dict` works
- [ ] FFN module names `w1`,`w2`,`w3` vs `ffn.w1`

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_transformer_block` | Judge |
| `tests/adapters.py::run_transformer_block` | Full key list |
| Handout §3.5 | Spec |

---

## Wiring reminder

Construct block with the given hyperparameters, `load_state_dict(weights)`, `forward(in_features)` (supply positions if your API needs them).
