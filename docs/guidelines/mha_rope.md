# Problem 012: Causal MHA with RoPE

**Difficulty:** Medium  
**Topic:** Attention / RoPE  
**Points:** (same MHA deliverable; extra adapter)  
**Implement in:** `cs336_basics/attention.py`  
**Wire via:** `tests/adapters.py::run_multihead_self_attention_with_rope`

---

## Description

Same as causal MHA, plus RoPE on **Q and K only**, with RoPE dimension = **head dim** `d_model // num_heads`. Head axis is a batch dim for RoPE (same rotation per head). Values are not rotated.

---

## Signature

```python
def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
```

Your MHA module can take an optional `token_positions` / RoPE module rather than duplicating the class.

---

## Input / Output

Same as MHA, plus:

| Param | Meaning |
|-------|---------|
| `max_seq_len` | RoPE buffer length (test: `n_keys=16`) |
| `theta` | `10000.0` |
| `token_positions` | Test rearranges `pos_ids` to `(1, seq)` |

`in_features` seq length is `n_queries=12`, which may be **shorter** than `max_seq_len`.

---

## Constraints

- RoPE `d_k` = `d_model // num_heads`, **not** full `d_model`
- Apply after Q/K projection and head split, before SDPA
- Causal mask still required
- Snapshot atol `1e-5`

---

## Examples

```bash
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -q
```

---

## Rules / Invariants

1. V unrotated
2. Every head sees the same RoPE angles for a given position
3. Positions come from `token_positions`, not assumed `arange(seq)` if provided

---

## Sub-problems

Pipeline: `MHA projections → split → RoPE(Q), RoPE(K) → causal SDPA → W_O`

### Sub-problem A — RoPE at head dim

**Tools / docs**

| What | Reference |
|------|-----------|
| RoPE class | `docs/guidelines/rope.md` |
| Handout | §3.4.5 “Applying RoPE” |

**Input:** Q,K `(..., h, seq, d_head)`, `token_positions`

**Output:** rotated Q,K

**Goal:** Position-encode keys/queries per head.

**Checkpoint:** `d_head == d_model // num_heads` passed into RoPE

### Sub-problem B — wire into MHA

**Tools / docs**

| What | Reference |
|------|-----------|
| Base MHA | `docs/guidelines/mha.md` |

**Input:** same as adapter

**Output:** `(..., seq, d_model)`

**Goal:** Causal MHA + RoPE.

**Checkpoint:** `uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| `token_positions is None` | Use `arange(seq)` on the sequence dim (for later LM use) |
| `seq < max_seq_len` | Do not require square `max_seq_len` inputs |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -q
```

---

## Complexity / Performance Targets

Same as MHA; RoPE is pairwise rotate, not a Python loop over heads.

---

## Debug Checklist

- [ ] RoPE on full `d_model` instead of head dim → WA
- [ ] RoPE on V
- [ ] `max_seq_len` vs actual `seq` confusion

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_multihead_self_attention_with_rope` | Judge |
| Handout §3.4.5 | Spec |

---

## Wiring reminder

Adapter constructs MHA-with-RoPE, loads four weights, passes `token_positions` into `forward`.
