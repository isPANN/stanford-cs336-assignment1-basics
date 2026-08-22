# Problem 008: Rotary Position Embedding (RoPE)

**Difficulty:** Medium  
**Topic:** Tensor ops / Positions  
**Points:** 2  
**Implement in:** `cs336_basics/model.py` (or `attention.py`)  
**Wire via:** `tests/adapters.py::run_rope`

---

## Description

Rotate pairs of query/key coordinates by an angle that depends on token position \(i\) and pair index \(k\). No learned parameters. Later, MHA applies this to **Q and K per head**, not to V.

Isolated test uses the full `d_model` as `d_k` (64). Inside MHA, `d_k = d_model // num_heads`.

---

## Signature

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor: ...
```

Adapter:

```python
def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
```

---

## Input / Output

| Param | Type | Meaning |
|-------|------|---------|
| `x` | `(..., seq, d_k)` | Query or key; extra leading dims OK |
| `token_positions` | `(..., seq)` | Integer positions along the sequence |
| **output** | same as `x` | Rotated vectors |

Handout: for pair \(k \in \{1,\ldots,d/2\}\),

\[
\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}
\]

Each pair \((x_{2k-1}, x_{2k})\) is rotated by the 2×2 block \(R^i_k\) (cos/sin of \(\theta_{i,k}\)). Do **not** materialize a full \(d\times d\) matrix.

---

## Constraints

- No `nn.Parameter`; optional `register_buffer(..., persistent=False)` for precomputed cos/sin of shape `(max_seq_len, d_k/2)`
- Slice buffers with `token_positions` (they need not be `0..seq-1` contiguous)
- `d_k` even
- `theta` in tests is `10000.0`
- Snapshot atol `1e-5`

---

## Examples

### Example — Test

`in_embeddings` `(4, 12, 64)`, `pos_ids = arange(12)`, `theta=10000`, `max_seq_len=12`. Snapshot `test_rope`.

---

## Rules / Invariants

1. Output shape == input shape
2. Position 0 is identity rotation (all \(\theta=0\))
3. Same rotation applied independently on every leading batch dim
4. RoPE is **not** applied to values in MHA (later)

---

## Sub-problems

Pipeline: `inv_freq → θ[pos, pair] → rotate even/odd pairs`

### Sub-problem A — frequencies

**Tools / docs**

| What | Reference |
|------|-----------|
| Inv freq | \(\Theta^{-2k/d}\) for 0-based pair index \(k\) (same as 1-based \((2k-2)/d\)) |
| Outer with positions | `einops` / broadcasting `(seq, d_k/2)` |
| Handout | §3.4.3 eq. (8)–(9) |

**Input:** `theta`, `d_k`, `max_seq_len`

**Output:** cos/sin tables

**Goal:** Precompute angles for all positions up to `max_seq_len`.

**Checkpoint:** `cos[0]` is all ones; `sin[0]` is all zeros.

### Sub-problem B — `forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| Pair rotate | even/odd (or stacked pairs) × 2×2 rotation |
| Index | `cos[token_positions]`, `sin[token_positions]` |
| Broadcast | positions `(..., seq)` vs `x` `(..., seq, d_k)` |

**Input:** `x`, `token_positions`

**Output:** rotated `x`

**Goal:** Apply \(R_i\) without a dense \(d\times d\) multiply.

**Checkpoint:** `uv run pytest tests/test_model.py::test_rope -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| `token_positions` not starting at 0 | Slice by those ids, not by `arange(seq)` |
| Extra batch dims on `x` | Rotate last two dims only |
| `seq < max_seq_len` | Fine — slice, do not require `seq == max_seq_len` |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_rope -q
```

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| Apply \(R_i\) | `(d,d) @ x` | Pairwise rotate + broadcast |

---

## Debug Checklist

- [ ] 1-based \(k\) vs 0-based: exponent is \(0, 2/d, 4/d, \ldots\)
- [ ] Pair grouping is consecutive dims `(0,1), (2,3), ...`
- [ ] Cos/sin not learned
- [ ] Adapter passes `d_k` as first positional arg (`run_rope(d_model, ...)` in the test)

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_rope` | Judge |
| Handout §3.4.3 | Spec |

---

## Wiring reminder

`run_rope` constructs `RotaryPositionalEmbedding` and calls `forward(in_query_or_key, token_positions)`.
