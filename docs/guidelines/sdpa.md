# Problem 010: Scaled Dot-Product Attention

**Difficulty:** Medium  
**Topic:** Attention / Broadcasting  
**Points:** 5  
**Implement in:** `cs336_basics/attention.py` (or `model.py`)  
**Wire via:** `tests/adapters.py::run_scaled_dot_product_attention`

---

## Description

Core attention (Vaswani §3.2.1 / handout eq. 11):

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

Optional boolean mask: **True = attend**, **False = do not** (add \(-\infty\) before softmax). Must work for 3D `(batch, seq, d)` and 4D `(batch, heads, seq, d)` via `...` batch dims.

---

## Signature

```python
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
```

---

## Input / Output

| Tensor | Shape | Notes |
|--------|-------|-------|
| `Q` | `(..., q, d_k)` | queries |
| `K` | `(..., k, d_k)` | keys |
| `V` | `(..., k, d_v)` | values; `k` matches K |
| `mask` | `(..., q, k)` or broadcastable | optional |
| **out** | `(..., q, d_v)` | |

Tests: `q` `(4, 12, 64)`, `k`/`v` `(4, 16, 64)`, `mask` `(4, 12, 16)` random `> 0.5`. 4D test rearranges to `(2, 2, seq, d)` with `head=2`.

---

## Constraints

- Scale by \(1/\sqrt{d_k}\) where `d_k = Q.shape[-1]`
- Use **your** `softmax` on the score axis (last dim of `QK^T`)
- Mask: `scores.masked_fill(~mask, float("-inf"))` (True keeps the score)
- Vectorized einsum; no Python over batch/heads/seq
- Snapshot atol `1e-5`

---

## Examples

### Example 1 — Handout mask

`[[True, True, False]]` → one query attends only to the first two keys; those two attention weights sum to 1.

### Example 2 — Tests

```bash
uv run pytest tests/test_model.py::test_scaled_dot_product_attention -q
uv run pytest tests/test_model.py::test_4d_scaled_dot_product_attention -q
```

---

## Rules / Invariants

1. `out.shape == Q.shape[:-1] + (V.shape[-1],)`
2. Rows of attention probs (where mask is True) sum to 1
3. False mask positions have probability 0
4. `q` and `k` sequence lengths may differ (12 vs 16 in the fixture)

---

## Sub-problems

Pipeline: `scores = QK^T / sqrt(d_k) → mask → softmax → @ V`

### Sub-problem A — scores

**Tools / docs**

| What | Reference |
|------|-----------|
| Einsum | `"... q d, ... k d -> ... q k"` |
| Scale | `d_k ** -0.5` |
| Handout | §3.4.4 eq. (11) |

**Input:** `Q`, `K`

**Output:** `(..., q, k)` scores

**Goal:** Scaled pairwise similarities.

**Checkpoint:** `scores.shape[-2:] == (n_queries, n_keys)`

### Sub-problem B — mask + softmax + values

**Tools / docs**

| What | Reference |
|------|-----------|
| Mask convention | True = information flows; False → \(-\infty\) |
| Softmax | `docs/guidelines/softmax.md` |
| Weighted sum | `"... q k, ... k d_v -> ... q d_v"` |

**Input:** scores, mask, `V`

**Output:** `(..., q, d_v)`

**Goal:** Masked attention.

**Checkpoint:** both SDPA pytest names above.

---

## Edge Cases

| Case | Expected |
|------|----------|
| `mask is None` | Attend to all keys |
| 4D heads | Treat `head` as a `...` batch dim |
| `d_k != d_v` | Allowed by the math; fixtures use equal dims |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_scaled_dot_product_attention -q
uv run pytest tests/test_model.py::test_4d_scaled_dot_product_attention -q
```

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| Heads | `for h in range(H)` | batched 4D einsum |

---

## Debug Checklist

- [ ] Missing \(1/\sqrt{d_k}\) → snapshot fail
- [ ] Mask polarity inverted (0/1 multiply vs True/False)
- [ ] Softmax on the wrong axis
- [ ] Using `F.scaled_dot_product_attention` as the solution — Blocker

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py` | 3D + 4D judges |
| `tests/conftest.py` | `q,k,v,mask` shapes |
| Handout §3.4.4 | Spec |

---

## Wiring reminder

`run_scaled_dot_product_attention` is a thin wrap of your function (no extra weights).
