# Problem 004: Embedding Module

**Difficulty:** Easy  
**Topic:** Tensor ops / nn.Module / Indexing  
**Points:** 1  
**Implement in:** `cs336_basics/model.py`  
**Wire via:** `tests/adapters.py::run_embedding`

---

## Description

Map integer token IDs to `d_model` vectors by indexing a `(vocab_size, d_model)` matrix. First layer of the Transformer LM. Do **not** use `nn.Embedding` / `F.embedding`.

---

## Signature

```python
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: ...

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: ...
```

Adapter:

```python
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
```

---

## Input / Output

### Input (`forward`)

| Param | Type | Meaning |
|-------|------|---------|
| `token_ids` | `(...,)` int64 | IDs in `[0, vocab_size)` |

### Output

| Return | Type | Meaning |
|--------|------|---------|
| embeddings | `(..., d_model)` | Row `token_ids[..., i]` of the matrix |

---

## Constraints

- Subclass `nn.Module`; store matrix as `nn.Parameter` with **`d_model` last**
- Init: truncated \(\mathcal{N}(0, 1)\) at \([-3, 3]\)
- Indexing, not a one-hot matmul (equivalent but wasteful)
- Adapter: construct → load `weights` → `forward`

---

## Examples

### Example 1 — Handout

`(batch, seq)` ids → `(batch, seq, d_model)` vectors.

### Example 2 — Test

Fixture `token_embeddings.weight` is `(10000, 64)`; `in_indices` is `(4, 12)` random ids. Snapshot: `tests/_snapshots/test_embedding.npz`.

---

## Rules / Invariants

1. `out.shape == token_ids.shape + (d_model,)`
2. `weight.shape == (num_embeddings, embedding_dim)`
3. Out-of-range ids are undefined (tests stay in range)

---

## Sub-problems

Pipeline: `construct Embedding → load_state_dict → index → return`

### Sub-problem A — `__init__`

**Tools / docs**

| What | Reference |
|------|-----------|
| Parameter | `nn.Parameter(torch.empty(vocab, d_model, ...))` |
| Init | `trunc_normal_(..., std=1, a=-3, b=3)` |
| Handout | §3.3.1, §3.3.3 |

**Input:** `num_embeddings`, `embedding_dim`

**Output:** `self.weight` shape `(vocab_size, d_model)`

**Goal:** Vocabulary lookup table as a parameter.

**Checkpoint:** `assert m.weight.shape[-1] == embedding_dim`

### Sub-problem B — `forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| Advanced index | `self.weight[token_ids]` |
| Handout | §3.3.3 |

**Input:** `token_ids: (...)`

**Output:** `(..., d_model)`

**Goal:** Gather embedding rows.

**Checkpoint:** `uv run pytest tests/test_model.py::test_embedding -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| 1D ids | Output `(seq, d_model)` |
| 2D batch | Output `(batch, seq, d_model)` |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_embedding -q
```

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| Lookup | one-hot @ matrix | direct index |

---

## Debug Checklist

- [ ] Not `nn.Embedding`
- [ ] Weight last dim is `d_model`
- [ ] `load_state_dict` key is `weight`

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_embedding` | Judge |
| Handout §3.3.3 | Spec |

---

## Wiring reminder

`tests/adapters.py::run_embedding` constructs your `Embedding`, loads `weights`, returns `forward`.
