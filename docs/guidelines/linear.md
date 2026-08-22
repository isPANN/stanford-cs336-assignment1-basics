# Problem 003: Linear Module

**Difficulty:** Easy  
**Topic:** Tensor ops / nn.Module  
**Points:** 1  
**Implement in:** `cs336_basics/model.py`  
**Wire via:** `tests/adapters.py::run_linear`

---

## Description

A bias-free linear map \(y = Wx\) used everywhere in the Transformer (QKV, FFN, LM head). Store \(W\) as shape `(out_features, in_features)` (not \(W^\top\)). Do **not** use `nn.Linear` / `F.linear`.

---

## Signature

```python
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Adapter:

```python
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
```

---

## Input / Output

### Input (`forward`)

| Param | Type | Meaning |
|-------|------|---------|
| `x` | `(..., d_in)` | Last dim is the feature dim; any leading batch dims |

### Output

| Return | Type | Meaning |
|--------|------|---------|
| `y` | `(..., d_out)` | \(y = x W^\top\) with stored `W` of shape `(d_out, d_in)` |

---

## Constraints

- Subclass `nn.Module`; call `super().__init__()`
- Parameter is `nn.Parameter` named so `load_state_dict({"weight": weights})` works (standard PyTorch name: `weight`)
- No bias
- Init (for training later, not judged by snapshot): truncated \(\mathcal{N}(0, 2/(d_\text{in}+d_\text{out}))\) at \([-3\sigma, 3\sigma]\) via `torch.nn.init.trunc_normal_`
- Vectorized: no Python loop over batch or sequence
- Adapter: construct → load given `weights` → `forward`

---

## Examples

### Example 1 — Handout

\(y = Wx\), no bias.

### Example 2 — Test

`test_linear` takes TinyStories fixture `layers.0.ffn.w1.weight` (`d_out=d_ff=128`, `d_in=d_model=64`) and `in_embeddings` of shape `(4, 12, 64)`. Output must match `tests/_snapshots/test_linear.npz` (default snapshot atol/rtol).

---

## Rules / Invariants

1. `y.shape == x.shape[:-1] + (d_out,)`
2. `module.weight.shape == (d_out, d_in)`
3. Same dtype/device as parameters after load

---

## Sub-problems

Pipeline: `construct Linear → load_state_dict → einsum/matmul → return`

### Sub-problem A — `Linear.__init__`

**Tools / docs**

| What | Reference |
|------|-----------|
| Parameter | `nn.Parameter(torch.empty(out, in, device=..., dtype=...))` |
| Truncated normal | `torch.nn.init.trunc_normal_(tensor, mean=0, std=..., a=..., b=...)` |
| Handout | §3.3.1–3.3.2 |

**Input:** `in_features`, `out_features`, `device`, `dtype`

**Output:** module with `self.weight` shape `(out_features, in_features)`

**Goal:** Own the matrix \(W\) as a trainable parameter.

**Checkpoint:** `assert list(m.parameters())[0].shape == (d_out, d_in)`

### Sub-problem B — `Linear.forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| Matmul | `einops.einsum(x, W, "... d_in, d_out d_in -> ... d_out")` or `x @ W.T` |
| Handout | §3.3.2 equation (3) |

**Input:** `x: (..., d_in)`

**Output:** `y: (..., d_out)`

**Goal:** Apply \(Wx\) with broadcasting over leading dims.

**Checkpoint:** `uv run pytest tests/test_model.py::test_linear -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Extra leading dims | Only the last dim is `d_in` |
| `d_in != d_out` | Rectangular \(W\) (FFN `w1` is this case) |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_linear -q
```

| Test | Verifies |
|------|----------|
| `test_linear` | Snapshot vs fixture `w1` applied to random embeddings |

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| Forward | Loop batch × seq | One matmul / einsum |

---

## Debug Checklist

- [ ] Stored `W` is `(d_out, d_in)`, not `(d_in, d_out)`
- [ ] Not `nn.Linear`
- [ ] Adapter loads weights then calls `forward`
- [ ] `einops` / `@` keeps `...` batch dims

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_linear` | Judge |
| `tests/fixtures/ts_tests/model.pt` | Weight source |
| Handout §3.3.2 | Spec |

---

## Wiring reminder

`tests/adapters.py::run_linear` should construct your `Linear`, load `weights`, and return `forward` — tests never import `cs336_basics` modules directly.
