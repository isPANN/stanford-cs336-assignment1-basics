# Problem 006: SiLU

**Difficulty:** Easy  
**Topic:** Tensor ops / Activation  
**Points:** (part of SwiGLU / FFN; judged by `test_silu_matches_pytorch`)  
**Implement in:** `cs336_basics/model.py` (or `nn_utils.py`)  
**Wire via:** `tests/adapters.py::run_silu`

---

## Description

SiLU / Swish: \(\mathrm{SiLU}(x) = x \cdot \sigma(x) = x / (1+e^{-x})\). Elementwise. Used inside SwiGLU. Handout allows `torch.sigmoid` for stability. There is no standalone written problem; the adapter exists so you can unit-test the activation before wiring SwiGLU.

---

## Signature

```python
def silu(in_features: torch.Tensor) -> torch.Tensor: ...
```

Adapter:

```python
def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
```

---

## Input / Output

| | Type | Meaning |
|--|------|---------|
| Input `in_features` | any shape | Elementwise |
| Output | same shape | `x * sigmoid(x)` |

---

## Constraints

- Elementwise; preserve shape
- Match `torch.nn.functional.silu` within `atol=1e-5`
- `torch.sigmoid` is allowed (handout §3.4.2)

---

## Examples

### Example — Test tensor

```python
x = torch.tensor([
    [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
    [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
])
assert allclose(run_silu(x), F.silu(x), atol=1e-5)
```

---

## Rules / Invariants

1. `out.shape == in.shape`
2. No reduction

---

## Sub-problems

Pipeline: `sigmoid → multiply by x`

### Sub-problem A — `silu`

**Tools / docs**

| What | Reference |
|------|-----------|
| Formula | §3.4.2 eq. (5) |
| Sigmoid | `torch.sigmoid(x)` |

**Input:** `x` any shape  

**Output:** same shape  

**Goal:** Elementwise SiLU.

**Checkpoint:** `uv run pytest tests/test_model.py::test_silu_matches_pytorch -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| Negative \(x\) | Smooth, not ReLU-zero |
| Large \|x\| | Saturates like sigmoid·x |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_silu_matches_pytorch -q
```

---

## Complexity / Performance Targets

Vectorized; no Python loops.

---

## Debug Checklist

- [ ] `x * sigmoid(x)`, not `sigmoid` alone
- [ ] Not ReLU

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py::test_silu_matches_pytorch` | Judge |
| Handout §3.4.2 | Formula |

---

## Wiring reminder

`tests/adapters.py::run_silu` delegates to your function/module.
