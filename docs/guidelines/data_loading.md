# Problem 019: Language-Modeling Data Loader

**Difficulty:** Easy  
**Topic:** Sampling / Devices  
**Points:** 2  
**Implement in:** `cs336_basics/data.py`  
**Wire via:** `tests/adapters.py::run_get_batch`

---

## Description

From a 1D token array \(x = (x_1,\ldots,x_n)\), sample `batch_size` windows of length `context_length`:

- inputs: \(x_{i:i+m}\)
- labels: \(x_{i+1:i+m+1}\)

Valid start \(i\) is in \(\{0, 1, \ldots, n-m-1\}\) inclusive (`n=100`, `m=7` → starts `0..92`). Place both tensors on `device`.

---

## Signature

```python
def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
```

---

## Input / Output

| Param | Meaning |
|-------|---------|
| `dataset` | 1D numpy integer token ids (may later be `np.memmap`) |
| `batch_size` | \(B\) |
| `context_length` | \(m\) |
| `device` | `'cpu'`, `'cuda:0'`, `'mps'`, … |
| **return** | `(x, y)` each `LongTensor` shape `(B, m)` on that device |

Handout example: \(B=1,m=3\), one sample `([x2,x3,x4],[x3,x4,x5])`.

---

## Constraints

- `y` is `x` shifted by +1 in token **values** when the dataset is `np.arange(100)` — the test checks `(x+1) == y`
- Starts must be uniformly random over the valid range (χ²-style count bounds ±5σ over 1000 iters × 32 batch)
- Invalid device `cuda:99` must **raise** (`RuntimeError` or `AssertionError`)
- dtype integer (`torch.long`)
- Later training: load corpus with `np.memmap` / `mmap_mode='r'` (not judged here)

---

## Examples

```python
dataset = np.arange(0, 100)
x, y = get_batch(dataset, batch_size=32, context_length=7, device="cpu")
assert x.shape == y.shape == (32, 7)
assert torch.equal(x + 1, y)
```

---

## Rules / Invariants

1. Never sample start \(> n-m-1\)
2. Both tensors on the requested device
3. Independent samples per call (not a sequential cursor)

---

## Sub-problems

Pipeline: `sample starts → slice windows → torch tensors on device`

### Sub-problem A — valid starts

**Tools / docs**

| What | Reference |
|------|-----------|
| High | `len(dataset) - context_length` possible starts; max start is that minus 1 |
| Sample | `torch.randint(0, high, (batch_size,))` |
| Handout | §5.1 |

**Input:** `n`, `m`, `B`

**Output:** start indices shape `(B,)`

**Goal:** Uniform valid \(i\).

**Checkpoint:** `starts.max() <= n - m - 1` and `starts.min() >= 0`

### Sub-problem B — windows + device

**Tools / docs**

| What | Reference |
|------|-----------|
| Slice | `dataset[i:i+m]`, `dataset[i+1:i+1+m]` |
| Device | `.to(device)` or `torch.tensor(..., device=device)` |

**Input:** dataset, starts, device

**Output:** `(x, y)`

**Goal:** LM pairs on device.

**Checkpoint:** `uv run pytest tests/test_data.py::test_get_batch -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| `device="cuda:99"` | Raise |
| memmap dataset | Same slicing API |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_data.py::test_get_batch -q
```

---

## Complexity / Performance Targets

O(`B * m`) copies; do not load extra dataset copies. For huge corpora, only index the memmap.

---

## Debug Checklist

- [ ] `high = n - m` (then max start is `high-1`) vs off-by-one `n-m+1`
- [ ] `x` and `y` the same slice
- [ ] Forgot `.to(device)`
- [ ] Float tensors

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_data.py` | Shape, shift, uniformity, bad device |
| Handout §5.1 | Spec + memmap tip |

---

## Wiring reminder

`run_get_batch` delegates to your sampler.
