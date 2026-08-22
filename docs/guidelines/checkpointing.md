# Problem 020: Checkpoint Save / Load

**Difficulty:** Easy  
**Topic:** Serialization  
**Points:** 1  
**Implement in:** `cs336_basics/checkpoint.py`  
**Wire via:** `tests/adapters.py::run_save_checkpoint` / `run_load_checkpoint`

---

## Description

Persist enough state to resume training: `model.state_dict()`, `optimizer.state_dict()`, and `iteration`. `torch.save` / `torch.load` to a path or file-like object.

Depends on a working `get_adamw_cls()` — the test trains 10 steps with your AdamW, saves, loads into a **fresh** model/optimizer, and compares weights + optimizer state + iteration.

---

## Signature

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None: ...

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int: ...
```

---

## Input / Output

| Function | Returns |
|----------|---------|
| `save_checkpoint` | `None` (writes `out`) |
| `load_checkpoint` | saved `iteration` (`int`); mutates `model` and `optimizer` |

Typical object:

```python
{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
```

Any layout is fine if load inverts it.

---

## Constraints

- Support path **and** file-like (PyTorch `torch.save(obj, out)` already does)
- `load_state_dict` on both model and optimizer
- Do not pickle the live `nn.Module` object as the only payload (restore goes into an already-constructed `new_model`)
- Test path: `tmp_path / "checkpoint.pt"`

---

## Examples

After 10 optimizer steps, `iteration=10`. Load must return `10`. Every tensor in `model.state_dict()` matches; optimizer `state` / `param_groups` match (`are_optimizers_equal`).

---

## Rules / Invariants

1. Round-trip is exact (allclose on tensors)
2. Optimizer moment buffers restore (AdamW `exp_avg` / `exp_avg_sq` or your keys)
3. Returned iteration is the value you saved, not `0`

---

## Sub-problems

Pipeline: `pack dict → torch.save` / `torch.load → load_state_dict → return t`

### Sub-problem A — save

**Tools / docs**

| What | Reference |
|------|-----------|
| `state_dict` | `model.state_dict()`, `optimizer.state_dict()` |
| Dump | `torch.save` |
| Handout | §5.2 |

**Input:** model, optimizer, iteration, `out`

**Output:** file on disk

**Goal:** Serializable blob.

**Checkpoint:** file exists and `torch.load` returns a dict with three pieces of state

### Sub-problem B — load

**Tools / docs**

| What | Reference |
|------|-----------|
| Load | `torch.load(src)` (map_location if you care about CPU/GPU later) |
| Restore | `model.load_state_dict(...)` |

**Input:** `src`, empty-ish new model & optimizer

**Output:** `iteration`

**Goal:** Resume-equivalent state.

**Checkpoint:** `uv run pytest tests/test_serialization.py::test_checkpointing -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| File-like `out` | Still works (`BytesIO`) — not in unit test but in the type signature |
| Weights-only save | Optimizer state missing → test fail |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_serialization.py::test_checkpointing -q
```

Requires AdamW AC first.

---

## Complexity / Performance Targets

O(parameter memory) write; do not also pickle Python source.

---

## Debug Checklist

- [ ] Forgot optimizer state
- [ ] Forgot iteration
- [ ] `torch.save(model)` instead of `state_dict`
- [ ] AdamW `state` keys incompatible with `load_state_dict`

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_serialization.py` | Round-trip judge |
| `get_adamw_cls` | Optimizer under test |
| Handout §5.2 | Spec |

---

## Wiring reminder

The two adapters call `save_checkpoint` / `load_checkpoint` in `cs336_basics`.
