# Problem 017: AdamW Optimizer

**Difficulty:** Medium  
**Topic:** Optimizer state / In-place updates  
**Points:** 2  
**Implement in:** `cs336_basics/optimizer.py`  
**Wire via:** `tests/adapters.py::get_adamw_cls`

---

## Description

AdamW as **Algorithm 1** in the handout (Loshchilov et al. algorithm 2): decoupled weight decay \(\theta \leftarrow \theta - \alpha\lambda\theta\), then Adam moments, then moment-adjusted step. Subclass `torch.optim.Optimizer`. \(t\) starts at **1**.

The judge accepts **either** match to PyTorch `AdamW` **or** the course snapshot (`atol=1e-4`) because float order of decay vs moments can differ.

---

## Signature

```python
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None: ...

    def step(self, closure=None): ...
```

Adapter: `get_adamw_cls() -> type` returning that class.

Test constructs:

```python
opt = get_adamw_cls()(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
```

---

## Input / Output

`step` updates `p.data` in place using `p.grad`. Return `loss` if `closure` was given (API); tests do not use closure.

Handout loop (per param), \(t = 1,\ldots,T\):

1. \(g \leftarrow \nabla_\theta \ell\)
2. \(\alpha_t \leftarrow \alpha \sqrt{1-\beta_2^t}/(1-\beta_1^t)\)
3. \(\theta \leftarrow \theta - \alpha\lambda\theta\)  (decoupled decay)
4. \(m \leftarrow \beta_1 m + (1-\beta_1)g\)
5. \(v \leftarrow \beta_2 v + (1-\beta_2)g^2\)
6. \(\theta \leftarrow \theta - \alpha_t m / (\sqrt{v}+\varepsilon)\)

Keep `m`, `v`, `t` in `self.state[p]`.

---

## Constraints

- `super().__init__(params, defaults)` with a dict of hyperparameters
- Skip `p.grad is None`
- Do **not** use `torch.optim.AdamW` as the class body
- Checkpointing tests also instantiate this class — state_dict must be restorable

---

## Examples

1000 steps on `nn.Linear(3, 2, bias=False)` with a synthetic quadratic-ish loss; seed 42. Match PyTorch or snapshot.

---

## Rules / Invariants

1. Weight decay is **not** `g ← g + λθ` (that's Adam, not AdamW as specified)
2. Bias correction uses the per-parameter step count starting at 1
3. `m` and `v` same shape as `p`, init 0

---

## Sub-problems

Pipeline: `Optimizer.__init__ → step: decay → moments → update`

### Sub-problem A — `Optimizer` boilerplate

**Tools / docs**

| What | Reference |
|------|-----------|
| Example SGD | Handout §4.2.1 class `SGD` |
| State | `state = self.state[p]` |

**Input:** `params`, `lr`, `betas`, `eps`, `weight_decay`

**Output:** constructed optimizer

**Goal:** Valid `Optimizer` subclass.

**Checkpoint:** `opt.param_groups[0]["lr"] == 1e-3`

### Sub-problem B — `step`

**Tools / docs**

| What | Reference |
|------|-----------|
| Algorithm 1 | Handout §4.3 |
| In-place | `p.data.add_(...)` / `mul_` |

**Input:** grads on parameters

**Output:** updated weights + state

**Goal:** AdamW.

**Checkpoint:** `uv run pytest tests/test_optimizer.py::test_adamw -q`

---

## Edge Cases

| Case | Expected |
|------|----------|
| First step `t=1` | Bias correction \(\sqrt{1-\beta_2}/(1-\beta_1)\) |
| `grad is None` | Skip |
| Matches PyTorch | Test returns early — still AC |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_optimizer.py::test_adamw -q
```

---

## Complexity / Performance Targets

One pass over parameters per step; no copies of the full model beyond moment buffers.

---

## Debug Checklist

- [ ] `t` starting at 0 (off-by-one on \(\beta^t\))
- [ ] Decay folded into `g`
- [ ] `sqrt(v)` vs `sqrt(v)+\varepsilon` placement (handout: \(m/(\sqrt{v}+\varepsilon)\))
- [ ] Forgot to increment `t` in `state`

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_optimizer.py::test_adamw` | Judge |
| Handout §4.3 Algorithm 1 | Spec |

---

## Wiring reminder

`get_adamw_cls` returns the class, not an instance.
