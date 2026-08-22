# Problem: learning_rate_tuning — Toy SGD LRs

**Difficulty:** Easy  
**Topic:** Optimization intuition  
**Points:** 1  
**Implement in:** (run the handout SGD snippet)  
**Wire via:** N/A — **manual deliverable**

---

## Description

The handout’s toy `SGD` on `loss = (weights**2).mean()` with `lr ∈ {1e1, 1e2, 1e3}` for **10** steps. Report whether loss decays faster, slower, or **diverges**.

This is **not** `test_get_lr_cosine_schedule`. It uses the example `SGD` class in §4.2.1.

---

## Signature

Copy the handout `SGD` + training loop; change `lr`.

---

## Input / Output

| lr | You run 10 iters and watch `loss` |
|----|-------------------------------------|
| `1e1` | write behavior |
| `1e2` | write behavior |
| `1e3` | write behavior |

Deliverable: 1–2 sentences covering all three.

---

## Constraints

- 10 iterations, not 100
- Same init recipe as the example (`5 * randn((10,10))` Parameter) — if you change the seed, say so
- Divergence = loss **increases** (or NaN)

---

## Examples

Handout loop with `opt = SGD([weights], lr=1)` printed loss each step. Repeat with the three LRs.

---

## Rules / Invariants

1. Too-large LR on a quadratic can explode
2. This does not tune the Transformer LR (that is §7 `learning_rate`)

---

## Sub-problems

Pipeline: `run three LRs → describe curves`

### Sub-problem A — run and describe

**Tools / docs**

| What | Reference |
|------|-----------|
| Example `SGD` | Handout §4.2.1 |
| Problem | §4.2.1 (learning_rate_tuning) |

**Input:** three learning rates

**Output:** 1–2 sentences

**Goal:** Feel LR sensitivity.

**Checkpoint:** Writeup submitted; numbers come from an actual run.

---

## Edge Cases

| Case | Expected |
|------|----------|
| NaN | Count as diverge |
| `lr=1` from the example | Not required in the deliverable |

---

## Acceptance Criteria (Judge)

**N/A.** One short paragraph.

---

## Complexity / Performance Targets

Seconds on CPU.

---

## Debug Checklist

- [ ] Ran 100 steps instead of 10
- [ ] Used AdamW instead of the toy SGD

---

## Related Files

| File | Why |
|------|-----|
| Handout §4.2.1 | Spec + code |
| `docs/guidelines/lr_schedule.md` | Later cosine schedule |

---

## Wiring reminder

No `adapters.py` hook.
