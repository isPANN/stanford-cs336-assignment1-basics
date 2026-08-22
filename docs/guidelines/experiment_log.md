# Problem: experiment_log — Tracking Runs

**Difficulty:** Easy  
**Topic:** Experiment hygiene  
**Points:** 3  
**Implement in:** logging in the training script + a writeup document  
**Wire via:** N/A — **manual deliverable**

---

## Description

Infrastructure so every §7 run has loss vs **gradient step** and vs **wall-clock**, plus a human log of what you tried (§7.1).

---

## Signature

No function contract. W&B, CSV, JSONL, or TensorBoard are all fine.

---

## Input / Output

| Deliverable | Content |
|-------------|---------|
| Logging code | records train/val loss, step, wall time, hparams |
| Experiment log | document of runs for §7 problems |

---

## Constraints

- Val loss must be **periodic**, not only final
- Wall-clock axis required (leaderboard later is 45 min)
- One run = one named config (seed, lr, B, architecture flags)

---

## Examples

Log columns: `step, wall_s, split, loss, lr, run_id`.

---

## Rules / Invariants

1. Curves for ablations must be comparable (same tokens or same wall time — state which)
2. Divergent runs still belong in the log

---

## Sub-problems

Pipeline: `instrument train loop → dump curves → keep a lab notebook`

### Sub-problem A — instrument

**Tools / docs**

| What | Reference |
|------|-----------|
| `time.perf_counter` | wall clock |
| W&B optional | wandb.ai |
| Handout | §7.1 Problem (experiment_log) |

**Input:** training loop

**Output:** time-series logs

**Goal:** Replot without rerunning.

**Checkpoint:** After a 20-step smoke run, you can plot loss vs step and vs seconds.

### Sub-problem B — written log

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §7.1 “document of all the things you tried” |

**Input:** later §7 experiments

**Output:** a markdown/PDF log

**Goal:** Staff can see search path, not only the best curve.

**Checkpoint:** Each §7 problem has an entry (hparams, outcome, link to curve).

---

## Edge Cases

| Case | Expected |
|------|----------|
| Crash mid-run | still keep partial logs |
| Resume from ckpt | wall time continues or is annotated |

---

## Acceptance Criteria (Judge)

**N/A.** Logging code + experiment document submitted with §7.

---

## Complexity / Performance Targets

Logging must not stall the GPU (async / infrequent val).

---

## Debug Checklist

- [ ] Only printed final loss
- [ ] Step index reset after checkpoint without noting it
- [ ] No wall-clock

---

## Related Files

| File | Why |
|------|-----|
| Handout §7.1 | Spec |
| `docs/guidelines/training_together.md` | Where to hook |
| `docs/guidelines/experiments.md` | Consumes the log |

---

## Wiring reminder

No pytest.
