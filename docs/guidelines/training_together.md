# Problem: training_together — Training Script

**Difficulty:** Medium-Hard  
**Topic:** Training loop  
**Points:** 4  
**Implement in:** a user script (e.g. `cs336_basics/train.py`) — **not** judged by pytest  
**Wire via:** N/A — **manual deliverable**

---

## Description

Glue tokenizer dumps, `TransformerLM`, AdamW, cosine LR, grad clip, `get_batch`, checkpointing, and logging into one configurable training loop (§5.3).

---

## Signature

CLI or config object is enough. Handout asks at least:

- Model + optimizer hyperparameters
- `np.memmap` train/val token arrays
- Checkpoint path
- Periodic train **and** val loss (console and/or W&B)

---

## Input / Output

| Input | memmap of `uint16` ids from `tokenizer_experiments` |
| Output | checkpoints + logged losses |

No unit test file.

---

## Constraints

- Device: `cpu` / `cuda` / `mps` consistently for model **and** batches
- Val loss: no grad, `model.eval()`; do not update optimizer
- LR from your cosine schedule each step; set optimizer `param_group["lr"]`
- Clip then `optimizer.step()`
- `total tokens ≈ B × steps × context` is a §7 concern; the script must make those knobs easy

---

## Examples

Per-step skeleton (structure only):

```
batch = get_batch(train_mmap, B, m, device)
logits = model(x)
loss = cross_entropy(logits.view(-1, vocab), y.view(-1))
loss.backward()
clip_gradients(...)
set lr
optimizer.step(); optimizer.zero_grad()
maybe val / save_checkpoint
```

---

## Rules / Invariants

1. Labels are next-token ids (`get_batch` contract)
2. Checkpoints restore **iteration** so the LR schedule can resume
3. Memmap: do not `.read()` the whole train set into RAM

---

## Sub-problems

Pipeline: `parse hparams → load mmap → loop → log/ckpt`

### Sub-problem A — configurable run

**Tools / docs**

| What | Reference |
|------|-----------|
| argparse / yaml | your choice |
| Handout | §5.3 |

**Input:** CLI

**Output:** a run that can change `d_model`, `lr`, `B`, paths without editing code

**Goal:** Experiment-ready script.

**Checkpoint:** Two runs with different `--lr` without touching source.

### Sub-problem B — loop body

**Tools / docs**

| What | Reference |
|------|-----------|
| Pieces | data_loading, cross_entropy, gradient_clipping, adamw, lr_schedule, checkpointing |
| Handout | §5.3 |

**Input:** mmap + model

**Output:** decreasing train loss on a tiny overfit batch (debug)

**Goal:** Correct backward path.

**Checkpoint:** Overfit one batch to ~0 (handout §7.2.3 tip). No pytest.

### Sub-problem C — val + ckpt + log

**Tools / docs**

| What | Reference |
|------|-----------|
| W&B optional | wandb.ai |
| Handout | §5.3 bullets 3–4 |

**Input:** val mmap, ckpt path

**Output:** files + curves vs **step** and **wall time**

**Goal:** Resume + experiment_log fodder.

**Checkpoint:** Kill and resume from checkpoint; iteration/LR continue.

---

## Edge Cases

| Case | Expected |
|------|----------|
| `mps` | same device string as batches; no TF32 `high` precision (later §7 tip) |
| Empty val | skip or error clearly |

---

## Acceptance Criteria (Judge)

**N/A.** Course staff look at the script + that you can train in §7.

Suggested local smoke (not in repo):

```bash
uv run python -m cs336_basics.train --help   # if you add a module
```

---

## Complexity / Performance Targets

Dataload/ckpt/val must not dominate GPU time (handout §7.2.2).

---

## Debug Checklist

- [ ] Train tokens on CPU, model on CUDA (silent slowness)
- [ ] Forgot `zero_grad`
- [ ] Val in train mode with dropout (this assignment’s tests removed dropout; still use eval)
- [ ] Saving full Python objects instead of `state_dict`

---

## Related Files

| File | Why |
|------|-----|
| Handout §5.3 | Spec |
| All unit-test guidelines 003–020 | Components |
| `docs/guidelines/experiment_log.md` | Logging requirement |

---

## Wiring reminder

No adapter. Do not put the training loop inside `tests/adapters.py`.
