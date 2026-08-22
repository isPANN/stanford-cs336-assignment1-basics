# Problem: adamw_accounting — Training Memory and MFU

**Difficulty:** Medium  
**Topic:** Memory / MFU  
**Points:** 2  
**Implement in:** (writeup)  
**Wire via:** N/A — **manual deliverable**

---

## Description

Peak memory and FLOPs of **one AdamW training step** in fp32: parameters, activations (listed components only), gradients, optimizer state. Then GPT-2 XL numbers, max batch on 80 GB, step FLOPs, and hours to 400k steps at 50% MFU on one H100 (495 TFLOP/s TF32). Backward = **2×** forward FLOPs (Kaplan / Hoffmann as cited).

---

## Signature

Algebra in terms of `batch_size` \(B\) and model hparams. `d_ff = 8/3 d_model`.

---

## Input / Output

Activation list to include (handout):

- Per Transformer block: RMSNorm(s); MHA: QKV, \(QK^\top\), softmax, weighted \(V\), output proj; SwiGLU: \(W_1,W_2\), SiLU, product, \(W_3\)
- Final RMSNorm, output embedding (lm_head), cross-entropy on logits

(a) algebraic expressions + total.  
(b) \(a\cdot B + b\) for XL, max \(B\) on 80 GB.  
(c) FLOPs of one AdamW **step** (optimizer arithmetic vs forward/backward — state what you include).  
(d) wall-clock hours.

---

## Constraints

- float32 everywhere
- AdamW state: typically **2 extra tensors per parameter** (`m` and `v`) plus params + grads
- Peak memory is not “params only”
- H100 peak 495e12 FLOP/s; observed = 0.5 × peak for (d)

---

## Examples

Params memory: `4 * P` bytes. Grads: another `4 * P` if stored full. AdamW `m,v`: `8 * P`. Activations scale with \(B \times T \times \cdots\).

---

## Rules / Invariants

1. (b) is affine in `batch_size` if activations are linear in \(B\)
2. MFU uses **model** FLOPs (fwd+bwd), not Python overhead
3. \(t\) in AdamW does not change the big-O of a step

---

## Sub-problems

Pipeline: `bytes(params, acts, grads, adam) → plug XL → max B → FLOPs/step → hours`

### Sub-problem A — peak memory decomposition

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §4.3 Problem (adamw_accounting) (a) |
| Activation list | exactly the bullet list in the PDF |

**Input:** symbolic hparams + \(B\)

**Output:** four expressions + total

**Goal:** Peak RAM model.

**Checkpoint:** Writeup (a) has four named terms.

### Sub-problem B — XL 80 GB

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | (b) |
| XL shape | same as transformer_accounting |

**Input:** (a) with XL numbers

**Output:** \(aB+b\) and max integer batch

**Goal:** Memory-limited \(B\).

**Checkpoint:** \(aB+b \le 80\times 2^{30}\) (or \(80\times 10^9\) if you state SI GB — pick one and be consistent).

### Sub-problem C — FLOPs per step

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | (c) |
| Fwd matmuls | transformer_accounting |

**Input:** architecture

**Output:** algebraic FLOPs + brief justification

**Goal:** Usually ~3× forward matmul FLOPs if bwd=2×fwd, plus tiny AdamW \(O(P)\).

**Checkpoint:** Writeup (c).

### Sub-problem D — 400k steps, B=1024, 50% MFU

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | (d) |
| H100 | 495 TFLOP/s |
| Tokens | \(400000 \times 1024 \times T\) if you reason in tokens; or use step FLOPs × 400k |

**Input:** (c), 50% MFU

**Output:** hours + justification

**Goal:** Wall-clock estimate.

**Checkpoint:** Writeup (d).

---

## Edge Cases

| Case | Note |
|------|------|
| Activation checkpointing | Not assumed unless you say so |
| Mixed precision | Out of scope (fp32) |

---

## Acceptance Criteria (Judge)

**N/A.** Four written parts.

---

## Complexity / Performance Targets

N/A.

---

## Debug Checklist

- [ ] Forgot optimizer state (2× params)
- [ ] Used 1× backward instead of 2×
- [ ] MFU applied to peak FLOP/s without the 50% factor
- [ ] 80 GB vs 80 GiB mix-up without stating units

---

## Related Files

| File | Why |
|------|-----|
| Handout §4.3 | Spec |
| `docs/guidelines/transformer_accounting.md` | Forward FLOPs |
| `docs/guidelines/adamw.md` | What state exists |

---

## Wiring reminder

No adapter. This is homework accounting, not a training run.
