# Problem: transformer_accounting — FLOPs and Parameters

**Difficulty:** Medium  
**Topic:** Resource accounting  
**Points:** 5  
**Implement in:** (calculator / writeup; optional script)  
**Wire via:** N/A — **manual deliverable**

---

## Description

Count trainable parameters and matmul FLOPs for **this assignment’s** Transformer (pre-norm, SwiGLU, RoPE, no bias), not necessarily stock GPT-2. Rule: \(A\in\mathbb{R}^{m\times n}\), \(B\in\mathbb{R}^{n\times p}\) costs **\(2mnp\)** FLOPs.

---

## Signature

No adapter. A spreadsheet or small Python counter is fine.

---

## Input / Output

GPT-2 XL-shaped **assignment** config (a):

| Hyperparam | Value |
|------------|-------|
| vocab_size | 50,257 |
| context_length | 1,024 |
| num_layers | 48 |
| d_model | 1,600 |
| num_heads | 25 |
| d_ff | 4,288 (\(\approx \tfrac83 \times 1600\), multiple of 64) |

(d) also GPT-2 small / medium / large shapes (12/768/12, 24/1024/16, 36/1280/20). (e) XL with context 16,384.

Assume float32: 4 bytes/parameter for (a) memory of **weights only**.

---

## Constraints

- Include: token embed, per-layer (RMSNorm gains, QKV, \(W_O\), SwiGLU \(W_{1,2,3}\), RMSNorm), final RMSNorm, lm_head
- RoPE has **no** parameters
- Attention scores \(QK^\top\) and \(\mathrm{attn}\,V\) **are** matmuls — include them in FLOPs
- Sequence length = `context_length` for one forward
- Do not use `nn.Linear` bias (there is none)

---

## Examples

Embedding params: `vocab_size * d_model`.  
One Linear `d_in → d_out`: `d_out * d_in` params; for a batch of \(T\) tokens, FLOPs \(2 \cdot T \cdot d_\text{in} \cdot d_\text{out}\).

---

## Rules / Invariants

1. Parameter count independent of `context_length` except you still use \(T\) in FLOPs
2. Attention FLOPs scale as \(T^2 d\) plus projections \(T d^2\)
3. (e) lengthening \(T\) grows attention share vs FFN

---

## Sub-problems

Pipeline: `list params → list matmuls → XL numbers → scale models → long context`

### Sub-problem A — parameter count + load memory

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §3.5 Problem (transformer_accounting) (a) |
| Bytes | `num_params * 4` for fp32 |

**Input:** XL config

**Output:** 1–2 sentences: #params and GiB to **load** the model

**Goal:** Weight memory only (not activations).

**Checkpoint:** Writeup (a).

### Sub-problem B — matmul list + total FLOPs

**Tools / docs**

| What | Reference |
|------|-----------|
| \(2mnp\) | handout Rule |
| Components | embed (gather, often ignored), QKV, \(QK^\top\), \(AV\), \(W_O\), SwiGLU ×3, lm_head |

**Input:** one forward, \(T=\) context_length

**Output:** list of matmuls with descriptions + total FLOPs

**Goal:** (b)

**Checkpoint:** Every SwiGLU and attention matmul appears.

### Sub-problem C — who dominates

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | (c) |

**Input:** (b) breakdown

**Output:** 1–2 sentences

**Goal:** Usually FFN + attention projections at this \(T\); say what **your** numbers show.

**Checkpoint:** Writeup (c).

### Sub-problem D — other GPT-2 sizes

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | (d) |
| d_ff | use \(\approx \tfrac83 d_\text{model}\) rounded to 64 if not specified |

**Input:** small / medium / large configs

**Output:** per-model FLOP **proportions** + 1–2 sentences on how shares move with size

**Goal:** Embeddings vs layers as \(d_\text{model}\) and depth grow.

**Checkpoint:** Writeup (d).

### Sub-problem E — context 16,384

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | (e) |
| Attention | \(QK^\top\) is \(O(T^2 d_\text{head} h)\) |

**Input:** XL widths, \(T=16384\)

**Output:** 1–2 sentences on total FLOPs and relative shares

**Goal:** See quadratic attention.

**Checkpoint:** Writeup (e).

---

## Edge Cases

| Case | Note |
|------|------|
| Weight tying | Not in the assignment LM tests; count embed + lm_head **separately** unless you state tying |
| RMSNorm | Tiny param count (`d_model` per norm); FLOPs are elementwise, not the \(2mnp\) rule |

---

## Acceptance Criteria (Judge)

**N/A.** Five written parts as in the handout.

---

## Complexity / Performance Targets

N/A (pencil-and-paper). Optional: a script that prints counts from your actual `TransformerLM`.

---

## Debug Checklist

- [ ] Counted GPT-2 **biases** / conv / dropout that this architecture does not have
- [ ] Forgot \(QK^\top\) and attn-weighted \(V\)
- [ ] Used \(d_\text{ff}=4d\) instead of SwiGLU \(4288\)
- [ ] Parameter memory included activations (that is AdamW accounting)

---

## Related Files

| File | Why |
|------|-----|
| Handout §3.5 | Spec |
| `docs/guidelines/transformer_lm.md` | What exists in the model |
| `docs/guidelines/adamw_accounting.md` | Activations / optimizer memory next |

---

## Wiring reminder

No pytest. Keep a short derivation in the writeup.
