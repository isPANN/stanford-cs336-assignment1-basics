# Problem: experiments — TinyStories, Ablations, OWT, Leaderboard

**Difficulty:** Hard  
**Topic:** Training / Ablations / GPU  
**Points:** see sub-problems (handout §7)  
**Implement in:** training script + architecture flags  
**Wire via:** N/A — **manual deliverable** (no unit tests)

---

## Description

Train the ~17M TinyStories default, tune LR/batch, generate text, ablate RMSNorm / pre-norm / RoPE / SwiGLU, then OWT and an optional 45-minute B200 leaderboard. **Judge = writeup + curves**, not pytest.

Low-resource: 40M tokens, val loss target 2.00 (CPU/MPS tips in §7.2.3). Full: ~327.68M tokens, val loss ≤ **1.45**, ~20–30 min on 1 B200 if the implementation is efficient.

---

## Signature

Reuse `training_together` + `decoding`. Architecture flags for ablations (remove RMSNorm, post-norm, NoPE, SiLU FFN with \(d_\text{ff}=4d_\text{model}\)).

---

## Input / Output

Default TinyStories hparams (§7.2.1):

| Knob | Default |
|------|---------|
| vocab | 10,000 |
| context | 256 |
| d_model | 512 |
| d_ff | 1344 |
| layers / heads | 4 / 16 |
| RoPE \(\Theta\) | 10,000 |
| tokens | \(B \times \text{steps} \times \text{context} \approx 327{,}680{,}000\) |

Tune: LR, warmup, AdamW \(\beta,\varepsilon,\lambda\).

---

## Constraints

- Log vs **steps and wall-clock** (`experiment_log`)
- Leaderboard: ≤ 45 min B200, **only** provided OWT train data, beat naive val loss **5.0**
- Online / limited GPU: continue ablations on TinyStories (handout)
- `torch.compile` tips: CPU default; MPS `backend="aot_eager"`; do **not** TF32-high on MPS

---

## Examples

Fluent TinyStories generation and the weaker 40M-token sample are in the PDF §7.2.3.

---

## Rules / Invariants

1. Ablations change **one** thing vs the base run when possible
2. SiLU FFN uses \(d_\text{ff}=4d\) to match SwiGLU param count
3. Post-norm: \(\mathrm{Norm}(x+\mathrm{sublayer}(x))\), not pre-norm
4. NoPE: causal mask only; no RoPE

---

## Sub-problems

Pipeline: `base train → LR sweep → batch sweep → generate → ablations → OWT → leaderboard`

### Sub-problem A — `learning_rate` (3 points, 2 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §7.2.3 Problem (learning_rate) |
| Cosine end | decay reaches \(\alpha_{\min}\) at step \(N\) |

**Input:** base architecture, several \(\alpha_{\max}\)

**Output:** learning curves; search strategy; a model with TS val loss ≤ 1.45 (or 2.00 low-resource); at least one **divergent** LR vs “edge of stability”

**Goal:** Find a stable-fast LR.

**Checkpoint:** Manual curves + val loss number.

### Sub-problem B — `batch_size_experiment` (1 point, 1 B200 hr)

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | Problem (batch_size_experiment) |

**Input:** \(B\) from 1 to GPU memory limit, including 64 and 128

**Output:** curves (re-tune LR if needed) + a few sentences

**Goal:** Throughput vs optimization noise.

**Checkpoint:** Writeup.

### Sub-problem C — `generate` (1 point)

**Tools / docs**

| What | Reference |
|------|-----------|
| Decoder | `docs/guidelines/decoding.md` |
| Handout | Problem (generate) |

**Input:** trained ckpt

**Output:** ≥ 256 tokens or until EOS; comment on fluency + **two factors** (temp, top-p, train tokens, val loss, …)

**Goal:** Show the LM speaks.

**Checkpoint:** Text dump in the writeup.

### Sub-problem D — `layer_norm_ablation` (1 point, 0.5 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | Problem (layer_norm_ablation) |

**Input:** no RMSNorm (blocks + final)

**Output:** curve at old LR; curve at best lower LR; commentary

**Goal:** See stability role of RMSNorm.

**Checkpoint:** Writeup.

### Sub-problem E — `pre_norm_ablation` (1 point, 0.5 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | eqs. (25)–(28) |

**Input:** post-norm block

**Output:** post- vs pre-norm curves

**Goal:** Norm **placement**.

**Checkpoint:** Writeup.

### Sub-problem F — `no_pos_emb` (1 point, 0.5 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | Problem (no_pos_emb) |

**Input:** MHA without RoPE, still causal

**Output:** RoPE vs NoPE curves

**Goal:** How much explicit position is worth.

**Checkpoint:** Writeup.

### Sub-problem G — `swiglu_ablation` (1 point, 0.5 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| SiLU FFN | \(W_2\mathrm{SiLU}(W_1 x)\), \(d_\text{ff}=4d_\text{model}\) |
| Handout | Problem (swiglu_ablation) |

**Input:** matched-param FFN

**Output:** SwiGLU vs SiLU curves + sentences

**Goal:** Gating vs extra width.

**Checkpoint:** Writeup.

### Sub-problem H — `main_experiment` OWT (2 points, 2 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §7.4 Problem (main_experiment) |

**Input:** same architecture **and** same step count as TinyStories; retune LR/B if needed

**Output:** OWT learning curve; loss comparison vs TS; generated text + why fluency is worse

**Goal:** Same compute, harder data.

**Checkpoint:** Writeup.

### Sub-problem I — `leaderboard` (6 points, 10 B200 hrs)

**Tools / docs**

| What | Reference |
|------|-----------|
| Rules | ≤ 45 min B200; OWT train only |
| Ideas | Llama/Qwen; NanoGPT speedrun; weight tying |
| Submit | github.com/stanford-cs336/assignment1-basics-leaderboard |
| Handout | §7.5 |

**Input:** your modified trainer

**Output:** final val loss, wall-clock curve **< 45 min**, description; beat 5.0

**Goal:** Minimize OWT val loss in 0.75 B200-hours (45 min).

**Checkpoint:** Leaderboard submission + writeup.

---

## Edge Cases

| Case | Expected |
|------|----------|
| Runtime ≫ 30 min on B200 for the default TS run | Profile dataloader/ckpt/val; check batching |
| MPS NaNs with TF32 high | Disable that path |
| Ablation diverges | Still log it |

---

## Acceptance Criteria (Judge)

**N/A — no `tests/test_experiments.py`.** Staff grade writeup, curves, generations, leaderboard.

Unit tests that must already be green before believing curves:

```bash
uv run pytest tests/test_model.py tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py -q
```

---

## Complexity / Performance Targets

| Run | Resource (handout) |
|-----|---------------------|
| Default TS | ~20–30 min / 1 B200; 327.68M tokens; val ≤ 1.45 |
| Low-resource | 40M tokens; val ≤ 2.00; ~36 min MPS / ~82 min CPU (staff solution) |
| Leaderboard | 45 min cap |

---

## Debug Checklist

- [ ] Val loss computed on train tokens
- [ ] Context length 16 from unit-test fixtures left in the trainer
- [ ] Ablation also changed LR/batch accidentally
- [ ] Leaderboard used extra data

---

## Related Files

| File | Why |
|------|-----|
| Handout §7 | Spec |
| `docs/guidelines/training_together.md` | Loop |
| `docs/guidelines/decoding.md` | Sampler |
| `docs/guidelines/experiment_log.md` | Curves |
| README `data/` | TinyStories / OWT downloads |

---

## Wiring reminder

No adapters. Keep experiment flags in **your** code, not in `tests/`.
