# Problem: decoding — Sampling from the LM

**Difficulty:** Medium  
**Topic:** Sampling  
**Points:** 3  
**Implement in:** e.g. `cs336_basics/decoding.py`  
**Wire via:** N/A — **manual deliverable** (no unit test)

---

## Description

Autoregressive decode: prompt → logits at last position → (temperature) softmax → optional nucleus(top-\(p\)) → sample until `<|endoftext|>` or max tokens (§6).

---

## Signature

Recommended:

```python
def decode(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> str: ...
```

---

## Input / Output

| Param | Meaning |
|-------|---------|
| prompt | \(x_{1:t}\) as text |
| max_new_tokens | hard stop |
| temperature \(\tau\) | \(v_i/\tau\) before softmax; \(\tau\to 0\) → argmax |
| top_p | smallest set \(V(p)\) with mass \(\ge p\); renormalize |

Output: completed string (prompt + generated), stopping at EOS.

Handout eqs. (21)–(24).

---

## Constraints

- Use **last** time step logits `(batch, vocab)` from `TransformerLM`
- Temperature applied **before** softmax
- Nucleus: sort probs descending, cut when cumulative \(\ge p\), zero the rest, renormalize
- Respect `context_length` (crop the left of the window if needed)
- `temperature=0` should be defined (argmax) even though math divides by \(\tau\)

---

## Examples

TinyStories fluent sample is in §7.2.3; low-resource 40M-token sample is worse but English-like. Decoder parameters (temp, \(p\)) are part of `generate` later.

---

## Rules / Invariants

1. EOS from your **special token** id, not a guessed string split
2. Sampling is sequential; each new id is appended to the context
3. Do not softmax the full `(seq, vocab)` then take `argmax` over sequence

---

## Sub-problems

Pipeline: `encode prompt → loop: forward → last logits → temp → top-p → sample → decode ids`

### Sub-problem A — one-step next-token

**Tools / docs**

| What | Reference |
|------|-----------|
| Last logits | `logits[:, -1, :]` |
| Softmax | your `softmax` |
| Handout | §6 eqs. (21)–(22) |

**Input:** token ids `(1, t)`

**Output:** distribution `(vocab,)`

**Goal:** \(P(x_{t+1} \mid x_{1:t})\).

**Checkpoint:** Manual — greedy decode of a trained TinyStories ckpt produces English-like words.

### Sub-problem B — temperature

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | eq. (23) |

**Input:** logits, \(\tau\)

**Output:** scaled softmax

**Goal:** Sharpen (\(\tau<1\)) or flatten (\(\tau>1\)).

**Checkpoint:** \(\tau\to 0\) matches argmax on a fixed logit vector.

### Sub-problem C — nucleus

**Tools / docs**

| What | Reference |
|------|-----------|
| Sort | `torch.sort(probs, descending=True)` |
| Handout | eq. (24), Holtzman et al. 2020 |

**Input:** probs, \(p\)

**Output:** truncated renormalized dist

**Goal:** Drop the long tail.

**Checkpoint:** With \(p=1\), identical to original dist (up to ties).

### Sub-problem D — generate loop

**Tools / docs**

| What | Reference |
|------|-----------|
| Stop | EOS id or `max_new_tokens` |
| Handout | Problem (decoding) bullets |

**Input:** prompt string

**Output:** completion text

**Goal:** User-facing decoder.

**Checkpoint:** Writeup later in `generate`; function itself has no pytest.

---

## Edge Cases

| Case | Expected |
|------|----------|
| Empty prompt | BOS/EOS-only or empty ids — define something sane |
| Context overflow | keep the last `context_length` tokens |
| `top_p=0` | undefined; avoid or treat as greedy |

---

## Acceptance Criteria (Judge)

**N/A.** Implement the four bullets in §6. Fluency is graded under §7 `generate`.

---

## Complexity / Performance Targets

Each step is one forward of current length; KV cache is optional (not required).

---

## Debug Checklist

- [ ] Sampled from position 0 logits
- [ ] Temperature after softmax
- [ ] Forgot to renormalize after masking nucleus
- [ ] Infinite loop without EOS/max

---

## Related Files

| File | Why |
|------|-----|
| Handout §6 | Spec |
| `docs/guidelines/transformer_lm.md` | Logits |
| `docs/guidelines/tokenizer.md` | encode/decode |
| `docs/guidelines/experiments.md` | `generate` problem |

---

## Wiring reminder

No adapter. Keep decoding out of `tests/`.
