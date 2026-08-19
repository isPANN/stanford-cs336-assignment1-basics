# Scoring rubric

Five dimensions × 0–4 = **Quality /20**.
Judge (`AC`/`WA`/…) is **separate** and is the gate.

## Scale (every dimension)

| Score | Meaning |
|------:|---------|
| 0 | Missing, stub, or actively wrong |
| 1 | Works on the toy test by accident / brute force; will fail the next scale jump |
| 2 | Acceptable homework: correct idea, obvious waste or a fragile edge |
| 3 | Course-ready: matches handout intent, safe to reuse next week |
| 4 | You would keep this in a training run: local updates, vectorized, numerically stable, thin API |

**Pass-for-now:** Judge `AC` and Quality ≥ 12 and **no Blocker**.
**Ready-for-later-assignments:** Quality ≥ 16 and **no Critical**.

Inflating a 1 to a 3 because "it passed pytest" is a skill failure.

## Dimensions

### 1. Contract (spec fidelity)

Does the code implement the handout + adapter docstring, not a nearby blog post?

- Right types (`bytes` vs `str` vs `int` token ids; `nn.Parameter` vs raw Tensor)
- Tie-breaks, merge order, special-token boundaries, in-place vs functional
- Vocab size counts specials + bytes + merges together when required
- Adapter is a thin wrapper (construct module, load weights, call `forward`)

0: signature not met or adapter still `raise NotImplementedError`.
4: a staff member could grade from the docstring alone and this would match.

### 2. Complexity / scale

Would this survive the next dataset or the next module?

- BPE: incremental pair updates vs full recount each merge
- Tokenizer: streaming `encode_iterable` vs loading the whole file
- Attention: batched heads vs Python loop over `B, H, T`
- Data: index into a memmap / numpy array vs copying the corpus to GPU

0: quadratic-in-the-wrong-variable or unbounded memory with no awareness.
4: cost is in the term the handout asks you to optimize; extra copies are gone.

### 3. Numerical / systems hygiene

- Softmax / cross-entropy: subtract max; prefer log-sum-exp
- RMSNorm: correct eps placement; no silent float64/float32 mix that tests hide
- Matmul: `@` / `einsum` on the last dims; no `for t in range(seq)`
- Device/dtype: output follows input; no accidental `.cpu()` in the hot path
- Grad clip / AdamW: in-place where the spec says in-place; decoupled decay

0: will NaN or silently wrong on realistic activations.
4: stable and allocation-aware.

### 4. Abstraction (will you reuse this?)

CS336 is one model built from pieces. Score whether the piece is a building block.

- `nn.Module` with `forward` for layers (Linear, Embedding, RMSNorm, Attention, Block, LM)
- Weights as `Parameter`; init isolated from forward
- No `torch.nn.Linear` / `Embedding` / `MultiheadAttention` / `functional.scaled_dot_product_attention` as the implementation
- Helpers named after pipeline steps, not `helper1`
- No god-function that inlines the entire Transformer

0: one-off script inside the adapter.
4: next problem can import this class without rewriting it.

### 5. Readability for future-you

This is **clarity of control flow**, not comments.

- One obvious path for the hot loop
- Invariants visible (pair counts only inside pre-tokens; mask meaning)
- Dead code, unused kwargs, commented-out experiments: penalize
- Over-engineering (generic framework, premature multiprocessing, extra classes) also penalize

0: cannot trace a merge / a forward pass.
4: a classmate can explain the algorithm from the function names.

## Severity of findings

Use **exactly** these labels. Budget: unlimited Blocker/Critical if real; **≤3 Improve**; **≤2 Nit** and only at the end.

| Label | When | User action |
|-------|------|-------------|
| **Blocker** | Wrong under the spec, or forbidden API, or tests fail | Must fix before calling it done |
| **Critical** | Tests may pass; design will collapse at real scale, GPU, or the next problem | Treat as not done |
| **Improve** | Transferable habit; cost is real but not catastrophic yet | Fix when touching the file |
| **Nit** | Style-only | Ignore unless they asked for polish |

A finding is not Critical unless you can name **which later failure** it causes (named test, TinyStories, OWT, training step, OOM, NaN).

## Finding format (mandatory for Blocker / Critical)

```
**Critical — <one-line diagnosis>**
- Evidence: `file.py` (function / short snippet)
- Later: <specific next problem, dataset, or failure mode>
- Principle: <one sentence they can reuse all quarter>
- Direction: <what to change, 1–3 sentences, no full rewrite>
```

## Long-term coaching (2–3 bullets max)

After findings, add a **Growth** section: principles that transfer (e.g. "local update when the edit is local", "vectorize the batch axis first", "adapters never own algorithms"). No generic "keep practicing".
