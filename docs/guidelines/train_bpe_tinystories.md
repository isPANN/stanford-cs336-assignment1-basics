# Problem: train_bpe_tinystories — BPE on TinyStories

**Difficulty:** Medium-Hard  
**Topic:** Systems / Profiling  
**Points:** 2  
**Implement in:** reuse `cs336_basics/bpe.py`  
**Wire via:** N/A — **manual deliverable** (uses your trained `train_bpe`)

---

## Description

Train byte-level BPE on TinyStories train split, `vocab_size=10000`, special token `<|endoftext|>`. Serialize vocab + merges. Report time, memory, longest token, and a profile of the bottleneck.

Prerequisite: `docs/guidelines/train_bpe.md` unit tests pass, including speed.

---

## Signature

Same as `train_bpe(input_path, vocab_size=10000, special_tokens=["<|endoftext|>"])`.

---

## Input / Output

| | |
|--|--|
| Input | `data/TinyStoriesV2-GPT4-train.txt` (see README wget) |
| Output | serialized `vocab` + `merges`; writeup sentences |
| Resource | ≤ 30 minutes CPU, ≤ 30 GB RAM; hint: **under 2 minutes** with multiprocessing pretok + incremental pair updates |

---

## Constraints

- Add `<|endoftext|>` to the vocab; it is a hard boundary (already required by `test_train_bpe_special_tokens`)
- Do not load the whole merge loop as a naive recount if you want to hit the time hint
- Downscale first: TinyStories **valid** (~22k docs) as a debug corpus
- Chunk at special-token boundaries (`pretokenization_example.py`)

---

## Examples

Handout hint: documents are delimited by `<|endoftext|>`; pretok parallel; merge stats never include the special token.

---

## Rules / Invariants

1. Same algorithm as unit-tested `train_bpe`
2. Longest token should be a frequent byte-string (often a word/morpheme), not a random unique sentence

---

## Sub-problems

Pipeline: `debug on valid → profile → full train → serialize → inspect longest token`

### Sub-problem A — train + serialize

**Tools / docs**

| What | Reference |
|------|-----------|
| Data | README `data/` wget |
| Parallel pretok | `cs336_basics/pretokenization_example.py` |
| Handout | §2.5 Problem (train_bpe_tinystories) (a) |

**Input:** TinyStories train path, vocab 10k

**Output:** files on disk + wall time + peak RAM

**Goal:** A reusable tokenizer artifact.

**Checkpoint:** Manual — training finishes in the resource envelope; `len(vocab)==10000`.

### Sub-problem B — longest token + sense-check

**Tools / docs**

| What | Reference |
|------|-----------|
| Max by `len(bytes)` | over `vocab.values()` |

**Input:** trained vocab

**Output:** the longest `bytes` token + 1–2 sentences whether it makes sense

**Goal:** Sanity-check BPE (usually a long frequent English piece, not a unique typo dump).

**Checkpoint:** Writeup (a) complete.

### Sub-problem C — profile

**Tools / docs**

| What | Reference |
|------|-----------|
| cProfile / py-spy | handout Low-Resource Tip |
| Handout | §2.5 (b) |

**Input:** a training run (debug set is enough if bottlenecks match)

**Output:** 1–2 sentences naming the hottest phase

**Goal:** Know whether pretok vs merge dominates.

**Checkpoint:** Writeup (b).

---

## Edge Cases

| Case | Expected |
|------|----------|
| OOM | chunked read / parallel workers; do not `read()` + copy endlessly |
| Special token in merges | Bug — should have been stripped before pair counts |

---

## Acceptance Criteria (Judge)

**N/A — no unit test.** Writeup:

- (a) time, memory, longest token + whether it makes sense (1–2 sentences)
- (b) profiler conclusion (1–2 sentences)

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| TinyStories train 10k vocab | tens of minutes / OOM | ≤ 2 min pretok-parallel + incremental merges |
| RAM | full token list copies | ≤ 30 GB |

---

## Debug Checklist

- [ ] Trained on valid only and reported that as the full experiment
- [ ] Global recount each merge (quality bar Critical even if `corpus.en` passed)
- [ ] Forgot `<|endoftext|>` in `special_tokens`

---

## Related Files

| File | Why |
|------|-----|
| `docs/guidelines/train_bpe.md` | Algorithm |
| `cs336_basics/pretokenization_example.py` | Chunk boundaries |
| Handout §2.5 | Spec |

---

## Wiring reminder

No extra adapter. Call your `train_bpe` from a script; save artifacts for `tokenizer_experiments`.
