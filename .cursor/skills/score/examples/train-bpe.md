# Example review: stub `train_bpe`

This is the expected **tone and depth**, not a live score of the user's current tree.

Input: `cs336_basics/bpe.py` returns undefined names; adapter already imports `train_bpe`.

---

# Review: train_bpe

**Date:** 2026-08-19
**Implement in:** `cs336_basics/bpe.py`
**Wire via:** `tests/adapters.py::run_train_bpe`

## Verdict

| | |
|--|--|
| Judge | `CE` |
| Quality | 0/20 |
| Ready to move on? | No — function body does not define `vocab` / `merges`; tests cannot run.

## Pytest

```bash
uv run pytest tests/test_train_bpe.py -q --tb=short
```

Result: 0/3  (`NameError: name 'vocab' is not defined` or equivalent)

## Scorecard

| Dimension | /4 | Note |
|-----------|----|------|
| Contract | 0 | Return values do not exist |
| Complexity / scale | 0 | No algorithm |
| Numerical / systems | 0 | N/A |
| Abstraction | 0 | Empty shell |
| Readability | 0 | Nothing to read |
| **Total** | **0/20** | |

## Findings

### Blocker

**Blocker — `train_bpe` is a stub**
- Evidence: `cs336_basics/bpe.py` ends with `return vocab, merges` and never assigns them
- Later: every tokenizer / LM experiment depends on this
- Principle: the judge only sees `adapters.run_train_bpe`; a stub is not a partial solution
- Direction: implement the pipeline in the guideline (`init_vocab` → split specials → pretok → merge loop). Do not put it in `adapters.py`

### Critical

None yet — there is no design to criticize. Do not invent performance commentary on empty code.

## Growth

- Wire first, then fill: `run_train_bpe` should already call `cs336_basics.bpe.train_bpe`
- Correctness on `corpus.en` is the first gate; incremental pair updates are the quality gate for TinyStories

## Next action

Implement sub-problems A–G from `docs/guidelines/train_bpe.md` until `test_train_bpe` and `test_train_bpe_special_tokens` pass. Optimize only if `test_train_bpe_speed` fails or you plan to train on TinyStories.

---

# Example review: AC but naive recount (the important case)

Input: tests pass; each merge does `pair_counts = count_pairs(all_pretokens)` from scratch.

## Verdict

| | |
|--|--|
| Judge | `AC` |
| Quality | 9/20 |
| Ready to move on? | No — fixture speed is not TinyStories speed.

## Scorecard

| Dimension | /4 | Note |
|-----------|----|------|
| Contract | 4 | Merges match reference; specials stripped |
| Complexity / scale | 1 | Global recount each merge |
| Numerical / systems | 2 | Single-process pretok is fine for `corpus.en` |
| Abstraction | 2 | One long function, but steps are visible |
| Readability | 3 | Clear loop |
| **Total** | **12/20** | AC on the fixture; Complexity 1 blocks "ready" |

Ready-to-move-on is **No**: Quality can be ≥12 and still fail if any Critical remains.

## Findings

### Blocker

None

### Critical

**Critical — full pair recount every merge**
- Evidence: `train_loop` body `for _ in range(num_merges): pair_counts = count_pairs(counts)`
- Later: TinyStories is ~2GB and thousands of merges; this is `O(num_merges · corpus_tokens)`. `test_train_bpe_speed` only times `corpus.en` / vocab 500 (~0.4s reference, 1.5s budget)
- Principle: a merge edits a local span; only pairs that touch that span change — update deltas, do not rescan the world
- Direction: keep `pair_counts` and an inverted index `pair → pre-tokens that contain it`. On merge `(a,b)→ab`, decrement old neighbors, increment new neighbors, delete empty keys

### Improve

- Pretokenization can be chunked with `find_chunk_boundaries` + `multiprocessing` before you touch OWT
- Split A–F into named helpers so the merge loop stays reviewable

## Growth

- **Local edit → local statistics.** Same idea as incremental attention caches and optimizer state
- Passing the speed unit test means "not toy-broken", not "production tokenizer trainer"

## Next action

Replace the recount with incremental pair updates; re-run `tests/test_train_bpe.py` to prove merges are unchanged.
