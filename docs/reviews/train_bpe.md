# Review: train_bpe

**Date:** 2026-08-21
**Implement in:** `cs336_basics/bpe.py`
**Wire via:** `tests/adapters.py::run_train_bpe`

## Verdict

| | |
|--|--|
| Judge | `AC` |
| Quality | 14/20 |
| Ready to move on? | No — fixture AC is not TinyStories speed; every merge still rewrites every pre-token. |

## Pytest

Command:

```bash
uv run pytest tests/test_train_bpe.py -q --tb=short
```

Result: 3/3 passed  Duration: 10.08s

`test_train_bpe` + `test_train_bpe_speed` alone: 2.50s (two full `corpus.en` / vocab 500 runs ≈ 1.2s each vs ~0.38s reference). `test_train_bpe_special_tokens` on `tinystories_sample_5M.txt` / vocab 1000 accounts for most of the 10s.

Failing tests (if any):

None

## Scorecard

| Dimension | /4 | Note |
|-----------|----|------|
| Contract | 4 | Merges match reference; specials stripped before PAT; lex-max tie-break; thin adapter |
| Complexity / scale | 2 | Pair deltas are incremental, but each merge still walks and copies all unique pre-tokens |
| Numerical / systems | 2 | Whole-file `read()`; new `counts` dict allocated every merge; zero-count pairs never deleted |
| Abstraction | 3 | Pipeline helpers match A–G; `apply_merge` mixes rewrite + stats; `__main__` self-tests in the module |
| Readability | 3 | `train_bpe` is a clear loop; `apply_merge` neighbor updates are the dense part |
| **Total** | **14/20** | AC on the fixture; Complexity 2 + Critical blocks "ready" |

## Findings

### Blocker

None

### Critical

**Critical — every merge rewrites every pre-token (no inverted index)**
- Evidence: `cs336_basics/bpe.py` `apply_merge` does `for key in counts.keys():` then rebuilds `new_token` / `new_counts[tuple(new_token)] = counts[key]` for **all** unique pre-tokens, whether or not they contain the merged pair. `train_bpe` calls this once per merge. Pair counts are updated with local deltas (good), but finding victims is still a full scan. `max(pair_counts, ...)` is also linear in the whole pair table, including keys whose count is already 0.
- Later: `test_train_bpe_speed` only times `corpus.en` / vocab 500 (budget 1.5s; this run is already ~1.2s). TinyStories (~2GB, thousands of merges) and OWT stay `O(num_merges · unique_pretokens · avg_len)`. The 5M-sample special-token test already dominates the 10s file runtime; full TinyStories is two orders of magnitude larger.
- Principle: a merge edits a local span; only pre-tokens that contain that pair change — index them, do not rescan the world.
- Direction: keep `pair_counts` and an inverted index `pair → pre-token keys that contain it`. On merge `(a,b)→ab`, rewrite only those keys, apply neighbor deltas, `del` empty / zero-count pair keys. Do not go back to `count_pairs` from scratch.

### Improve

- Drop keys with count `<= 0` from `pair_counts` after a merge, and make `select_best_pair` return `None` (and break) if nothing positive remains. Dead zeros inflate every `max()` and can invent phantom merges if you ever exhaust real pairs before `vocab_size`.
- Pre-tokenize in chunks (`find_chunk_boundaries` + `multiprocessing`) before OWT. `open(...).read()` of the whole corpus is acceptable for `corpus.en` / 5M, not for multi-GB.
- `new_counts[tuple(new_token)] = counts[key]` should be `+=` if you ever collapse two keys onto the same tuple. Unlikely while each pre-token's bytes are unique, but `=` silently drops frequency if it happens.

### Nit

- `if __name__ == "__main__"` fixtures in `bpe.py` are leftover self-checks; they do not belong in the library module once pytest is green.
- `byte_id` in `apply_merge` is a token index, not a byte index, after the first merge.

## Growth

- **Local edit → local index.** Incremental pair deltas are necessary but not sufficient; you still need a way to *find* the edited rows (same idea as sparse optimizer state, not a full param scan).
- Passing `test_train_bpe_speed` means "not toy-broken", not "production tokenizer trainer". The reference is ~0.38s; a solution sitting near 1.5s will not survive the next dataset.

## Next action

Add `pair → set of pre-token keys` and only run the `apply_merge` rewrite / neighbor updates on those keys; `del` pair keys whose count hits 0. Re-run `tests/test_train_bpe.py` to prove merges are unchanged.
