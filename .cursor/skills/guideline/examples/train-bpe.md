# Problem 001: Train BPE Tokenizer

**Difficulty:** Medium-Hard
**Topic:** String / Hash Map / Greedy / Simulation
**Points:** 15
**Implement in:** `cs336_basics/bpe.py`
**Wire via:** `tests/adapters.py::run_train_bpe`

---

## Description

Train a byte-level BPE tokenizer on a text file: initialize byte vocabulary,
pre-tokenize with GPT-2 regex, iteratively merge the most frequent adjacent byte
pairs within each pre-token, return vocab and ordered merges.

---

## Signature

```python
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
```

---

## Input / Output (top-level)

### Input

| Param | Type | Meaning |
|-------|------|---------|
| `input_path` | `str \| PathLike` | UTF-8 text corpus file |
| `vocab_size` | `int` | Target vocab size (bytes + merges + specials) |
| `special_tokens` | `list[str]` | Strings that become atomic vocab items and hard boundaries |

### Output

| Return | Type | Meaning |
|--------|------|---------|
| `vocab` | `dict[int, bytes]` | Token id → byte sequence |
| `merges` | `list[tuple[bytes, bytes]]` | Merge history in creation order |

---

## Constraints

- 256 initial byte tokens + special tokens in vocab
- Pre-tokenize with handout `PAT` using `regex` package + `finditer`
- Never count or merge pairs across pre-token boundaries
- Strip/split special tokens before pre-tokenization; they do not enter merge stats
- Tie-break equal-frequency pairs: **lexicographically greater** wins
- Stop when `len(vocab) == vocab_size`
- `merges` order must match training order exactly
- Speed: `corpus.en`, vocab 500, `< 1.5s`

---

## Examples

### Example 1 — Handout (simplified pretokenize = whitespace)

Corpus: `low×5`, `lower×2`, `widest×3`, `newest×6`

First merge among tied pairs: `('s','t')` over `('e','s')`.

First six merges: `st`, `est`, `ow`, `low`, `west`, `ne`.

### Example 2 — Test fixture

```python
train_bpe("tests/fixtures/corpus.en", vocab_size=500, special_tokens=["<|endoftext|>"])
```

`merges` must exactly equal `tests/fixtures/train-bpe-reference-merges.txt`
(decoded to bytes). Vocab key/value **sets** must match reference vocab JSON.

---

## Sub-problems

Pipeline: `init_vocab → split_specials → pretokenize_and_count → [select → merge] × N → return`

### Sub-problem A — `init_vocab`

**Tools / docs**

| What | Reference |
|------|-----------|
| Single-byte token | `bytes([i])` for `i in range(256)` |
| Special token bytes | `s.encode("utf-8")` |

**Input:** `special_tokens: list[str]`

**Output:** `vocab: dict[int, bytes]`

**Goal:** ids 0–255 → single bytes; each special token appended with a new id.

**Checkpoint:** `assert len(vocab) == 256 + len(special_tokens)`

---

### Sub-problem B — `split_by_special_tokens`

**Tools / docs**

| What | Reference |
|------|-----------|
| Regex split | `import regex as re` → `re.split(pattern, text)` |
| Escape special chars | `re.escape(s)` |
| Build pattern | `"\|".join(re.escape(t) for t in special_tokens)` |
| Handout | §2.5 "Removing special tokens before pre-tokenization" |

**Input:** `text: str`, `special_tokens: list[str]`

**Output:** `segments: list[str]`

**Goal:** Hard boundaries; specials never enter merge stats.

**Checkpoint:** `split("Doc1<|endoftext|>Doc2", [...]) == ["Doc1", "Doc2"]`

---

### Sub-problem C — `pretokenize_and_count`

**Tools / docs**

| What | Reference |
|------|-----------|
| GPT-2 pattern | `PAT = r"""'(?:[sdmt]\|ll\|ve\|re)\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+"""` |
| Iteration | `re.finditer(PAT, segment)` → `match.group()` |
| Encode | `.encode("utf-8")` → `tuple(bytes([b]) for b in encoded)` |
| Accumulate | `collections.Counter` |

**Input:** `segments: list[str]`

**Output:** `counts: dict[tuple[bytes, ...], int]`

**Goal:** Weighted byte-tuple representation of all pre-tokens.

**Checkpoint:** `re.findall(PAT, "some text that i'll pre-tokenize")` returns the 8-element handout list.

---

### Sub-problem D — `count_pairs`

**Tools / docs**

| What | Reference |
|------|-----------|
| Adjacent pairs | `zip(seq, seq[1:])` |
| Weighted count | multiply by `counts[pre_token]` |

**Input:** `counts: dict[tuple[bytes, ...], int]`

**Output:** `pair_counts: dict[tuple[bytes, bytes], int]`

**Goal:** Aggregate weighted frequency of every adjacent pair.

**Checkpoint:** Handout round-1: `pair_counts[(b'e', b's')] == 9` and `pair_counts[(b's', b't')] == 9`

---

### Sub-problem E — `select_best_pair`

**Tools / docs**

| What | Reference |
|------|-----------|
| Max with key | `max(pair_counts, key=lambda p: (pair_counts[p], p))` |
| Lex order on `bytes` | `bytes` supports `<`, `>` natively |

**Input:** `pair_counts: dict`

**Output:** `tuple[bytes, bytes] | None`

**Goal:** Max-count pair; lex max on ties.

**Checkpoint:** `select({(b'A',b'B'):5, (b'BA',b'A'):5}) == (b'BA', b'A')`

---

### Sub-problem F — `apply_merge`

**Tools / docs**

| What | Reference |
|------|-----------|
| Non-overlapping scan | advance index by 2 after a match, 1 otherwise |
| Merge bytes | `token_a + token_b` |

**Input:** `counts`, `pair: tuple[bytes, bytes]`

**Output:** `new_counts: dict[tuple[bytes, ...], int]`

**Goal:** Replace every non-overlapping `(a, b)` with `a+b` in each pre-token.

**Checkpoint:** `{(b'n',b'e',b'w',b'e',b's',b't'):6}` after `('s','t')` → `{(b'n',b'e',b'w',b'e',b'st'):6}`

---

### Sub-problem G — `train_loop`

**Tools / docs**

| What | Reference |
|------|-----------|
| Read file | `open(path, encoding="utf-8").read()` |
| Loop condition | `while len(vocab) < vocab_size` |
| Extend vocab | `vocab[next_id] = token_a + token_b` |

**Input:** `input_path`, `vocab_size`, `special_tokens`

**Output:** `vocab`, `merges`

**Goal:** Orchestrate A–F; record each merge in order.

**Checkpoint:** `uv run pytest tests/test_train_bpe.py::test_train_bpe -q`

---

### Sub-problem H — optimize (only if speed test fails)

**Tools / docs**

| What | Reference |
|------|-----------|
| Incremental update | only pairs adjacent to merged span change; skip full recount |
| Inverted index | `pair → set of pre_token keys` |
| Parallelism | `multiprocessing.Pool` + `find_chunk_boundaries` in `pretokenization_example.py` |
| Profiler | `python -m cProfile -s cumulative` or `py-spy record` |
| Handout | §2.5 "Optimizing the merging step" |

**Checkpoint:** `uv run pytest tests/test_train_bpe.py::test_train_bpe_speed -q`

---

## Acceptance Criteria

```bash
uv run pytest tests/test_train_bpe.py -q
```

---

## Debug Checklist

- [ ] `import regex as re` not stdlib `re`
- [ ] Single byte = `bytes([b])` not `int`
- [ ] Lex **max** tie-break, not min
- [ ] Special tokens split out before regex
- [ ] `adapters.py` calls `train_bpe`

---

This file is the reference **quality bar** for generated guidelines. Every
sub-problem must specify Input and Output tables (or equivalent inline form).
Do not add solution code.
