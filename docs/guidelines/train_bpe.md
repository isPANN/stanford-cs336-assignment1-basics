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


| Param            | Type             | Meaning                                                           |
| ---------------- | ---------------- | ----------------------------------------------------------------- |
| `input_path`     | `str | PathLike` | UTF-8 text corpus file                                            |
| `vocab_size`     | `int`            | Target vocab size (bytes + merges + specials counted together)    |
| `special_tokens` | `list[str]`      | Strings that become atomic vocab items and act as hard boundaries |




### Output


| Return   | Type                        | Meaning                                       |
| -------- | --------------------------- | --------------------------------------------- |
| `vocab`  | `dict[int, bytes]`          | Token id → byte sequence; `len == vocab_size` |
| `merges` | `list[tuple[bytes, bytes]]` | `(token_a, token_b)` in creation order        |


---



## Constraints

- 256 initial byte tokens, then special tokens, then merge-produced tokens
- Pre-tokenize with handout `PAT` using `regex` (not stdlib `re`) + `finditer`
- Never count or merge pairs that cross pre-token boundaries
- Strip/split special tokens **before** pre-tokenization; they do not enter merge stats
- Tie-break equal-frequency pairs: **lexicographically greater** wins
- Stop when `len(vocab) == vocab_size`
- `merges` order must match training order exactly
- Speed constraint: `corpus.en`, vocab 500 → **< 1.5 s**

---



## Examples



### Example 1 — Handout (simplified pretokenize = whitespace)

Corpus: `low×5`, `lower×2`, `widest×3`, `newest×6`

Pair counts round 1: `{lo:7, ow:7, we:8, er:2, wi:3, id:3, de:3, es:9, st:9, ne:6, ew:6}`

Tie between `('e','s')` and `('s','t')` → pick `('s','t')` (lex max).

First six merges: `st, est, ow, low, west, ne`

### Example 2 — Test fixture

```python
train_bpe("tests/fixtures/corpus.en", vocab_size=500, special_tokens=["<|endoftext|>"])
```

`merges` must exactly equal `tests/fixtures/train-bpe-reference-merges.txt` decoded to bytes.  
Vocab key and value **sets** must match `tests/fixtures/train-bpe-reference-vocab.json`.

---



## Sub-problems

Overall pipeline:

```
init_vocab → read & split_specials → pretokenize_and_count
  → [select_best_pair → apply_merge → update vocab] × N → return
```

Each sub-problem specifies **Tools / docs**, **Input**, **Output**, **Goal**, and **Checkpoint**.
Algorithm logic is left for you to implement.

---



### Sub-problem A — `init_vocab`

**Tools / docs**


| What                   | Reference                                    |
| ---------------------- | -------------------------------------------- |
| Single-byte token      | `bytes([i])` for `i in range(256)`           |
| Special token encoding | `s.encode("utf-8")` → `bytes`                |
| Ordered dict           | plain `dict` (Python 3.7+ insertion-ordered) |


**Input**


| Name             | Type        | Meaning                  |
| ---------------- | ----------- | ------------------------ |
| `special_tokens` | `list[str]` | e.g. `["<|endoftext|>"]` |


**Output**


| Name    | Type               | Meaning                                                             |
| ------- | ------------------ | ------------------------------------------------------------------- |
| `vocab` | `dict[int, bytes]` | ids 0–255 → `bytes([i])`; each special token appended with a new id |


**Goal:** Build the starting vocabulary before any merges.

**Checkpoint:**

```python
assert len(vocab) == 256 + len(special_tokens)
assert vocab[0] == bytes([0]) and vocab[255] == bytes([255])
assert b"<|endoftext|>" in vocab.values()
```

---



### Sub-problem B — `split_by_special_tokens`

**Tools / docs**


| What                 | Reference                                              |
| -------------------- | ------------------------------------------------------ |
| Regex split          | `import regex as re` → `re.split(pattern, text)`       |
| Escape special chars | `re.escape(s)` — needed because `                      |
| Build pattern        | `"|".join(re.escape(t) for t in special_tokens)`       |
| Handout              | §2.5 "Removing special tokens before pre-tokenization" |


**Input**


| Name             | Type        | Meaning                                           |
| ---------------- | ----------- | ------------------------------------------------- |
| `text`           | `str`       | Raw decoded corpus text (whole file or one chunk) |
| `special_tokens` | `list[str]` | Boundary strings                                  |


**Output**


| Name       | Type        | Meaning                                                          |
| ---------- | ----------- | ---------------------------------------------------------------- |
| `segments` | `list[str]` | Text pieces between special tokens; empty strings may be dropped |


**Goal:** Ensure special tokens act as hard segmentation boundaries and never contribute to pair counts.

**Checkpoint:**

```python
result = split_by_special_tokens("Doc1<|endoftext|>Doc2", ["<|endoftext|>"])
assert result == ["Doc1", "Doc2"]
assert all("<|endoftext|>" not in s for s in result)
```

---



### Sub-problem C — `pretokenize_and_count`

**Tools / docs**


| What               | Reference                                                                                             |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| GPT-2 regex        | `PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""` (handout §2.4) |
| Regex iteration    | `re.finditer(PAT, segment)` → `match.group()` per match                                               |
| String → bytes     | `match.group().encode("utf-8")`                                                                       |
| Bytes → byte tuple | `tuple(bytes([b]) for b in encoded)`                                                                  |
| Accumulate counts  | `collections.Counter` or `dict` with `d[key] = d.get(key, 0) + freq`                                  |
| Handout note       | Use `finditer` not `findall` — avoid materialising all matches upfront                                |


**Input**


| Name       | Type        | Meaning          |
| ---------- | ----------- | ---------------- |
| `segments` | `list[str]` | Output of step B |


**Output**


| Name     | Type                           | Meaning                                              |
| -------- | ------------------------------ | ---------------------------------------------------- |
| `counts` | `dict[tuple[bytes, ...], int]` | Pre-token (tuple of single-byte `bytes`) → frequency |


**Goal:** Convert text into weighted byte-sequences for pair statistics.

**Checkpoint:**

```python
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
assert re.findall(PAT, "some text that i'll pre-tokenize") == \
    ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

---



### Sub-problem D — `count_pairs`

**Tools / docs**


| What           | Reference                                              |
| -------------- | ------------------------------------------------------ |
| Adjacent pairs | `zip(seq, seq[1:])` over each pre-token tuple          |
| Weighted count | multiply each pair's occurrence by `counts[pre_token]` |
| Accumulate     | `collections.Counter` or `dict`                        |


**Input**


| Name     | Type                           | Meaning                         |
| -------- | ------------------------------ | ------------------------------- |
| `counts` | `dict[tuple[bytes, ...], int]` | Weighted pre-tokens from step C |


**Output**


| Name          | Type                             | Meaning                                                        |
| ------------- | -------------------------------- | -------------------------------------------------------------- |
| `pair_counts` | `dict[tuple[bytes, bytes], int]` | Adjacent pair → total weighted frequency across all pre-tokens |


**Goal:** Know how often every adjacent byte-token pair appears across the whole corpus.

**Checkpoint (handout round-1 on stylized corpus):**

```python
assert pair_counts[(b'e', b's')] == 9
assert pair_counts[(b's', b't')] == 9
assert pair_counts[(b'l', b'o')] == 7
```

---



### Sub-problem E — `select_best_pair`

**Tools / docs**


| What                   | Reference                                             |
| ---------------------- | ----------------------------------------------------- |
| Max by count then lex  | `max(pair_counts, key=lambda p: (pair_counts[p], p))` |
| `bytes` comparison     | `bytes` supports `<` `>` natively (lexicographic)     |
| Handout tie-break rule | §2.4 "preferring the lexicographically greater pair"  |
| Verify lex order       | `max([(b"A",b"B"), (b"BA",b"A")])` → `(b'BA', b'A')`  |


**Input**


| Name          | Type                             | Meaning                  |
| ------------- | -------------------------------- | ------------------------ |
| `pair_counts` | `dict[tuple[bytes, bytes], int]` | Current pair frequencies |


**Output**


| Name        | Type                         | Meaning                                                  |
| ----------- | ---------------------------- | -------------------------------------------------------- |
| `best_pair` | `tuple[bytes, bytes] | None` | Max-count pair, lex-max on ties; `None` if dict is empty |


**Goal:** Deterministically choose the next merge target.

**Checkpoint:**

```python
pairs = {(b'A', b'B'): 5, (b'BA', b'A'): 5, (b'A', b'C'): 5}
assert select_best_pair(pairs) == (b'BA', b'A')
```

---



### Sub-problem F — `apply_merge`

**Tools / docs**


| What                  | Reference                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| Non-overlapping scan  | iterate with index; when `seq[i]==a and seq[i+1]==b` emit `a+b` and advance by 2, else advance by 1 |
| `bytes` concatenation | `token_a + token_b` produces the merged `bytes` object                                              |
| Rebuild pre-token     | build a new `list`, then `tuple(result)`                                                            |
| Rebuild `counts`      | new `dict`; key is rebuilt tuple, value is same frequency                                           |


**Input**


| Name     | Type                           | Meaning                       |
| -------- | ------------------------------ | ----------------------------- |
| `counts` | `dict[tuple[bytes, ...], int]` | Current pre-token multiset    |
| `pair`   | `tuple[bytes, bytes]`          | `(token_a, token_b)` to merge |


**Output**


| Name         | Type                           | Meaning                                                                                    |
| ------------ | ------------------------------ | ------------------------------------------------------------------------------------------ |
| `new_counts` | `dict[tuple[bytes, ...], int]` | Same structure; every non-overlapping `(token_a, token_b)` replaced by `token_a + token_b` |


**Goal:** Apply one BPE merge across all pre-tokens without crossing boundaries.

**Checkpoint (handout after merge** `('s','t')`**):**

```python
before = {(b'n', b'e', b'w', b'e', b's', b't'): 6}
after = apply_merge(before, (b's', b't'))
assert after == {(b'n', b'e', b'w', b'e', b's' + b't'): 6}
```

---



### Sub-problem G — `train_loop`

**Tools / docs**


| What           | Reference                                          |
| -------------- | -------------------------------------------------- |
| Read file      | `open(input_path, encoding="utf-8").read()`        |
| Loop condition | `while len(vocab) < vocab_size`                    |
| Extend vocab   | `vocab[next_id] = token_a + token_b; next_id += 1` |
| Record merge   | `merges.append((token_a, token_b))`                |
| Early exit     | if `select_best_pair` returns `None`, break        |


**Input**


| Name             | Type        | Meaning                   |
| ---------------- | ----------- | ------------------------- |
| `input_path`     | `PathLike`  | Corpus file               |
| `vocab_size`     | `int`       | Hard stop target          |
| `special_tokens` | `list[str]` | Passed through to A and B |


**Output**


| Name     | Type                        | Meaning                                |
| -------- | --------------------------- | -------------------------------------- |
| `vocab`  | `dict[int, bytes]`          | Final vocabulary                       |
| `merges` | `list[tuple[bytes, bytes]]` | One entry per merge, in creation order |


**Goal:** Orchestrate A–F into the full training loop.

**Checkpoint:**

```bash
uv run pytest tests/test_train_bpe.py::test_train_bpe -q
uv run pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -q
```

---



### Sub-problem H — optimize (only if speed test fails)

**Tools / docs**


| What                    | Reference                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| Incremental pair update | Only pairs **adjacent to the merged span** change — skip full recount                              |
| Inverted index          | `pair → set of pre_token keys` to locate affected pre-tokens quickly                               |
| Parallel pretokenize    | `multiprocessing.Pool` + `find_chunk_boundaries` in `cs336_basics/pretokenization_example.py`      |
| Profiling               | `python -m cProfile -s cumulative script.py` or `py-spy record -o profile.svg -- python script.py` |
| Handout                 | §2.5 "Optimizing the merging step" and "Parallelizing pre-tokenization"                            |


**Input / Output:** same as G.

**Checkpoint:**

```bash
uv run pytest tests/test_train_bpe.py::test_train_bpe_speed -q
# must complete corpus.en vocab=500 in < 1.5 s
```

---



## Acceptance Criteria

```bash
uv run pytest tests/test_train_bpe.py -q
```


| Test                            | Verifies                                                 |
| ------------------------------- | -------------------------------------------------------- |
| `test_train_bpe`                | Exact merge order + vocab key/value sets match reference |
| `test_train_bpe_special_tokens` | No `b"<|"` inside any non-special vocab entry            |
| `test_train_bpe_speed`          | Wall time < 1.5 s on `corpus.en`                         |


---



## Debug Checklist

- [ ] `import regex as re` not `import re`
- [ ] Single byte is `bytes([b])`, not `int`
- [ ] Tie-break is lex **max**, not min or first-seen
- [ ] Special tokens split out **before** regex runs
- [ ] Pair counting stays **inside** each pre-token — never across boundaries
- [ ] `merges` appended in creation order, not sorted afterwards
- [ ] `tests/adapters.py::run_train_bpe` calls your `train_bpe`

---



## Related Files


| File                                            | Why read it                                 |
| ----------------------------------------------- | ------------------------------------------- |
| `tests/test_train_bpe.py`                       | Judge — read to understand exact assertions |
| `tests/fixtures/corpus.en`                      | Small integration input (1016 lines)        |
| `tests/fixtures/train-bpe-reference-merges.txt` | Expected merge order                        |
| `tests/fixtures/train-bpe-reference-vocab.json` | Expected vocab sets                         |
| Handout §2.4–2.5                                | Full spec + optimization hints              |
| `cs336_basics/pretokenization_example.py`       | Chunk-boundary helper for parallelism       |


