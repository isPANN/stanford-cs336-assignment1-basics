# Problem 002: BPE Tokenizer Encode / Decode

**Difficulty:** Medium-Hard  
**Topic:** String / Hash Map / Greedy / Streaming  
**Points:** 15  
**Implement in:** `cs336_basics/tokenizer.py`  
**Wire via:** `tests/adapters.py::get_tokenizer`

---

## Description

Load a trained byte-level BPE vocabulary and merge list, then convert text ↔ token IDs. Encoding reuses the same GPT-2 pre-tokenizer as training, then applies merges **in creation order** inside each pre-token. Decoding concatenates vocab byte strings and UTF-8-decodes them. Special tokens stay atomic. `encode_iterable` must stream so a multi-GB corpus never sits in memory.

This is the runtime counterpart of Problem 001 (`train_bpe`). Training produced `(vocab, merges)`; this class consumes them.

---

## Signature

Factory used by the judge:

```python
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
```

Recommended class (handout §2.6):

```python
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None: ...

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer: ...

    def encode(self, text: str) -> list[int]: ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: ...

    def decode(self, ids: list[int]) -> str: ...
```

`get_tokenizer` should return an instance of this class. Tests only call `.encode`, `.decode`, and `.encode_iterable`.

---

## Input / Output (top-level)

### Input (`__init__` / `get_tokenizer`)

| Param            | Type                          | Meaning                                                                 |
| ---------------- | ----------------------------- | ----------------------------------------------------------------------- |
| `vocab`          | `dict[int, bytes]`            | Token id → byte sequence (same object `train_bpe` returned)             |
| `merges`         | `list[tuple[bytes, bytes]]`   | `(token_a, token_b)` in **creation order**; index 0 is the first merge  |
| `special_tokens` | `list[str] \| None`           | Strings that must never be split; append to vocab if missing            |

The judge loads GPT-2 fixtures, **converts the GPT-2 unicode remapping back to raw `bytes`**, then calls `get_tokenizer`. Your class works on raw bytes. You do **not** need `gpt2_bytes_to_unicode` inside `Tokenizer`.

### Output

| Method              | Type              | Meaning                                                                 |
| ------------------- | ----------------- | ----------------------------------------------------------------------- |
| `encode`            | `list[int]`       | Token ids for the whole string                                          |
| `encode_iterable`   | `Iterator[int]`   | Same ids as `encode("".join(iterable))`, yielded lazily                 |
| `decode`            | `str`             | Concatenated vocab bytes, UTF-8 decoded                                 |
| `from_files`        | `Tokenizer`       | Constructed from serialized vocab + merges (handout; no unit test)      |

---

## Constraints

- Same pre-tokenizer as training: handout `PAT` via `import regex as re` + `finditer`
- Never merge across pre-token boundaries
- Apply merges in **training / creation order**, not by live frequency
- Special tokens are atomic: do not run `PAT` or BPE on them; map each to a single id
- If a special token's UTF-8 bytes are not already a vocab value, **append** it with `id = len(vocab)`
- Overlapping specials: **longest match wins**
- `decode`: `bytes.decode("utf-8", errors="replace")` — invalid UTF-8 → U+FFFD (`�`)
- `encode_iterable` must be a generator; extra memory is O(chunk), not O(file)
- Linux memory test: `encode_iterable` on `tinystories_sample_5M.txt` under **1 MB** RSS headroom
- `Tokenizer.encode` is **not** required to be memory-tight (`test_encode_memory_usage` is `xfail`)
- Must match `tiktoken.get_encoding("gpt2")` on the GPT-2 vocab/merges fixtures (when specials are passed, tiktoken uses `allowed_special={"<|endoftext|>"}`)
- Do not implement by wrapping `tiktoken` / HuggingFace

---

## Examples

### Example 1 — Handout (`bpe_encoding`)

Input string: `'the cat ate'`

Vocabulary:

```
0: b' '   1: b'a'   2: b'c'   3: b'e'   4: b'h'   5: b't'
6: b'th'  7: b' c'  8: b' a'  9: b'the' 10: b' at'
```

Merges (creation order):

```
(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')
```

Pre-tokens: `['the', ' cat', ' ate']`

| Pre-token | Bytes after merges        | Ids        |
| --------- | ------------------------- | ---------- |
| `'the'`   | `[b'the']`                | `[9]`      |
| `' cat'`  | `[b' c', b'a', b't']`     | `[7, 1, 5]`|
| `' ate'`  | `[b' at', b'e']`          | `[10, 3]`  |

Final `encode` result: `[9, 7, 1, 5, 10, 3]`

Walk-through for `'the'`: start `[b't', b'h', b'e']` → first applicable merge `(b't', b'h')` → `[b'th', b'e']` → next applicable `(b'th', b'e')` → `[b'the']`.

### Example 2 — Round-trip / tiktoken fixtures

```python
tok = get_tokenizer(gpt2_vocab_bytes, gpt2_merges_bytes)  # loaded by the test helper
assert tok.encode("") == []
assert tok.encode("s")  # equals tiktoken gpt2
assert tok.decode(tok.encode("🙃")) == "🙃"
assert tok.decode(tok.encode("Hello, how are you?")) == "Hello, how are you?"
```

Per-token decode of `"Hello, how are you?"` must be:

```python
["Hello", ",", " how", " are", " you", "?"]
```

### Example 3 — Overlapping special tokens

```python
tok = get_tokenizer(
    vocab, merges,
    special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
)
ids = tok.encode("Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>")
pieces = [tok.decode([i]) for i in ids]
assert pieces.count("<|endoftext|>") == 1
assert pieces.count("<|endoftext|><|endoftext|>") == 1
assert tok.decode(ids) == "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
```

---

## Rules / Invariants

1. `decode(encode(text)) == text` for the Unicode strings in the tests (empty, ASCII, combining marks, emoji, German, TinyStories, address).
2. `encode(text)` with GPT-2 fixtures equals `tiktoken.get_encoding("gpt2").encode(...)` (with `allowed_special` when `<|endoftext|>` is present).
3. Merges never apply across a `PAT` pre-token boundary or across a special-token boundary.
4. Merge application order is the list order from training, **not** "most frequent pair in this string".
5. A special token is exactly one id; `PAT` must not see it.
6. When two specials overlap, the **longer** string is one token.
7. Consecutive specials with nothing between them produce consecutive special ids (empty segments are dropped, not encoded).
8. `encode_iterable(f)` on a text file handle yields the same id sequence as `encode(f.read())`, without materializing the whole file.
9. `decode` never raises on malformed UTF-8; it substitutes U+FFFD.
10. Inverse maps (`bytes → id`, `pair → rank`) are derived once in `__init__`, not rebuilt per `encode` call.

---

## Sub-problems

Overall pipeline:

```
__init__: invert vocab, rank merges, register specials
encode:
    split_keeping_specials → (special → id | pretok → bytes → apply_merges → ids)
decode:
    ids → concat vocab bytes → utf-8 (errors='replace')
encode_iterable:
    for each chunk: yield from encode(chunk)   # without joining the whole stream
```

Each sub-problem specifies **Tools / docs**, **Input**, **Output**, **Goal**, and **Checkpoint**. Algorithm logic is left for you to implement.

You may reuse `PAT` from `cs336_basics/bpe.py`. Do **not** reuse training `split_by_special_tokens` unchanged: training **drops** specials, encoding must **keep** them.

---

### Sub-problem A — `__init__` indexes

**Tools / docs**

| What                         | Reference                                                                 |
| ---------------------------- | ------------------------------------------------------------------------- |
| Invert vocab                 | `{token_bytes: token_id for token_id, token_bytes in vocab.items()}`      |
| Merge rank                   | `{pair: i for i, pair in enumerate(merges)}` — smaller `i` = earlier merge |
| Special → bytes              | `s.encode("utf-8")`                                                       |
| Append missing special       | `vocab[len(vocab)] = special_bytes` (copy or mutate; tests already append, handout still requires this) |
| Handout                      | §2.6 "Implementing the tokenizer"                                         |

**Input**

| Name             | Type                        | Meaning              |
| ---------------- | --------------------------- | -------------------- |
| `vocab`          | `dict[int, bytes]`          | id → bytes           |
| `merges`         | `list[tuple[bytes, bytes]]` | creation-ordered     |
| `special_tokens` | `list[str] \| None`         | may be `None` or `[]`|

**Output**

| Name              | Type                             | Meaning                                      |
| ----------------- | -------------------------------- | -------------------------------------------- |
| `self.vocab`      | `dict[int, bytes]`               | possibly extended with new specials          |
| `self.byte_to_id` | `dict[bytes, int]`               | inverse map used by encode                   |
| `self.ranks`      | `dict[tuple[bytes, bytes], int]` | pair → merge rank                            |
| `self.specials`   | `list[str]`                      | empty list if `None` was passed              |

**Goal:** Precompute everything `encode`/`decode` need so each call is a lookup + merge loop, not a rebuild.

**Checkpoint:**

```python
tok = Tokenizer({0: b"a", 1: b"b"}, [(b"a", b"b")], special_tokens=["<s>"])
assert b"<s>" in tok.vocab.values()
assert tok.ranks[(b"a", b"b")] == 0
```

---

### Sub-problem B — `split_keeping_specials`

**Tools / docs**

| What              | Reference                                                                 |
| ----------------- | ------------------------------------------------------------------------- |
| Escape            | `regex.escape(s)`                                                         |
| Longest first     | `sorted(special_tokens, key=len, reverse=True)` before joining with `\|`  |
| Keep delimiters   | capturing group: `re.split(f"({pattern})", text)` or `re.finditer`        |
| Drop empties      | consecutive specials produce `""` pieces — skip them                      |
| No specials       | if the list is empty, the whole `text` is one ordinary segment            |
| Handout           | §2.6.1 "Special tokens"                                                   |
| Contrast training | `bpe.split_by_special_tokens` discards the delimiters; encode must not    |

**Input**

| Name             | Type        | Meaning                         |
| ---------------- | ----------- | ------------------------------- |
| `text`           | `str`       | Full string passed to `encode`  |
| `special_tokens` | `list[str]` | User-defined atomic strings     |

**Output**

| Name     | Type                         | Meaning                                                                 |
| -------- | ---------------------------- | ----------------------------------------------------------------------- |
| `pieces` | `list[tuple[str, bool]]`     | `(surface, is_special)` in left-to-right order, empties omitted         |

(The pair type is a suggestion; any structure that distinguishes special vs ordinary is fine.)

**Goal:** Carve specials out as atoms **before** `PAT`, preferring the longest special when they overlap.

**Checkpoint:**

```python
text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
specials = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
pieces = split_keeping_specials(text, specials)
surfaces = [p[0] if isinstance(p, tuple) else p for p in pieces]
assert "<|endoftext|><|endoftext|>" in surfaces
assert surfaces.count("<|endoftext|>") == 1
```

---

### Sub-problem C — `pretokenize_bytes`

**Tools / docs**

| What            | Reference                                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| GPT-2 regex     | `PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""` (same as §2.4) |
| Iteration       | `re.finditer(PAT, segment)` → `match.group()`                                                         |
| String → bytes  | `match.group().encode("utf-8")`                                                                       |
| Byte tokens     | `tuple(bytes([b]) for b in encoded)` — same representation as training                                |
| Package         | `import regex as re`, not stdlib `re` (`\p{L}` is not in stdlib)                                      |

**Input**

| Name      | Type  | Meaning                                      |
| --------- | ----- | -------------------------------------------- |
| `segment` | `str` | One **non-special** piece from step B        |

**Output**

| Name         | Type                         | Meaning                                      |
| ------------ | ---------------------------- | -------------------------------------------- |
| `pretokens`  | `list[tuple[bytes, ...]]`    | One tuple of single-byte `bytes` per match   |

**Goal:** Get the same pre-token byte sequences training would have produced for that substring.

**Checkpoint:**

```python
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
assert re.findall(PAT, "the cat ate") == ["the", " cat", " ate"]
assert re.findall(PAT, "Hello, how are you?") == [
    "Hello", ",", " how", " are", " you", "?"
]
```

---

### Sub-problem D — `apply_merges`

**Tools / docs**

| What                 | Reference                                                                 |
| -------------------- | ------------------------------------------------------------------------- |
| Handout algorithm    | §2.6.1 Step 2: walk the merge **list in order**; apply each if present    |
| Equivalent (faster)  | repeatedly merge the adjacent pair whose **rank is smallest**             |
| Rank lookup          | `self.ranks.get((left, right))` — missing pair is not mergeable           |
| Concatenate          | `left + right` is the merged `bytes` token                                |
| Stop                 | no remaining adjacent pair is in `ranks`                                  |
| Single-symbol token  | length-1 sequence is already done                                         |

Naive "for each of 50k GPT-2 merges, rescan the pre-token" will pass tiny tests and crawl on real data. Rank-based search on the **current** sequence is the intended approach.

**Input**

| Name       | Type                             | Meaning                         |
| ---------- | -------------------------------- | ------------------------------- |
| `seq`      | `tuple[bytes, ...] \| list`      | Single-byte tokens of one pretok |
| `ranks`    | `dict[tuple[bytes, bytes], int]` | pair → creation index           |

**Output**

| Name         | Type               | Meaning                                      |
| ------------ | ------------------ | -------------------------------------------- |
| `merged_seq` | `list[bytes]`      | Tokens after all applicable merges           |

**Goal:** Reproduce training-time BPE on one pre-token so the pieces are vocab keys.

**Checkpoint (handout `'the'`):**

```python
ranks = {
    (b"t", b"h"): 0,
    (b" ", b"c"): 1,
    (b" ", b"a"): 2,
    (b"th", b"e"): 3,
    (b" a", b"t"): 4,
}
assert apply_merges((b"t", b"h", b"e"), ranks) == [b"the"]
assert apply_merges((b" ", b"c", b"a", b"t"), ranks) == [b" c", b"a", b"t"]
assert apply_merges((b" ", b"a", b"t", b"e"), ranks) == [b" at", b"e"]
```

---

### Sub-problem E — `encode`

**Tools / docs**

| What            | Reference                                          |
| --------------- | -------------------------------------------------- |
| Map to ids      | `self.byte_to_id[token_bytes]` after merges        |
| Special path    | skip C/D; `self.byte_to_id[special.encode("utf-8")]` |
| Concatenate     | extend one `list[int]` in left-to-right piece order |
| Empty string    | no pieces → `[]`                                   |

**Input**

| Name   | Type  | Meaning        |
| ------ | ----- | -------------- |
| `text` | `str` | Arbitrary text |

**Output**

| Name  | Type        | Meaning        |
| ----- | ----------- | -------------- |
| `ids` | `list[int]` | Token id sequence |

**Goal:** Orchestrate B–D into the public `encode` method.

**Checkpoint:**

```bash
uv run pytest tests/test_tokenizer.py::test_empty_matches_tiktoken -q
uv run pytest tests/test_tokenizer.py::test_single_character_matches_tiktoken -q
uv run pytest tests/test_tokenizer.py::test_ascii_string_matches_tiktoken -q
```

---

### Sub-problem F — `decode`

**Tools / docs**

| What              | Reference                                                                 |
| ----------------- | ------------------------------------------------------------------------- |
| Lookup            | `self.vocab[token_id]` → `bytes`                                          |
| Concatenate       | `b"".join(...)`                                                           |
| UTF-8             | `.decode("utf-8", errors="replace")`                                      |
| Replacement char  | U+FFFD; handout §2.6.2 and CHANGELOG hint on invalid UTF-8                |
| Unknown id        | spec does not define it; tests only pass ids that exist in GPT-2 vocab    |

**Input**

| Name  | Type        | Meaning              |
| ----- | ----------- | -------------------- |
| `ids` | `list[int]` | Token ids to decode  |

**Output**

| Name   | Type  | Meaning                         |
| ------ | ----- | ------------------------------- |
| `text` | `str` | Unicode string, never raises    |

**Goal:** Invert `encode` for valid sequences; degrade gracefully on malformed bytes.

**Checkpoint:**

```bash
uv run pytest tests/test_tokenizer.py::test_roundtrip_unicode_string -q
uv run pytest tests/test_tokenizer.py::test_roundtrip_single_unicode_character -q
```

Local check for the replacement marker:

```python
# bytes [0x80] is invalid as UTF-8 start; after errors='replace' you get '\ufffd'
assert "\ufffd" in Tokenizer({0: bytes([0x80])}, []).decode([0])
```

---

### Sub-problem G — special-token encode cases

**Tools / docs**

| What                    | Reference                                                         |
| ----------------------- | ----------------------------------------------------------------- |
| Adjacent specials       | `Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>`     |
| Trailing newlines       | `tests/fixtures/special_token_trailing_newlines.txt`              |
| Double newline + text   | `tests/fixtures/special_token_double_newlines_non_whitespace.txt` |
| tiktoken reference      | `encode(..., allowed_special={"<|endoftext|>"})`                  |

Whitespace after a special is a **separate** `PAT` pre-token (`\n`, `\n\n`, …), not part of the special.

**Input / Output:** same as `encode`.

**Goal:** Specials stay one id even when repeated, overlapping, or sitting next to newlines.

**Checkpoint:**

```bash
uv run pytest tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken -q
uv run pytest tests/test_tokenizer.py::test_overlapping_special_tokens -q
uv run pytest tests/test_tokenizer.py::test_encode_special_token_trailing_newlines -q
uv run pytest tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace -q
```

---

### Sub-problem H — `encode_iterable`

**Tools / docs**

| What                 | Reference                                                                 |
| -------------------- | ------------------------------------------------------------------------- |
| Signature            | `Iterable[str] → Iterator[int]` (a generator, not a list)                 |
| File handle          | iterating `open(path)` yields **lines** (including `\n`)                  |
| Equivalence          | yielded ids == `encode(full_file_text)`                                   |
| Memory limit         | `@memory_limit(int(1e6))` around the generator; Linux only                |
| Forbidden            | `text = "".join(iterable)` then `encode(text)` — blows the 1 MB cap       |
| Handout              | §2.6.1 "Memory considerations" — tokens must not silently cross chunks    |

For the provided tests, encoding **each line** (or each yielded string) independently matches `encode` of the whole TinyStories sample, because `PAT` does not glue letters across `\n`. Joining the stream first is what the memory test is designed to catch.

**Input**

| Name        | Type             | Meaning                                      |
| ----------- | ---------------- | -------------------------------------------- |
| `iterable`  | `Iterable[str]`  | e.g. a text file object                      |

**Output**

| Name     | Type            | Meaning                         |
| -------- | --------------- | ------------------------------- |
| `ids`    | `Iterator[int]` | Lazy stream of token ids        |

**Goal:** Tokenize data larger than RAM with constant extra memory.

**Checkpoint:**

```bash
uv run pytest tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken -q
uv run pytest tests/test_tokenizer.py::test_encode_iterable_memory_usage -q
```

`test_encode_memory_usage` is marked `xfail` — `encode` is allowed to read the whole 5 MB string.

---

### Sub-problem I — `from_files` (handout-only)

**Tools / docs**

| What         | Reference                                                                 |
| ------------ | ------------------------------------------------------------------------- |
| Classmethod  | `@classmethod`; `Tokenizer.from_files(...)` not `instance.from_files`     |
| Format       | "same format that your BPE training code output" (handout §2.6)           |
| Tests        | **none** — the judge never calls this; still required for later experiments |

Serialize however you dumped `train_bpe`'s `(vocab, merges)` (JSON, pickle, two files, …). GPT-2 fixture JSON uses a different (unicode-remapped) layout; do not assume `from_files` must parse `gpt2_vocab.json`.

**Checkpoint:** no pytest. Manual: dump vocab/merges from `train_bpe`, reload via `from_files`, `encode` a short string, round-trip.

---

### Sub-problem J — adapter + GPT-2 integration

**Tools / docs**

| What            | Reference                                      |
| --------------- | ---------------------------------------------- |
| Adapter         | `tests/adapters.py::get_tokenizer`             |
| Test helper     | `get_tokenizer_from_vocab_merges_path` already converts GPT-2 files → raw `bytes` |
| Fixtures        | `tests/fixtures/gpt2_vocab.json`, `gpt2_merges.txt` |

**Input / Output:** `get_tokenizer(...)` returns your `Tokenizer`.

**Goal:** Tests import only the adapter, never `cs336_basics.tokenizer`.

**Checkpoint:**

```bash
uv run pytest tests/test_tokenizer.py::test_tinystories_matches_tiktoken -q
uv run pytest tests/test_tokenizer.py::test_german_matches_tiktoken -q
uv run pytest tests/test_tokenizer.py::test_address_matches_tiktoken -q
```

---

## Edge Cases

| Case | Expected behavior |
|------|-------------------|
| `text == ""` | `encode` → `[]`; `decode([])` → `""` |
| Single ASCII char `"s"` | One (or more) ids that tiktoken GPT-2 also emits; round-trip |
| Emoji `"🙃"` | Multi-byte UTF-8; ids match tiktoken; round-trip |
| `"Héllò hôw are ü? 🙃"` | Exact tiktoken id sequence |
| Special only at EOF / with trailing `\n` | Special is one id; newline is ordinary whitespace pretok |
| `<\|endoftext\|><\|endoftext\|>` with both lengths registered | One double-token, not two singles |
| Same string with only the short special registered | Two short specials |
| `special_tokens=None` | No special handling; `<\|endoftext\|>` would be BPE-split if present in text (GPT-2 vocab still has the token as a normal vocab entry — tests that omit `special_tokens` also omit the substring, except overlapping/special tests which pass the list) |
| Invalid UTF-8 from `decode` | Replacement character, no exception |
| `encode_iterable` on a file | Same ids as `encode(full_text)`; generator; < 1 MB extra |

---

## Acceptance Criteria (Judge)

```bash
# smoke
uv run pytest tests/test_tokenizer.py::test_empty_matches_tiktoken -q

# specials
uv run pytest tests/test_tokenizer.py::test_overlapping_special_tokens -q

# streaming
uv run pytest tests/test_tokenizer.py::test_encode_iterable_memory_usage -q

# full module
uv run pytest tests/test_tokenizer.py -q
```

| Test | What it verifies |
|------|------------------|
| `test_roundtrip_*` | `decode(encode(x)) == x` |
| `test_*_matches_tiktoken` | Exact id list vs `tiktoken` GPT-2 |
| `test_ascii_string_matches_tiktoken` | Per-id decode is `Hello` / `,` / ` how` / … (id equality assert is commented out) |
| `test_overlapping_special_tokens` | Longest special wins |
| `test_encode_special_token_trailing_newlines` | Special + trailing newlines match tiktoken |
| `test_encode_special_token_double_newline_non_whitespace` | Special, blank line, then `testing!` |
| `test_address_*` / `test_german_*` / `test_tinystories_*` | Longer Unicode / English / TinyStories sample |
| `test_encode_iterable_*` | Streaming ids equal `encode` / tiktoken |
| `test_encode_iterable_memory_usage` | Linux `RLIMIT_AS` + 1 MB; must **not** slurp the 5 MB file |
| `test_encode_memory_usage` | `xfail` — ignore |

---

## Complexity / Performance Targets

| Phase | Naive | Target | Notes |
|-------|-------|--------|-------|
| Apply merges on one pretok | Scan all `\|merges\|` (~50k) each time | Repeatedly merge min-rank pair in the current sequence | Quality bar: O(n · \|vocab\|) byte-scan is Critical on OWT |
| `encode` of `tinystories_sample.txt` (~4 KB) | Instant | Instant | First correctness gate |
| `encode_iterable` of 5 MB fixture | Join-then-encode uses ≫ 1 MB | Generator, ~line buffer | Judge memory test |
| Later OWT / TinyStories train set | Whole-file `encode` | Stream with `encode_iterable` | No unit test; needed in §2.7 |

---

## Debug Checklist

- [ ] `import regex as re`, same `PAT` as `train_bpe`
- [ ] Training split **drops** specials; encode split **keeps** them
- [ ] Overlapping specials: sort by **length descending** before building the regex
- [ ] Capturing split / `finditer` so delimiters are not thrown away
- [ ] Empty pieces between consecutive specials are skipped
- [ ] Merges applied in **list order** (rank), not by frequency in the query string
- [ ] After merges, lookup `bytes → id`; tokens are `bytes`, not `str`
- [ ] Inverse vocab built from the **post-append** vocab (new specials included)
- [ ] `decode(..., errors="replace")`, not `"strict"`
- [ ] `encode_iterable` is `yield` / `yield from`, not `return list(...)`
- [ ] Do not `"".join(iterable)` inside `encode_iterable`
- [ ] `get_tokenizer` returns `Tokenizer(...)`; logic lives in `cs336_basics/tokenizer.py`, not inlined in the adapter
- [ ] Do not wrap `tiktoken` to “pass the tests”

---

## Related Files

| File | Why read it |
|------|-------------|
| `tests/test_tokenizer.py` | Judge — exact strings, tiktoken compare, memory limits |
| `tests/adapters.py::get_tokenizer` | Factory signature to implement |
| `tests/common.py::gpt2_bytes_to_unicode` | Used **by tests** to decode GPT-2 files; not required in your class |
| `tests/fixtures/gpt2_vocab.json` / `gpt2_merges.txt` | GPT-2 reference tokenizer (unicode-mapped; tests convert to bytes) |
| `tests/fixtures/tinystories_sample.txt` | Integration + iterable round-trip |
| `tests/fixtures/tinystories_sample_5M.txt` | 5 MB streaming / memory |
| `tests/fixtures/special_token_*.txt` | Newline-adjacent specials |
| `tests/fixtures/address.txt`, `german.txt` | Longer Unicode round-trip |
| Handout §2.6 | Spec + worked `'the cat ate'` example |
| `cs336_basics/bpe.py` | `PAT` and pretok helpers to reuse carefully |
| `docs/guidelines/train_bpe.md` | Training-side contract this class consumes |

---

## Wiring reminder

`tests/adapters.py::get_tokenizer` should delegate to `cs336_basics.tokenizer.Tokenizer` — tests never import implementation modules directly.
