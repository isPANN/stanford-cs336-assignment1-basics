# Problem: unicode2 — Unicode Encodings

**Difficulty:** Easy  
**Topic:** UTF-8 / bytes  
**Points:** 3  
**Implement in:** (none — written answers)  
**Wire via:** N/A — **manual deliverable**

---

## Description

Why BPE trains on **UTF-8 bytes** (0–255), not UTF-16/32 or raw code points. Handout §2.2.

---

## Signature

No adapter. Use `.encode` / `.decode` / `list(bytes)`.

---

## Input / Output

Written answers for (a)–(c).

---

## Constraints

- Compare encodings on several strings (ASCII, CJK, emoji)
- (b) needs a **counterexample** for the per-byte `decode` function
- (c) a 2-byte sequence that is not valid UTF-8

---

## Examples

```python
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
list(utf8_encoded)  # ints 0–255
len(test_string), len(utf8_encoded)  # 13 vs 23
```

Incorrect decoder from the handout:

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

Works on `"hello".encode("utf-8")`; fails on multi-byte characters.

---

## Rules / Invariants

1. One Unicode character can be 1–4 UTF-8 bytes
2. UTF-8 is prefix-free; you cannot decode each byte in isolation if any code point is non-ASCII
3. Not every byte sequence is valid UTF-8

---

## Sub-problems

Pipeline: `compare encodings → break naive decode → invalid sequence`

### Sub-problem A — UTF-8 vs 16/32

**Tools / docs**

| What | Reference |
|------|-----------|
| Encode | `s.encode("utf-8" | "utf-16" | "utf-32")` |
| Handout | §2.2 (a) |

**Input:** several strings (ASCII-only and CJK)

**Output:** 1–2 sentences on why UTF-8 is preferred for tokenizer training

**Goal:** Smaller / ASCII-compatible / dominant on the web — argue from encodings you measured.

**Checkpoint:** `len(s.encode("utf-8"))` vs `utf-16`/`utf-32` on `"hello"` and `"こんにちは"`.

### Sub-problem B — wrong per-byte decode

**Tools / docs**

| What | Reference |
|------|-----------|
| `bytes([b]).decode("utf-8")` | fails for continuation bytes |
| Handout | §2.2 (b) |

**Input:** a `bytes` object

**Output:** that example + one sentence why the function is wrong

**Goal:** Show UTF-8 is a **variable-width** encoding.

**Checkpoint:** Call `decode_utf8_bytes_to_str_wrong` on a non-ASCII encoded string and catch `UnicodeDecodeError` (or wrong text).

### Sub-problem C — invalid two-byte sequence

**Tools / docs**

| What | Reference |
|------|-----------|
| UTF-8 well-formedness | continuation bytes `10xxxxxx`; overlong / lone continuation |
| Handout | §2.2 (c) |

**Input:** two integers in `0..255`

**Output:** the two-byte sequence + one-sentence explanation

**Goal:** Exhibit bytes that `.decode("utf-8")` rejects.

**Checkpoint:** `bytes([...]).decode("utf-8")` raises `UnicodeDecodeError`.

---

## Edge Cases

| Case | Note |
|------|------|
| ASCII-only | Naive decoder accidentally works |
| Emoji | 4-byte UTF-8 |

---

## Acceptance Criteria (Judge)

**N/A — no unit test.** Writeup:

- (a) 1–2 sentences
- (b) example bytes + 1 sentence
- (c) 2-byte example + 1 sentence

---

## Complexity / Performance Targets

N/A.

---

## Debug Checklist

- [ ] (b) used only `"hello"` (the function succeeds there)
- [ ] Confused code points with UTF-8 code units

---

## Related Files

| File | Why |
|------|-----|
| Handout §2.2 | Spec |
| `docs/guidelines/train_bpe.md` | Byte vocab 0–255 |

---

## Wiring reminder

No adapter. Answers go in the writeup.
