# Problem: unicode1 — Understanding Unicode

**Difficulty:** Easy  
**Topic:** Unicode / Python strings  
**Points:** 1  
**Implement in:** (none — written answers)  
**Wire via:** N/A — **manual deliverable**

---

## Description

Play with `ord` / `chr` and the NUL character `chr(0)` so later BPE-on-bytes makes sense. Handout §2.1.

---

## Signature

No code adapter. Python REPL exploration.

---

## Input / Output

### Input

The character produced by `chr(0)`, and strings that contain it.

### Output

Written answers (one sentence each) for (a)–(c).

---

## Constraints

- Use the Python interpreter; do not guess without running
- Deliverable is prose, not a pytest

---

## Examples

Handout:

```python
ord("牛")  # 29275
chr(29275)  # '牛'
chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")
```

---

## Rules / Invariants

1. Unicode maps characters ↔ code points; `chr`/`ord` are inverses on valid code points
2. `__repr__` of a string can differ from what `print` shows

---

## Sub-problems

Pipeline: `chr(0) → inspect repr vs print → embed in a longer string`

### Sub-problem A — `chr(0)` identity

**Tools / docs**

| What | Reference |
|------|-----------|
| `chr` / `ord` | Python builtins |
| Handout | §2.1 Problem (unicode1) (a) |

**Input:** integer `0`

**Output:** the Unicode character (written description)

**Goal:** Name what `chr(0)` is.

**Checkpoint:** Run `chr(0)` in a REPL; one-sentence answer submitted in the writeup.

### Sub-problem B — repr vs printed form

**Tools / docs**

| What | Reference |
|------|-----------|
| `__repr__` | `repr(chr(0))` vs `print(chr(0))` |
| Handout | §2.1 (b) |

**Input:** that character

**Output:** one sentence on how string repr differs from printed representation

**Goal:** See why NUL is awkward in text.

**Checkpoint:** Compare `chr(0)` (REPL display) to `print(chr(0))`.

### Sub-problem C — NUL inside text

**Tools / docs**

| What | Reference |
|------|-----------|
| Concatenation | handout snippet with `"this is a test" + chr(0) + "string"` |
| Handout | §2.1 (c) |

**Input:** a Python `str` containing U+0000

**Output:** one sentence on what happens when this character occurs in text

**Goal:** Observe truncation / invisibility / length vs print.

**Checkpoint:** Compare `len(...)` and `print(...)` on the concatenated string.

---

## Edge Cases

| Case | Expected (explore, then write) |
|------|--------------------------------|
| `print` vs REPL | Often looks like an empty/invisible char |
| Embedded NUL | `str` still contains it; some displays stop or skip |

---

## Acceptance Criteria (Judge)

**N/A — no unit test.** Course writeup:

- (a) one sentence: what `chr(0)` returns
- (b) one sentence: repr vs printed
- (c) one sentence: behavior in text

---

## Complexity / Performance Targets

N/A.

---

## Debug Checklist

- [ ] Answered from memory without running the snippets
- [ ] Confused Python `str` with C-style NUL-terminated buffers

---

## Related Files

| File | Why |
|------|-----|
| Handout §2.1 | Spec |
| `docs/guidelines/tokenizer.md` | Later: decode invalid UTF-8 |

---

## Wiring reminder

Nothing in `adapters.py`. Put answers in your assignment writeup.
