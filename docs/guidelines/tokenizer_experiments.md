# Problem: tokenizer_experiments — Encode Corpora

**Difficulty:** Medium  
**Topic:** Compression / Throughput  
**Points:** 4  
**Implement in:** reuse `cs336_basics/tokenizer.py`  
**Wire via:** N/A — **manual deliverable** (needs encode/decode AC first)

---

## Description

Use the TinyStories 10k and OWT 32k tokenizers: compression ratios, cross-domain tokenization, throughput, and dump train/dev token ids as `uint16` numpy arrays for the LM.

---

## Signature

`Tokenizer.encode` / `encode_iterable` from `docs/guidelines/tokenizer.md`. Prefer `encode_iterable` + memmap-friendly dumps for large files.

---

## Input / Output

| Part | Deliverable |
|------|-------------|
| (a) | 10 docs from each corpus; bytes/token for each tokenizer |
| (b) | OWT sample encoded with **TinyStories** tokenizer — ratio + qualitative |
| (c) | bytes/s throughput; time to tokenize the Pile (825 GB) |
| (d) | `.npy` / memmap of token ids, dtype **uint16**, plus why uint16 |

---

## Constraints

- Compression ratio = **UTF-8 bytes / number of tokens** (handout: bytes/token)
- `uint16` max 65535; both vocabs (10k and 32k) fit
- Do not materialize giant Python `list[int]` for full OWT if you can stream to `np.memmap`
- `encode_iterable` must stay under memory like the unit test (1 MB is a fixture; full data needs streaming)

---

## Examples

Sample documents split on `<|endoftext|>`. Encode with both vocabs. Cross-domain: TinyStories BPE will fragment OWT names/URLs into many short tokens → worse compression.

---

## Rules / Invariants

1. Same tokenizer used later for LM training — dump the **matching** train/dev ids
2. Throughput estimate should use a large enough sample, then scale to 825 GB

---

## Sub-problems

Pipeline: `sample docs → ratios → cross-tokenize → bench → dump uint16`

### Sub-problem A — in-domain compression

**Tools / docs**

| What | Reference |
|------|-----------|
| `encode` | tokenizer guideline |
| Handout | §2.7 (a) |

**Input:** 10 TS docs, 10 OWT docs, two tokenizers

**Output:** two (or four) compression ratios + 1–2 sentences

**Goal:** Measure bytes/token.

**Checkpoint:** Ratios > 1; TS tokenizer on TS stories better than on OWT (typically).

### Sub-problem B — cross-domain

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §2.7 (b) |

**Input:** OWT sample + TinyStories tokenizer

**Output:** 1–2 sentences

**Goal:** See domain mismatch (UNK-like fragmentation, odd merges).

**Checkpoint:** Writeup (b).

### Sub-problem C — throughput

**Tools / docs**

| What | Reference |
|------|-----------|
| `time.perf_counter` | wall time vs `os.path.getsize` |
| Handout | §2.7 (c) |

**Input:** a sizable file encode

**Output:** bytes/s and extrapolated Pile time

**Goal:** Order-of-magnitude estimate.

**Checkpoint:** Writeup (c); use `encode_iterable` so the bench is realistic.

### Sub-problem D — serialize ids

**Tools / docs**

| What | Reference |
|------|-----------|
| `np.uint16` | 0..65535 |
| `np.save` / memmap | later `get_batch` |
| Handout | §2.7 (d) |

**Input:** full train/dev text per corpus

**Output:** arrays on disk + 1–2 sentences why uint16

**Goal:** LM-ready data.

**Checkpoint:** `array.max() < vocab_size`; dtype uint16.

---

## Edge Cases

| Case | Expected |
|------|----------|
| Token id ≥ 65536 | Would not fit uint16 — should not happen at 32k vocab |
| Empty documents | Skip or encode as empty id list |

---

## Acceptance Criteria (Judge)

**N/A.** Four short writeup answers + tokenized arrays you will reuse in §5–7.

---

## Complexity / Performance Targets

| Phase | Naive | Target |
|-------|-------|--------|
| Full OWT encode | `encode(whole_file)` RAM blowup | `encode_iterable` → memmap |
| Pile estimate | N/A | linear scale from measured bytes/s |

---

## Debug Checklist

- [ ] Ratio inverted (tokens/byte)
- [ ] int32 dump “because PyTorch”
- [ ] Tokenized with tiktoken instead of your BPE

---

## Related Files

| File | Why |
|------|-----|
| `docs/guidelines/tokenizer.md` | Encoder |
| `docs/guidelines/data_loading.md` | Consumes the arrays |
| Handout §2.7 | Spec |

---

## Wiring reminder

No adapter. Your `Tokenizer.from_files` is useful here even though unit tests skip it.
