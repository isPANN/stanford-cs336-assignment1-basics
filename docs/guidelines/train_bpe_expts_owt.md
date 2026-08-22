# Problem: train_bpe_expts_owt — BPE on OpenWebText

**Difficulty:** Medium-Hard  
**Topic:** Systems / Domain shift  
**Points:** 2  
**Implement in:** reuse `cs336_basics/bpe.py`  
**Wire via:** N/A — **manual deliverable**

---

## Description

Train byte-level BPE on OpenWebText sample, `vocab_size=32000`. Serialize. Report longest token; contrast with the TinyStories 10k tokenizer.

---

## Signature

`train_bpe(owt_path, vocab_size=32000, special_tokens=["<|endoftext|>"])` (add whatever specials the corpus uses).

---

## Input / Output

| | |
|--|--|
| Input | `data/owt_train.txt` (README: huggingface `owt-sample`) |
| Output | serialized vocab/merges; writeup |
| Resource | ≤ 12 hours CPU, ≤ 100 GB RAM |

---

## Constraints

- Same BPE contract as unit tests
- Look at the raw data first (web text ≠ children’s stories)
- Incremental merges required; naive recount will miss the wall-clock budget

---

## Examples

OWT is news/forums/code-ish English; expect different long tokens (URLs, HTML leftovers, names) vs TinyStories (`Once`, `upon`, etc.).

---

## Rules / Invariants

1. 32k vocab including 256 bytes + specials + merges
2. Comparison in (b) should mention domain (vocabulary contents), not only runtime

---

## Sub-problems

Pipeline: `inspect OWT → train 32k → longest token → compare to TinyStories 10k`

### Sub-problem A — train OWT 32k

**Tools / docs**

| What | Reference |
|------|-----------|
| Data | README `owt_train.txt.gz` |
| Handout | §2.5 Problem (train_bpe_expts_owt) (a) |

**Input:** OWT train file

**Output:** serialized tokenizer + longest token + 1–2 sentences

**Goal:** A 32k web tokenizer.

**Checkpoint:** Manual — job finishes; `len(vocab)==32000`.

### Sub-problem B — compare tokenizers

**Tools / docs**

| What | Reference |
|------|-----------|
| Handout | §2.5 (b) |

**Input:** TinyStories 10k vocab vs OWT 32k vocab

**Output:** 1–2 sentences contrast (domain, token length, punctuation, names, code, …)

**Goal:** See that BPE vocab reflects the corpus.

**Checkpoint:** Writeup (b).

---

## Edge Cases

| Case | Expected |
|------|----------|
| Different special-token formatting | Still split on whatever delimiter the file uses |
| Memory | memmap/chunk; same pretok parallelism |

---

## Acceptance Criteria (Judge)

**N/A.** Writeup:

- (a) longest token + does it make sense
- (b) TinyStories vs OWT contrast

---

## Complexity / Performance Targets

| Phase | Target |
|-------|--------|
| Wall time | ≤ 12 h |
| RAM | ≤ 100 GB |

---

## Debug Checklist

- [ ] Compared runtimes only, not vocab contents
- [ ] Used TinyStories vocab size 10k on OWT by mistake

---

## Related Files

| File | Why |
|------|-----|
| `docs/guidelines/train_bpe.md` | Algorithm |
| `docs/guidelines/train_bpe_tinystories.md` | 10k counterpart |
| Handout §2.5 | Spec |

---

## Wiring reminder

No adapter. Save OWT merges for `tokenizer_experiments` and LM training.
