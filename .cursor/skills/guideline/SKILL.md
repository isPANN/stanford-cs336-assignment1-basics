---
name: guideline
description: >-
  Read the CS336 Assignment 1 handout and tests, then generate a LeetCode-style
  problem guideline for a specific sub-problem. Use when the user asks for a
  guideline, learning path, problem breakdown, acceptance criteria, or
  LeetCode-style guide for cs336_basics / adapters / train_bpe / tokenizer /
  transformer / optimizer tasks in this repo. Always save markdown to
  docs/guidelines/. Do not implement solutions unless explicitly asked.
---

# CS336 LeetCode-Style Guideline Generator

Generate **study guidelines**, not solutions. The user implements code in
`cs336_basics/` and wires tests through `tests/adapters.py`.

## When to use

- User asks for a guideline, roadmap, or "LeetCode-style" breakdown
- User starts a new problem and wants constraints + acceptance criteria
- User says "引导我" / "验收标准" / "第一步做什么"

## When NOT to use

- User asks you to implement, debug, or complete their code
- User only wants a quick factual answer (path, command, one-liner)

## Workflow

1. **Identify the target problem**
   - Ask which problem if unclear, or infer from open file / recent context
   - Look up mapping in [problem-map.md](problem-map.md)

2. **Read sources (in order)**
   - Handout: `cs336_assignment1_basics.pdf` — relevant section only
   - Tests: `tests/test_<module>.py` — assertions, fixtures, timeouts, snapshots
   - Adapter: `tests/adapters.py` — function signature + docstring contract
   - Fixtures: `tests/fixtures/` — concrete inputs/outputs when referenced by tests
   - Starter code: `cs336_basics/` — only to note where user should implement

   PDF extraction (when needed):

   ```bash
   uv run --with pypdf python -c "
   from pypdf import PdfReader
   r = PdfReader('cs336_assignment1_basics.pdf')
   text = '\n'.join((p.extract_text() or '') for p in r.pages)
   start = text.find('SECTION_START')
   end = text.find('SECTION_END')
   print(text[start:end])
   "
   ```

3. **Synthesize the guideline** using [template.md](template.md)
   - Fill every section from handout + tests, not from memory
   - Sub-problems must each define **Input, Output, Goal, Checkpoint**
   - Acceptance criteria must cite **exact pytest commands**

4. **Hard rules for output**
   - ❌ No complete implementations, no copy-paste-ready solution code
   - ❌ No filling in `adapters.py` or `cs336_basics/` for the user
   - ✅ Pseudocode / data-structure hints OK if high-level
   - ✅ Handout examples and test fixture behavior OK
   - ✅ Common WA causes and debug checklist OK
   - ✅ Suggested file/module name under `cs336_basics/`

5. **Always save markdown**
   - Write to `docs/guidelines/<problem-id>.md`
   - Also show a short summary in chat (do not dump the full file unless asked)
   - Overwrite if the file already exists
   - Do not commit unless user asks

## Repo conventions (always mention)

| Layer | Path | Role |
|-------|------|------|
| Implementation | `cs336_basics/*.py` | User writes algorithms/modules here |
| Adapter | `tests/adapters.py` | Thin wrapper called by tests |
| Tests | `tests/test_*.py` | Judge — do not modify |
| Handout | `cs336_assignment1_basics.pdf` | Spec |
| Guidelines | `docs/guidelines/*.md` | Saved study artifacts |
| Data | `data/` | TinyStories / OWT (gitignored) |

Run tests: `uv run pytest tests/test_<name>.py -q`

## Difficulty heuristic

| Signal | Difficulty |
|--------|------------|
| Single function, no state, numpy snapshot | Easy |
| Multiple sub-steps, shape contracts | Medium |
| Training loops, speed tests, memory limits | Medium-Hard |
| Full LM + experiments + GPU hours | Hard |

## Reference example

See [examples/train-bpe.md](examples/train-bpe.md) for the expected output
quality and depth. Saved copy: `docs/guidelines/train_bpe.md`.

## After the user implements

If they ask to 评分 / review / check quality, use the companion skill `score`
(do not implement the solution; run pytest and write `docs/reviews/<problem-id>.md`).

## Quick problem picker

If user says "next problem" after BPE training, suggest this order:

1. `train_bpe` → 2. `get_tokenizer` → 3. `linear`/`embedding`/`rmsnorm`/`silu`
→ 4. `softmax` → 5. attention stack → 6. `transformer_lm`
→ 7. training utilities → 8. experiments (handout only, no unit tests)

Full mapping: [problem-map.md](problem-map.md)
