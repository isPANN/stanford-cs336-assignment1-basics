---
name: score
description: >-
  Score a CS336 Assignment 1 solution for both pytest correctness and critical
  code quality. Use when the user asks to 评分, 打分, grade, score, review, critique,
  or check code quality of cs336_basics / adapters implementations (train_bpe,
  tokenizer, transformer, optimizer, etc.). Runs tests, writes a strict review
  to docs/reviews/, and does not implement fixes unless explicitly asked.
---

# CS336 Solution Scorer

Score the user's implementation like a strict TA: **passing tests is necessary,
not sufficient**. Surface critical quality issues that will hurt later
assignments, TinyStories/OWT scale, and GPU training.

Companion to `guideline`. Do **not** write the solution.

## When to use

- User asks to 评分 / 打分 / score / grade / review / critique their solution
- User says they finished a problem and wants feedback
- User asks "代码质量怎么样" / "能不能过" / "帮我看看"

## When NOT to use

- User only wants a study guideline → use `cs336-leetcode-guideline`
- User explicitly asks you to implement or rewrite the solution
- Pure factual question (path, pytest command)

## Hard rules

- Always **run pytest** before claiming AC. Never infer pass/fail from reading code.
- Adapters stay thin: logic belongs in `cs336_basics/`, not `tests/adapters.py`.
- Do **not** modify `tests/test_*.py` or fixtures.
- Do **not** implement the fix unless the user explicitly asks.
- Do **not** inflate scores. Stub / `NotImplementedError` / `NameError` → Judge `CE` or `NotImplemented`, Quality `0`.
- Lead with **Blocker / Critical**. Nits (naming, extra blank lines, "add a docstring") are optional and never first.
- Every Critical finding must include: **evidence**, **why it hurts later**, **reusable principle**, **fix direction** (sketch, not a rewrite).
- Reply in the **user's language**.

## Workflow

Copy this checklist and track it:

```
Score Progress:
- [ ] Identify problem
- [ ] Read spec sources
- [ ] Run pytest
- [ ] Review code against rubric + quality bars
- [ ] Write report (chat + docs/reviews/)
```

### 1. Identify the problem

Infer from the open file, recent chat, or user message. If still ambiguous, ask.

Look up test file + adapter in `guideline/problem-map.md`
(same table; do not duplicate it here).

### 2. Read sources (in order)

1. Implementation: `cs336_basics/<module>.py` (and any helpers it imports)
2. Wiring: `tests/adapters.py` — corresponding `run_*` / `get_*` only
3. Judge: `tests/test_<module>.py` — assertions, timeouts, memory limits
4. Guideline if present: `docs/guidelines/<problem-id>.md`
5. Quality bars: [quality-bars.md](quality-bars.md) for this problem
6. Rubric: [rubric.md](rubric.md)
7. Handout excerpt only if the code's intent is unclear

### 3. Run pytest

Use the **specific** tests for this problem, not the whole suite (other problems are still stubs).

```bash
uv run pytest tests/test_<file>.py::<test_name> -q --tb=short
```

For a module with several tests (e.g. `test_train_bpe.py`), run the file:

```bash
uv run pytest tests/test_train_bpe.py -q --tb=short
```

Map failures:

| Pytest signal | Judge |
|---------------|-------|
| all passed | `AC` |
| AssertionError / snapshot mismatch | `WA` |
| speed / time assert | `TLE` |
| MemoryError / RLIMIT / killed | `MLE` |
| NotImplementedError | `NotImplemented` |
| NameError / SyntaxError / ImportError | `CE` |
| other exception | `RE` |

Record the exact command and pass/fail counts in the report.

If Judge ≠ `AC`, correctness is the gate. Still scan for **one or two** Critical design issues if they are already visible — do not wait until tests pass to mention a design that cannot scale.

### 4. Review quality

Score each rubric dimension 0–4 using [rubric.md](rubric.md).

Apply problem-specific bars in [quality-bars.md](quality-bars.md). A solution can be `AC` and still score **1–2** on Complexity if it is asymptotically or numerically unsafe for later work.

Ignore starter files the user did not edit (`pretokenization_example.py` unless they copy-pasted it badly).

### 5. Write the report

Fill [template.md](template.md). Then:

- Write `docs/reviews/<problem-id>.md` (overwrite if it exists)
- In chat: short verdict + scorecard + Critical findings. Do **not** dump the full file unless asked.
- Do not commit unless asked.

## What "critical quality" means here

These are **course-blocking habits**, not style nits:

- Tests pass on a toy fixture but the algorithm is `O(wrong)` for TinyStories / OWT / GPU
- Forbidden or course-defeating shortcuts (`torch.nn.Linear` / `Embedding`, stdlib `re` for GPT-2 pretok)
- Numerical instability (softmax/CE without max / log-sum-exp)
- Python loops over batch / heads / sequence when the op is a matmul
- Fat adapters: entire algorithm inside `tests/adapters.py`
- Hidden state / extra copies that will OOM in `encode_iterable` or training
- Contract drift: wrong tie-break, merge order, in-place vs copy, dtype/device

Do **not** spend review budget on:

- Pep8-only nits, quote style, whether to use `X | Y` vs `Optional`
- "Consider adding more comments"
- Rewriting working vectorized code into a "cleaner" one-liner

## After the review

If Judge ≠ `AC`: tell them the failing test and the likely invariant (from the guideline debug checklist), then stop. Do not fix the code.

If Judge = `AC` but Critical findings exist: say clearly **do not treat this as done**. Point at the highest-leverage Critical item as the next edit.

If no Blocker/Critical and Quality ≥ 12/20: say it is good enough to proceed to the next problem; list at most two Improve items.

## Reference

- Rubric: [rubric.md](rubric.md)
- Per-problem bars: [quality-bars.md](quality-bars.md)
- Report template: [template.md](template.md)
- Example: [examples/train-bpe.md](examples/train-bpe.md)
