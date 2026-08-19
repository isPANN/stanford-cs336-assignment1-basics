# Problem NNN: <Title>

**Difficulty:** Easy | Medium | Medium-Hard | Hard
**Topic:** <e.g. String / Hash Map / Tensor ops / Training loop>
**Points:** <from handout, if stated>
**Implement in:** `cs336_basics/<module>.py`
**Wire via:** `tests/adapters.py::<function>`

---

## Description

<2–4 sentences. What the function does and why it exists in the pipeline.>

---

## Signature

```python
<copy from adapters.py docstring, with types>
```

---

## Input / Output

### Input

| Param | Type | Meaning |
|-------|------|---------|
| ... | ... | ... |

### Output

| Return | Type | Meaning |
|--------|------|---------|
| ... | ... | ... |

---

## Constraints

<Bullet list from handout + tests: shapes, dtypes, invariants, libraries to use,
time/memory limits, determinism rules, what is forbidden.>

---

## Examples

### Example 1 — Handout

<Worked mini-example from PDF if available. Show input → intermediate → output.>

### Example 2 — Test fixture

<Concrete call using `tests/fixtures/` paths and expected check behavior.>

---

## Rules / Invariants

Numbered list of properties that must always hold. These are the "hidden tests"
in the spec.

1. ...
2. ...

---

## Sub-problems

Show the overall pipeline first:

```
step_A → step_B → ... → final return value
```

Each sub-problem must have **four sections** in this order. Do not write the
algorithm — only specify the contract.

### Sub-problem A — `<name>`

**Tools / docs**

| What | Reference |
|------|-----------|
| <library or built-in> | <function signature or doc link> |
| <relevant handout section> | §X.Y "<quote>" |

**Input**

| Name | Type | Meaning |
|------|------|---------|
| ... | ... | ... |

**Output**

| Name | Type | Meaning |
|------|------|---------|
| ... | ... | ... |

**Goal:** One sentence: what this step produces and why it is needed.

**Checkpoint:** Concrete `assert` or `uv run pytest` command that passes iff this
step is correct — no hand-wavy descriptions.

### Sub-problem B — `<name>`

(repeat Tools / Input / Output / Goal / Checkpoint)

---

## Edge Cases

| Case | Expected behavior |
|------|-------------------|
| ... | ... |

---

## Acceptance Criteria (Judge)

```bash
# minimal
uv run pytest tests/test_<module>.py::test_<name> -q

# full module
uv run pytest tests/test_<module>.py -q
```

| Test | What it verifies |
|------|------------------|
| `test_...` | ... |

---

## Complexity / Performance Targets

| Phase | Naive | Target | Notes |
|-------|-------|--------|-------|
| ... | ... | ... | from speed/memory tests if any |

---

## Debug Checklist

Common wrong answers for this problem:

- [ ] ...
- [ ] ...

---

## Related Files

| File | Why read it |
|------|-------------|
| `tests/test_....py` | Judge |
| `tests/fixtures/...` | Sample I/O |
| Handout §X.Y | Spec |
| `cs336_basics/...` | Where to implement |

---

## Wiring reminder

`tests/adapters.py::<fn>` should delegate to `cs336_basics` — tests never import
implementation modules directly.
