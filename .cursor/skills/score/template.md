# Review template

Write this to `docs/reviews/<problem-id>.md`. In chat, show Verdict + Scorecard + findings; skip Related files unless asked.

```markdown
# Review: <problem-id>

**Date:** <YYYY-MM-DD>
**Implement in:** `cs336_basics/<module>.py`
**Wire via:** `tests/adapters.py::<fn>`

## Verdict

| | |
|--|--|
| Judge | `AC` / `WA` / `TLE` / `MLE` / `RE` / `CE` / `NotImplemented` / `N/A` |
| Quality | <n>/20 |
| Ready to move on? | Yes / No — <one sentence> |

## Pytest

Command:

\`\`\`bash
uv run pytest <args>
\`\`\`

Result: <passed>/<total>  Duration: <if relevant>

Failing tests (if any):

- `test_...`: <one-line cause>

## Scorecard

| Dimension | /4 | Note |
|-----------|----|------|
| Contract | | |
| Complexity / scale | | |
| Numerical / systems | | |
| Abstraction | | |
| Readability | | |
| **Total** | **/20** | |

## Findings

### Blocker
<or "None">

**Blocker — ...**
- Evidence:
- Later:
- Principle:
- Direction:

### Critical
<or "None">

**Critical — ...**
- Evidence:
- Later:
- Principle:
- Direction:

### Improve
<max 3>

### Nit
<max 2, or omit section>

## Growth

- <transferable principle 1>
- <transferable principle 2>

## Next action

One concrete edit. Not a rewrite. Not "keep going".
```
