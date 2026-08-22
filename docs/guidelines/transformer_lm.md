# Problem 014: Transformer Language Model

**Difficulty:** Medium  
**Topic:** Composition / LM head  
**Points:** 3  
**Implement in:** `cs336_basics/transformer.py`  
**Wire via:** `tests/adapters.py::run_transformer_lm`

---

## Description

Full decoder-only LM (Figure 1):

```
token embed → TransformerBlock × num_layers → RMSNorm → Linear (lm_head) → logits
```

No softmax in the module (loss does that). No extra learned absolute position embedding — RoPE lives inside attention. `sequence_length` may be **shorter** than `context_length` (`test_transformer_lm_truncated_input`).

---

## Signature

```python
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        ...
    ) -> None: ...

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor: ...
```

Adapter: `run_transformer_lm(...)` — full state-dict key list is in the adapter docstring (`token_embeddings.weight`, `layers.{i}.*`, `ln_final.weight`, `lm_head.weight`).

---

## Input / Output

| Param | Shape |
|-------|-------|
| `in_indices` | `(batch, seq)` token ids, `seq ≤ context_length` |
| **logits** | `(batch, seq, vocab_size)` unnormalized next-token scores |

Fixture config (`tests/fixtures/ts_tests/model_config.json`): vocab 10000, context 16, d_model 64, layers 3, heads 4, d_ff 128, rope_theta 10000.

---

## Constraints

- Shared architecture with your `Embedding`, `TransformerBlock`, `RMSNorm`, `Linear`
- `lm_head` is a bias-free Linear `d_model → vocab_size` (weight `(vocab_size, d_model)`). Weight tying with embeddings is **not** required by tests (fixture has a separate `lm_head.weight`)
- Truncated seq: still valid causal LM; do not index RoPE as if `seq == context_length` without using actual positions `0..seq-1`
- Snapshot: full LM `atol=1e-4, rtol=1e-2`; truncated `atol=1e-4`

---

## Examples

```bash
uv run pytest tests/test_model.py::test_transformer_lm -q
uv run pytest tests/test_model.py::test_transformer_lm_truncated_input -q
```

Truncation: `in_indices[..., : seq//2]`.

---

## Rules / Invariants

1. Logits are **not** softmaxed
2. Each position \(i\) predicts token \(i+1\) (training); causal mask enforces that
3. `ln_final` after the last block, before `lm_head`

---

## Sub-problems

Pipeline: `embed → blocks → ln_final → lm_head`

### Sub-problem A — construct + load

**Tools / docs**

| What | Reference |
|------|-----------|
| Keys | `run_transformer_lm` docstring |
| `nn.ModuleList` | `self.layers = nn.ModuleList([...])` so keys are `layers.0....` |
| Handout | §3.1, §3.5 |

**Input:** hyperparameters + `weights`

**Output:** module whose `state_dict` keys match the fixture

**Goal:** Load the reference weights without remapping every tensor by hand (prefer matching names).

**Checkpoint:** `set(model.state_dict())` covers `token_embeddings.weight`, `ln_final.weight`, `lm_head.weight`, `layers.0.attn.q_proj.weight`

### Sub-problem B — `forward`

**Tools / docs**

| What | Reference |
|------|-----------|
| Positions | `arange(seq, device=...)` broadcast to batch if needed |

**Input:** `(batch, seq)` ids

**Output:** `(batch, seq, vocab)` logits

**Goal:** Full LM forward.

**Checkpoint:** both `test_transformer_lm*` pytest commands.

---

## Edge Cases

| Case | Expected |
|------|----------|
| `seq < context_length` | Works; RoPE uses actual positions |
| `seq == context_length` | Full context test |

---

## Acceptance Criteria (Judge)

```bash
uv run pytest tests/test_model.py::test_transformer_lm -q
uv run pytest tests/test_model.py::test_transformer_lm_truncated_input -q
```

Optional full model file:

```bash
uv run pytest tests/test_model.py -q
```

---

## Complexity / Performance Targets

Batched; `num_layers` is a Python loop over **layers** only (3 in tests), not over tokens.

---

## Debug Checklist

- [ ] Embedding vs lm_head names
- [ ] `layers.i` vs `layer.i`
- [ ] `output_proj` vs `o_proj` vs `attn.output_proj.weight`
- [ ] Softmax accidentally applied to logits
- [ ] Assuming `seq == context_length` in RoPE buffers

---

## Related Files

| File | Why |
|------|-----|
| `tests/test_model.py` | Two LM tests |
| `tests/fixtures/ts_tests/model.pt` | Weights |
| `tests/fixtures/ts_tests/model_config.json` | Hparams |
| Handout §3.5 | Spec |

---

## Wiring reminder

`run_transformer_lm` constructs `TransformerLM`, `load_state_dict(weights)`, returns `forward(in_indices)`.
