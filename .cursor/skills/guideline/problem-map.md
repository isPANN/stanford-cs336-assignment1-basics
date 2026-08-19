# CS336 Assignment 1 — Problem Map

Use this table to locate handout sections, tests, and adapter entry points.

Persistent guideline output: always write to `docs/guidelines/<problem-id>.md`

| ID | Adapter function | Test file | Handout § | Suggested module | Notes |
|----|------------------|-----------|-----------|------------------|-------|
| train_bpe | `run_train_bpe` | `test_train_bpe.py` | 2.4–2.5 | `bpe.py` | Speed + special-token tests |
| tokenizer | `get_tokenizer` | `test_tokenizer.py` | 2.6 | `tokenizer.py` | Match tiktoken; memory limits |
| linear | `run_linear` | `test_model.py` | 3.3.2 | `model.py` | NumPy snapshot |
| embedding | `run_embedding` | `test_model.py` | 3.3.3 | `model.py` | NumPy snapshot |
| rmsnorm | `run_rmsnorm` | `test_model.py` | 3.4.1 | `model.py` | |
| silu | `run_silu` | `test_model.py` | 3.4.2 | `model.py` | Compare PyTorch |
| swiglu | `run_swiglu` | `test_model.py` | 3.4.2 | `model.py` | |
| rope | `run_rope` | `test_model.py` | 3.4.3 | `model.py` | |
| sdpa | `run_scaled_dot_product_attention` | `test_model.py` | 3.4.4 | `attention.py` | 2D + 4D tests |
| mha | `run_multihead_self_attention` | `test_model.py` | 3.4.5 | `attention.py` | No RoPE variant |
| mha_rope | `run_multihead_self_attention_with_rope` | `test_model.py` | 3.4.5 | `attention.py` | |
| transformer_block | `run_transformer_block` | `test_model.py` | 3.4 | `transformer.py` | Pre-norm + RoPE |
| transformer_lm | `run_transformer_lm` | `test_model.py` | 3.5 | `transformer.py` | Includes truncated input test |
| softmax | `run_softmax` | `test_nn_utils.py` | 4.1 | `nn_utils.py` | Numerical stability |
| cross_entropy | `run_cross_entropy` | `test_nn_utils.py` | 4.1 | `nn_utils.py` | |
| gradient_clipping | `run_gradient_clipping` | `test_nn_utils.py` | 4.5 | `nn_utils.py` | In-place grad modify |
| adamw | `get_adamw_cls` | `test_optimizer.py` | 4.3 | `optimizer.py` | Custom Optimizer class |
| lr_schedule | `run_get_lr_cosine_schedule` | `test_optimizer.py` | 4.4 | `optimizer.py` | Warmup + cosine |
| data_loading | `run_get_batch` | `test_data.py` | 5.1 | `data.py` | Random LM batches |
| checkpointing | `run_save_checkpoint` / `run_load_checkpoint` | `test_serialization.py` | 5.2 | `checkpoint.py` | |

## Handout-only (no unit test in repo)

Written answers or GPU experiments — guideline should say "manual deliverable":

- `unicode1`, `unicode2` — §2.1–2.2
- `train_bpe_tinystories`, `train_bpe_expts_owt` — §2.5
- `tokenizer_experiments` — §2.7
- `transformer_accounting`, `adamw_accounting` — §3 / §4
- `training_together`, `decoding`, `experiment_log` — §5–6
- §7 ablations, leaderboard — GPU experiments

## PDF section anchors

Search these strings in `cs336_assignment1_basics.pdf` text:

| § | Start anchor |
|---|--------------|
| 2.4 BPE training | `2.4 BPE Tokenizer Training` |
| 2.6 Encode/decode | `2.6 BPE Tokenizer: Encoding and Decoding` |
| 3.3 Linear/Embedding | `3.3 Basic Building Blocks` |
| 3.4 Transformer block | `3.4 Pre-Norm Transformer Block` |
| 3.5 Full LM | `3.5 The Full Transformer LM` |
| 4 Training | `4 Training a Transformer LM` |
| 5 Training loop | `5 Training loop` |
