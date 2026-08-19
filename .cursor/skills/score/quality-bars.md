# Per-problem quality bars

Use with the rubric. **AC on the unit test is not the bar.**
If a row says Critical, emit a Critical finding when the anti-pattern is present.

## train_bpe

| Bar | Anti-pattern (Critical unless noted) |
|-----|--------------------------------------|
| Pair counts never cross pre-token or special-token boundaries | Merging across `"<|endoftext|>"` or whitespace-pre-tokens |
| GPT-2 pretok uses `regex` + `finditer` + handout `PAT` | `import re`; `findall` on a huge string |
| Tie-break = max count, then **lexicographically greater** pair | `min`, first-seen, or `sorted` without `(count, pair)` |
| Incremental merge statistics | Full `count_pairs` over all pre-tokens every merge → Critical (TLE on TinyStories even if `corpus.en` is AC) |
| `merges` is creation order | Sorting merges at the end |
| Adapter thin | Entire trainer inlined in `run_train_bpe` → Blocker/Improve depending on size |

Speed test (`corpus.en`, vocab 500, < 1.5s) is a **lower bound**. Mention TinyStories/OWT if they still recount globally.

## tokenizer

| Bar | Anti-pattern |
|-----|----------------|
| `encode_iterable` is streaming | Materializing the whole corpus → Critical (`test_encode_iterable_memory_usage`, 1MB) |
| Encode uses merge ranks, not "scan vocab every byte" | O(n · \|vocab\|) → Critical at OWT |
| Special tokens are atomic | Splitting them with PAT |
| Decode is inverse of encode | Dropping bytes / UTF-8 errors swallowed silently |

`test_encode_memory_usage` is **xfail** — do not fail the user for `encode` using more than 1MB.

## linear / embedding / rmsnorm / silu / swiglu

| Bar | Anti-pattern |
|-----|----------------|
| From-scratch `nn.Module` | `torch.nn.Linear` / `nn.Embedding` / `F.linear` as the implementation → **Blocker** (defeats the assignment) |
| Weights are `Parameter` | Buffers you forget to train; or re-creating tensors in `forward` |
| Forward is vectorized | `for` over batch or sequence |
| RMSNorm eps | `rsqrt(mean(x^2) + eps)` vs mean of `(x^2 + eps)` — match handout |
| SwiGLU | Wrong W1/W2/W3 multiply order |

Adapter should: construct module → load weights → `forward`. Logic in the module, not the adapter.

## rope / sdpa / mha / mha_rope

| Bar | Anti-pattern |
|-----|----------------|
| Scale by `1/sqrt(d_k)` | Missing scale → Blocker/WA |
| Mask | Treating boolean mask as multiplicative 0/1 without checking test convention; wrong broadcast |
| Batched heads | Loop `for h in range(num_heads)` → Critical (will not fly in LM training) |
| QKV as one (or three) big matmuls then view as heads | Per-head Linear modules |
| RoPE on **head** dim (`d_model // num_heads`) | Applying RoPE to full `d_model` |
| Softmax numerical stability | Softmax without max on the score axis |

## transformer_block / transformer_lm

| Bar | Anti-pattern |
|-----|----------------|
| Pre-norm | Post-norm residual → WA against snapshots |
| Residual around sublayers | Replacing the stream instead of `x + sublayer(norm(x))` |
| Token + (RoPE in attention), no extra learned pos unless spec says so | Absolute `nn.Embedding` positions if handout is RoPE-only |
| Truncated / short inputs | Assuming `seq == max_seq_len` always |

## softmax / cross_entropy / gradient_clipping

| Bar | Anti-pattern |
|-----|----------------|
| Softmax max-subtract | `exp(x) / sum` on raw logits → **Critical** (NaNs later) |
| CE with log-sum-exp | `log(softmax)` separately (less stable) |
| CE reduction | Mean vs sum vs none — match tests |
| Grad clip in-place on `.grad` | Returning a new tensor and leaving `.grad` unchanged → Blocker |

## adamw / lr_schedule

| Bar | Anti-pattern |
|-----|----------------|
| Decoupled weight decay | Decay inside the moment update (Adam, not AdamW) → Blocker |
| Bias correction | Missing `1 - beta**t` |
| `t` starts at the value tests use | Off-by-one vs `train_steps` |
| Cosine schedule | Warmup linear then cosine to `min_lr`, not to 0 if spec says `min_lr` |

## data_loading / checkpointing

| Bar | Anti-pattern |
|-----|----------------|
| Batch is `x[i:i+ctx], x[i+1:i+ctx+1]` | Same sequence for inputs and targets |
| Random starts valid | `i` in `0 .. len-seq_len-1` inclusive |
| Save/load round-trip | Pickling whole Python objects; missing optimizer / `iteration` |
| `torch.save` of `state_dict` | Saving the module object only |

## Handout-only experiments

If there is no unit test: Judge = `N/A`. Score design/reproducibility: seed, logged hyperparameters, wall-clock, whether they trained the **student** model rather than importing HuggingFace.
