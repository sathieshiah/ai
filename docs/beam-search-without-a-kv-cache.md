# Why this beam search runs without a KV cache

Measured 2026-08-26 on a Colab A100, `notebooks/cloud/naive-beam-search-colab.ipynb`,
bfloat16, seed 0. Mechanism confirmed against `transformers` 5.15.1; the notebook
installs the latest with `%pip install -U transformers`, so the version on Colab
may be newer.

**Four of seven models cannot beam-search with a KV cache.** They are not
obscure: they are the hybrid SSM/attention architectures the comparison exists to
study. The cache was removed rather than made conditional, because an
optimisation that applies to three models out of seven is a confound in a
cross-architecture comparison.

## What happened

| Model | Revision | Cache reorder |
|---|---|---|
| `ibm-granite/granite-4.0-h-tiny` | `791e0d3d28c8` | `AttributeError` |
| `LiquidAI/LFM2.5-8B-A1B` | `5dd22602c2e9` | `AttributeError` |
| `tiiuae/Falcon-H1R-7B` | `a6f74bf18138` | `RuntimeError` |
| `Zyphra/Zaya1-8B` | `67d34da515b3` | `RuntimeError` |
| `Qwen/Qwen3-8B` | `b968826d9c46` | worked |
| `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f9779` | worked |
| `deepseek-ai/DeepSeek-V2-Lite` | `604d5664dddd` | worked |

```
AttributeError: 'LinearAttentionLayer' object has no attribute 'batch_select_indices'

RuntimeError: Sizes of tensors must match except in dimension 2.
              Expected size 1 but got size 10 for tensor number 1 in the list.
```

The three that worked are the three conventional attention architectures — dense
GQA, plain dense, and MLA. Every failure is a model with recurrent or
convolutional state.

## Why

Beam search re-parents its hypotheses at every step: the beam in row 3 after a
step may descend from the beam that was in row 7 before it. A cache is indexed by
batch row, so it must be permuted to match, or every later forward pass applies
one beam's history to another beam's tokens.

`transformers` implements that permutation per cache-layer:

```python
# transformers/cache_utils.py
def batch_select_indices(self, indices):
    for layer_idx in range(len(self.layers)):
        self.layers[layer_idx].batch_select_indices(indices)
```

`DynamicLayer` and `StaticLayer` implement `batch_select_indices`.
`LinearAttentionCacheLayerMixin` — the base class for linear-attention and SSM
layer caches — does not. It defines `crop`, `get_max_length` and
`lazy_initialization`, and nothing that permutes the batch dimension. Hence the
`AttributeError`.

The second signature is the same gap presenting differently: the call is accepted
but leaves a recurrent or convolutional state at batch 1 while the beam fans out
to `beam_width`, which surfaces one step later as a shape mismatch on the concat.
`Expected size 1 but got size 10` — 10 is the beam width.

This is not a bug so much as an unexercised interface. Beam search is close to
the only consumer of `batch_select_indices`; sampling and greedy decoding never
re-parent, so nothing else notices these layers cannot be permuted. A batch-wise
permute of SSM state is well defined in principle — the recurrent state is a
tensor with a batch dimension like any other — so this is an implementation gap
rather than a mathematical obstruction.

## What was done about it

The cached path was removed entirely: `reorder_cache()`, the `use_kv_cache`
option, the `past_key_values` threading. The search now recomputes from the whole
sequence at every step, which holds no state and is therefore correct on every
architecture. `use_cache=False` is passed explicitly on each forward so no cache
is built and discarded.

**The cost is O(n²) in generated length.** At the notebook's 100 tokens that is
roughly 40× the work of 10 tokens rather than 10×, and it now applies to all
seven models rather than the four that had no choice.

**It does not change the results.** Measured on `gpt2` at float32 on CPU, the
cached and uncached searches returned identical token sequences in identical rank
order, with cumulative log probabilities agreeing to ~1.7e-05 — float32
reassociation, not a difference in the search:

```
#  ids match    uncached score   cached score       delta
0  True       -11.1490440369  -11.1490612030    1.72e-05
1  True       -11.4294395447  -11.4294414520    1.91e-06
2  True       -11.6611747742  -11.6611919403    1.72e-05
```

After the cache was removed, the same prompt reproduced the uncached scores
digit for digit, confirming the refactor moved nothing.

## Consequences for reading the results

- `search_s` is now the same algorithm on every architecture, so it is
  comparable across models. It was not while the cache applied to only three.
- Wall-clock is not a fair proxy for inference cost. These are O(n²) numbers, and
  a production deployment of any of these models would use a cache.
- Any future work that wants the cache back needs a per-architecture batch
  permute for SSM and convolutional state. That belongs upstream in
  `transformers`, not in this notebook.

## Reproducing

`CONFIG` in section 3: width 10, 100 tokens, top-10 candidates per beam,
temperature 1.0, `early_stopping=True`, seed 0, bfloat16 on an A100. Chat
template applied where the tokenizer has one; `used_chat_template` is recorded
per model in `run-metadata.json`.
