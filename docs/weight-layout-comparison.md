# How the four shortlisted models actually lay out their weights

Measured 2026-08-21 by `scripts/compare_architectures.py`, which read
safetensors headers directly from the Hub. **That script has since been deleted**
in the cleanup that reduced this repo to the Colab notebook, so the manifests
below are no longer re-derivable from this repository; they are kept as the
record of the measurement. **Nothing was downloaded** — the
header carries every tensor name, shape and dtype, so a full pull of all four
(~60 GB) would have bought nothing this analysis needed.

Per-model manifests: `results/<model>/tensor-manifest.csv`.
Combined table: `results/architecture-comparison.csv`.

## Measured

| Model | Total params | LM params | Tensors | LM blocks | LM roles | Components |
|---|---|---|---|---|---|---|
| Falcon-H1R-7B | 7,585,648,736 | 7,585,648,736 | 751 | 44 | 20 | text only |
| Qwen3.5-9B | 9,653,104,368 | 9,197,093,888 | 775 | 32 | 38 | text + vision |
| gemma-4-E4B | 7,996,157,418 | 7,523,967,274 | 2,130 | 42 | 24 | text + vision + audio |
| LFM2.5-8B-A1B | 8,467,856,832 | 8,467,856,832 | 2,302 | 24 | 21 | text only |

Mechanisms, detected from tensor names actually present rather than from
published descriptions:

| Model | softmax attn | SSM / linear attn | sparse MoE | short conv | PLE / altup | QAT ranges |
|---|---|---|---|---|---|---|
| Falcon-H1R-7B | yes | **yes** | no | yes | no | no |
| Qwen3.5-9B | yes | **yes** | no | yes | no | no |
| gemma-4-E4B | yes | no | no | yes | **yes** | **yes** |
| LFM2.5-8B-A1B | yes | no | **yes** | yes | no | no |

## Correction: Qwen3.5-9B is not a dense transformer

The shortlist originally described Qwen3.5-9B as a "dense transformer + GQA"
baseline, taken from a release-tracker summary. **The weights say otherwise.**
Alongside the expected `self_attn.{q,k,v,o}_proj`, every decoder layer carries:

```
model.language_model.layers.N.linear_attn.A_log
model.language_model.layers.N.linear_attn.dt_bias
model.language_model.layers.N.linear_attn.conv1d.weight
model.language_model.layers.N.linear_attn.in_proj_{a,b,z,qkv}.weight
```

`A_log` and `dt_bias` are state-space parameters — the log-decay and timestep
bias of an SSM recurrence. This is the same mechanism family as Falcon-H1R's
`mamba.A_log` / `mamba.dt_bias`, under a different name.

**Consequence:** Falcon-H1R and Qwen3.5 overlap on their core sequence-mixing
mechanism far more than the shortlist claimed. The set contains three genuinely
distinct families, not four:

1. **attention + state-space** — Falcon-H1R, Qwen3.5
2. **attention + per-layer embeddings** — Gemma 4 E4B
3. **attention + sparse MoE** — LFM2.5

## The larger finding: none of the four is a plain transformer

Every model in this set augments softmax attention with a second sequence-mixing
or capacity mechanism. There is no vanilla dense decoder among the four ~7B
open-weight releases of 2026 that were selected for architectural spread — the
"baseline" slot could not be filled from this cohort. Anyone wanting a plain
dense transformer as a control must reach back to an earlier generation.

## Layout observations worth building on

**Tensor count does not track parameter count.** LFM2.5 has the most tensors
(2,302) but sits mid-pack on parameters. MoE is why: each of its 24 blocks
replicates `experts.N.w1/w2/w3` across many experts, so the tensor count scales
with expert count while only ~1.5 B parameters activate per token.

**Role count measures architectural regularity.** Falcon-H1R needs only **20
distinct roles** across 44 blocks — the leanest, most uniform layout here, and
therefore the easiest to write generic analysis against. Qwen3.5 needs 38 for
its 32 LM blocks, because attention layers and linear-attention layers have
different tensor sets and both appear.

**Gemma 4 E4B's complexity is in its towers, not its language model.** Only 24
distinct roles in the LM, but 2,130 tensors overall across vision and audio
encoders. It is also the only model carrying `input_max` / `output_max` range
tensors — quantisation-aware training baked into the checkpoint.

**Only Qwen3.5 and Gemma 4 mix dtypes.** Qwen3.5 keeps 48 tensors in F32
(the SSM `A_log` decay terms among them) and LFM2.5 keeps 22; Falcon-H1R and
Gemma 4 are uniformly BF16. Numerically sensitive parameters being held at
higher precision is a pattern worth checking before any quantisation work.

## What this means for the model-agnostic code

Predicted earlier and now confirmed: `find(model, "attn|self_attn")` returns
fewer matches than there are blocks on Falcon-H1R and Qwen3.5, because only some
layers are attention. Any code assuming `n_blocks == n_attention_layers` is
wrong on half this set. `blocks()` — which discovers the stack structurally —
remains the correct entry point.
