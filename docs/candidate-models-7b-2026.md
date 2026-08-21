# Four ~7B open-weight models from 2026, with four different architectures

Compiled 2026-08-21. Selected for **architectural diversity** rather than
benchmark scores: the point is that comparing them exercises genuinely
different mechanisms, not four variations of the same decoder stack.

## The shortlist

| Model | Released | Params | Architecture family | Context |
|---|---|---|---|---|
| **Falcon-H1R 7B** (TII) | 5 Jan 2026 | 7B dense | Hybrid: attention + Mamba-2 SSM interleaved | 256K |
| **Qwen3.5-9B** (Alibaba) | 2 Mar 2026 | 9B dense | Dense transformer + GQA (the baseline) | 262K |
| **Gemma 4 E4B** (Google) | 2 Apr 2026 | 8B total / 4.5B effective | MatFormer nesting + Per-Layer Embeddings | 128K |
| **LFM2.5-8B-A1B** (Liquid AI) | 28 May 2026 | 8.3B total / 1.5B active | Sparse MoE on the LFM2 conv-hybrid backbone | 128K |

> **Correction (2026-08-21):** the "Qwen3.5-9B = dense transformer" claim below
> came from a release-tracker summary and is **wrong**. Its weights carry
> `linear_attn.A_log` / `dt_bias` state-space parameters in every decoder layer,
> putting it in the same mechanism family as Falcon-H1R. The set therefore spans
> three distinct architecture families, not four. Measured evidence in
> [weight-layout-comparison.md](weight-layout-comparison.md).

## All four are autoregressive — verified

Confirmed against vendor documentation, not assumed:

| Model | Decoder-only, next-token AR? | Source of the claim |
|---|---|---|
| Falcon-H1R 7B | Yes | described as a decoder-only LLM; Mamba-2 layers are causal recurrences |
| Qwen3.5-9B | Yes | conventional dense causal decoder |
| Gemma 4 E4B | Yes | "Gemma 4 models follow a decoder-only Transformer architecture" |
| LFM2.5-8B-A1B | Yes | "retains a decoder-only LFM2 backbone ... decoder-only autoregressive" |

The distinction worth holding onto: **architecture family is orthogonal to the
generation objective.** SSM layers, MoE routing, per-layer embeddings and short
convolutions all change *how a token is computed*; none of them change the fact
that tokens are produced one at a time, left to right, each conditioned on its
predecessors. Every one of these is trained on next-token prediction.

Two things that look like exceptions but are not:

- **Gemma 4 ships a draft model for speculative decoding.** That is a decoding
  optimisation — the draft proposes tokens the main model verifies — and the
  output distribution is unchanged. Still autoregressive.
- **Mamba/SSM layers are recurrent, not attentional.** Recurrence is if anything
  *more* strictly sequential than attention. Still autoregressive.

The genuine non-autoregressive alternative is a **diffusion language model**,
which denoises a whole sequence in parallel over several passes rather than
extending it token by token. None of the four are of that kind. If a non-AR
comparison is ever wanted, that is the family to look in — and it would need
sourcing separately, as none was surveyed here.

## Why each one is architecturally distinct

**Falcon-H1R 7B** interleaves standard transformer attention layers with Mamba-2
state-space layers in one stack. The SSM layers carry a fixed-size recurrent
state instead of a growing KV cache, which is what buys the 256K context. For
weight-level work this is the interesting one: the parameter *shapes* in an SSM
layer (state-transition, input/output projections, per-channel decay) have no
analogue in an attention block.

**Qwen3.5-9B** is the control. A conventional dense decoder with grouped-query
attention and native vision. Include it precisely because it is ordinary — it is
the reference every other measurement is read against.

**Gemma 4 E4B** carries 8B total parameters but activates ~4.5B, and does so by a
mechanism unlike MoE routing. Per-Layer Embeddings feed each decoder layer its
own auxiliary embedding signal rather than relying only on the input embedding,
and MatFormer trains nested sub-models inside one checkpoint. So "how many
parameters does this model have" has three different defensible answers, which
makes it a good stress test for any parameter-counting code.

**LFM2.5-8B-A1B** is sparse MoE — 8.3B total, ~1.5B active per token — built on
Liquid AI's LFM2 backbone rather than a plain transformer. Routing means the
per-token compute and the parameter count diverge sharply, and expert weights
are a distinct tensor layout from dense FFN weights.

## Feasibility on this machine

Established earlier: 16 GB unified RAM, CPU only. That splits cleanly.

| Task | Feasible? |
|---|---|
| `raw_tensors()` / `read_tensor()` — shapes, dtypes, per-tensor stats | **Yes, all four.** Nothing is materialised, so size does not matter. |
| Quantised inference via Ollama (Q4, ~5 GB) | **Yes**, where a GGUF exists — expect ~5 tok/s. |
| Full `torch` load for activations / hooks | **No.** 7B at float32 is 28 GB; bfloat16 is ~14 GB against 15.61 GB total. |

So these four are usable for **weight-level** research immediately, and for
activation-level work only via their smaller siblings (Qwen3.5-2B, Gemma 4 E2B).

## Warning for the model-agnostic code

This shortlist is a real test of the "changing model is a text change" rule, and
parts of it will fail — by design:

- `models.blocks()` finds the block stack structurally, so it should hold across
  all four.
- `models.find(model, "attn|self_attn")` will return **fewer matches than there
  are blocks** on Falcon-H1R (only some layers are attention) and may return
  nothing useful on LFM2.5's conv blocks. That is the correct outcome — a loud
  mismatch rather than silently analysing the wrong tensors.
- Any code assuming `n_layers == len(attention_layers)` breaks on Falcon-H1R.
- Parameter-count code must decide explicitly between total, active, and
  effective parameters. Gemma 4 E4B and LFM2.5-8B-A1B disagree on all three.

## Verification status

Release dates, parameter counts, and architecture claims were taken from the
sources below rather than from model memory. **Licences were not individually
verified** — Qwen3.5 and Gemma 4 are reported as Apache-2.0; Falcon-H1R and
LFM2.5 ship under their vendors' own licences and should be checked before any
commercial use.

Two candidates were considered and rejected: **Mamba-3** (Apache-2.0,
17 Mar 2026) is a genuine pure-SSM architecture but published checkpoints stop
around 1.5B, so it does not meet the ~7B bar; **Nemotron 3 Nano** is a
Mamba-2/attention MoE hybrid but is 31.6B total (3B active), too large here.
