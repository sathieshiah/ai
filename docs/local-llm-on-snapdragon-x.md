# Which AI models can run on this laptop

Measured 2026-08-21 on the host machine. Numbers marked **measured** came from
`ollama run --verbose`; others are extrapolations and labelled as such.

## 1. The hardware, and what actually constrains it

| | |
|---|---|
| Machine | ASUS Vivobook 16 X1607QA |
| SoC | Snapdragon X (X1-26-100), 8-core Oryon @ 2.96 GHz |
| GPU | Adreno X1-45 (integrated, no dedicated VRAM) |
| NPU | Hexagon, ~45 TOPS |
| RAM | 16 GB LPDDR5X-8448, soldered, 15.61 GB usable |
| OS | Windows 11 Home ARM64 (build 26200) |
| Disk | C: 141 GB free · D: 591 GB free |

Three constraints matter, in this order:

1. **16 GB of *unified* RAM is the hard ceiling.** There is no discrete GPU and no
   separate VRAM pool — model weights, KV cache, Windows, and the browser all draw
   on the same 15.61 GB. At the time of measurement only **1.86 GB was free**.
   Realistic budget for a model: **~9 GB**, and that assumes closing most apps.
2. **Everything runs on the CPU.** `ollama ps` reports `100% CPU` while a model is
   loaded. The Adreno GPU and the 45-TOPS NPU are both idle. This is not a
   misconfiguration — Ollama and llama.cpp have no NPU backend on Snapdragon, and
   GGUF weights cannot execute on the Hexagon NPU at all.
3. **Throughput is memory-bandwidth bound**, so tokens/sec scales inversely with
   the size of the weights being read per token.

## 2. Measured baseline

| Model | Quant | On disk | Load time | Generation |
|---|---|---|---|---|
| `gemma3:1b` | Q4 | 815 MB | 10.1 s | **37.9 tok/s** (measured) |
| `deepseek-r1:8b` | Q4 | 5.2 GB | 15.1 s | **5.85 tok/s** (measured) |

Prompt-eval on the 8B ran at 2.9 tok/s, so long prompts are expensive too — a
2,000-token context costs roughly 11 minutes just to ingest.

## 3. What fits

Estimates below extrapolate from the two measured points.

| Class | Example (Aug 2026) | Size @ Q4 | Verdict |
|---|---|---|---|
| 1–2 B | `gemma3:1b` | 0.8 GB | Comfortable · ~38 tok/s (measured) |
| 4 B | `gemma4:e2b` (7.2 GB), Qwen 3.5 4B | 2.5–3 GB | Good · ~12 tok/s (est.) |
| 8–9 B | `deepseek-r1:8b`, Qwen 3.5 9B | 5–6 GB | **Sweet spot** · 5.9 tok/s (measured) |
| 12 B | `gemma4:12b` | 7.6 GB | Marginal · ~4 tok/s (est.), close all apps |
| 26–31 B | `gemma4:26b` (18 GB), `gemma4:31b` (20 GB), `qwen3.8:27b` (18 GB) | 18–20 GB | **Does not fit** — exceeds 15.61 GB total RAM |

**MoE does not help here.** `gemma4:26b` activates only 4B parameters per token, so
it is compute-cheap — but all 26B of weights must still be resident. Mixture-of-
experts trades memory for speed, which is exactly the wrong trade on a 16 GB machine.

### Already-installed model that cannot work

`qwen3:30b` is a 30.5B MoE at Q4_K_M, **18 GB on disk against 15.61 GB of RAM**.
It cannot be held in memory and will page against the SSD. It should be removed:

```bash
ollama rm qwen3:30b
```

`deepseek-v3.1:671b-cloud` is a *cloud* tag — it runs on Ollama's servers, not
locally, so it is unaffected by any of the above.

## 4. Using the NPU

The 45-TOPS NPU is reachable, but only outside the GGUF ecosystem. It requires
ONNX weights compiled for Qualcomm's QNN backend. The practical route is
**Microsoft Foundry Local**, which detects the Qualcomm NPU and pulls the
NPU-optimised ONNX variant automatically:

```bash
winget install Microsoft.FoundryLocal
```

Model choice there is much narrower than Ollama's library (Phi-family, some Qwen
and Llama builds) and the models cannot be mixed with Ollama's. The payoff is
throughput at far lower power draw — meaningful on battery.

## 5. Recommendation

- Keep an **8–9 B model at Q4** as the daily driver. That is the largest class
  that leaves the machine usable while loaded.
- Keep **`gemma3:1b`** for anything latency-sensitive — at ~38 tok/s it is the
  only model on this machine that feels interactive.
- Ignore everything above ~12 B regardless of how it benchmarks elsewhere.
- Try Foundry Local if battery life or sustained throughput matters.

## Caveats

`gemma4` tag sizes were read from ollama.com on 2026-08-21 and are quantised
downloads, not parameter counts. Ollama's "newest" listing also showed
`ornith-1.5` (9b/35b/397b), `muse-glimmer:30b`, and `laguna-xs-2.1:33b`; these
were not individually verified, and only the 9B tier would be viable here anyway.
