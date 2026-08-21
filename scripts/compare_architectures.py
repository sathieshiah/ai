"""Compare how different architectures lay out their weights.

Reads safetensors headers straight from the Hub, so nothing is downloaded: the
header carries every tensor name, shape and dtype, which is all a layout
comparison needs. Downloading these four models in full would be ~60 GB for
information that is a few hundred KB.

Writes a per-model tensor manifest to results/<model>/ and a combined table to
results/architecture-comparison.csv.

    uv run python scripts/compare_architectures.py
"""

from __future__ import annotations

import re
from collections import Counter

import research  # noqa: F401  - pins the HF cache to D:/research/models
import pandas as pd
from huggingface_hub import get_safetensors_metadata

from research import paths

MODELS = {
    "tiiuae/Falcon-H1R-7B": "attention + Mamba-2 SSM",
    "Qwen/Qwen3.5-9B": "attention + linear attention",
    "google/gemma-4-E4B": "attention + per-layer embeddings",
    "LiquidAI/LFM2.5-8B-A1B": "attention + short conv + sparse MoE",
}

# Patterns taken from tensor names actually observed in these repos, not
# guessed. Guessed patterns matched everything and were useless.
MECHANISMS = {
    "softmax attention": r"self_attn\.",
    "state-space / linear attn": r"\bA_log\b|\bdt_bias\b",
    "sparse MoE": r"\.experts\.|expert_bias",
    "short convolution": r"conv1d|\.conv\.conv\.",
    "per-layer embedding / altup": r"altup|per_layer|laurel",
    "vision tower": r"\bvisual\b|vision_tower",
    "audio tower": r"audio_tower",
    "QAT ranges": r"input_max|output_max",
}

# Which part of the network a tensor belongs to.
COMPONENTS = {
    "vision": r"\bvisual\b|vision_tower",
    "audio": r"audio_tower",
}


def role(name: str) -> str:
    """Collapse block indices so 'layers.0.q' and 'layers.7.q' share a role."""
    return re.sub(r"\.\d+\.", ".N.", name)


def component(name: str) -> str:
    for label, pattern in COMPONENTS.items():
        if re.search(pattern, name):
            return label
    return "language model"


def layer_index(name: str) -> float:
    """Index of the decoder/encoder layer this tensor belongs to, else NaN."""
    match = re.search(r"layers\.(\d+)\.", name)
    return float(match.group(1)) if match else float("nan")


def numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def summarise(repo_id: str, note: str) -> dict:
    meta = get_safetensors_metadata(repo_id)

    tensors = {}
    for file_meta in meta.files_metadata.values():
        tensors.update(file_meta.tensors)

    frame = pd.DataFrame(
        [
            {
                "name": name,
                "role": role(name),
                "component": component(name),
                "layer": layer_index(name),
                "dtype": info.dtype,
                "shape": tuple(info.shape),
                "params": numel(info.shape),
            }
            for name, info in tensors.items()
        ]
    ).sort_values("name")

    out = paths.results_dir(repo_id)
    frame.to_csv(out / "tensor-manifest.csv", index=False)

    lm = frame[frame["component"] == "language model"]
    lm_layers = lm["layer"].dropna()
    joined = " ".join(frame["name"])

    return {
        "model": repo_id,
        "architecture": note,
        "tensors": len(frame),
        "params": int(frame["params"].sum()),
        "lm params": int(lm["params"].sum()),
        "lm blocks": int(lm_layers.max()) + 1 if len(lm_layers) else 0,
        "lm roles": lm["role"].nunique(),
        "components": sorted(frame["component"].unique()),
        "dtypes": dict(Counter(frame["dtype"])),
        "mechanisms": [
            label
            for label, pattern in MECHANISMS.items()
            if re.search(pattern, joined)
        ],
        "manifest": out / "tensor-manifest.csv",
    }


def main() -> None:
    summaries = []
    for repo_id, note in MODELS.items():
        try:
            summaries.append(summarise(repo_id, note))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL {repo_id}: {type(exc).__name__}: {exc}")

    if not summaries:
        print("nothing to compare")
        return

    for s in summaries:
        print(f"=== {s['model']}")
        print(f"    total params : {s['params']:>15,}   ({s['lm params']:,} in the LM)")
        print(f"    tensors      : {s['tensors']:>15,}")
        print(f"    LM blocks    : {s['lm blocks']:>15}   ({s['lm roles']} distinct roles)")
        print(f"    components   : {', '.join(s['components'])}")
        print(f"    dtypes       : {s['dtypes']}")
        print(f"    mechanisms   : {', '.join(s['mechanisms'])}")
        print()

    table = pd.DataFrame(
        [
            {
                "model": s["model"].split("/")[-1],
                "params": s["params"],
                "lm_params": s["lm params"],
                "tensors": s["tensors"],
                "lm_blocks": s["lm blocks"],
                "lm_roles": s["lm roles"],
                "components": "+".join(s["components"]),
                "mechanisms": ", ".join(s["mechanisms"]),
            }
            for s in summaries
        ]
    )
    combined = paths.RESULTS / "architecture-comparison.csv"
    combined.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(combined, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {combined}")


if __name__ == "__main__":
    main()
