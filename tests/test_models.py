import pytest
import torch
from torch import nn

from research import models


class Tiny(nn.Module):
    """Small stand-in so the core tests need no downloaded weights."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.block = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8))
        self.head = nn.Linear(8, 16)

    def forward(self, x):
        return self.head(self.block(self.embed(x)))


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return Tiny()


def test_layers_lists_named_submodules(tiny):
    names = [n for n, _ in models.layers(tiny)]
    assert "embed" in names and "block.0" in names and "head" in names


def test_layers_filters_by_class(tiny):
    linear = [n for n, _ in models.layers(tiny, nn.Linear)]
    assert set(linear) == {"block.0", "block.2", "head"}


def test_layer_table_skips_paramless_modules(tiny):
    rows = models.layer_table(tiny)
    names = {r["name"] for r in rows}
    assert "block.1" not in names  # GELU has no parameters
    assert rows[0]["shapes"]["weight"] == (16, 8)
    assert sum(r["params"] for r in rows) == sum(p.numel() for p in tiny.parameters())


class TiedTiny(nn.Module):
    """Weight tying, as GPT-2 does between lm_head and the token embedding."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight  # tied

    def forward(self, x):
        return self.head(self.embed(x))


def test_layer_table_counts_tied_weights_once():
    """Double-counting a tied embedding inflates gpt2 from 124M to 163M."""
    model = TiedTiny()
    rows = models.layer_table(model)
    assert sum(r["params"] for r in rows) == sum(p.numel() for p in model.parameters())


def test_layer_table_records_what_a_tied_weight_is_tied_to():
    rows = {r["name"]: r for r in models.layer_table(TiedTiny())}
    assert rows["embed"]["tied"] is False
    assert rows["head"]["tied"] is True
    assert rows["head"]["tied_to"] == "embed"
    assert rows["head"]["params"] == 0
    assert rows["head"]["shapes"]["weight"] == (16, 8)  # still reported


def test_weight_returns_detached_tensor(tiny):
    w = models.weight(tiny, "head.weight")
    assert w.shape == (16, 8)
    assert not w.requires_grad


def test_capture_records_activations(tiny):
    with models.capture(tiny, ["block", "head"]) as acts:
        tiny(torch.tensor([[1, 2, 3]]))
    assert acts["block"].shape == (1, 3, 8)
    assert acts["head"].shape == (1, 3, 16)


def test_capture_removes_hooks_afterwards(tiny):
    with models.capture(tiny, ["head"]):
        pass
    assert tiny.head._forward_hooks == {}


def test_capture_removes_hooks_even_on_error(tiny):
    with pytest.raises(RuntimeError):  # noqa: PT012
        with models.capture(tiny, ["head"]):
            raise RuntimeError("boom")
    assert tiny.head._forward_hooks == {}


def test_capture_rejects_unknown_module(tiny):
    with pytest.raises(KeyError):
        with models.capture(tiny, ["nope"]):
            pass


def test_raw_tensors_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        models.raw_tensors(tmp_path)


def test_ollama_blob_rejects_unknown_tag():
    with pytest.raises(FileNotFoundError):
        models.ollama_blob("definitely-not-a-real-model:9b")


@pytest.mark.slow
def test_ollama_blob_and_gguf_roundtrip():
    blob = models.ollama_blob("gemma3:1b")
    assert blob.is_file()
    tensors = models.gguf_tensors(blob)
    assert any(t["name"] == "token_embd.weight" for t in tensors)


def test_hf_cache_is_pinned_to_project_models_dir():
    """Importing `research` must redirect HF downloads onto D:, not C:."""
    import os

    from research.paths import MODELS

    assert os.environ["HF_HUB_CACHE"] == str(MODELS)


def test_huggingface_hub_actually_resolves_to_project_models_dir():
    """The pin is only real if huggingface_hub picked it up at import time."""
    from pathlib import Path

    import huggingface_hub.constants as hub_constants

    from research.paths import MODELS

    assert Path(hub_constants.HF_HUB_CACHE) == MODELS
