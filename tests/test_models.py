import os
from pathlib import Path

import pytest
import torch
from torch import nn

from research import models, paths


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


# --- every model must live under <project>/models -------------------------


def test_assert_is_local_accepts_a_hub_repo_id():
    models.assert_is_local("gpt2")           # not a path; resolves via the cache
    models.assert_is_local("google/gemma-3-1b-it")


def test_assert_is_local_accepts_a_path_inside_models():
    from research.paths import MODELS

    inside = MODELS / "models--gpt2"
    inside.mkdir(parents=True, exist_ok=True)
    models.assert_is_local(inside)


def test_assert_is_local_rejects_a_path_outside_models(tmp_path):
    external = tmp_path / "some-model"
    external.mkdir()
    with pytest.raises(models.ExternalModelError, match="outside"):
        models.assert_is_local(external)


def test_assert_is_local_rejects_the_profile_cache_on_c():
    """The specific thing this guards: reusing weights from ~/.cache/huggingface."""
    profile_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not profile_cache.exists():
        pytest.skip("no profile cache on this machine")
    with pytest.raises(models.ExternalModelError):
        models.assert_is_local(profile_cache)


def test_assert_cache_is_local_passes_under_normal_import_order():
    models.assert_cache_is_local()


def test_assert_cache_is_local_detects_a_late_pin(monkeypatch):
    """Simulates transformers being imported before research."""
    import huggingface_hub.constants as hub_constants

    elsewhere = str(Path.home() / ".cache" / "huggingface" / "hub")
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", elsewhere)
    with pytest.raises(models.ExternalModelError, match="Import `research` before"):
        models.assert_cache_is_local()


def test_pin_overrides_an_ambient_env_var():
    """The pin is forced, not a default - a stray env var must not win."""
    import subprocess
    import sys

    from research.paths import MODELS

    env = dict(os.environ, HF_HUB_CACHE="C:/somewhere/else")
    out = subprocess.run(
        [sys.executable, "-c", "import research, os; print(os.environ['HF_HUB_CACHE'])"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert out.stdout.strip() == str(MODELS)


# --- architecture-agnostic discovery ---------------------------------------


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp(self.attn(x))


class GPT2Style(nn.Module):
    """Blocks under transformer.h, as GPT-2 names them."""

    def __init__(self, n=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(Block() for _ in range(n))
        self.transformer.ln_f = nn.LayerNorm(8)


class LlamaStyle(nn.Module):
    """Blocks under model.layers, as Llama/Gemma/Qwen name them."""

    def __init__(self, n=6):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Block() for _ in range(n))
        self.model.norm = nn.LayerNorm(8)


def test_blocks_found_under_gpt2_naming():
    assert models.block_names(GPT2Style(4)) == [f"transformer.h.{i}" for i in range(4)]


def test_blocks_found_under_llama_naming():
    """The same code must work across families - that is the whole point."""
    assert models.block_names(LlamaStyle(6)) == [f"model.layers.{i}" for i in range(6)]


def test_block_count_matches_the_stack():
    assert len(models.blocks(LlamaStyle(6))) == 6


def test_blocks_raises_when_there_is_no_stack(tiny):
    with pytest.raises(LookupError):
        models.blocks(tiny)


def test_find_matches_across_naming_conventions():
    pattern = "attn|c_attn"
    assert len(models.find(GPT2Style(4), pattern)) == 4
    assert len(models.find(LlamaStyle(6), pattern)) == 6


def test_find_returns_nothing_for_a_non_match():
    assert models.find(GPT2Style(2), "definitely_not_a_module") == []


# --- results land in results/<model>/ ---------------------------------------


def test_model_slug_handles_org_prefixed_ids():
    assert paths.model_slug("google/gemma-3-1b-it") == "google_gemma-3-1b-it"


def test_model_slug_handles_ollama_tags():
    assert paths.model_slug("gemma3:1b") == "gemma3_1b"


def test_model_slug_leaves_a_plain_id_alone():
    assert paths.model_slug("gpt2") == "gpt2"


def test_model_slug_rejects_an_empty_id():
    with pytest.raises(ValueError):
        paths.model_slug("///")


def test_results_dir_is_under_results_and_named_for_the_model():
    d = paths.results_dir("google/gemma-3-1b-it")
    assert d.parent == paths.RESULTS
    assert d.name == "google_gemma-3-1b-it"
    assert d.is_dir()


def test_results_dir_separates_models():
    assert paths.results_dir("gpt2") != paths.results_dir("distilgpt2")


def test_results_dir_can_skip_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    d = paths.results_dir("gpt2", create=False)
    assert not d.exists()
