"""Weighted-embedding feedback, checked against arithmetic done by hand.

The notebook feeds back a probability-weighted average of the whole embedding
table instead of one token's row. These tests execute the notebook's own tagged
cells rather than a copy of them - a copy would drift, and the copy is not what
anyone runs.

The toy model's embedding table is the identity matrix, which is what makes the
claims checkable exactly rather than approximately: with ``E = I`` the mixed
vector *is* the probability vector, so ``p @ E == p`` and the model's logits for
the next step are ``p @ log P`` - arithmetic that can be written down. Feeding a
one-hot recovers ordinary hard decoding through the same code path, which is why
the greedy control belongs in the batch rather than in a run of its own.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from torch import nn

# Resolved here rather than through research.paths: the notebook this suite
# tests is self-sufficient and the src/research package is gone, so the tests
# must not depend on it either.
NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"

IMPLEMENTATION_TAG = "weighted-embedding-implementation"


def tagged_cells(path):
    """The implementation cells of one notebook, in order."""
    notebook = nbformat.read(path, as_version=4)
    return [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and IMPLEMENTATION_TAG in cell.metadata.get("tags", [])
    ]


def notebooks_with_an_implementation():
    """Every notebook carrying a tagged copy of the algorithm.

    Recursive, so copies in subfolders are covered. Hidden directories are
    skipped: ``.ipynb_checkpoints/`` holds Jupyter's autosave copies, which are
    stale by nature and would fail the moment one lags the notebook it shadows.
    """
    return [
        path
        for path in sorted(NOTEBOOKS.rglob("*.ipynb"))
        if not any(part.startswith(".") for part in path.relative_to(NOTEBOOKS).parts)
        if tagged_cells(path)
    ]


IMPLEMENTATIONS = notebooks_with_an_implementation()


def test_at_least_one_notebook_carries_the_implementation():
    assert IMPLEMENTATIONS, (
        f"no notebook under {NOTEBOOKS} has a cell tagged {IMPLEMENTATION_TAG!r}; "
        "the tests below would silently test nothing"
    )


@pytest.fixture(scope="module", params=IMPLEMENTATIONS, ids=lambda path: path.stem)
def wef(request):
    """Execute one notebook's tagged cells and expose the names they define."""
    namespace = {"__name__": "weighted_embedding_notebook"}
    for index, source in enumerate(tagged_cells(request.param)):
        exec(compile(source, f"{request.param.name}:implementation{index}", "exec"), namespace)
    return SimpleNamespace(**namespace)


VOCAB = "abcd"
V = len(VOCAB)

# Row i is P(next token | the input vector was one-hot on i). Deliberately
# skewed so the argmax is unambiguous at every step and no two rows agree.
PROBS = torch.tensor(
    [
        [0.05, 0.80, 0.10, 0.05],  # from a
        [0.10, 0.10, 0.20, 0.60],  # from b
        [0.05, 0.75, 0.15, 0.05],  # from c
        [0.42, 0.30, 0.20, 0.08],  # from d
    ]
)


class ToyTokenizer:
    """Character-level stand-in: token id i is the character ``VOCAB[i]``."""

    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[VOCAB.index(char) for char in text]])}

    def decode(self, ids):
        return "".join(VOCAB[int(i)] for i in ids)


class ToyModel(nn.Module):
    """Logits are the last input vector times ``log PROBS``, and calls are recorded.

    The embedding table is the identity, so embedding token i yields the one-hot
    ``e_i`` and the logits become ``log PROBS[i]`` exactly - already normalised,
    so the ``log_softmax`` inside the decoder returns them unchanged. A mixed
    input ``p`` yields ``p @ log PROBS``, which is what makes the soft path
    checkable by hand.
    """

    #: How this architecture applies its multiplier, if it has one. The three
    #: values reproduce the three conventions found in transformers.
    multiplier = 1.0
    applies_to = "none"     # "none" | "ids-only" | "always"

    def __init__(self, probs=PROBS, eos_token_id=None):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(torch.eye(V), freeze=True)
        self.log_probs = probs.log()
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.config = SimpleNamespace(hidden_size=V,
                                      embedding_multiplier=self.multiplier)
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def embed_input(self, input_ids, inputs_embeds):
        """Resolve the two input forms exactly as transformers models do."""
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
            if self.applies_to == "ids-only":
                inputs_embeds = inputs_embeds * self.multiplier
        if self.applies_to == "always":
            inputs_embeds = inputs_embeds * self.multiplier
        return inputs_embeds

    def logits_from(self, hidden):
        return hidden @ self.log_probs

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                logits_to_keep=0, **kwargs):
        hidden = self.embed_input(input_ids, inputs_embeds)
        self.calls.append(
            {
                "batch": hidden.shape[0],
                "length": hidden.shape[1],
                "logits_to_keep": logits_to_keep,
                "by_ids": inputs_embeds is None,
                "kwargs": dict(kwargs),
            }
        )
        logits = self.logits_from(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class FalconStyleToyModel(ToyModel):
    """Multiplies only inside the ``inputs_embeds is None`` branch (FalconH1)."""

    multiplier = 5.656854249492381
    applies_to = "ids-only"


class GraniteStyleToyModel(ToyModel):
    """Multiplies after the branch, so it applies either way (GraniteMoeHybrid)."""

    multiplier = 12.0
    applies_to = "always"


class UnreconcilableToyModel(ToyModel):
    """Transforms its input in a way no single scalar can undo."""

    def embed_input(self, input_ids, inputs_embeds):
        if inputs_embeds is None:
            return self.embed(input_ids) + 3.0      # a shift, not a scale
        return inputs_embeds


class ToyModelWithoutLogitsToKeep(ToyModel):
    """An architecture whose forward has no logits-trimming argument."""

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        return super().forward(input_ids, inputs_embeds, attention_mask,
                               logits_to_keep=0, **kwargs)


def hard_greedy(prompt, num_tokens, probs=PROBS):
    """Ordinary greedy decoding, done directly. The reference to match."""
    last = VOCAB.index(prompt[-1])
    tokens = []
    for _ in range(num_tokens):
        last = int(torch.argmax(probs[last]))
        tokens.append(last)
    return tokens


@pytest.fixture
def toy():
    return ToyModel(), ToyTokenizer()


# --- the notebook is the implementation -----------------------------------


def test_the_notebook_exposes_the_expected_entry_points(wef):
    """If a rename breaks this, every other test below is testing nothing."""
    for name in (
        "GREEDY",
        "SOFT",
        "COMMITTED",
        "DecodeConfig",
        "build_configs",
        "validate_configs",
        "mixture_weights",
        "mix_embeddings",
        "rescale_rows",
        "nearest_token",
        "weighted_embedding_decode",
        "divergence_step",
    ):
        assert hasattr(wef, name), f"the notebook no longer defines {name!r}"


# --- the configuration cross ----------------------------------------------


def test_build_configs_crosses_every_axis_and_ends_with_the_control(wef):
    configs = wef.build_configs([0.5, 1.0], ["full", "top5"], ["raw", "rescaled"],
                                [wef.SOFT, wef.COMMITTED])

    assert len(configs) == 2 * 2 * 2 * 2 + 1
    assert configs[-1].mixture == wef.GREEDY, "the control must be the last row"
    assert sum(1 for c in configs if c.mixture == wef.GREEDY) == 1

    soft = [c for c in configs if c.mixture != wef.GREEDY]
    assert {c.temperature for c in soft} == {0.5, 1.0}
    assert {c.mixture for c in soft} == {"full", "top5"}
    assert {c.rescale for c in soft} == {True, False}
    assert {c.history for c in soft} == {wef.SOFT, wef.COMMITTED}
    assert configs[-1].history == wef.SOFT, (
        "the control needs no history variant: its mixture is already one-hot"
    )
    assert len({c.name for c in configs}) == len(configs), "names must be unique"


def test_validate_rejects_configurations_that_cannot_mean_what_they_say(wef):
    good = wef.build_configs([1.0], ["full"], ["raw"])

    with pytest.raises(ValueError, match="at least 1"):
        wef.validate_configs(good, 0, 10, 5)
    with pytest.raises(ValueError, match="temperature must be positive"):
        wef.validate_configs(
            [wef.DecodeConfig("bad", 0.0, "full", False)], 10, 10, 5)
    with pytest.raises(ValueError, match="unknown mixture"):
        wef.validate_configs(
            [wef.DecodeConfig("bad", 1.0, "nonsense", False)], 10, 10, 5)
    with pytest.raises(ValueError, match="unknown history"):
        wef.validate_configs(
            [wef.DecodeConfig("bad", 1.0, "full", False, "sometimes")], 10, 10, 5)
    with pytest.raises(ValueError, match="unique"):
        wef.validate_configs(
            [wef.DecodeConfig("same", 1.0, "full", False),
             wef.DecodeConfig("same", 1.0, "top5", False)], 10, 10, 5)


# --- the mixture itself ----------------------------------------------------


def test_full_mixture_is_the_distribution_unchanged(wef):
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    configs = [wef.DecodeConfig("full", 1.0, "full", False)]

    weights = wef.mixture_weights(probs, configs, mixture_top_k=2)

    assert torch.allclose(weights, probs), "every token must contribute its own mass"


def test_truncated_mixture_keeps_k_tokens_and_renormalises(wef):
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    configs = [wef.DecodeConfig("top", 1.0, "top5", False)]

    weights = wef.mixture_weights(probs, configs, mixture_top_k=2)

    assert int((weights > 0).sum()) == 2, "only the top k may contribute"
    assert weights[0, 0] == 0 and weights[0, 1] == 0, "the tail must be dropped"
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-6), "must renormalise"
    # 0.4 and 0.3 renormalised over their own total.
    assert math.isclose(float(weights[0, 3]), 0.4 / 0.7, rel_tol=1e-6)


def test_greedy_mixture_is_one_hot_on_the_argmax(wef):
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    configs = [wef.DecodeConfig("g", 1.0, wef.GREEDY, False)]

    weights = wef.mixture_weights(probs, configs, mixture_top_k=2)

    assert float(weights[0, 3]) == 1.0
    assert float(weights.sum()) == 1.0, "a one-hot carries all the mass"


def test_each_row_is_mixed_according_to_its_own_configuration(wef):
    """The whole point of batching: one forward pass, three different rules."""
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 3)
    configs = [
        wef.DecodeConfig("a", 1.0, "full", False),
        wef.DecodeConfig("b", 1.0, "top5", False),
        wef.DecodeConfig("c", 1.0, wef.GREEDY, False),
    ]

    weights = wef.mixture_weights(probs, configs, mixture_top_k=2)

    assert int((weights[0] > 0).sum()) == 4
    assert int((weights[1] > 0).sum()) == 2
    assert int((weights[2] > 0).sum()) == 1


def test_mixing_against_an_identity_table_returns_the_weights(wef):
    """``p @ I == p`` - the arithmetic the rest of the suite leans on."""
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    mixed = wef.mix_embeddings(weights, torch.eye(V))

    assert torch.allclose(mixed, weights, atol=1e-6)


# --- magnitude -------------------------------------------------------------


def test_rescaling_touches_only_the_rows_that_ask_for_it(wef):
    fed = torch.tensor([[3.0, 4.0], [3.0, 4.0]])       # both norm 5
    configs = [
        wef.DecodeConfig("raw", 1.0, "full", False),
        wef.DecodeConfig("scaled", 1.0, "full", True),
    ]

    out = wef.rescale_rows(fed, configs, target_norm=torch.tensor(1.0))

    assert torch.allclose(out[0], fed[0]), "the raw row must be untouched"
    assert math.isclose(float(out[1].norm()), 1.0, rel_tol=1e-6)
    # Direction is preserved; only the magnitude changes.
    assert torch.allclose(out[1] * 5.0, fed[1], atol=1e-5)


def test_a_flatter_mixture_is_shorter_than_a_peaked_one(wef):
    """The reason the magnitude axis exists, stated as a test."""
    table = torch.eye(V)
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    flat = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    assert (wef.mix_embeddings(flat, table).norm()
            < wef.mix_embeddings(peaked, table).norm())


def test_nearest_token_is_one_for_a_real_embedding(wef):
    table = torch.eye(V)
    norms = table.norm(dim=-1)

    cosine, ids = wef.nearest_token(table[2:3].clone(), table, norms)

    assert math.isclose(float(cosine[0]), 1.0, rel_tol=1e-5)
    assert int(ids[0]) == 2


# --- end to end ------------------------------------------------------------


def test_the_control_row_reproduces_ordinary_greedy_decoding(wef, toy):
    """The strongest claim here: one-hot feedback is hard decoding, exactly."""
    model, tokenizer = toy
    configs = [wef.DecodeConfig("greedy", 1.0, wef.GREEDY, False)]

    result = wef.weighted_embedding_decode(
        model, tokenizer, "a", configs, num_tokens_to_generate=6,
        record_top_k=2, mixture_top_k=2)

    assert result.readouts["greedy"] == hard_greedy("a", 6)


def test_a_near_zero_temperature_full_mixture_collapses_onto_greedy(wef, toy):
    """As temperature falls the mixture approaches the argmax's own embedding."""
    model, tokenizer = toy
    configs = [
        wef.DecodeConfig("cold", 0.01, "full", False),
        wef.DecodeConfig("greedy", 1.0, wef.GREEDY, False),
    ]

    result = wef.weighted_embedding_decode(
        model, tokenizer, "a", configs, num_tokens_to_generate=6,
        record_top_k=2, mixture_top_k=2)

    assert result.readouts["cold"] == result.readouts["greedy"]
    cold = [s for s in result.steps if s["config"] == "cold"]
    assert all(s["cosine_to_nearest"] > 0.99 for s in cold), (
        "a cold mixture should sit almost exactly on a real token embedding"
    )


def test_a_hot_full_mixture_leaves_the_vocabulary(wef, toy):
    """The finding the notebook exists to measure, in miniature."""
    model, tokenizer = toy
    configs = [
        wef.DecodeConfig("hot", 100.0, "full", False),
        wef.DecodeConfig("greedy", 1.0, wef.GREEDY, False),
    ]

    result = wef.weighted_embedding_decode(
        model, tokenizer, "a", configs, num_tokens_to_generate=4,
        record_top_k=2, mixture_top_k=2)

    hot = [s for s in result.steps if s["config"] == "hot"]
    greedy = [s for s in result.steps if s["config"] == "greedy"]
    assert all(s["cosine_to_nearest"] > 0.99 for s in greedy)
    assert all(s["cosine_to_nearest"] < 0.99 for s in hot), (
        "a flat mixture is not any real token's embedding"
    )


def test_every_row_runs_the_full_budget_even_after_the_argmax_is_eos(wef):
    """EOS is recorded and ignored, so the rows stay step-aligned."""
    model = ToyModel(eos_token_id=3)          # "d", which greedy reaches at step 2
    tokenizer = ToyTokenizer(eos_token_id=3)
    configs = wef.build_configs([1.0], ["full"], ["raw"])

    result = wef.weighted_embedding_decode(
        model, tokenizer, "a", configs, num_tokens_to_generate=7,
        record_top_k=2, mixture_top_k=2)

    assert any(s["argmax_is_eos"] for s in result.steps), "the fixture must hit EOS"
    for config in configs:
        assert len(result.readouts[config.name]) == 7, (
            f"{config.name} stopped early; every row must run the full budget"
        )


def test_records_have_one_row_per_config_step_and_rank(wef, toy):
    model, tokenizer = toy
    configs = wef.build_configs([0.5, 1.0], ["full", "top5"], ["raw", "rescaled"])
    steps, top_k = 5, 3

    result = wef.weighted_embedding_decode(
        model, tokenizer, "a", configs, num_tokens_to_generate=steps,
        record_top_k=top_k, mixture_top_k=2)

    assert len(result.steps) == len(configs) * steps
    assert len(result.top_tokens) == len(configs) * steps * top_k
    assert {row["rank"] for row in result.top_tokens} == {1, 2, 3}
    # Recorded probabilities must be ordered within a step.
    first = [r for r in result.top_tokens
             if r["config"] == configs[0].name and r["step"] == 1]
    assert first == sorted(first, key=lambda r: -r["probability"])


def test_every_configuration_shares_the_first_step(wef, toy):
    """All rows start from the same prompt, so step 1 cannot differ."""
    model, tokenizer = toy
    configs = wef.build_configs([0.5, 2.0], ["full", "top5"], ["raw", "rescaled"])

    result = wef.weighted_embedding_decode(
        model, tokenizer, "a", configs, num_tokens_to_generate=3,
        record_top_k=4, mixture_top_k=2)

    step_one = {row["config"]: row["argmax_token_id"]
                for row in result.steps if row["step"] == 1}
    assert len(set(step_one.values())) == 1, (
        "temperature is monotonic and cannot reorder the first step's logits"
    )


def test_the_sequence_grows_by_exactly_one_vector_per_step(wef, toy):
    model, tokenizer = toy
    configs = wef.build_configs([1.0], ["full"], ["raw"])

    result = wef.weighted_embedding_decode(
        model, tokenizer, "ab", configs, num_tokens_to_generate=4,
        record_top_k=2, mixture_top_k=2)

    # Calibration runs first: one reference pass by token ids, then one probe
    # per candidate scale. The decode loop starts after those.
    decoding = model.calls[1 + len(result.scale_errors):]
    lengths = [call["length"] for call in decoding]
    assert lengths == [2, 3, 4, 5], "one continuous vector appended per step"
    assert all(call["batch"] == len(configs) for call in decoding)
    assert all(call["kwargs"].get("use_cache") is False for call in decoding)


def test_it_runs_on_an_architecture_without_logits_to_keep(wef):
    """Not every causal LM exposes the logits-trimming argument."""
    model = ToyModelWithoutLogitsToKeep()
    configs = wef.build_configs([1.0], ["full"], ["raw"])

    result = wef.weighted_embedding_decode(
        model, ToyTokenizer(), "a", configs, num_tokens_to_generate=3,
        record_top_k=2, mixture_top_k=2)

    assert len(result.steps) == len(configs) * 3


def test_an_empty_prompt_is_refused(wef, toy):
    model, tokenizer = toy
    configs = wef.build_configs([1.0], ["full"], ["raw"])

    with pytest.raises(ValueError, match="tokenised to nothing"):
        wef.weighted_embedding_decode(model, tokenizer, "", configs,
                                      num_tokens_to_generate=2)


# --- divergence ------------------------------------------------------------


def test_divergence_is_none_for_a_row_that_never_leaves_the_control(wef):
    readouts = {"greedy": [1, 2, 3], "same": [1, 2, 3], "late": [1, 2, 0]}

    diverged = wef.divergence_step(readouts, control="greedy")

    assert diverged["same"] is None
    assert diverged["late"] == 3
    assert "greedy" not in diverged, "the control cannot diverge from itself"


# --- history ---------------------------------------------------------------


class CausalMeanToyModel(ToyModel):
    """Logits depend on the running mean of the context, so history matters.

    ``ToyModel`` reads only the last position, which makes it blind to the
    history axis by construction: committing a passed position cannot change an
    output that never looked at it. This averages every position up to and
    including each one - the crudest possible stand-in for attention - so a
    context of mixtures and a context of committed tokens genuinely differ.
    """

    def logits_from(self, hidden):
        divisor = torch.arange(1, hidden.shape[1] + 1, device=hidden.device).view(1, -1, 1)
        return (hidden.cumsum(dim=1) / divisor) @ self.log_probs


class RecordingModel(ToyModel):
    """Keeps every ``inputs_embeds`` it was handed, so the context can be inspected."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            self.seen.append(inputs_embeds.detach().clone())
        return super().forward(input_ids, inputs_embeds, **kwargs)


def one_hot_positions(sequence):
    """Which positions of a (length, vocab) context are exact one-hots.

    Only meaningful because the toy's embedding table is the identity: a real
    token's embedding is a one-hot, and a mixture of two or more is not.
    """
    return [
        bool(torch.isclose(row.max(), torch.tensor(1.0), atol=1e-4)
             and int((row.abs() > 1e-4).sum()) == 1)
        for row in sequence
    ]


def test_a_soft_history_keeps_every_generated_position_continuous(wef):
    model, tokenizer = RecordingModel(), ToyTokenizer()
    configs = [wef.DecodeConfig("soft", 2.0, "full", False, wef.SOFT)]

    wef.weighted_embedding_decode(model, tokenizer, "ab", configs,
                                  num_tokens_to_generate=5, record_top_k=2,
                                  mixture_top_k=2)

    final = model.seen[-1][0]                       # (length, vocab), last call
    hard = one_hot_positions(final)
    assert hard[:2] == [True, True], "the prompt must stay discrete"
    assert not any(hard[2:]), (
        "with a soft history every generated position stays a mixture"
    )


def test_a_committed_history_leaves_exactly_one_continuous_position(wef):
    model, tokenizer = RecordingModel(), ToyTokenizer()
    configs = [wef.DecodeConfig("committed", 2.0, "full", False, wef.COMMITTED)]

    wef.weighted_embedding_decode(model, tokenizer, "ab", configs,
                                  num_tokens_to_generate=5, record_top_k=2,
                                  mixture_top_k=2)

    final = model.seen[-1][0]
    hard = one_hot_positions(final)
    assert hard[:2] == [True, True], "the prompt must stay discrete"
    assert all(hard[2:-1]), "every passed position must have been committed"
    assert not hard[-1], "the newest position is still the mixture"
    assert sum(1 for h in hard if not h) == 1


def test_committed_positions_hold_the_token_their_own_step_named(wef):
    """Committing writes back the argmax that was recorded, not something else."""
    model, tokenizer = RecordingModel(), ToyTokenizer()
    configs = [wef.DecodeConfig("committed", 2.0, "full", False, wef.COMMITTED)]

    result = wef.weighted_embedding_decode(model, tokenizer, "ab", configs,
                                           num_tokens_to_generate=5, record_top_k=2,
                                           mixture_top_k=2)

    final = model.seen[-1][0]
    readout = result.readouts["committed"]
    # Generated positions start at index 2; the last is still soft, so the
    # committed ones correspond to readout[0], readout[1], ...
    for offset, token_id in enumerate(readout[: final.shape[0] - 3]):
        assert int(torch.argmax(final[2 + offset])) == token_id


def test_the_two_histories_are_identical_until_the_second_step(wef):
    """They can only differ once there is a passed position to commit."""
    model, tokenizer = CausalMeanToyModel(), ToyTokenizer()
    configs = [
        wef.DecodeConfig("soft", 2.0, "full", False, wef.SOFT),
        wef.DecodeConfig("committed", 2.0, "full", False, wef.COMMITTED),
    ]

    result = wef.weighted_embedding_decode(model, tokenizer, "a", configs,
                                           num_tokens_to_generate=6, record_top_k=2,
                                           mixture_top_k=2)

    assert result.readouts["soft"][:2] == result.readouts["committed"][:2]
    assert result.readouts["soft"] != result.readouts["committed"], (
        "an accumulating history must eventually diverge from a committed one"
    )


def test_the_history_axis_is_recorded_on_every_step_row(wef, toy):
    model, tokenizer = toy
    configs = wef.build_configs([1.0], ["full"], ["raw"], [wef.SOFT, wef.COMMITTED])

    result = wef.weighted_embedding_decode(model, tokenizer, "a", configs,
                                           num_tokens_to_generate=3, record_top_k=2,
                                           mixture_top_k=2)

    assert {row["history"] for row in result.steps} == {wef.SOFT, wef.COMMITTED}


def test_committing_does_not_change_the_greedy_control(wef):
    """Its mixture is already one-hot, so a commit rewrites a vector to itself."""
    model, tokenizer = ToyModel(), ToyTokenizer()
    plain = wef.DecodeConfig("g-soft", 1.0, wef.GREEDY, False, wef.SOFT)
    committed = wef.DecodeConfig("g-committed", 1.0, wef.GREEDY, False, wef.COMMITTED)

    result = wef.weighted_embedding_decode(model, tokenizer, "a", [plain, committed],
                                           num_tokens_to_generate=6, record_top_k=2,
                                           mixture_top_k=2)

    assert result.readouts["g-soft"] == result.readouts["g-committed"]
    assert result.readouts["g-soft"] == hard_greedy("a", 6)


def test_a_last_only_history_keeps_a_single_position_context(wef):
    """The context is replaced, not appended to: one position from step 2 on."""
    model, tokenizer = RecordingModel(), ToyTokenizer()
    configs = [wef.DecodeConfig("window", 2.0, "full", False, wef.LAST_ONLY)]

    result = wef.weighted_embedding_decode(model, tokenizer, "abc", configs,
                                           num_tokens_to_generate=5, record_top_k=2,
                                           mixture_top_k=2)

    lengths = [seen.shape[1] for seen in model.seen[len(result.scale_errors):]]
    assert lengths == [3, 1, 1, 1, 1], (
        "step 1 sees the prompt because nothing else exists yet; after that the "
        "context is the newest mixture alone - not even the prompt survives"
    )


def test_last_only_rows_run_as_their_own_forward_pass(wef):
    """Two context lengths cannot share a batch, so there are two passes."""
    model, tokenizer = RecordingModel(), ToyTokenizer()
    configs = [
        wef.DecodeConfig("grow", 2.0, "full", False, wef.SOFT),
        wef.DecodeConfig("window", 2.0, "full", False, wef.LAST_ONLY),
    ]

    result = wef.weighted_embedding_decode(model, tokenizer, "ab", configs,
                                           num_tokens_to_generate=4, record_top_k=2,
                                           mixture_top_k=2)

    decoding = model.seen[len(result.scale_errors):]
    lengths = [seen.shape[1] for seen in decoding]
    assert lengths == [2, 2, 3, 1, 4, 1, 5, 1], (
        "each step runs the growing group then the one-position group"
    )
    assert all(seen.shape[0] == 1 for seen in decoding), "one row in each group"


def test_a_last_only_history_diverges_from_a_full_one(wef):
    model, tokenizer = CausalMeanToyModel(), ToyTokenizer()
    configs = [
        wef.DecodeConfig("soft", 1.0, "full", False, wef.SOFT),
        wef.DecodeConfig("window", 1.0, "full", False, wef.LAST_ONLY),
    ]

    result = wef.weighted_embedding_decode(model, tokenizer, "ab", configs,
                                           num_tokens_to_generate=8, record_top_k=2,
                                           mixture_top_k=2)

    assert result.readouts["soft"][0] == result.readouts["window"][0], (
        "step 1 is the prompt for both, so it cannot differ"
    )

    # Compared on the distributions, not the readouts: two trajectories can
    # settle on the same argmax while being computed from different contexts,
    # and it is the computation the axis changes.
    def entropies(name):
        return [round(s["entropy"], 6) for s in result.steps if s["config"] == name]

    assert entropies("soft")[0] == entropies("window")[0]
    assert entropies("soft")[1:] != entropies("window")[1:], (
        "throwing the context away must change what the model computes"
    )


def test_every_history_is_recorded_and_rows_stay_aligned(wef):
    model, tokenizer = CausalMeanToyModel(), ToyTokenizer()
    configs = wef.build_configs([1.0], ["full"], ["raw"],
                                [wef.SOFT, wef.COMMITTED, wef.LAST_ONLY])

    result = wef.weighted_embedding_decode(model, tokenizer, "ab", configs,
                                           num_tokens_to_generate=6, record_top_k=2,
                                           mixture_top_k=2)

    assert {row["history"] for row in result.steps} == {
        wef.SOFT, wef.COMMITTED, wef.LAST_ONLY}
    for config in configs:
        assert len(result.readouts[config.name]) == 6, (
            f"{config.name} must run the full budget like every other row"
        )


# --- embedding scale calibration -------------------------------------------
#
# Architectures disagree about where an embedding multiplier is applied, and
# the disagreement is silent. These reproduce the three conventions found in
# transformers and check the calibration measures each one correctly.


def test_a_model_without_a_multiplier_calibrates_to_one(wef):
    model = ToyModel()
    ids = torch.tensor([[0, 1]])

    scale, errors = wef.calibrate_embedding_scale(model, ids, logits_arg="logits_to_keep")

    assert scale == 1.0
    assert errors[1.0] < 1e-6, "the two input paths already agree"


def test_a_falcon_style_multiplier_is_recovered(wef):
    """Applied only for input_ids, so feeding embeddings must scale them up."""
    model = FalconStyleToyModel()
    ids = torch.tensor([[0, 1]])

    scale, errors = wef.calibrate_embedding_scale(model, ids, logits_arg="logits_to_keep")

    assert math.isclose(scale, model.multiplier, rel_tol=1e-9)
    assert errors[1.0] > 0.02, "the unscaled path must be detectably wrong"


def test_a_granite_style_multiplier_needs_no_correction(wef):
    """Applied either way, so the raw table is already right."""
    model = GraniteStyleToyModel()
    ids = torch.tensor([[0, 1]])

    scale, _ = wef.calibrate_embedding_scale(model, ids, logits_arg="logits_to_keep")

    assert scale == 1.0, (
        "this convention applies the multiplier to whatever it is handed, so "
        "scaling the table too would apply it twice"
    )


def test_an_unreconcilable_model_raises_rather_than_running_wrong(wef):
    model = UnreconcilableToyModel()
    ids = torch.tensor([[0, 1]])

    with pytest.raises(RuntimeError, match="cannot reconcile"):
        wef.calibrate_embedding_scale(model, ids, logits_arg="logits_to_keep")


def test_scale_candidates_come_from_the_model_config(wef):
    candidates = wef.scale_candidates(
        SimpleNamespace(hidden_size=16, embedding_multiplier=7.0))

    assert candidates[0] == 1.0, "no correction is always tried first"
    assert 7.0 in candidates, "the model's own multiplier must be a candidate"
    assert 4.0 in candidates, "sqrt(hidden_size) covers the Gemma convention"
    assert len(candidates) == len(set(candidates)), "no duplicates"


def test_the_decode_feeds_the_calibrated_scale(wef):
    """End to end: a Falcon-style model must be driven at its own input scale."""
    model, tokenizer = FalconStyleToyModel(), ToyTokenizer()
    configs = [wef.DecodeConfig("greedy", 1.0, wef.GREEDY, False)]

    result = wef.weighted_embedding_decode(model, tokenizer, "a", configs,
                                           num_tokens_to_generate=5, record_top_k=2,
                                           mixture_top_k=2)

    assert math.isclose(result.embedding_scale, model.multiplier, rel_tol=1e-9)
    # With the scale applied, one-hot feedback still reproduces hard decoding.
    assert result.readouts["greedy"] == hard_greedy("a", 5)


def test_an_uncalibrated_decode_would_have_decoded_differently(wef):
    """Proof the bug was real: without the correction the readout changes."""
    model, tokenizer = FalconStyleToyModel(), ToyTokenizer()
    configs = [wef.DecodeConfig("greedy", 1.0, wef.GREEDY, False)]

    calibrated = wef.weighted_embedding_decode(model, tokenizer, "a", configs,
                                               num_tokens_to_generate=5,
                                               record_top_k=2, mixture_top_k=2)

    # Same model with the multiplier neutralised stands in for the old code
    # path, which fed the raw table and never applied it.
    naive_model = FalconStyleToyModel()
    naive_model.multiplier = 1.0
    naive = wef.weighted_embedding_decode(naive_model, tokenizer, "a", configs,
                                          num_tokens_to_generate=5,
                                          record_top_k=2, mixture_top_k=2)

    assert calibrated.embedding_scale != naive.embedding_scale
