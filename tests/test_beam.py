"""Beam search, checked against brute force on a model small enough to enumerate.

Notebooks are self-sufficient by convention, so the algorithm lives in the
notebooks rather than in `src/research/`, and more than one notebook may carry
its own copy. These tests execute each notebook's own tagged cells instead of a
copy of them - a copy would drift, and the copy is not what anyone runs. Every
copy found is put through the whole suite, so duplication is allowed but a
duplicate that stops being correct is not.

The toy model's next-token distribution depends only on the last token, so every
continuation of a prompt can be scored by hand. That makes the interesting claims
testable exactly rather than approximately: that candidates are compared globally
instead of per beam, that the cumulative log probabilities add up, and that the
returned sequences really are the highest-probability ones.
"""

import itertools
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

IMPLEMENTATION_TAG = "beam-search-implementation"


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

    Notebooks are self-sufficient by convention, so more than one may hold its
    own copy. Each copy is tested in full: duplication is allowed, drifting into
    a *broken* copy is not.

    Recursive, so copies in subfolders are covered too - ``notebooks/cloud/``
    holds runtime-specific variants that carry the same tagged implementation,
    and an untested copy there would drift silently.

    Hidden directories are skipped: ``.ipynb_checkpoints/`` holds Jupyter's
    autosave copies, which are stale by nature. Testing them adds no coverage
    and fails the moment a checkpoint lags the notebook it shadows.
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
def beam(request):
    """Execute one notebook's tagged cells and expose the names they define."""
    namespace = {"__name__": "beam_search_notebook"}
    for index, source in enumerate(tagged_cells(request.param)):
        exec(compile(source, f"{request.param.name}:implementation{index}", "exec"), namespace)
    return SimpleNamespace(**namespace)

VOCAB = "abcd"
V = len(VOCAB)

# Row i is P(next token | last token was i). Deliberately skewed: from "a" the
# token "b" is worth 0.80, which is what makes the global-vs-per-beam test bite.
# The values are also chosen so that no two short continuations score exactly
# alike - with a tie, the ranking of equal-probability sequences is arbitrary and
# the brute-force comparison below would be testing tie-breaking, not search.
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
    """Returns log P(next | last token) as logits, and records how it was called.

    Feeding log probabilities in as logits is exact, not a fudge: ``log_softmax``
    of a normalised log-probability vector is that vector again, so the search
    sees precisely the numbers in ``PROBS``.
    """

    def __init__(self, probs=PROBS, eos_token_id=None):
        super().__init__()
        self.device_marker = nn.Parameter(torch.zeros(1))
        self.log_probs = probs.log()
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kwargs):
        self.calls.append(
            {
                "batch": input_ids.shape[0],
                "length": input_ids.shape[1],
                "logits_to_keep": logits_to_keep,
                "kwargs": dict(kwargs),
            }
        )
        wanted = input_ids[:, -logits_to_keep:] if logits_to_keep else input_ids
        return SimpleNamespace(logits=self.log_probs[wanted])


class ToyModelWithoutLogitsToKeep(ToyModel):
    """An architecture whose forward has no logits-trimming argument."""

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return super().forward(input_ids, attention_mask, logits_to_keep=0, **kwargs)


def brute_force(prompt, num_tokens, probs=PROBS):
    """Every continuation of ``prompt``, scored exactly. The reference to match."""
    scored = []
    for candidate in itertools.product(range(V), repeat=num_tokens):
        last = VOCAB.index(prompt[-1])
        total = 0.0
        for token in candidate:
            total += math.log(float(probs[last, token]))
            last = token
        scored.append((total, "".join(VOCAB[t] for t in candidate)))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored


@pytest.fixture
def toy():
    return ToyModel(), ToyTokenizer()


# --- the notebook is the implementation -----------------------------------


def test_the_notebook_exposes_the_expected_entry_points(beam):
    """If a rename breaks this, every other test below is testing nothing."""
    for name in ("beam_search", "token_level_rows", "eos_token_ids", "ROOT_BEAM_ID"):
        assert hasattr(beam, name), f"{name} is missing from the notebook implementation"


# --- the search finds the highest-probability sequences -------------------


def test_exhaustive_search_matches_brute_force(beam, toy):
    """With a beam wide enough to hold every hypothesis, the answer must be exact."""
    model, tokenizer = toy
    result = beam.beam_search(
        model,
        tokenizer,
        "a",
        num_tokens_to_generate=3,
        beam_width=V**3,  # wide enough that nothing is ever pruned
        top_k_per_beam=None,
        num_return_sequences=8,
    )

    expected = brute_force("a", 3)[:8]
    assert [row["generated_text"] for row in result.sequences] == [text for _, text in expected]
    assert [row["total_log_probability"] for row in result.sequences] == pytest.approx(
        [score for score, _ in expected]
    )


def test_narrow_beam_scores_are_still_exactly_right(beam, toy):
    """A narrow beam may miss a sequence, but must never misprice the ones it keeps."""
    model, tokenizer = toy
    result = beam.beam_search(
        model, tokenizer, "a", num_tokens_to_generate=4, beam_width=3, top_k_per_beam=3,
        num_return_sequences=3,
    )
    reference = {text: score for score, text in brute_force("a", 4)}

    for row in result.sequences:
        assert row["total_log_probability"] == pytest.approx(reference[row["generated_text"]])


def test_sequences_are_ranked_by_cumulative_log_probability(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(
        model, tokenizer, "a", num_tokens_to_generate=4, beam_width=6, top_k_per_beam=6,
        num_return_sequences=6,
    )
    scores = [row["total_log_probability"] for row in result.sequences]
    assert scores == sorted(scores, reverse=True)
    assert [row["rank"] for row in result.sequences] == [1, 2, 3, 4, 5, 6]


def test_greedy_is_the_degenerate_case(beam, toy):
    """beam_width=1, top_k=1 is argmax at every step - a useful sanity anchor."""
    model, tokenizer = toy
    result = beam.beam_search(
        model, tokenizer, "a", num_tokens_to_generate=4, beam_width=1, top_k_per_beam=1,
        num_return_sequences=1,
    )

    expected, last = [], VOCAB.index("a")
    for _ in range(4):
        last = int(PROBS[last].argmax())
        expected.append(VOCAB[last])
    assert result.sequences[0]["generated_text"] == "".join(expected)


# --- candidates are compared globally, not one winner per beam ------------


def test_candidates_are_compared_globally_not_per_beam(beam, toy):
    """Both survivors come from one parent; the other beam's own best is dropped.

    From "a" the search keeps "b" (0.80) and "c" (0.10). At step 2 the two best
    candidates overall are b->d (0.48) and b->c (0.16), both children of "b" -
    while "c"'s best child, c->b, scores only 0.075. Taking the best token from
    each beam separately would keep that instead, which is the mistake this
    guards against.
    """
    model, tokenizer = toy
    result = beam.beam_search(
        model, tokenizer, "a", num_tokens_to_generate=2, beam_width=2, top_k_per_beam=4,
        num_return_sequences=2,
    )

    assert [row["generated_text"] for row in result.sequences] == ["bd", "bc"]
    step2 = [row for row in result.trace if row["step"] == 2]
    assert len({row["parent_beam_id"] for row in step2}) == 1  # one parent, two children


def test_first_step_expands_the_prompt_once(beam, toy):
    """All beams are identical before step 1; expanding each would clone one token."""
    model, tokenizer = toy
    result = beam.beam_search(
        model, tokenizer, "a", num_tokens_to_generate=2, beam_width=4, top_k_per_beam=4,
        num_return_sequences=4,
    )

    assert model.calls[0]["batch"] == 1  # the prompt alone, not four copies of it
    step1 = [row for row in result.trace if row["step"] == 1]
    assert len({row["token_id"] for row in step1}) == 4  # four distinct continuations


def test_only_beam_width_hypotheses_stay_alive(beam, toy):
    model, tokenizer = toy
    width = 3
    result = beam.beam_search(
        model, tokenizer, "a", num_tokens_to_generate=4, beam_width=width, top_k_per_beam=4,
        num_return_sequences=width,
    )

    per_step = {}
    for row in result.trace:
        per_step[row["step"]] = per_step.get(row["step"], 0) + 1
    assert set(per_step.values()) <= {width}
    assert all(call["batch"] <= width for call in model.calls)


# --- no KV cache: the model recomputes from the whole sequence -------------


def test_no_kv_cache_is_used(beam, toy):
    """The stated constraint: use_cache off, no past_key_values, ever."""
    model, tokenizer = toy
    beam.beam_search(model, tokenizer, "abc", num_tokens_to_generate=4, beam_width=2,
                     top_k_per_beam=2, num_return_sequences=2)

    assert model.calls, "the model was never called"
    for call in model.calls:
        assert call["kwargs"]["use_cache"] is False
        assert "past_key_values" not in call["kwargs"]


def test_every_step_recomputes_the_full_sequence(beam, toy):
    """Without a cache, each step must see prompt + everything generated so far."""
    model, tokenizer = toy
    beam.beam_search(model, tokenizer, "abc", num_tokens_to_generate=4, beam_width=2,
                     top_k_per_beam=2, num_return_sequences=2)

    assert [call["length"] for call in model.calls] == [3, 4, 5, 6]


def test_logits_are_trimmed_to_the_last_position_when_supported(beam, toy):
    model, tokenizer = toy
    beam.beam_search(model, tokenizer, "abc", num_tokens_to_generate=2, beam_width=2,
                     top_k_per_beam=2, num_return_sequences=2)
    assert all(call["logits_to_keep"] == 1 for call in model.calls)


def test_works_on_a_model_without_a_logits_trimming_argument(beam):
    """Not every architecture takes logits_to_keep; the search must not require it."""
    model, tokenizer = ToyModelWithoutLogitsToKeep(), ToyTokenizer()
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=3, beam_width=4,
                              top_k_per_beam=4)

    assert result.sequences[0]["generated_text"] == brute_force("a", 3)[0][1]
    assert all(call["logits_to_keep"] == 0 for call in model.calls)


# --- the trace explains the search ----------------------------------------


def test_trace_arithmetic_is_consistent(beam, toy):
    """previous + this token = cumulative, on every row. If this drifts, ranking is wrong."""
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=4,
                              top_k_per_beam=4)

    for row in result.trace:
        assert row["previous_log_probability"] + row["token_log_probability"] == pytest.approx(
            row["cumulative_log_probability"]
        )
        assert row["token_probability"] == pytest.approx(math.exp(row["token_log_probability"]))


def test_trace_records_a_full_parent_chain(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=3,
                              top_k_per_beam=4, num_return_sequences=3)

    beam_ids = [row["beam_id"] for row in result.trace]
    assert len(beam_ids) == len(set(beam_ids)), "beam ids must be unique across the search"

    known = {beam.ROOT_BEAM_ID}
    for step in range(1, 5):
        rows = [row for row in result.trace if row["step"] == step]
        assert all(row["parent_beam_id"] in known for row in rows)
        known = {row["beam_id"] for row in rows}


def test_trace_marks_which_beams_were_discarded(beam, toy):
    """A beam that no later row claims as a parent, and is not alive at the end, died."""
    model, tokenizer = toy
    # Returning every surviving beam makes "alive at the end" observable from the
    # result, so `survived` can be checked exactly rather than one-sidedly.
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=3,
                              top_k_per_beam=4, num_return_sequences=3)

    parents = {row["parent_beam_id"] for row in result.trace}
    alive_at_the_end = {row["beam_id"] for row in result.sequences}
    for row in result.trace:
        assert row["survived"] == (row["beam_id"] in parents or row["beam_id"] in alive_at_the_end)
    assert any(not row["survived"] for row in result.trace), "nothing was ever pruned"


def test_final_path_rows_reconstruct_the_winning_sequence(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=4,
                              top_k_per_beam=4, num_return_sequences=1)

    path = [row for row in sorted(result.trace, key=lambda r: r["step"]) if row["on_final_path"]]
    assert "".join(row["token"] for row in path) == result.sequences[0]["generated_text"]
    assert path[-1]["cumulative_log_probability"] == pytest.approx(
        result.sequences[0]["total_log_probability"]
    )


def test_trace_can_be_switched_off_without_changing_the_result(beam, toy):
    """Analysis mode and memory mode must search identically."""
    model, tokenizer = toy
    kwargs = dict(num_tokens_to_generate=4, beam_width=4, top_k_per_beam=4,
                  num_return_sequences=3)

    with_trace = beam.beam_search(model, tokenizer, "a", record_trace=True, **kwargs)
    without = beam.beam_search(ToyModel(), tokenizer, "a", record_trace=False, **kwargs)

    assert without.trace == []
    assert [row["generated_text"] for row in without.sequences] == [
        row["generated_text"] for row in with_trace.sequences
    ]
    assert [row["total_log_probability"] for row in without.sequences] == pytest.approx(
        [row["total_log_probability"] for row in with_trace.sequences]
    )


def test_the_trace_holds_no_tensors(beam, toy):
    """Storing a tensor per row is the memory mistake the trace exists to avoid."""
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=3, beam_width=3,
                              top_k_per_beam=3, num_return_sequences=3)

    for row in result.trace:
        for key, value in row.items():
            assert not isinstance(value, torch.Tensor), f"{key} is a tensor"
            assert isinstance(value, int | float | str | bool), f"{key} is a {type(value)}"


# --- token-level detail ----------------------------------------------------


def test_token_log_probabilities_sum_to_the_sequence_score(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=5, beam_width=4,
                              top_k_per_beam=4)

    for row in result.sequences:
        assert sum(row["token_log_probabilities"]) == pytest.approx(row["total_log_probability"])
        assert len(row["token_log_probabilities"]) == row["num_generated_tokens"] == 5


def test_average_token_probability_is_the_geometric_mean(beam, toy):
    """exp(mean log p): the per-token probability that multiplies back to the total."""
    model, tokenizer = toy
    row = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=2,
                           top_k_per_beam=2, num_return_sequences=1).sequences[0]

    assert row["average_token_probability"] ** row["num_generated_tokens"] == pytest.approx(
        row["sequence_probability"]
    )


def test_token_level_rows_track_the_running_score(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=3, beam_width=4,
                              top_k_per_beam=4, num_return_sequences=2)
    rows = beam.token_level_rows(result)

    assert len(rows) == 6
    assert [row["position"] for row in rows] == [1, 2, 3, 1, 2, 3]
    assert rows[2]["cumulative_log_probability"] == pytest.approx(
        result.sequences[0]["total_log_probability"]
    )


def test_cumulative_probability_is_the_running_product(beam, toy):
    """The greedy baseline reports it directly, so it has to be the actual product."""
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=2,
                              top_k_per_beam=2, num_return_sequences=1)

    running = 1.0
    for row in beam.token_level_rows(result):
        running *= row["token_probability"]
        assert row["cumulative_probability"] == pytest.approx(running)
        assert row["cumulative_probability"] == pytest.approx(
            math.exp(row["cumulative_log_probability"])
        )
    assert running == pytest.approx(result.sequences[0]["sequence_probability"])


def test_greedy_baseline_takes_the_argmax_at_every_step(beam, toy):
    """The baseline the notebook reports is this search at width 1, not a second code path."""
    model, tokenizer = toy
    greedy = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=1,
                              top_k_per_beam=1, num_return_sequences=1)
    rows = beam.token_level_rows(greedy)

    last = VOCAB.index("a")
    for row in rows:
        assert row["token_id"] == int(PROBS[last].argmax())
        assert row["token_probability"] == pytest.approx(float(PROBS[last].max()))
        last = row["token_id"]
    assert len(rows) == 4


# --- EOS and early stopping ------------------------------------------------


def test_eos_finishes_a_beam_and_stops_the_search(beam):
    """Every path leads straight to EOS, so the search should end as soon as all beams have."""
    eos = VOCAB.index("d")
    probs = torch.tensor([[0.02, 0.03, 0.05, 0.90]] * V)
    model, tokenizer = ToyModel(probs, eos_token_id=eos), ToyTokenizer()

    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=6, beam_width=2,
                              top_k_per_beam=2, num_return_sequences=2)

    assert result.stopped_early is True
    assert result.steps_run == 2  # the second beam ("c") only reached EOS one step later
    assert all(row["finished"] for row in result.sequences)
    assert result.sequences[0]["generated_text"] == "d"


def test_a_finished_beam_carries_forward_without_changing_its_score(beam):
    """Beam 1 hits EOS immediately; beam 2 keeps going. The finished score must freeze."""
    eos = VOCAB.index("d")
    probs = torch.tensor(
        [
            [0.05, 0.30, 0.05, 0.60],  # from a: EOS is the single best continuation
            [0.30, 0.30, 0.35, 0.05],  # from b: EOS is unlikely, so this beam runs on
            [0.05, 0.75, 0.15, 0.05],
            [0.42, 0.30, 0.20, 0.08],
        ]
    )
    model, tokenizer = ToyModel(probs, eos_token_id=eos), ToyTokenizer()

    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=3, beam_width=2,
                              top_k_per_beam=2, num_return_sequences=2)

    assert result.stopped_early is False  # one beam was still generating
    finished = [row for row in result.sequences if row["finished"]]
    assert len(finished) == 1
    assert finished[0]["generated_text"] == "d"  # no filler tokens leaked into the text
    assert finished[0]["num_generated_tokens"] == 1
    assert finished[0]["total_log_probability"] == pytest.approx(math.log(0.60))

    carried = [row for row in result.trace if row["carried_forward"]]
    assert carried, "the finished beam should have been carried forward"
    assert all(row["token_log_probability"] == 0.0 for row in carried)


def test_early_stopping_off_keeps_generating_the_full_length(beam):
    eos = VOCAB.index("d")
    probs = torch.tensor([[0.02, 0.03, 0.05, 0.90]] * V)
    model, tokenizer = ToyModel(probs, eos_token_id=eos), ToyTokenizer()

    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=4, beam_width=2,
                              top_k_per_beam=2, early_stopping=False, num_return_sequences=2)

    assert result.stopped_early is False
    assert result.steps_run == 4
    assert result.sequences[0]["num_generated_tokens"] == 1  # still only the EOS token


def test_eos_ids_come_from_the_generation_config_first(beam):
    model = ToyModel(eos_token_id=[1, 2])
    assert beam.eos_token_ids(model, ToyTokenizer(eos_token_id=3)) == {1, 2}


def test_eos_ids_fall_back_to_the_tokenizer(beam):
    assert beam.eos_token_ids(ToyModel(), ToyTokenizer(eos_token_id=3)) == {3}


def test_a_tokenizer_without_an_eos_token_is_not_assumed(beam, toy):
    """Never assume the tokenizer has one - some do not, and nothing should break."""
    model, tokenizer = toy
    assert beam.eos_token_ids(model, tokenizer) == set()
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=3, beam_width=2,
                              top_k_per_beam=2, num_return_sequences=2)
    assert all(not row["finished"] for row in result.sequences)


# --- configuration ---------------------------------------------------------


def test_temperature_flattens_the_distribution(beam, toy):
    """A high temperature must move the recorded probabilities towards uniform."""
    model, tokenizer = toy
    kwargs = dict(num_tokens_to_generate=1, beam_width=4, top_k_per_beam=4,
                  num_return_sequences=1)

    sharp = beam.beam_search(model, tokenizer, "a", temperature=1.0, **kwargs)
    flat = beam.beam_search(model, tokenizer, "a", temperature=5.0, **kwargs)

    assert sharp.sequences[0]["sequence_probability"] == pytest.approx(0.80)
    assert flat.sequences[0]["sequence_probability"] < sharp.sequences[0]["sequence_probability"]
    assert flat.sequences[0]["sequence_probability"] > 1 / V


def test_top_k_per_beam_bounds_the_candidate_pool(beam, toy):
    """top_k=1: only one child per beam exists, so only one candidate is ever scored."""
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=3, beam_width=1,
                              top_k_per_beam=1, num_return_sequences=1)
    assert len([row for row in result.trace if row["step"] == 2]) == 1


def test_full_vocabulary_is_allowed(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=2, beam_width=4,
                              top_k_per_beam=None, num_return_sequences=4)
    assert len(result.sequences) == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_tokens_to_generate": 0},
        {"beam_width": 0},
        {"num_return_sequences": 0},
        {"num_return_sequences": 11},  # more than beam_width=10 survive
        {"top_k_per_beam": 2},  # below beam_width: step 1 could not fill the beam
        {"temperature": 0.0},
    ],
)
def test_impossible_configurations_are_rejected(beam, toy, overrides):
    model, tokenizer = toy
    with pytest.raises(ValueError):
        beam.beam_search(model, tokenizer, "a", **overrides)
    assert model.calls == [], "the configuration must be rejected before any compute"


def test_the_resolved_configuration_is_recorded(beam, toy):
    """The result has to carry what produced it, or it is not reproducible."""
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "a", num_tokens_to_generate=2, beam_width=2,
                              top_k_per_beam=2, num_return_sequences=2)

    assert result.config["beam_width"] == 2
    assert result.config["temperature"] == 1.0  # defaults are recorded too, not just overrides


def test_a_device_the_model_is_not_on_is_refused(beam, toy):
    model, tokenizer = toy
    with pytest.raises(ValueError, match="the model is on"):
        beam.beam_search(model, tokenizer, "a", device="meta")


def test_an_empty_prompt_is_refused(beam, toy):
    model, tokenizer = toy
    with pytest.raises(ValueError, match="tokenised to nothing"):
        beam.beam_search(model, tokenizer, "", num_tokens_to_generate=2, beam_width=2,
                         top_k_per_beam=2, num_return_sequences=2)


def test_the_prompt_is_reported_back(beam, toy):
    model, tokenizer = toy
    result = beam.beam_search(model, tokenizer, "abc", num_tokens_to_generate=2, beam_width=2,
                              top_k_per_beam=2, num_return_sequences=2)
    assert result.prompt == "abc"
    assert result.prompt_token_ids == [0, 1, 2]
    assert result.sequences[0]["full_text"].startswith("abc")
