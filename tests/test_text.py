"""Tests for src/randomization_tests/_text.py.

Covers docs_to_tfidf, docs_to_topics, coherence_score, exclusivity_score,
select_n_components, and text_mmd_test.

Test design principles:
- Deterministic: all stochastic calls use random_state=42 for reproducibility.
- Fast: n_permutations=199 throughout; small corpora (≤40 docs).
- Signal tests: clearly distinct lexicons yield p < 0.05.
- Null tests: shuffled-split corpora use p > 0.01 to avoid false failures.
- Diagnostic tests: verify range properties and direction of effect, not
  exact values (which are corpus- and seed-dependent).
"""

from __future__ import annotations

import numpy as np
import pytest

from randomization_tests._results import KernelTestResult
from randomization_tests._text import (
    coherence_score,
    docs_to_tfidf,
    docs_to_topics,
    exclusivity_score,
    select_n_components,
    text_mmd_test,
)

# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #

# Two clearly distinct lexicons — no word overlap between topic A and B.
_TOPIC_A_WORDS = [
    "alpha",
    "atom",
    "molecule",
    "electron",
    "proton",
    "neutron",
    "quantum",
    "energy",
    "wave",
    "particle",
]
_TOPIC_B_WORDS = [
    "market",
    "stock",
    "dividend",
    "portfolio",
    "bond",
    "equity",
    "investor",
    "profit",
    "capital",
    "fund",
]

# Politically distinct corpora for MMD signal tests.
_SPORTS_DOCS = [
    "football game touchdown score winning team",
    "basketball player dunk assist rebounds court",
    "soccer goal penalty kick referee stadium",
    "baseball pitcher batter homerun inning diamond",
    "tennis serve volley racket match championship",
    "swimming pool lap stroke freestyle backstroke",
    "track sprint relay hurdles marathon finish",
    "hockey puck goalie penalty powerplay rink",
    "volleyball spike serve rotation block setter",
    "rugby tackle scrum lineout prop hooker try",
    "cricket wicket bowler batsman over innings",
    "golf birdie eagle par fairway green putt",
    "cycling peloton sprint climb stage breakaway",
    "rowing crew stroke oar coxswain regatta",
    "wrestling pin takedown submission grapple mat",
    "boxing round punch jab uppercut knockout",
    "fencing foil epee sabre parry thrust lunge",
    "archery bow arrow target quiver release draw",
    "rowing crew stroke oar coxswain regatta final",
    "gymnastics vault bar beam floor routine",
]

_POLITICS_DOCS = [
    "election president vote congress senate bill",
    "policy economy tax budget government spending",
    "candidate campaign debate primary endorsement",
    "legislation committee hearing amendment clause",
    "diplomat treaty alliance foreign minister summit",
    "protest rally demonstration petition activist",
    "judiciary court ruling precedent constitution",
    "minister cabinet coalition parliament opposition",
    "sanction embargo tariff trade negotiation deal",
    "referendum ballot initiative proposition recall",
    "lobbyist donor fundraiser super-pac influence",
    "regulation agency enforcement compliance rule",
    "mayor council district ward zoning ordinance",
    "governor legislature veto override session bill",
    "tribunal indictment verdict acquittal sentence",
    "secretary state department bureau directive",
    "polling survey approval rating electorate",
    "filibuster cloture quorum recess adjourn vote",
    "amendment repeal statute regulation executive",
    "delegate convention platform plank resolution",
]


def _make_distinct_corpus(
    n_per_topic: int = 20,
    words_per_doc: int = 6,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Two disjoint-lexicon document groups."""
    rng = np.random.default_rng(seed)
    docs_a = [
        " ".join(rng.choice(_TOPIC_A_WORDS, size=words_per_doc, replace=True))
        for _ in range(n_per_topic)
    ]
    docs_b = [
        " ".join(rng.choice(_TOPIC_B_WORDS, size=words_per_doc, replace=True))
        for _ in range(n_per_topic)
    ]
    return docs_a, docs_b


def _make_random_corpus(
    n_docs: int = 40, words_per_doc: int = 6, seed: int = 42
) -> list[str]:
    """Corpus drawn uniformly from the union of both lexicons (no topic structure)."""
    all_words = _TOPIC_A_WORDS + _TOPIC_B_WORDS
    rng = np.random.default_rng(seed)
    return [
        " ".join(rng.choice(all_words, size=words_per_doc, replace=True))
        for _ in range(n_docs)
    ]


# ------------------------------------------------------------------ #
# TestDocsToTfidf
# ------------------------------------------------------------------ #


class TestDocsToTfidf:
    """Tests for docs_to_tfidf."""

    _DOCS = [
        "the quick brown fox jumps over the lazy dog",
        "a fast red fox leaps across a sleepy hound",
        "the brown dog barks loudly at the red fox",
        "a lazy hound sleeps while foxes play nearby",
    ]

    def test_tfidf_shape(self) -> None:
        M = docs_to_tfidf(self._DOCS)
        assert M.ndim == 2
        assert M.shape[0] == len(self._DOCS)
        assert M.shape[1] > 0

    def test_tfidf_max_features(self) -> None:
        M = docs_to_tfidf(self._DOCS, max_features=5)
        assert M.shape[1] == 5

    def test_tfidf_ngram_range(self) -> None:
        M_uni = docs_to_tfidf(self._DOCS, stop_words=None, ngram_range=(1, 1))
        M_bi = docs_to_tfidf(self._DOCS, stop_words=None, ngram_range=(1, 2))
        # Bigrams add extra columns
        assert M_bi.shape[1] > M_uni.shape[1]

    def test_tfidf_stop_words_none(self) -> None:
        # stop_words=None retains common words — vocabulary is at least as large
        M_with = docs_to_tfidf(self._DOCS, stop_words="english")
        M_without = docs_to_tfidf(self._DOCS, stop_words=None)
        assert M_without.shape[1] >= M_with.shape[1]

    def test_tfidf_pooled_vocabulary(self) -> None:
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=5)
        combined = docs_a + docs_b
        M_pool = docs_to_tfidf(combined)
        M_a_from_pool = M_pool[: len(docs_a)]
        M_a_solo = docs_to_tfidf(docs_a)
        # Pooled fit sees both lexicons → more columns
        assert M_a_from_pool.shape[1] >= M_a_solo.shape[1]
        # And the shapes are different (different column space)
        assert M_a_from_pool.shape[1] != M_a_solo.shape[1]

    def test_tfidf_output_is_float(self) -> None:
        M = docs_to_tfidf(self._DOCS)
        assert M.dtype == np.float64


# ------------------------------------------------------------------ #
# TestDocsToTopics
# ------------------------------------------------------------------ #


class TestDocsToTopics:
    """Tests for docs_to_topics."""

    def test_topics_shape(self) -> None:
        docs_a, docs_b = _make_distinct_corpus()
        docs = docs_a + docs_b
        W = docs_to_topics(docs, n_components=3, random_state=42)
        assert isinstance(W, np.ndarray)
        assert W.shape == (len(docs), 3)

    def test_topics_nonnegative(self) -> None:
        docs_a, docs_b = _make_distinct_corpus()
        docs = docs_a + docs_b
        W = docs_to_topics(docs, n_components=2, random_state=42)
        assert isinstance(W, np.ndarray)
        assert float(np.min(W)) >= 0.0

    def test_topics_reproducible(self) -> None:
        docs_a, docs_b = _make_distinct_corpus()
        docs = docs_a + docs_b
        W1 = docs_to_topics(docs, n_components=2, random_state=42)
        W2 = docs_to_topics(docs, n_components=2, random_state=42)
        assert isinstance(W1, np.ndarray)
        assert isinstance(W2, np.ndarray)
        np.testing.assert_array_equal(W1, W2)

    def test_topics_return_components(self) -> None:
        docs_a, docs_b = _make_distinct_corpus()
        docs = docs_a + docs_b
        result = docs_to_topics(
            docs, n_components=2, random_state=42, return_components=True
        )
        assert isinstance(result, tuple)
        W, H = result
        assert W.shape == (len(docs), 2)
        # H: (n_components, vocab_size)
        assert H.shape[0] == 2
        assert H.shape[1] > 0

    def test_topics_pooled_vocabulary(self) -> None:
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=10)
        # Fitting on the union gives more vocabulary columns in H
        W_pool, H_pool = docs_to_topics(  # type: ignore[misc]
            docs_a + docs_b, 2, random_state=42, return_components=True
        )
        W_solo, H_solo = docs_to_topics(  # type: ignore[misc]
            docs_a, 2, random_state=42, return_components=True
        )
        # Pooled H sees both lexicons → more vocabulary
        assert H_pool.shape[1] > H_solo.shape[1]


# ------------------------------------------------------------------ #
# TestDiagnosticFunctions
# ------------------------------------------------------------------ #


class TestDiagnosticFunctions:
    """Tests for coherence_score, exclusivity_score, select_n_components."""

    def test_coherence_score_range(self) -> None:
        docs_a, docs_b = _make_distinct_corpus()
        docs = docs_a + docs_b
        score = coherence_score(docs, n_components=2, random_state=42)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_coherence_score_distinct_topics(self) -> None:
        # Corpus with two disjoint lexicons should score higher coherence
        # than a corpus drawn uniformly from the union of those lexicons.
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=20)
        distinct_docs = docs_a + docs_b
        random_docs = _make_random_corpus(n_docs=40)
        score_distinct = coherence_score(distinct_docs, n_components=2, random_state=42)
        score_random = coherence_score(random_docs, n_components=2, random_state=42)
        assert score_distinct > score_random

    def test_exclusivity_score_range(self) -> None:
        docs_a, docs_b = _make_distinct_corpus()
        W, H = docs_to_topics(  # type: ignore[misc]
            docs_a + docs_b, 2, random_state=42, return_components=True
        )
        score = exclusivity_score(H)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_exclusivity_distinct_topics(self) -> None:
        # Well-separated topics should have higher exclusivity than random topics.
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=20)
        distinct_docs = docs_a + docs_b
        random_docs = _make_random_corpus(n_docs=40)

        _, H_distinct = docs_to_topics(  # type: ignore[misc]
            distinct_docs, 2, random_state=42, return_components=True
        )
        _, H_random = docs_to_topics(  # type: ignore[misc]
            random_docs, 2, random_state=42, return_components=True
        )
        assert exclusivity_score(H_distinct) > exclusivity_score(H_random)

    def test_select_n_components_returns_int(self) -> None:
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=15)
        k = select_n_components(docs_a + docs_b, [2, 3, 4], random_state=42)
        assert isinstance(k, int)
        assert k in {2, 3, 4}

    def test_select_n_components_distinct(self) -> None:
        # A 2-topic corpus should not over-split to the maximum candidate.
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=20)
        docs = docs_a + docs_b
        k = select_n_components(docs, range(2, 6), random_state=42)
        # Should prefer fewer topics; assert it doesn't always pick the max
        assert k < 5

    def test_select_n_components_empty_raises(self) -> None:
        docs_a, _ = _make_distinct_corpus(n_per_topic=5)
        with pytest.raises(ValueError, match="non-empty"):
            select_n_components(docs_a, [], random_state=42)


# ------------------------------------------------------------------ #
# TestTextMMDTest
# ------------------------------------------------------------------ #


class TestTextMMDTest:
    """Tests for text_mmd_test."""

    def test_text_mmd_returns_result(self) -> None:
        result = text_mmd_test(
            _SPORTS_DOCS[:5],
            _POLITICS_DOCS[:5],
            n_permutations=99,
            random_state=42,
        )
        assert isinstance(result, KernelTestResult)
        assert result.method == "mmd"

    def test_text_mmd_different_corpora_signal(self) -> None:
        # Sports vs politics — distinct lexicons should yield significant result.
        result = text_mmd_test(
            _SPORTS_DOCS,
            _POLITICS_DOCS,
            n_permutations=199,
            random_state=42,
        )
        assert result.p_value < 0.05

    def test_text_mmd_same_corpus_null(self) -> None:
        # Random halves of a shuffled corpus — should not reject under H₀.
        rng = np.random.default_rng(42)
        all_docs = _SPORTS_DOCS + _POLITICS_DOCS
        idx = rng.permutation(len(all_docs))
        half = len(all_docs) // 2
        docs_a = [all_docs[i] for i in idx[:half]]
        docs_b = [all_docs[i] for i in idx[half:]]
        result = text_mmd_test(docs_a, docs_b, n_permutations=199, random_state=42)
        assert result.p_value > 0.01

    def test_text_mmd_with_topics_signal(self) -> None:
        docs_a, docs_b = _make_distinct_corpus(n_per_topic=20)
        result = text_mmd_test(
            docs_a,
            docs_b,
            n_components=2,
            n_permutations=199,
            random_state=42,
        )
        assert result.p_value < 0.05

    def test_text_mmd_preprocessor(self) -> None:
        # A preprocessor that returns a fixed random matrix should be accepted.
        rng = np.random.default_rng(0)
        n_total = len(_SPORTS_DOCS[:5]) + len(_POLITICS_DOCS[:5])
        fixed_M = rng.standard_normal((n_total, 8))

        def dummy_preprocessor(docs: list[str]) -> np.ndarray:
            return fixed_M

        result = text_mmd_test(
            _SPORTS_DOCS[:5],
            _POLITICS_DOCS[:5],
            preprocessor=dummy_preprocessor,
            n_permutations=99,
            random_state=42,
        )
        assert isinstance(result, KernelTestResult)

    def test_text_mmd_preprocessor_overrides_tfidf(self) -> None:
        # Preprocessor path should produce a different statistic than the TF-IDF path.
        docs_x = _SPORTS_DOCS[:10]
        docs_y = _POLITICS_DOCS[:10]
        n_total = len(docs_x) + len(docs_y)

        rng = np.random.default_rng(7)
        random_emb = rng.standard_normal((n_total, 4))

        result_tfidf = text_mmd_test(docs_x, docs_y, n_permutations=99, random_state=42)
        result_custom = text_mmd_test(
            docs_x,
            docs_y,
            preprocessor=lambda _docs: random_emb,
            n_permutations=99,
            random_state=42,
        )
        # Statistics come from different representations — they must differ
        assert result_tfidf.statistic != result_custom.statistic

    def test_text_mmd_reproducible(self) -> None:
        r1 = text_mmd_test(
            _SPORTS_DOCS, _POLITICS_DOCS, n_permutations=99, random_state=42
        )
        r2 = text_mmd_test(
            _SPORTS_DOCS, _POLITICS_DOCS, n_permutations=99, random_state=42
        )
        assert r1.p_value == r2.p_value
        assert r1.statistic == r2.statistic
