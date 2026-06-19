"""Text feature extraction and MMD-based two-sample tests for document corpora.

Provides utilities for converting raw document lists to numerical representations
(TF-IDF or NMF topic proportions) and running kernel-based statistical tests on them.

The key design principle is **pooled vocabulary**: when comparing two document groups
X and Y, the vectoriser must be fitted on the *union* ``docs_x + docs_y``.  Fitting
separately on each group produces incompatible column spaces.  :func:`text_mmd_test`
enforces this automatically; :func:`docs_to_tfidf` and :func:`docs_to_topics` expose
it as an explicit contract for users who need the raw embeddings.

Representation choices
----------------------
* **TF-IDF + CosineKernel** (default): sensible for short documents where term
  frequencies carry the signal.  Cosine similarity is invariant to document length.
* **KL-NMF topic proportions + LinearKernel**: appropriate when documents are long
  enough to support latent topic inference.  KL divergence NMF assumes Poisson counts
  (raw ``CountVectorizer`` output), *not* TF-IDF.
* **Custom preprocessor**: pass any ``Callable[[list[str]], np.ndarray]`` to
  :func:`text_mmd_test` to use sentence-transformers, OpenAI embeddings, or any other
  representation that produces an ``(n_docs, d)`` numpy array.

Diagnostic tools
----------------
:func:`coherence_score` and :func:`exclusivity_score` quantify topic quality.
:func:`select_n_components` runs a grid search maximising their product, providing a
principled way to choose NMF topic count before running :func:`text_mmd_test`.

References
----------
Mimno, D., Wallach, H. M., Talley, E., Leenders, M., & McCallum, A. (2011).
    Optimizing semantic coherence in topic models.  *EMNLP*, 262–272.

Blei, D. M., & Lafferty, J. D. (2009).  Topic models.  In A. N. Srivastava &
    M. Sahami (Eds.), *Text Mining: Classification, Clustering, and Applications*.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ._kernel_tests import mmd_test
from ._kernels import CosineKernel, GaussianKernel, Kernel, LinearKernel
from ._results import KernelTestResult

__all__ = [
    "docs_to_tfidf",
    "docs_to_topics",
    "coherence_score",
    "exclusivity_score",
    "select_n_components",
    "text_mmd_test",
]


# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


def _fit_nmf(
    docs: list[str],
    n_components: int,
    *,
    random_state: int | None,
    max_iter: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit CountVectorizer + NMF(KL) on *docs*.

    Returns
    -------
    W : (n_docs, n_components) — document–topic matrix
    H : (n_components, vocab_size) — topic–word matrix
    bin_arr : (n_docs, vocab_size) float — binary document–term matrix
        (needed for NPMI coherence without a second CountVectorizer pass)
    """
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(min_df=1)
    counts = vec.fit_transform(docs)

    nmf = NMF(
        n_components=n_components,
        beta_loss="kullback-leibler",
        solver="mu",
        init="random",
        random_state=random_state,
        max_iter=max_iter,
    )
    W: np.ndarray = nmf.fit_transform(counts)
    H: np.ndarray = nmf.components_
    bin_arr: np.ndarray = (counts > 0).toarray().astype(float)
    return W, H, bin_arr


def _npmi_coherence(bin_arr: np.ndarray, H: np.ndarray, top_n: int) -> float:
    """Compute mean within-corpus NPMI coherence (Mimno et al. 2011).

    Parameters
    ----------
    bin_arr : (n_docs, vocab_size) float
        Binary document–term matrix (1 if term appears, 0 otherwise).
    H : (n_components, vocab_size) float
        NMF topic–word matrix.
    top_n : int
        Number of top words per topic to use for coherence.

    Returns
    -------
    float
        Mean NPMI in [−1, 1].  Higher is better; ~0.1 indicates
        coherent topics for typical corpora.
    """
    n_docs, vocab_size = bin_arr.shape
    n_topics = H.shape[0]
    actual_top = min(top_n, vocab_size)

    # Collect all unique top-word indices to avoid an n×n co-occurrence matrix
    top_word_lists: list[list[int]] = [
        list(np.argsort(H[k])[-actual_top:]) for k in range(n_topics)
    ]
    unique_words: list[int] = sorted({w for lst in top_word_lists for w in lst})

    if not unique_words:
        return 0.0

    idx_map: dict[int, int] = {w: i for i, w in enumerate(unique_words)}

    # Small co-occurrence sub-matrix over the unique top words only
    bin_sub = bin_arr[:, unique_words]  # (n_docs, n_unique)
    cooc = bin_sub.T @ bin_sub  # (n_unique, n_unique)
    doc_freq = np.diag(cooc)  # D(w) for each unique word

    topic_scores: list[float] = []
    for k in range(n_topics):
        top = top_word_lists[k]
        pair_scores: list[float] = []
        for ii in range(len(top)):
            for jj in range(ii):
                wi = idx_map[top[ii]]
                wj = idx_map[top[jj]]
                d_wi = doc_freq[wi]
                d_wj = doc_freq[wj]
                d_wiwj = cooc[wi, wj]
                # +1 smoothing on the pair count; individual word counts use max(·,1)
                log_joint = np.log((d_wiwj + 1.0) / n_docs)
                log_pi = np.log(max(d_wi, 1.0) / n_docs)
                log_pj = np.log(max(d_wj, 1.0) / n_docs)
                denom = -log_joint
                npmi = 1.0 if denom == 0.0 else (log_joint - log_pi - log_pj) / denom
                pair_scores.append(npmi)
        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(topic_scores)) if topic_scores else 0.0


# ------------------------------------------------------------------ #
# Representation helpers
# ------------------------------------------------------------------ #


def docs_to_tfidf(
    docs: list[str],
    *,
    max_features: int | None = None,
    min_df: int = 1,
    stop_words: str | list[str] | None = "english",
    ngram_range: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Convert documents to a TF-IDF matrix.

    Fits and transforms in a single pass on the full corpus supplied.
    For multi-group comparisons, **always** pass the concatenated corpus
    ``docs_x + docs_y`` and then slice the result — fitting each group
    independently produces incompatible column spaces.

    Parameters
    ----------
    docs : list[str]
        Raw document strings.
    max_features : int or None
        If set, keep only the top *max_features* terms by corpus TF.
    min_df : int
        Minimum document frequency for a term to be included.
        Default 1 (all terms).
    stop_words : str, list[str], or None
        Stop-word list passed to :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        ``"english"`` removes common English words (default).
        Pass ``None`` to retain all words, or a custom list for other languages.
    ngram_range : tuple[int, int]
        Lower and upper boundary of the n-gram range.  ``(1, 2)`` adds
        bigrams, which can capture phrase-level signal (e.g. "not significant").

    Returns
    -------
    np.ndarray
        Shape ``(n_docs, vocab_size)``.  Float64, L2-normalised rows,
        sublinear TF scaling.  Ready for :class:`~._kernels.CosineKernel`.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    result: np.ndarray = vec.fit_transform(docs).toarray()
    return result


def docs_to_topics(
    docs: list[str],
    n_components: int,
    *,
    random_state: int | None = None,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert documents to NMF topic-proportion vectors.

    Applies raw term-frequency counting (``CountVectorizer``) followed by
    Non-negative Matrix Factorisation with KL-divergence loss — the correct
    loss for Poisson count data.

    The pooled-vocabulary contract applies here too: pass the concatenated
    corpus for multi-group comparisons and slice the result.

    Parameters
    ----------
    docs : list[str]
        Raw document strings.
    n_components : int
        Number of latent topics.  Use :func:`select_n_components` to choose
        this value from data.
    random_state : int or None
        Seed for reproducible NMF initialisation.
    return_components : bool
        If ``True``, return ``(W, H)`` where H is the ``(n_components,
        vocab_size)`` topic–word matrix.  H is the input to
        :func:`exclusivity_score`.

    Returns
    -------
    W : np.ndarray
        Shape ``(n_docs, n_components)``.  Non-negative topic-proportion
        matrix.  Rows do *not* sum to 1 (NMF does not normalise).
    (W, H) : tuple[np.ndarray, np.ndarray]
        Only when ``return_components=True``.
    """
    W, H, _ = _fit_nmf(docs, n_components, random_state=random_state)
    if return_components:
        return W, H
    return W


# ------------------------------------------------------------------ #
# Diagnostic / model-selection tools
# ------------------------------------------------------------------ #


def coherence_score(
    docs: list[str],
    n_components: int,
    *,
    top_n: int = 10,
    random_state: int | None = None,
) -> float:
    """Compute mean within-corpus NPMI coherence for NMF topics.

    A higher score indicates that the top words of each topic genuinely
    co-occur in the same documents.  Values above approximately 0.1 are
    typical of meaningful topics; negative values indicate poor fit.

    Parameters
    ----------
    docs : list[str]
        Raw document strings used for both fitting NMF and computing
        co-occurrence statistics.
    n_components : int
        Number of NMF topics.
    top_n : int
        Number of top words per topic used in NPMI calculation.
    random_state : int or None
        Seed for NMF initialisation.

    Returns
    -------
    float
        Mean NPMI in [−1, 1].
    """
    _, H, bin_arr = _fit_nmf(docs, n_components, random_state=random_state)
    return _npmi_coherence(bin_arr, H, top_n)


def exclusivity_score(H: np.ndarray, *, top_n: int = 10) -> float:
    """Compute mean topic exclusivity from an NMF topic–word matrix.

    Exclusivity of word *w* to topic *k* is its share of the total NMF
    weight for that word across all topics:
    ``exclusivity(w, k) = H[k, w] / Σ_k H[k, w]``.

    A score close to 1 means each topic's top words appear almost
    exclusively in that topic; a score below ~0.3 suggests heavy overlap
    and may indicate too many topics.

    Parameters
    ----------
    H : np.ndarray
        Shape ``(n_components, vocab_size)``.  Topic–word matrix from
        :func:`docs_to_topics` with ``return_components=True`` or from
        ``NMF.components_``.
    top_n : int
        Number of top words per topic to average over.

    Returns
    -------
    float
        Mean exclusivity in [0, 1].
    """
    H_arr = np.asarray(H, dtype=float)
    n_components, vocab_size = H_arr.shape
    actual_top = min(top_n, vocab_size)

    col_sums = H_arr.sum(axis=0)
    col_sums = np.where(col_sums > 0.0, col_sums, 1.0)  # avoid division by zero
    excl = H_arr / col_sums[np.newaxis, :]  # (n_components, vocab_size)

    topic_scores: list[float] = []
    for k in range(n_components):
        top_idx = np.argsort(H_arr[k])[-actual_top:]
        topic_scores.append(float(np.mean(excl[k, top_idx])))

    return float(np.mean(topic_scores)) if topic_scores else 0.0


def select_n_components(
    docs: list[str],
    n_components_range: Iterable[int],
    *,
    top_n: int = 10,
    random_state: int | None = None,
) -> int:
    """Grid-search for the NMF topic count that maximises coherence × exclusivity.

    Fits NMF once per candidate value in *n_components_range* and selects the
    count that maximises ``coherence_score × exclusivity_score`` — a joint
    criterion that rewards topics that are both semantically tight and mutually
    distinct.

    Intended use: run this once before :func:`text_mmd_test` to select a
    principled ``n_components``, then pass that value to the test.

    Parameters
    ----------
    docs : list[str]
        Raw document strings.
    n_components_range : Iterable[int]
        Candidate topic counts to evaluate (e.g. ``range(2, 11)``).
    top_n : int
        Number of top words per topic passed to the diagnostic functions.
    random_state : int or None
        Seed for NMF initialisation (same seed for all candidates).

    Returns
    -------
    int
        The topic count with the highest joint coherence–exclusivity score.

    Raises
    ------
    ValueError
        If *n_components_range* is empty.
    """
    best_k: int | None = None
    best_score = -np.inf

    for k in n_components_range:
        _, H, bin_arr = _fit_nmf(docs, k, random_state=random_state)
        coh = _npmi_coherence(bin_arr, H, top_n)
        excl = exclusivity_score(H, top_n=top_n)
        score = coh * excl
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise ValueError("n_components_range must be non-empty")
    return int(best_k)


# ------------------------------------------------------------------ #
# Main test wrapper
# ------------------------------------------------------------------ #


def text_mmd_test(
    docs_x: list[str],
    docs_y: list[str],
    *,
    n_components: int | None = None,
    kernel: Kernel | None = None,
    preprocessor: Callable[[list[str]], np.ndarray] | None = None,
    n_permutations: int = 5000,
    max_landmarks: int | None = None,
    random_state: int | None = None,
) -> KernelTestResult:
    """MMD two-sample test on document corpora.

    Converts two document lists to numerical embeddings and runs
    :func:`~._kernel_tests.mmd_test`.  The pooled-vocabulary fit is handled
    internally.

    Three embedding paths are available, selected in priority order:

    1. **Custom preprocessor** (``preprocessor`` is not ``None``): calls
       ``preprocessor(docs_x + docs_y)`` to obtain an ``(n, d)`` embedding
       matrix.  Default kernel: :class:`~._kernels.GaussianKernel` with
       median bandwidth.  Use this path for sentence-transformers, OpenAI
       embeddings, or any other representation.

    2. **NMF topic proportions** (``n_components`` is set): fits
       ``CountVectorizer`` + KL-NMF on the pooled corpus, yielding an
       ``(n, n_components)`` topic-proportion matrix.  Default kernel:
       :class:`~._kernels.LinearKernel` (inner products measure shared topic
       mass).

    3. **TF-IDF** (default): fits ``TfidfVectorizer`` on the pooled corpus,
       yielding an ``(n, vocab_size)`` L2-normalised matrix.  Default kernel:
       :class:`~._kernels.CosineKernel`.

    In all paths, an explicit ``kernel`` argument overrides the default.

    Parameters
    ----------
    docs_x : list[str]
        Documents from group X (distribution P).
    docs_y : list[str]
        Documents from group Y (distribution Q).
    n_components : int or None
        If set, use NMF topic proportions instead of TF-IDF.
        Use :func:`select_n_components` to choose this value.
    kernel : Kernel or None
        Kernel override.  If ``None``, the default is chosen per path above.
    preprocessor : Callable[[list[str]], np.ndarray] or None
        Custom embedding function.  Receives the full pooled document list
        ``docs_x + docs_y`` and must return an ``(n_x + n_y, d)`` array.
    n_permutations : int
        Number of MMD permutations.  Default 5000.
    max_landmarks : int or None
        Nyström approximation landmark count.
    random_state : int or None
        Seed for permutations and Nyström landmark selection.

    Returns
    -------
    KernelTestResult
        ``method="mmd"``.  ``statistic`` is the unbiased MMD² estimate.
    """
    all_docs = list(docs_x) + list(docs_y)
    n_x = len(docs_x)

    if preprocessor is not None:
        M = np.asarray(preprocessor(all_docs), dtype=float)
        if kernel is None:
            kernel = GaussianKernel(sigma="median")
    elif n_components is not None:
        M = np.asarray(
            docs_to_topics(all_docs, n_components, random_state=random_state),
            dtype=float,
        )
        if kernel is None:
            kernel = LinearKernel()
    else:
        M = docs_to_tfidf(all_docs)
        if kernel is None:
            kernel = CosineKernel()

    X_emb = M[:n_x]
    Y_emb = M[n_x:]

    return mmd_test(
        X_emb,
        Y_emb,
        kernel=kernel,
        n_permutations=n_permutations,
        max_landmarks=max_landmarks,
        random_state=random_state,
    )
