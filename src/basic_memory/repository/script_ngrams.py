"""Application-owned lexical analysis for scripts without reliable word boundaries."""

import unicodedata
from dataclasses import dataclass
from typing import Protocol


_SCRIPT_BOUNDARY = "bm_script_boundary"


@dataclass(frozen=True, slots=True)
class ScriptQuery:
    """The word-oriented and script-oriented parts of one user query."""

    word_text: str | None
    gram_phrases: tuple[tuple[str, ...], ...]
    # Fork (ADR 0001): phrases a zero-result relaxed retry may match any one of. Empty
    # keeps every gram phrase required in the retry, as upstream does.
    relaxed_phrases: tuple[tuple[str, ...], ...] = ()


def is_script_search_character(character: str) -> bool:
    codepoint = ord(character)
    return any(
        lower <= codepoint <= upper
        for lower, upper in (
            (0x1100, 0x11FF),  # Hangul Jamo
            (0x0E00, 0x0EFF),  # Thai and Lao
            (0x0F00, 0x0FFF),  # Tibetan
            (0x1000, 0x109F),  # Myanmar
            (0x1780, 0x17FF),  # Khmer
            (0x19E0, 0x19FF),  # Khmer symbols
            (0x3005, 0x3007),  # Ideographic iteration, closing, and zero marks
            (0x3021, 0x3029),  # Hangzhou numerals
            (0x3031, 0x3035),  # Vertical kana repeat marks
            (0x3038, 0x303B),  # Hangzhou tens and vertical iteration mark
            (0x3040, 0x30FF),  # Hiragana and Katakana
            (0x3100, 0x318F),  # Bopomofo and Hangul compatibility Jamo
            (0x31A0, 0x31BF),  # Bopomofo extended
            (0x31F0, 0x31FF),  # Katakana phonetic extensions
            (0x3400, 0x4DBF),  # CJK unified ideographs extension A
            (0x4E00, 0x9FFF),  # CJK unified ideographs
            (0xA960, 0xA97F),  # Hangul Jamo extended A
            (0xA9E0, 0xA9FF),  # Myanmar extended B
            (0xAA60, 0xAA7F),  # Myanmar extended A
            (0xAC00, 0xD7AF),  # Hangul syllables
            (0xD7B0, 0xD7FF),  # Hangul Jamo extended B
            (0xF900, 0xFAFF),  # CJK compatibility ideographs
            (0x1AFF0, 0x1AFFF),  # Katakana extended B
            (0x1B000, 0x1B16F),  # Kana supplements and extensions
            (0x20000, 0x2FFFF),  # Supplementary CJK ideographs
            (0x30000, 0x323AF),  # CJK unified ideographs extensions G-H
        )
    )


def script_runs(text: str) -> tuple[tuple[str, ...], ...]:
    """Return normalized script runs as grapheme-like searchable units."""
    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", text):
        # Join controls shape neighboring script characters without introducing a searchable
        # unit. Keeping the current run open preserves ordered matching across the control.
        if character in {"\u200c", "\u200d"}:
            continue
        if current and unicodedata.category(character) in {"Mn", "Mc", "Me"}:
            current[-1] += character
            continue
        if is_script_search_character(character):
            current.append(character)
            continue
        if current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return tuple(runs)


def script_run_grams(run: tuple[str, ...]) -> tuple[str, ...]:
    """Use bigrams for context and retain a searchable single-character run."""
    if len(run) == 1:
        return run
    return tuple(first + second for first, second in zip(run, run[1:], strict=False))


def build_script_ngrams(*texts: str | None) -> str:
    """Build portable index text without depending on a database tokenizer."""
    if _installed_segmentation is not None:
        return _installed_segmentation.build_index_text(*texts)
    gram_runs: list[str] = []
    for text in texts:
        if not text:
            continue
        for run in script_runs(text):
            # Unigrams make a single-character query searchable inside a longer run. Keeping
            # bigrams together after them preserves ordered phrase matching for longer queries.
            index_terms = run if len(run) == 1 else (*run, *script_run_grams(run))
            gram_runs.append(" ".join(index_terms))
    return f" {_SCRIPT_BOUNDARY} ".join(gram_runs)


def analyze_script_query(text: str) -> ScriptQuery:
    """Split a natural-language query into word text and ordered script grams."""
    normalized = unicodedata.normalize("NFKC", text)
    # Explicit Boolean expressions keep the existing backend parser. Splitting one
    # operand into a second SQL channel would otherwise change OR/NOT semantics.
    # Compatibility forms are natural-language text because neither backend parses
    # their normalized equivalents as operators.
    padded_text = f" {text} "
    if '"' in text or any(f" {operator} " in padded_text for operator in ("AND", "OR", "NOT")):
        return ScriptQuery(word_text=text, gram_phrases=())

    word_tokens: list[str] = []
    for token in text.split():
        normalized_token = unicodedata.normalize("NFKC", token)
        # Keep mixed-script tokens on the established word-search path. The auxiliary channel
        # adds script substring recall without inventing new cross-script token semantics.
        has_word_character = any(
            character.isalnum() and not is_script_search_character(character)
            for character in normalized_token
        )
        if has_word_character:
            word_tokens.append(token.lower() if token in {"AND", "OR", "NOT"} else token)

    gram_phrases = tuple(script_run_grams(run) for run in script_runs(normalized))
    # Preserve punctuation-only input as an explicit backend query. Dropping it would make
    # the repositories confuse user text with the intentional no-predicate wildcard path.
    word_text = " ".join(word_tokens) or (text if not gram_phrases else None)
    query = ScriptQuery(
        word_text=word_text,
        gram_phrases=gram_phrases,
    )
    if _installed_segmentation is not None:
        return _installed_segmentation.refine_query(text, query)
    return query


# --- Fork (ADR 0001): installable script segmentation strategy ---


class ScriptSegmentation(Protocol):
    """A strategy that replaces the n-grams above for script index writes and queries.

    A query matches only the terms its own strategy writes into the ``script_ngrams``
    column, so index writes and queries must use the same strategy.
    """

    def build_index_text(self, *texts: str | None) -> str:
        """Return space-separated index terms for the script runs in ``texts``."""
        ...

    def refine_query(self, text: str, query: ScriptQuery) -> ScriptQuery:
        """Replace the n-gram terms of ``query``, the natural-language analysis of ``text``."""
        ...


_installed_segmentation: ScriptSegmentation | None = None


def install_script_segmentation(segmentation: ScriptSegmentation | None) -> None:
    """Route every later script index write and query of this process through ``segmentation``.

    ``None`` restores the n-grams above. The strategy is process-wide because every search
    backend reaches this module through module-level functions rather than an injected
    dependency; ``script_segmentation.startup`` installs it when the database is opened.
    """
    global _installed_segmentation
    _installed_segmentation = segmentation
