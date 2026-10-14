"""Dictionary-word strategy backed by jieba."""

import hashlib
import io
import logging
from pathlib import Path

import jieba

from basic_memory.repository.script_ngrams import (
    ScriptQuery,
    is_script_search_character,
    script_runs,
)

# Part of the fingerprint: bump it when the index text layout changes.
_INDEX_LAYOUT_VERSION = 1
# With a single distinctive word, an OR retry would only broaden the strict query.
_MIN_RELAXED_PHRASES = 2


class JiebaSegmenter:
    """Index jieba search-mode words plus single characters; query with jieba words.

    The strict query requires every query word as its own phrase: a multi-character
    word hits only notes where jieba produced that word, while a single character still
    matches anywhere it occurs. A natural-language sentence rarely has all of its words
    in one note, so the zero-result relaxed retry may match any one of its distinctive
    words instead: the multi-character ones, since single characters are mostly
    particles and pronouns. Han text is segmented with jieba's dictionary (plus the
    optional user dictionary); jieba leaves kana, Hangul and other scripts as single
    characters. HMM new-word discovery stays off on both sides, so every query word is a
    dictionary word that indexing can produce too.
    """

    def __init__(self, *, cache_dir: Path, user_dict: Path | None = None) -> None:
        """Load the dictionaries now so configuration errors surface at startup.

        Args:
            cache_dir: Directory for jieba's prefix-dictionary cache.
            user_dict: Optional UTF-8 user dictionary, one ``word [freq] [tag]`` per line.
        """
        jieba.setLogLevel(logging.WARNING)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = jieba.Tokenizer()
        # jieba is untyped and initializes tmp_dir to None.
        tokenizer.tmp_dir = str(cache_dir)  # pyright: ignore[reportAttributeAccessIssue]
        tokenizer.initialize()
        fingerprint = f"jieba-{jieba.__version__}-layout{_INDEX_LAYOUT_VERSION}"
        if user_dict is not None:
            content = user_dict.read_bytes()
            # A file object instead of the path: jieba never closes a file it opened itself.
            tokenizer.load_userdict(io.BytesIO(content))
            fingerprint += f"-userdict-{hashlib.sha256(content).hexdigest()}"
        self._tokenizer = tokenizer
        self.fingerprint = fingerprint

    def build_index_text(self, *texts: str | None) -> str:
        terms: list[str] = []
        for text in texts:
            if not text:
                continue
            for run in script_runs(text):
                terms.extend(run)
                words = self._tokenizer.cut_for_search("".join(run), HMM=False)
                terms.extend(word for word in words if len(word) > 1)
        return " ".join(terms)

    def refine_query(self, text: str, query: ScriptQuery) -> ScriptQuery:
        if not query.gram_phrases:
            return query
        words = tuple(
            dict.fromkeys(
                word
                for run in script_runs(text)
                for word in self._tokenizer.cut("".join(run), HMM=False)
            )
        )
        distinctive = tuple((word,) for word in words if len(word) > 1)
        return ScriptQuery(
            word_text=_without_script_text(query.word_text),
            gram_phrases=tuple((word,) for word in words),
            relaxed_phrases=distinctive if len(distinctive) >= _MIN_RELAXED_PHRASES else (),
        )


def _without_script_text(word_text: str | None) -> str | None:
    """Split the word channel's mixed tokens at their script characters.

    Upstream keeps a token such as ``ReadTimeoutError，连不上`` whole on the word channel,
    where it matches only that exact adjacency; its script part is already a jieba
    phrase, so only the Latin part stays a word. A part spelled like an FTS5 operator is
    lowercased, as upstream does for whole tokens.
    """
    if word_text is None:
        return None
    parts = [
        part
        for token in word_text.split()
        for part in "".join(
            " " if is_script_search_character(character) else character for character in token
        ).split()
        if any(character.isalnum() for character in part)
    ]
    return (
        " ".join(part.lower() if part in {"AND", "OR", "NOT"} else part for part in parts) or None
    )
