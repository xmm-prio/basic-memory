"""Dictionary-word strategy backed by jieba."""

import hashlib
import io
import logging
from pathlib import Path

import jieba

from basic_memory.repository.script_ngrams import ScriptQuery, analyze_ngram_query, script_runs

# Part of the fingerprint: bump it when the index text layout changes.
_INDEX_LAYOUT_VERSION = 1


class JiebaSegmenter:
    """Index jieba search-mode words plus single characters; query with jieba words.

    Every query word is its own phrase and all of them must match. A multi-character
    word therefore hits only notes where jieba produced that word, while a single
    character still matches anywhere it occurs. Han text is segmented with jieba's
    dictionary (plus the optional user dictionary); jieba leaves kana, Hangul and
    other scripts as single characters. HMM new-word discovery stays off on both
    sides, so every query word is a dictionary word that indexing can produce too.
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

    def analyze_query(self, text: str) -> ScriptQuery:
        query = analyze_ngram_query(text)
        if not query.gram_phrases:
            return query
        words = dict.fromkeys(
            word
            for run in script_runs(text)
            for word in self._tokenizer.cut("".join(run), HMM=False)
        )
        return ScriptQuery(
            word_text=query.word_text,
            gram_phrases=tuple((word,) for word in words),
        )
