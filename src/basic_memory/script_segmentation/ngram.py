"""Upstream strategy: script characters and ordered bigrams."""

from basic_memory.repository.script_ngrams import (
    ScriptQuery,
    analyze_ngram_query,
    build_ngram_index_text,
)


class NgramSegmenter:
    """Index every script character and bigram; query with ordered bigram phrases."""

    fingerprint = "ngram"

    def build_index_text(self, *texts: str | None) -> str:
        return build_ngram_index_text(*texts)

    def analyze_query(self, text: str) -> ScriptQuery:
        return analyze_ngram_query(text)
