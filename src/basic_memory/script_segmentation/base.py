"""The strategy contract shared by script index writes and script queries."""

from typing import Protocol

from basic_memory.repository.script_ngrams import ScriptQuery


class ScriptSegmenter(Protocol):
    """Turns text in scripts without word spaces into index terms and query phrases.

    A query matches only the terms its own strategy writes into the ``script_ngrams``
    column, so index writes and queries must use the same strategy.
    """

    @property
    def fingerprint(self) -> str:
        """Identity of the index text this strategy writes; a change makes stored text stale."""
        ...

    def build_index_text(self, *texts: str | None) -> str:
        """Return space-separated index terms for the script runs in ``texts``."""
        ...

    def analyze_query(self, text: str) -> ScriptQuery:
        """Split a query into word-channel text and script phrases that must all match."""
        ...
