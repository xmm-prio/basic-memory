"""The process-wide script segmentation strategy.

``repository.script_ngrams`` delegates every script index write and query to it, so
all search backends share one strategy without being handed it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basic_memory.script_segmentation.base import ScriptSegmenter

_active: "ScriptSegmenter | None" = None


def active_script_segmenter() -> "ScriptSegmenter":
    """Return the installed strategy; upstream n-grams until another one is installed."""
    global _active
    if _active is None:
        # Deferred: the n-gram strategy wraps repository.script_ngrams, which imports this
        # module while it initializes.
        from basic_memory.script_segmentation.ngram import NgramSegmenter

        _active = NgramSegmenter()
    return _active


def use_script_segmenter(segmenter: "ScriptSegmenter") -> None:
    """Install ``segmenter`` for every later script index write and query in this process."""
    global _active
    _active = segmenter
