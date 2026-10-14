"""Compose the configured script segmentation strategy when the database is opened."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.script_segmentation.active import use_script_segmenter
from basic_memory.script_segmentation.base import ScriptSegmenter
from basic_memory.script_segmentation.jieba_segmenter import JiebaSegmenter
from basic_memory.script_segmentation.ngram import NgramSegmenter
from basic_memory.script_segmentation.sqlite_index import align_sqlite_script_index


def create_script_segmenter(app_config: BasicMemoryConfig) -> ScriptSegmenter:
    """Build the strategy named by ``script_segmentation``."""
    if app_config.script_segmentation == "jieba":
        user_dict = app_config.script_segmentation_user_dict
        return JiebaSegmenter(
            cache_dir=app_config.data_dir_path,
            user_dict=Path(user_dict) if user_dict else None,
        )
    return NgramSegmenter()


async def prepare_script_segmentation(
    session_maker: async_sessionmaker[AsyncSession],
    app_config: BasicMemoryConfig,
    database_backend: DatabaseBackend,
) -> None:
    """Install the configured strategy and rebuild script index text written by another.

    Runs whenever a process opens the database, before anything is indexed or searched,
    so index writes and queries in every entry point use the same strategy.
    """
    if database_backend != DatabaseBackend.SQLITE and app_config.script_segmentation != "ngram":
        # Postgres also derives script text per FTS chunk; rebuilding that is not implemented.
        raise ValueError(
            f"script_segmentation={app_config.script_segmentation!r} requires the SQLite "
            f"backend; {database_backend.value} supports only 'ngram'"
        )
    segmenter = create_script_segmenter(app_config)
    if database_backend == DatabaseBackend.SQLITE:
        await align_sqlite_script_index(session_maker, segmenter)
    use_script_segmenter(segmenter)
