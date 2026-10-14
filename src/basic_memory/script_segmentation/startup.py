"""Install the configured script segmentation strategy when the database is opened."""

from typing import assert_never

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.repository.script_ngrams import install_script_segmentation
from basic_memory.script_segmentation.jieba_segmenter import JiebaSegmenter
from basic_memory.script_segmentation.sqlite_index import (
    NGRAM_FINGERPRINT,
    align_sqlite_script_index,
)


async def prepare_script_segmentation(
    session_maker: async_sessionmaker[AsyncSession],
    app_config: BasicMemoryConfig,
    database_backend: DatabaseBackend,
) -> None:
    """Install the configured strategy and rebuild script index text written by another.

    Runs whenever a process opens the database, before anything is indexed or searched,
    so index writes and queries in every entry point use the same strategy.
    """
    match app_config.script_segmentation:
        case "ngram":
            install_script_segmentation(None)
            fingerprint = NGRAM_FINGERPRINT
        case "jieba":
            if database_backend != DatabaseBackend.SQLITE:
                # Postgres also derives script text per FTS chunk; rebuilding that is not
                # implemented.
                raise ValueError(
                    "script_segmentation='jieba' requires the SQLite backend; "
                    f"{database_backend.value} supports only 'ngram'"
                )
            segmenter = JiebaSegmenter(
                cache_dir=app_config.data_dir_path,
                user_dict=app_config.script_segmentation_user_dict,
            )
            install_script_segmentation(segmenter)
            fingerprint = segmenter.fingerprint
        case unsupported:  # pragma: no cover - the config field is a closed Literal
            assert_never(unsupported)
    if database_backend == DatabaseBackend.SQLITE:
        await align_sqlite_script_index(session_maker, fingerprint)
