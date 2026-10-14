"""Keep SQLite's stored script index text in step with the installed strategy."""

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.repository.script_ngrams import build_script_ngrams

# Created here rather than by an Alembic revision so the fork adds nothing to
# upstream's migration chain.
_CREATE_STATE_TABLE = text(
    """
    CREATE TABLE IF NOT EXISTS script_segmentation_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        fingerprint TEXT NOT NULL
    )
    """
)
_SELECT_FINGERPRINT = text("SELECT fingerprint FROM script_segmentation_state WHERE id = 1")
_UPSERT_FINGERPRINT = text(
    """
    INSERT INTO script_segmentation_state (id, fingerprint) VALUES (1, :fingerprint)
    ON CONFLICT (id) DO UPDATE SET fingerprint = excluded.fingerprint
    """
)
_SELECT_ROWS = text(
    """
    SELECT rowid, title, content_stems, content_snippet
    FROM search_index
    WHERE rowid > :after
    ORDER BY rowid
    LIMIT :limit
    """
)
_UPDATE_ROW = text("UPDATE search_index SET script_ngrams = :script_ngrams WHERE rowid = :rowid")

_BATCH_SIZE = 500
_MIN_ROWID = -(2**63)

NGRAM_FINGERPRINT = "ngram"
# Index text written before the state table existed came from upstream n-grams.
_UNRECORDED_FINGERPRINT = NGRAM_FINGERPRINT


async def align_sqlite_script_index(
    session_maker: async_sessionmaker[AsyncSession],
    fingerprint: str,
) -> None:
    """Rebuild the ``script_ngrams`` column in place when another strategy wrote it.

    ``fingerprint`` identifies the strategy installed in this process, which rewrites
    the rows through ``build_script_ngrams``. Every SQLite search row derives the column
    from its own title, content_stems and content_snippet, so the rebuild rereads no
    files and leaves embeddings untouched. The rows and the recorded fingerprint change
    in one transaction.
    """
    async with db.scoped_session(session_maker) as session:
        await session.execute(_CREATE_STATE_TABLE)
        recorded = (await session.execute(_SELECT_FINGERPRINT)).scalar_one_or_none()
        previous = recorded or _UNRECORDED_FINGERPRINT
        if previous == fingerprint:
            return
        logger.info("Rebuilding script index text", previous=previous, current=fingerprint)
        rebuilt = await _rebuild_rows(session)
        await session.execute(_UPSERT_FINGERPRINT, {"fingerprint": fingerprint})
    logger.info("Rebuilt script index text", rows=rebuilt, current=fingerprint)


async def _rebuild_rows(session: AsyncSession) -> int:
    rebuilt = 0
    after = _MIN_ROWID
    while rows := (
        await session.execute(_SELECT_ROWS, {"after": after, "limit": _BATCH_SIZE})
    ).all():
        await session.execute(
            _UPDATE_ROW,
            [
                {
                    "rowid": row.rowid,
                    "script_ngrams": build_script_ngrams(
                        row.title, row.content_stems, row.content_snippet
                    ),
                }
                for row in rows
            ],
        )
        rebuilt += len(rows)
        after = rows[-1].rowid
    return rebuilt
