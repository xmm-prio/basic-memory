"""Fixtures for configuring the process-wide script segmentation strategy."""

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.repository.script_ngrams import install_script_segmentation
from basic_memory.repository.search_index_row import SearchIndexRow
from basic_memory.script_segmentation.startup import prepare_script_segmentation


@pytest_asyncio.fixture
async def configure_segmentation(
    session_maker, app_config: BasicMemoryConfig, db_backend: str
) -> AsyncIterator[Callable[..., Awaitable[None]]]:
    """Apply a strategy through the startup step, then restore the upstream n-grams."""
    if db_backend == "postgres":
        pytest.skip("Non-default script segmentation strategies are SQLite-only")

    async def configure(strategy: str, user_dict: Path | None = None) -> None:
        config = app_config.model_copy(
            update={"script_segmentation": strategy, "script_segmentation_user_dict": user_dict}
        )
        await prepare_script_segmentation(session_maker, config, DatabaseBackend.SQLITE)

    yield configure
    install_script_segmentation(None)


@pytest.fixture
def index_note(search_repository) -> Callable[[int, str, str], Awaitable[None]]:
    """Index one entity row whose body is `content`."""

    async def index(row_id: int, title: str, content: str) -> None:
        now = datetime.now(timezone.utc)
        await search_repository.index_item(
            SearchIndexRow(
                project_id=search_repository.project_id,
                id=row_id,
                type="entity",
                file_path=f"notes/{row_id}.md",
                title=title,
                content_stems=content,
                content_snippet=content,
                permalink=f"notes/{row_id}",
                created_at=now,
                updated_at=now,
            )
        )

    return index


@pytest.fixture
def search_ids(search_repository) -> Callable[[str], Awaitable[list[int]]]:
    """Run a full-text search and return the matching row ids in ascending order."""

    async def search(query: str) -> list[int]:
        return sorted(result.id for result in await search_repository.search(query))

    return search
