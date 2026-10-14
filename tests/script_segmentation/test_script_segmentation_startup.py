"""Opening the database applies the configured script segmentation strategy."""

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend, ProjectEntry
from basic_memory.repository.script_ngrams import install_script_segmentation
from basic_memory.repository.search_index_row import SearchIndexRow
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.script_segmentation.startup import prepare_script_segmentation


@pytest_asyncio.fixture
async def restore_segmenter() -> AsyncIterator[None]:
    yield
    install_script_segmentation(None)


def make_config(config_home, **overrides) -> BasicMemoryConfig:
    return BasicMemoryConfig(
        env="test",
        projects={"main": ProjectEntry(path=str(config_home / "main"))},
        default_project="main",
        **overrides,
    )


def test_strategy_and_user_dictionary_are_read_from_the_environment(
    config_home, monkeypatch
) -> None:
    monkeypatch.setenv("BASIC_MEMORY_SCRIPT_SEGMENTATION", "jieba")
    monkeypatch.setenv("BASIC_MEMORY_SCRIPT_SEGMENTATION_USER_DICT", str(config_home / "t.txt"))

    config = make_config(config_home)

    assert config.script_segmentation == "jieba"
    assert config.script_segmentation_user_dict == config_home / "t.txt"


def test_upstream_ngram_strategy_is_the_default(config_home) -> None:
    assert make_config(config_home).script_segmentation == "ngram"


def test_an_unknown_strategy_is_refused_by_the_config(config_home) -> None:
    with pytest.raises(ValidationError, match="script_segmentation"):
        make_config(config_home, script_segmentation="bigram")


@pytest.mark.asyncio
async def test_opening_the_database_applies_the_configured_strategy(
    config_home, restore_segmenter
) -> None:
    config = make_config(config_home, script_segmentation="jieba")
    _, session_maker = await db.get_or_create_db(config.database_path, config=config)
    try:
        repository = SQLiteSearchRepository(session_maker, project_id=1, app_config=config)
        now = datetime.now(timezone.utc)
        await repository.index_item(
            SearchIndexRow(
                project_id=1,
                id=1,
                type="entity",
                file_path="notes/origin.md",
                title="起源笔记",
                content_stems="研究生命起源的方法",
                content_snippet="研究生命起源的方法",
                permalink="notes/origin",
                created_at=now,
                updated_at=now,
            )
        )

        assert [result.id for result in await repository.search("生命起源")] == [1]
        assert await repository.search("研究生") == []
    finally:
        await db.shutdown_db()


@pytest.mark.asyncio
async def test_opening_the_database_fails_when_the_user_dictionary_is_missing(
    config_home, restore_segmenter
) -> None:
    missing = config_home / "no-such-terms.txt"
    config = make_config(
        config_home, script_segmentation="jieba", script_segmentation_user_dict=missing
    )

    try:
        with pytest.raises(FileNotFoundError, match="no-such-terms.txt"):
            await db.get_or_create_db(config.database_path, config=config)
    finally:
        await db.shutdown_db()


@pytest.mark.asyncio
async def test_jieba_is_rejected_on_postgres(config_home, restore_segmenter) -> None:
    config = make_config(config_home, script_segmentation="jieba")

    # An unbound session maker: rejection must happen before any database access.
    with pytest.raises(ValueError, match="SQLite"):
        await prepare_script_segmentation(async_sessionmaker(), config, DatabaseBackend.POSTGRES)
