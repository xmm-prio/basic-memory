"""Stored script index text follows the configured strategy across restarts."""

import pytest


@pytest.mark.asyncio
async def test_switching_strategy_rebuilds_existing_index_text(
    configure_segmentation, index_note, search_ids
) -> None:
    # Written by the upstream n-gram default before any strategy was recorded.
    await index_note(1, "起源笔记", "研究生命起源的方法")
    await index_note(2, "国家", "中华人民共和国成立")
    assert await search_ids("研究生") == [1]

    await configure_segmentation("jieba")
    assert await search_ids("研究生") == []
    # The word 共和国 exists only in text rebuilt by jieba.
    assert await search_ids("共和国") == [2]

    await configure_segmentation("ngram")
    # The bigram phrase 研究 究生 exists only in text rebuilt by n-grams.
    assert await search_ids("研究生") == [1]


@pytest.mark.asyncio
async def test_changing_the_user_dictionary_rebuilds_existing_index_text(
    configure_segmentation, index_note, search_ids, tmp_path
) -> None:
    await configure_segmentation("jieba")
    await index_note(1, "术语", "踩坑经验库收录常见问题")

    user_dict = tmp_path / "terms.txt"
    user_dict.write_text("踩坑经验库\n", encoding="utf-8")
    await configure_segmentation("jieba", user_dict)

    assert await search_ids("踩坑经验库") == [1]
