"""Full-text search behavior under each configured script segmentation strategy."""

import pytest


@pytest.mark.asyncio
async def test_jieba_matches_chinese_words_rather_than_character_fragments(
    configure_segmentation, index_note, search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_note(1, "起源笔记", "研究生命起源的方法")

    assert await search_ids("生命起源") == [1]
    assert await search_ids("起源 方法") == [1]
    # 研究生 is a dictionary word; the note says 研究/生命, so it is not a word match.
    assert await search_ids("研究生") == []


@pytest.mark.asyncio
async def test_jieba_requires_latin_terms_of_mixed_queries_on_the_word_channel(
    configure_segmentation, index_note, search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_note(1, "watcher", "watchfiles 无法识别中文路径")
    await index_note(2, "路径", "中文路径的编码问题")

    assert await search_ids("中文路径") == [1, 2]
    assert await search_ids("watchfiles 中文路径") == [1]
    assert await search_ids("inotify 中文路径") == []


@pytest.mark.asyncio
async def test_jieba_user_dictionary_terms_match_only_as_whole_words(
    configure_segmentation, index_note, search_ids, tmp_path
) -> None:
    user_dict = tmp_path / "terms.txt"
    user_dict.write_text("踩坑经验库\n", encoding="utf-8")
    await configure_segmentation("jieba", user_dict)
    await index_note(1, "术语", "踩坑经验库收录常见问题")
    await index_note(2, "拆开", "经验库里记录了踩坑过程")

    assert await search_ids("踩坑经验库") == [1]


@pytest.mark.asyncio
async def test_jieba_without_user_dictionary_matches_unknown_terms_by_their_pieces(
    configure_segmentation, index_note, search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_note(1, "术语", "踩坑经验库收录常见问题")
    await index_note(2, "拆开", "经验库里记录了踩坑过程")

    assert await search_ids("踩坑经验库") == [1, 2]
