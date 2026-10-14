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


PIP_QUESTION = "pip install 一直报 ReadTimeoutError，连不上 PyPI，怎么解决？"
POWERSHELL_QUESTION = "PowerShell 脚本里的中文字符串报解析错误"


async def index_lessons(index_note) -> None:
    await index_note(
        1, "pip 安装超时", "pip install 报 ReadTimeoutError：连接 PyPI 超时，需要配置企业代理"
    )
    await index_note(
        2,
        "PowerShell 中文乱码",
        "PowerShell 5.1 读取无 BOM 的 UTF-8 脚本时，中文字符串被当成乱码导致解析失败",
    )
    await index_note(3, "电脑唤醒后断网", "Windows 电脑从休眠唤醒后网络适配器断开连接")
    await index_note(4, "秋天", "落叶和桂花香的季节")


@pytest.mark.asyncio
async def test_jieba_natural_sentences_keyword_match_notes_sharing_their_distinctive_words(
    configure_segmentation, index_note, search_ids, relaxed_search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_lessons(index_note)

    # Every word of a whole sentence rarely appears in one note, so the strict query
    # finds nothing and the zero-result retry hybrid search allows has to match.
    assert await search_ids(PIP_QUESTION) == []
    assert await relaxed_search_ids(PIP_QUESTION) == [1]
    assert await relaxed_search_ids(POWERSHELL_QUESTION) == [2]
    assert await relaxed_search_ids("我的 PowerShell 脚本里的中文字符串报解析错误了吗？") == [2]
    assert await relaxed_search_ids("电脑唤醒以后网络就断开了，怎么办？") == [3]


@pytest.mark.asyncio
async def test_jieba_relaxed_retry_reads_latin_words_glued_to_chinese_text(
    configure_segmentation, index_note, relaxed_search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_lessons(index_note)

    assert await relaxed_search_ids("下载依赖时一直报ReadTimeoutError，怎么处理") == [1]


@pytest.mark.asyncio
async def test_jieba_relaxed_retry_ranks_notes_sharing_more_words_first(
    configure_segmentation, index_note, relaxed_search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_lessons(index_note)
    await index_note(5, "通用", "遇到问题先想想怎么解决")

    assert await relaxed_search_ids(PIP_QUESTION) == [1, 5]


@pytest.mark.asyncio
async def test_jieba_relaxed_retry_keeps_short_queries_strict(
    configure_segmentation, index_note, relaxed_search_ids
) -> None:
    await configure_segmentation("jieba")
    await index_lessons(index_note)

    # One distinctive word alongside one Latin word would broaden to either under OR.
    assert await relaxed_search_ids("PowerShell 季节") == []


@pytest.mark.asyncio
async def test_ngram_default_relaxed_retry_still_requires_every_script_phrase(
    index_note, relaxed_search_ids
) -> None:
    await index_lessons(index_note)

    assert await relaxed_search_ids(PIP_QUESTION) == []
    assert await relaxed_search_ids(POWERSHELL_QUESTION) == []
