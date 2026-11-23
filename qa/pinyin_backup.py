"""拼音近邻。备用，不接入 ask.py。命中多个时只返回候选，不选位置。"""


def candidates(text, vocab):
    from pypinyin import lazy_pinyin

    compact = (
        text.replace(" ", "")
        .replace("？", "")
        .replace("?", "")
        .replace("。", "")
    )
    for suffix in ("在哪里", "在哪", "放哪了", "放哪", "呢"):
        if compact.endswith(suffix) and len(compact) > len(suffix):
            compact = compact[: -len(suffix)]
            break
    target = tuple(lazy_pinyin(compact))
    if not target:
        return []
    hits = []
    for obj in vocab["objects"]:
        for alias in obj["aliases"]:
            if tuple(lazy_pinyin(alias)) == target:
                hits.append(obj["id"])
                break
    return list(dict.fromkeys(hits))
