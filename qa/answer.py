"""问答只填词表。位置句不由模型生成。"""


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[-1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def band(confidence):
    if confidence is None or confidence < 0.40:
        return "low"
    if confidence < 0.60:
        return "mid"
    return "high"


def speak_time(ts):
    return f"{int(ts[5:7])}月{int(ts[8:10])}日 {ts[11:16]}"


def _clean(text):
    compact = text.replace(" ", "").replace("？", "").replace("?", "").replace("。", "")
    for suffix in ("在哪里", "在哪", "放哪了", "放哪", "呢"):
        if compact.endswith(suffix) and len(compact) > len(suffix):
            return compact[: -len(suffix)]
    return compact


NO_RECORD = "我没有这个物品的记录。"


def find_object(text, vocab, last_object_id):
    raw = text.replace(" ", "").replace("？", "").replace("?", "").replace("。", "")
    if raw in {"它在哪", "它在哪里", "那个在哪", "那个在哪里", "它呢", "那个呢"}:
        if last_object_id is None:
            return None, "请说出物品的名字。"
        return last_object_id, None

    compact = _clean(text)
    objects = vocab["objects"]
    # 全词表按别名长度从长到短试。否则“家门钥匙”会先被更短的“钥匙”截走。
    ranked = []
    for obj in objects:
        for alias in obj["aliases"]:
            if alias:
                ranked.append((len(alias), obj["id"], alias))
    ranked.sort(key=lambda item: item[0], reverse=True)
    for _, object_id, alias in ranked:
        if alias in compact:
            return object_id, None

    hits = []
    for obj in objects:
        for alias in obj["aliases"]:
            if min(len(alias), len(compact)) < 2:
                continue
            if abs(len(alias) - len(compact)) > 1:
                continue
            if edit_distance(alias, compact) <= 1:
                hits.append(obj["id"])
                break
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        names = "、".join(_name(vocab, item) for item in hits)
        return None, f"你说的是哪一个：{names}？"
    return None, NO_RECORD


def _name(vocab, object_id):
    for obj in vocab["objects"]:
        if obj["id"] == object_id:
            return obj["canonical_name"]
    return object_id


def _slot(vocab, slot_id):
    if not slot_id:
        return None
    for slot in vocab["slots"]:
        if slot["id"] == slot_id:
            return slot["canonical_name"]
    return None


def _visual(state):
    if not state or not state.get("visual_action"):
        return None
    return {
        "action": state["visual_action"],
        "slot_id": state.get("visual_slot_id"),
        "ts": state.get("visual_ts"),
        "confidence": state.get("visual_confidence"),
    }


def speech_differs(state):
    """口述列和视觉列的槽位或动作不一样时才加第二句。"""
    if not state:
        return False
    note = (state.get("speech_note") or "").strip()
    has_speech = bool(state.get("speech_action") or note)
    if not has_speech:
        return False
    if not state.get("visual_action"):
        return True
    if state.get("speech_action") and state["speech_action"] != state["visual_action"]:
        return True
    visual_slot = state.get("visual_slot_id") or ""
    speech_slot = state.get("speech_slot_id") or ""
    if note and not speech_slot:
        return True
    if speech_slot and visual_slot and speech_slot != visual_slot:
        return True
    return False


def speech_clause(vocab, state):
    note = (state.get("speech_note") or "").strip()
    slot_id = state.get("speech_slot_id") or ""
    action = state.get("speech_action")
    if note and not slot_id:
        body = note[:-1] if note.endswith("。") else note
        return f"您后来说{body}。"
    slot = _slot(vocab, slot_id)
    if action == "removed":
        return "您后来说已经拿走。"
    if action == "placed" and slot:
        return f"您后来说放在{slot}。"
    if action == "occluded":
        return "您后来说被挡住了。"
    if note:
        body = note[:-1] if note.endswith("。") else note
        return f"您后来说{body}。"
    return "您后来说了另一件事。"


def history_slot_question(vocab, slot_ids):
    """备用的第 4 级。只生成一次选择，不把出现次数说成当前位置。"""
    names = []
    for slot_id in slot_ids:
        name = _slot(vocab, slot_id)
        if name and name not in names:
            names.append(name)
    if len(names) < 2:
        return None
    return "你是说" + "，还是".join(names) + "？"


def _visual_sentence(name, visual, vocab):
    level = band(visual["confidence"])
    slot = _slot(vocab, visual["slot_id"])
    when = speak_time(visual["ts"]) if visual.get("ts") else ""
    status = visual["action"]
    if level == "low" or slot is None:
        return "我不能确定。请查看该时刻保存的照片。"
    if status == "placed":
        if level == "high":
            return f"摄像头看到{name}在{slot}。"
        return f"最后一次摄像头看到{name}是在{slot}，时间是 {when}。"
    if status == "removed":
        return f"{name}最后出现在{slot}，{when} 被拿走了。"
    if status == "occluded":
        return f"{name}最后出现在{slot}，后来被挡住了，我不能确定还在不在。"
    if status == "conflict":
        return f"{slot}上同时有两个{name}，我不能确定是哪一个。"
    return f"我没有{name}的记录。"


def answer(text, vocab, state, last_object_id):
    object_id, early = find_object(text, vocab, last_object_id)
    if early:
        return early, last_object_id
    return sentence_for_object(vocab, object_id, state), object_id


def sentence_for_object(vocab, object_id, state):
    """按节点 H 的表填句子。节点 H2 召回到唯一物品 id 后也走这里，不另写一套位置句。"""
    name = _name(vocab, object_id)
    visual = _visual(state)
    if visual is None:
        if speech_differs(state):
            return f"摄像头没有{name}的记录。{speech_clause(vocab, state)}"
        return f"我没有{name}的记录。"
    sentence = _visual_sentence(name, visual, vocab)
    if speech_differs(state):
        sentence += speech_clause(vocab, state)
    return sentence
