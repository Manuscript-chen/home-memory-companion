"""从数据库回答一句问话。位置只来自词表。

顺序：节点 H 的第 1 到 3 级先走；拿到唯一物品 id 且没有时间词时直接读当前状态行。
第 1 到 3 级失败，或问句里带时间词时，进入节点 H2 的向量召回（说明书 10.5）。
每次问答把“精确命中”还是“向量召回”写进 recall_log。
"""

import sys
from pathlib import Path

import yaml

from answer import NO_RECORD, find_object, sentence_for_object
from memory import connect, get_state, get_value, log_recall, set_value

ROOT = Path(__file__).resolve().parent


def load_vocab():
    with (ROOT / "config" / "vocab.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _name(vocab, object_id):
    for obj in vocab["objects"]:
        if obj["id"] == object_id:
            return obj["canonical_name"]
    return object_id


def ask_once(conn, vocab, text, last_object_id):
    """返回 (回答, 新的上一件物品, 路径)。路径是“精确命中”或“向量召回”。"""
    import vec_memory as vec

    time_range = vec.parse_time_range(vec.clean_question(text))
    object_id, early = find_object(text, vocab, last_object_id)

    # 第 2、3 级的提问句直接返回。没有上一件的“它”也在这里。
    if early is not None and early != NO_RECORD:
        return early, last_object_id, "精确命中"

    if object_id is not None and time_range is None:
        state = get_state(conn, object_id)
        return sentence_for_object(vocab, object_id, state), object_id, "精确命中"

    # 第 1 到 3 级失败，或问句带时间词：向量召回。
    cleaned = vec.clean_question(text)
    try:
        vec.prepare(conn)
        result = vec.recall_question(conn, cleaned, time_range, v=vocab)
    except Exception as exc:
        if object_id is not None:
            state = get_state(conn, object_id)
            reply = sentence_for_object(vocab, object_id, state)
            return reply, object_id, f"向量不可用（{exc}）"
        return NO_RECORD, None, f"向量不可用（{exc}）"

    if result["verdict"] == "ambiguous":
        return result["ask"], last_object_id, "向量召回"

    if result["verdict"] == "none":
        if object_id is not None:
            name = _name(vocab, object_id)
            return f"我没有{name}在这个时间段的记录。", object_id, "向量召回"
        return NO_RECORD, None, "向量召回"

    # hit：先定物品 id。没有时间词时槽位回到当前状态行，事件行不进句子。
    resolved, event_rows = vec.answer_rows_for_recall(result, v=vocab)
    name = _name(vocab, resolved)
    if time_range is not None:
        if not event_rows:
            return f"我没有{name}在这个时间段的记录。", resolved, "向量召回"
        visual_rows = [r for r in event_rows if r.get("source") == "visual"]
        speech_rows = [r for r in event_rows if r.get("source") == "user_speech"]
        if visual_rows:
            reply = vec.event_row_sentence(visual_rows[-1], v=vocab)
            if speech_rows:
                reply += vec.event_row_sentence(speech_rows[-1], v=vocab)
        else:
            reply = vec.event_row_sentence(speech_rows[-1], v=vocab)
        return reply, resolved, "向量召回"
    state = get_state(conn, resolved)
    return sentence_for_object(vocab, resolved, state), resolved, "向量召回"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法：python ask.py 钥匙在哪")
    text = "".join(sys.argv[1:])
    vocab = load_vocab()
    conn = connect(ROOT / "data" / "memory.db")
    try:
        last = get_value(conn, "last_object")
        reply, new_last, path = ask_once(conn, vocab, text, last)
        if new_last:
            set_value(conn, "last_object", new_last)
        log_recall(conn, text, path, new_last)
        print(reply)
        print(f"路径：{path}", file=sys.stderr)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
