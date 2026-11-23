"""节点 H2：问句向量化与回答筛选（说明书第 10 节）。

只嵌入由数据库字段拼出来的规范句，不嵌入画面，也不嵌入模型写的日记。
问句和记忆句用同一个中文嵌入模型，保证同一空间。
向量与 events.id 存在 SQLite 向量扩展 sqlite-vec 的 vec_events 里。
第一道筛检索结果在生成之前；第二道筛生成结果只对模型句子执行，模板句天然通过。

模型签名（目录名 + 权重修改时间 + 维度）变化会清空向量，由 prepare 重建。
"""

import sys
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
if str(_repo / "common") not in sys.path:
    sys.path.insert(0, str(_repo / "common"))
import nodepath  # noqa: E402
from paths import REPO_ROOT as ROOT  # noqa: E402

import sqlite3  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from answer import NO_RECORD, _clean, speak_time  # noqa: E402


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def recall_config():
    return _load(ROOT / "config" / "recall.yaml")


RECALL = recall_config()

ACTION_WORDS = {"placed": "放下", "removed": "拿走", "occluded": "挡住", "conflict": "同时出现"}
SOURCE_WORDS = {"visual": "摄像头", "user_speech": "口述"}

_VOCAB = None
_EMBEDDER = None


def vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _load(ROOT / "config" / "vocab.yaml")
    return _VOCAB


def object_names(v):
    return {item["id"]: item["canonical_name"] for item in v["objects"]}


def slot_names(v):
    return {item["id"]: item["canonical_name"] for item in v["slots"]}


class Embedder:
    """延迟加载的本地嵌入模型。所有句子和问句都经过这里。"""

    def __init__(self):
        self._model = None
        self.dim = int(RECALL.get("dim", 512))
        self.prefix = str(RECALL.get("query_prefix") or "")

    @property
    def model_dir(self):
        return (ROOT / RECALL["model_dir"]).resolve()

    def signature(self):
        weight = self.model_dir / "model.safetensors"
        if not weight.exists():
            weight = self.model_dir / "pytorch_model.bin"
        if not weight.exists():
            return "missing"
        return f"{self.model_dir.name}|{weight.stat().st_mtime_ns}|{self.dim}"

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(str(self.model_dir))
        return self._model

    def encode(self, texts, query=False):
        model = self.load()
        if query and self.prefix:
            texts = [self.prefix + str(t) for t in texts]
        vec = model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def encode_one(self, text, query=False):
        return self.encode([text], query=query)[0]


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def _blob(vec):
    import sqlite_vec

    return sqlite_vec.serialize_float32(np.ascontiguousarray(vec, dtype=np.float32))


def load_vec_extension(conn):
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def ensure_vec(conn, emb=None):
    """建表。模型签名变化时清空向量并返回 True，等待 prepare 重建。"""
    load_vec_extension(conn)
    dim = int(RECALL.get("dim", 512))
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_events USING "
        f"vec0(embedding float[{dim}] distance_metric=cosine)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alias_vec (
             alias TEXT PRIMARY KEY,
             object_id TEXT NOT NULL,
             embedding BLOB NOT NULL,
             updated_at TEXT NOT NULL
           )"""
    )
    conn.execute("CREATE TABLE IF NOT EXISTS vec_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    if emb is None:
        return False
    sig = emb.signature()
    row = conn.execute("SELECT value FROM vec_meta WHERE key = 'model_sig'").fetchone()
    if row is None or row[0] != sig:
        conn.execute("DELETE FROM vec_events")
        conn.execute("DELETE FROM alias_vec")
        conn.execute(
            "INSERT OR REPLACE INTO vec_meta (key, value) VALUES ('model_sig', ?)", (sig,)
        )
        conn.commit()
        return True
    return False


def _as_dict(row):
    if isinstance(row, dict):
        return row
    return {key: row[key] for key in row.keys()}


def canonical_sentence(row, v=None):
    """由该行字段拼出规范句。例如“11月17日 08:12，摄像头，放下，钥匙，玄关柜。”
    口述行把“摄像头”换成“口述”。空槽位不写第五段。"""
    row = _as_dict(row)
    v = v or vocab()
    obj = object_names(v).get(row["object_id"], row["object_id"])
    slot = slot_names(v).get(row.get("slot_id")) if row.get("slot_id") else None
    parts = [
        speak_time(row["ts"]),
        SOURCE_WORDS.get(row["source"], str(row["source"])),
        ACTION_WORDS.get(row["action"], str(row["action"])),
        obj,
    ]
    if slot:
        parts.append(slot)
    return "，".join(parts) + "。"


def index_event_row(conn, event_id, ts, object_id, slot_id, action, source, emb=None):
    """apply_event 插入成功后调用。失败抛给调用方决定是否降级。"""
    emb = emb or embedder()
    ensure_vec(conn)
    sentence = canonical_sentence(
        {
            "ts": ts,
            "object_id": object_id,
            "slot_id": slot_id,
            "action": action,
            "source": source,
        }
    )
    conn.execute(
        "INSERT OR REPLACE INTO vec_events (rowid, embedding) VALUES (?, ?)",
        (event_id, _blob(emb.encode_one(sentence))),
    )
    conn.commit()


def build_alias_index(conn, v=None, emb=None):
    """每个别名单独一条短向量，指向物品 id。没有登记的说法不进索引。"""
    v = v or vocab()
    emb = emb or embedder()
    ensure_vec(conn)
    conn.execute("DELETE FROM alias_vec")
    items = []
    for obj in v["objects"]:
        for alias in obj["aliases"]:
            if alias:
                items.append((alias, obj["id"]))
    if items:
        vecs = emb.encode([alias for alias, _ in items])
        now = datetime.now().isoformat(timespec="seconds")
        conn.executemany(
            "INSERT OR REPLACE INTO alias_vec (alias, object_id, embedding, updated_at)"
            " VALUES (?, ?, ?, ?)",
            [(alias, object_id, _blob(vec), now) for (alias, object_id), vec in zip(items, vecs)],
        )
    conn.commit()
    return len(items)


def rebuild_index(conn, v=None, emb=None):
    """从 events 表全量重建事件向量和别名向量。幂等，模型换了之后调用。"""
    v = v or vocab()
    emb = emb or embedder()
    ensure_vec(conn, emb)
    rows = conn.execute(
        "SELECT id, ts, object_id, slot_id, action, source FROM events ORDER BY id"
    ).fetchall()
    conn.execute("DELETE FROM vec_events")
    if rows:
        sentences = [canonical_sentence(row, v) for row in rows]
        vecs = emb.encode(sentences)
        conn.executemany(
            "INSERT OR REPLACE INTO vec_events (rowid, embedding) VALUES (?, ?)",
            [(row["id"], _blob(vec)) for row, vec in zip(rows, vecs)],
        )
    conn.commit()
    aliases = build_alias_index(conn, v, emb)
    return len(rows), aliases


def prepare(conn):
    """查询前调用。签名变化或向量缺失时重建，平时只做一次空表检查。"""
    emb = embedder()
    changed = ensure_vec(conn, emb)
    if changed:
        return rebuild_index(conn, emb=emb)
    empty = conn.execute("SELECT rowid FROM vec_events LIMIT 1").fetchone() is None
    events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    aliases = conn.execute("SELECT COUNT(*) FROM alias_vec").fetchone()[0]
    if (events > 0 and empty) or aliases == 0:
        return rebuild_index(conn, emb=emb)
    return None


def delete_event_vectors(conn, event_ids):
    ensure_vec(conn)
    for event_id in event_ids:
        conn.execute("DELETE FROM vec_events WHERE rowid = ?", (event_id,))
    conn.commit()


def clear_vectors(conn):
    """清除命令或删除事件时，向量一起删。"""
    try:
        ensure_vec(conn)
        conn.execute("DELETE FROM vec_events")
        conn.execute("DELETE FROM alias_vec")
        conn.commit()
    except (ImportError, sqlite3.OperationalError, AttributeError):
        pass


def _hm(text):
    hour, minute = text.split(":")
    return int(hour), int(minute)


def parse_time_range(text, now=None):
    """把“早上、昨天早上”收成时间区间。返回 (start, end) 的 ISO 字符串或 None。"""
    words = RECALL.get("time_words", {})
    day_word = None
    hour_word = None
    for word in sorted(words, key=len, reverse=True):
        if word not in text:
            continue
        spec = words[word] or {}
        if spec.get("day", 0) != 0 and day_word is None:
            day_word = spec
        elif spec.get("hours") and hour_word is None:
            hour_word = spec
    if day_word is None and hour_word is None:
        return None
    now = now or datetime.now()
    base = now + timedelta(days=int((day_word or {}).get("day", 0)))
    start = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    if hour_word is not None:
        h1, m1 = _hm(hour_word["hours"][0])
        h2, m2 = _hm(hour_word["hours"][1])
        start = start.replace(hour=h1, minute=m1)
        end = start.replace(hour=h2, minute=m2)
        if end <= start:
            end += timedelta(days=1)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


def has_time_word(text):
    return parse_time_range(text) is not None


def _event_row(conn, event_id):
    return conn.execute(
        """SELECT id, ts, object_id, slot_id, action, source, confidence, evidence_path
           FROM events WHERE id = ?""",
        (event_id,),
    ).fetchone()


def _alias_sims(conn, qvec):
    out = []
    for row in conn.execute("SELECT alias, object_id, embedding FROM alias_vec").fetchall():
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        out.append(("alias", float(vec @ qvec), {"object_id": row["object_id"], "alias": row["alias"]}))
    return out


def recall_question(conn, question, time_range=None, v=None, emb=None, now=None):
    """第一道筛选（说明书 10.3）。返回 dict：
    verdict 取 hit / ambiguous / none；rows 是通过筛选的 (kind, 相似度, 行) 列表。
    有时间词时只在区间内检索；时间条件是标量过滤，不交给嵌入模型。
    前两名物品 id 不同才提问。同一物品的两行分差再小也不提问，
    否则问句会退化成“你是说钥匙，还是钥匙”，答案也不受影响。
    """
    v = v or vocab()
    emb = emb or embedder()
    ensure_vec(conn)
    top_k = int(RECALL.get("top_k", 5))
    min_sim = float(RECALL.get("min_sim", 0.55))
    qvec = emb.encode_one(question, query=True)
    candidates = []
    if time_range:
        rows = conn.execute(
            """SELECT id, ts, object_id, slot_id, action, source, confidence, evidence_path
               FROM events WHERE ts >= ? AND ts < ? ORDER BY ts DESC""",
            time_range,
        ).fetchall()
        scored = []
        for row in rows:
            stored = conn.execute(
                "SELECT embedding FROM vec_events WHERE rowid = ?", (row["id"],)
            ).fetchone()
            if stored is None:
                continue
            vec = np.frombuffer(stored[0], dtype=np.float32)
            scored.append((float(vec @ qvec), row))
        scored.sort(key=lambda item: -item[0])
        candidates = [("event", sim, dict(row)) for sim, row in scored[:top_k]]
    else:
        for hit in conn.execute(
            "SELECT rowid, distance FROM vec_events WHERE embedding MATCH ? AND k = ?",
            (_blob(qvec), top_k),
        ).fetchall():
            row = _event_row(conn, hit["rowid"])
            if row is None:
                continue
            candidates.append(("event", 1.0 - float(hit["distance"]), dict(row)))
    candidates += _alias_sims(conn, qvec)
    candidates.sort(key=lambda item: -item[1])
    if not candidates:
        return {"verdict": "none", "rows": [], "time_range": time_range}

    # 筛 2：物品 id 或槽位 id 不在词表里的行丢掉。空槽位允许（口述行可以没有槽位）。
    valid_objects = {item["id"] for item in v["objects"]}
    valid_slots = {item["id"] for item in v["slots"]}
    kept = []
    for kind, sim, payload in candidates:
        if payload.get("object_id") not in valid_objects:
            continue
        slot_id = payload.get("slot_id")
        if kind == "event" and slot_id and slot_id not in valid_slots:
            continue
        kept.append((kind, sim, payload))
    # 筛 3：相似度低于 0.55 的丢掉。
    kept = [item for item in kept if item[1] >= min_sim]
    if not kept:
        return {"verdict": "none", "rows": [], "time_range": time_range}

    # 筛 4：前两名物品 id 不同，不作答，改问一次。
    if len(kept) >= 2 and kept[0][2]["object_id"] != kept[1][2]["object_id"]:
        name_of = object_names(v)
        first = name_of.get(kept[0][2]["object_id"], kept[0][2]["object_id"])
        second = name_of.get(kept[1][2]["object_id"], kept[1][2]["object_id"])
        return {
            "verdict": "ambiguous",
            "rows": kept,
            "time_range": time_range,
            "ask": f"你是说{first}，还是{second}？",
        }
    return {"verdict": "hit", "rows": kept, "time_range": time_range}


def answer_rows_for_recall(result, v=None):
    """hit 时该物品通过筛选的全部事件行（含视觉和口述，按时间升序）。
    没有时间词时槽位回到当前状态行，事件行不参与句子。"""
    object_id = result["rows"][0][2]["object_id"]
    event_rows = sorted(
        (
            payload
            for kind, sim, payload in result["rows"]
            if kind == "event" and payload["object_id"] == object_id
        ),
        key=lambda row: row["ts"],
    )
    return object_id, event_rows


def event_row_sentence(row, v=None):
    """用事件行自己的字段填模板，不查当前状态。视觉行走节点 H 的表。"""
    row = _as_dict(row)
    v = v or vocab()
    name = object_names(v).get(row["object_id"], row["object_id"])
    when = speak_time(row["ts"])
    if row.get("source") == "user_speech":
        slot = slot_names(v).get(row.get("slot_id")) if row.get("slot_id") else None
        action = row.get("action")
        if action == "placed":
            return f"您在{when}说过{name}放在{slot}。" if slot else f"您在{when}说过{name}放下了。"
        if action == "removed":
            return f"您在{when}说过{name}被拿走了。"
        if action == "occluded":
            return f"您在{when}说过{name}被挡住了。"
        return f"您在{when}说过{name}。"
    from answer import _visual_sentence

    visual = {
        "action": row.get("action"),
        "slot_id": row.get("slot_id"),
        "ts": row.get("ts"),
        "confidence": row.get("confidence"),
    }
    return _visual_sentence(name, visual, v)


def filter_generated(sentence, rows, v=None):
    """第二道筛选（说明书 10.3 的表）。只筛模型生成的句子，模板句不经过这里。

    rows 是通过第一道筛选的 (kind, 相似度, 行) 列表。
    返回 (是否通过, 原因列表)。任何一条不满足都丢弃整句，改念模板。
    """
    v = v or vocab()
    reasons = []
    for word in RECALL.get("forbidden_words", []):
        if word and word in sentence:
            reasons.append(f"句中有“{word}”")

    row_object_ids = {payload["object_id"] for _, _, payload in rows}
    row_slot_ids = {
        payload.get("slot_id") for _, _, payload in rows if payload.get("slot_id")
    }

    def mentions(items, text):
        found = []
        for item in items:
            names = [item["canonical_name"]] + [a for a in item.get("aliases", []) if a]
            for name in names:
                if name and name in text:
                    found.append((item["id"], name))
                    break
        return found

    for object_id, name in mentions(v["objects"], sentence):
        if object_id not in row_object_ids:
            reasons.append(f"物品“{name}”不在这些行里")
    for slot_id, name in mentions(v["slots"], sentence):
        if slot_id not in row_slot_ids:
            reasons.append(f"位置“{name}”不在这些行里")
    if row_slot_ids and not mentions(v["slots"], sentence):
        reasons.append("行里有槽位，句子里没有行内槽位")

    has_visual = any(payload.get("source") == "visual" for _, _, payload in rows)
    has_speech = any(payload.get("source") == "user_speech" for _, _, payload in rows)
    if "摄像头" in sentence and not has_visual:
        reasons.append("句子里说摄像头，行里没有摄像头来源")
    if ("口述" in sentence or "您" in sentence) and not has_speech:
        reasons.append("句子里说口述，行里没有口述来源")
    return (not reasons), reasons


def clean_question(text):
    return _clean(text)
