"""只读资料室。打开现有的库、证据照片和训练存档，不写业务数据，不重新训练。"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
VOCAB_PATH = ROOT / "参考实现" / "config" / "vocab.yaml"

EVIDENCE_ROOTS = {
    "ref": ROOT / "参考实现" / "data" / "evidence",
    "lab": HERE / "data" / "evidence",
}
CLOSED_ROOT = HERE / "data" / "closed_set" / "images"
REAL_ROOT = HERE / "data" / "real_set"
MODELS = ROOT / "工具" / "models"

LABEL_TO_ID = {
    "主库": "lab",
    "契约库": "contract",
    "清除后的副本": "deleted",
    "H2 库": "h2",
    "第4级库": "level4",
    "拼音对照库": "pinyin",
    "按键库": "voice",
    "分界侧库": "band",
}

DATABASES = [
    {
        "id": "memory",
        "title": "正式记忆库",
        "path": ROOT / "参考实现" / "data" / "memory.db",
        "blurb": "提问、回放和摄像头线束写进这一只。这是参考实现正在用的库。",
    },
    {
        "id": "lab",
        "title": "实验主库",
        "path": HERE / "data" / "lab.db",
        "blurb": "仿真全流程留下的库。整库清除做在副本上，这一只还留着当时的行。",
    },
    {
        "id": "deleted",
        "title": "清除后的副本",
        "path": HERE / "data" / "deleted.db",
        "blurb": "从实验主库复制出来之后执行了整库清除。事件行已经不在文件里。",
    },
    {
        "id": "contract",
        "title": "契约库",
        "path": HERE / "data" / "contract" / "contract.db",
        "blurb": "用来核对“事件只追加、清除才整表删掉”。清除之后文件里的事件是空的。",
    },
    {
        "id": "h2",
        "title": "向量召回库",
        "path": HERE / "data" / "h2.db",
        "blurb": "问句走向量召回时用的仿真库。别名向量在 alias_vec。",
    },
    {
        "id": "level4",
        "title": "第4级库",
        "path": HERE / "data" / "level4.db",
        "blurb": "同一把钥匙先后出现在两个槽位，用来核对第 4 级只提问、不改当前槽位。",
    },
    {
        "id": "pinyin",
        "title": "拼音对照库",
        "path": HERE / "data" / "pinyin.db",
        "blurb": "拼音近邻不进入主路径的那一次对照。",
    },
    {
        "id": "band",
        "title": "分界侧库",
        "path": HERE / "data" / "band.db",
        "blurb": "和主库分开的一侧对照，避免两条流程写进同一只文件。",
    },
    {
        "id": "voice",
        "title": "按键库",
        "path": HERE / "data" / "voice.db",
        "blurb": "只收录按键按下到松开之间那一句的仿真库。",
    },
    {
        "id": "roundtrip",
        "title": "语音回环库",
        "path": ROOT / "参考实现" / "data" / "voice_roundtrip.db",
        "blurb": "合成语音问 20 句之后写下的输出日志。不是真人麦克风录音。",
    },
]

VISIBLE_TABLES = ("events", "object_states", "plans", "session", "recall_log", "alias_vec")
ACTIONS = {"placed": "放下", "removed": "拿走", "occluded": "挡住", "conflict": "同时出现"}
SOURCES = {"visual": "摄像头", "user_speech": "口述"}
FIELD_LABELS = {
    "id": "编号",
    "ts": "时间",
    "object_id": "物品",
    "slot_id": "槽位",
    "action": "动作",
    "source": "来源",
    "confidence": "置信度",
    "cam_id": "摄像头",
    "evidence_path": "照片路径",
    "created_at": "写入时间",
    "visual_slot_id": "视觉槽位",
    "visual_action": "视觉动作",
    "visual_ts": "视觉时间",
    "visual_confidence": "视觉置信度",
    "speech_slot_id": "口述槽位",
    "speech_action": "口述动作",
    "speech_ts": "口述时间",
    "speech_note": "口述原话",
    "updated_at": "更新时间",
    "title": "事项",
    "due_ts": "到点",
    "done": "是否完成",
    "done_source": "完成来源",
    "confirm_ts": "确认时间",
    "key": "键",
    "value": "值",
    "question": "问句",
    "path": "路径",
    "alias": "别名",
    "embedding": "向量",
    "step": "删除发生在",
}

_NAMES = None
_DELETIONS = None
_CORPUS = None


def human_size(size):
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def speak_time(value):
    if not value or "T" not in str(value):
        return value or "—"
    day, clock = str(value).split("T", 1)
    parts = day.split("-")
    if len(parts) != 3:
        return str(value)
    hour, minute = (clock.split(":") + ["00", "00"])[:2]
    return f"{int(parts[1])}月{int(parts[2])}日 {hour}:{minute}"


def names():
    global _NAMES
    if _NAMES is not None:
        return _NAMES
    objects, slots = {}, {}
    try:
        import yaml

        with VOCAB_PATH.open(encoding="utf-8") as handle:
            vocab = yaml.safe_load(handle)
        objects = {item["id"]: item["canonical_name"] for item in vocab.get("objects", [])}
        slots = {item["id"]: item["canonical_name"] for item in vocab.get("slots", [])}
    except Exception:
        objects = {
            "keys": "钥匙",
            "glasses_case": "眼镜盒",
            "medicine_box": "药盒",
            "remote_control": "遥控器",
            "insurance_card": "医保卡",
        }
        slots = {
            "entry.cabinet.top": "玄关柜",
            "living.coffee.top": "茶几垫",
            "dining.table.top": "餐桌垫",
        }
    _NAMES = {"objects": objects, "slots": slots}
    return _NAMES


def object_name(value):
    if not value:
        return "—"
    label = names()["objects"].get(value)
    return f"{label}" if label else str(value)


def slot_name(value):
    if not value:
        return "—"
    label = names()["slots"].get(value)
    return label if label else str(value)


def action_name(value):
    if not value:
        return "—"
    return ACTIONS.get(value, str(value))


def source_name(value):
    if not value:
        return "—"
    return SOURCES.get(value, str(value))


def db_by_id(db_id):
    for item in DATABASES:
        if item["id"] == db_id:
            return item
    return None


def open_ro(path):
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_count(conn, table):
    found = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if found is None:
        return None
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except sqlite3.Error:
        return None


def evidence_url(path_str):
    if not path_str:
        return None
    path = Path(str(path_str))
    if not path.is_file():
        return None
    resolved = path.resolve()
    for key, root in EVIDENCE_ROOTS.items():
        if not root.exists():
            continue
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"/media/evidence/{key}/{path.name}"
    return None


def list_evidence(root, key):
    if not root.exists():
        return []
    files = []
    for path in sorted(root.glob("*.jpg")):
        files.append(
            {
                "name": path.name,
                "url": f"/media/evidence/{key}/{path.name}",
                "bytes": path.stat().st_size,
                "size": human_size(path.stat().st_size),
            }
        )
    return files


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def catalog():
    deleted = deletion_index()
    libraries = []
    for spec in DATABASES:
        path = spec["path"]
        ready = path.exists()
        counts = {name: 0 for name in VISIBLE_TABLES}
        vector_rows = 0
        size = 0
        modified = None
        if ready:
            size = path.stat().st_size
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            conn = open_ro(path)
            for name in VISIBLE_TABLES:
                counts[name] = table_count(conn, name) or 0
            vector_rows = table_count(conn, "vec_events_rowids") or 0
            conn.close()
        evidence_key = "ref" if spec["id"] in {"memory", "roundtrip"} else "lab"
        libraries.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "blurb": spec["blurb"],
                "path": str(path),
                "relative": relative(path),
                "ready": ready,
                "bytes": size,
                "size": human_size(size) if ready else "文件不在",
                "modified": modified,
                "counts": {
                    "events": counts["events"],
                    "states": counts["object_states"],
                    "plans": counts["plans"],
                    "session": counts["session"],
                    "recall": counts["recall_log"],
                    "aliases": counts["alias_vec"],
                    "deleted": len(deleted.get(spec["id"], [])),
                    "vectors": vector_rows,
                },
                "evidence": list_evidence(EVIDENCE_ROOTS[evidence_key], evidence_key)
                if spec["id"] in {"memory", "lab"}
                else [],
            }
        )
    training = read_json(RESULTS / "training.json") or {}
    finetune = read_json(RESULTS / "embed_finetune.json") or {}
    summary = read_json(RESULTS / "summary.json") or {}
    return {
        "libraries": libraries,
        "training_verdict": "未达到训练级",
        "training_line": "嵌入微调和闭集检测都还是演示规模。生产仍用基础嵌入模型。打开「训练」看对照。",
        "generated": {
            "experiment": (summary.get("generated_at") if summary else None),
            "finetune": finetune.get("generated_at"),
            "closed_set": (training.get("generated_at") if training else None),
        },
    }


def cell(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def event_fields(row):
    keys = [
        "id", "ts", "object_id", "slot_id", "action", "source",
        "confidence", "cam_id", "evidence_path", "created_at",
    ]
    return pack_fields(row, keys)


def pack_fields(row, keys):
    fields = []
    for key in keys:
        if key not in row:
            continue
        raw = row[key]
        if isinstance(raw, bytes):
            shown = f"{len(raw)} 字节，页面不展开向量"
        elif key in {"object_id"}:
            shown = object_name(raw)
        elif key in {"slot_id", "visual_slot_id", "speech_slot_id"}:
            shown = slot_name(raw)
        elif key in {"action", "visual_action", "speech_action"}:
            shown = action_name(raw)
        elif key == "source":
            shown = source_name(raw)
        elif key in {"ts", "created_at", "updated_at", "visual_ts", "speech_ts", "due_ts", "confirm_ts"}:
            shown = speak_time(raw)
        elif key == "done":
            shown = "已确认" if raw else "未完成"
        elif key == "confidence" or key == "visual_confidence":
            shown = cell(raw)
        else:
            shown = cell(raw)
        fields.append({"label": FIELD_LABELS.get(key, key), "value": shown, "raw": cell(raw)})
    return fields


def event_sentence(row):
    slot = slot_name(row.get("slot_id")) if row.get("slot_id") else ""
    parts = [
        speak_time(row.get("ts")),
        source_name(row.get("source")),
        action_name(row.get("action")),
        object_name(row.get("object_id")),
    ]
    if slot and slot != "—":
        parts.append(slot)
    return "，".join(parts) + "。"


def row_payload(sentence, fields, evidence=None, stamp=None):
    return {
        "sentence": sentence,
        "fields": fields,
        "evidence": evidence,
        "stamp": stamp,
    }


def fetch_table(db_id, kind):
    spec = db_by_id(db_id)
    if spec is None:
        return None
    if kind == "deleted":
        return deleted_table(db_id)
    if not spec["path"].exists():
        return {"note": "这只库的文件不在磁盘上。", "rows": [], "count": 0}
    conn = open_ro(spec["path"])
    try:
        if kind == "events":
            rows = [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY id")]
            payload = [
                row_payload(event_sentence(row), event_fields(row), evidence_url(row.get("evidence_path")))
                for row in rows
            ]
            note = "事件只追加。一行就是一次放下、拿走、挡住或冲突。旧行不会被改写。"
        elif kind == "states":
            rows = [dict(row) for row in conn.execute("SELECT * FROM object_states ORDER BY object_id")]
            payload = []
            for row in rows:
                visual = f"视觉：{action_name(row.get('visual_action'))} · {slot_name(row.get('visual_slot_id'))}"
                speech = row.get("speech_action")
                if speech:
                    spoken = f"口述：{action_name(speech)}"
                    if row.get("speech_note"):
                        spoken += f"，原话「{row['speech_note']}」"
                else:
                    spoken = "口述：没有"
                payload.append(
                    row_payload(
                        f"{object_name(row.get('object_id'))}。{visual}。{spoken}。",
                        pack_fields(
                            row,
                            [
                                "object_id", "visual_slot_id", "visual_action", "visual_ts",
                                "visual_confidence", "evidence_path", "speech_slot_id",
                                "speech_action", "speech_ts", "speech_note", "updated_at",
                            ],
                        ),
                        evidence_url(row.get("evidence_path")),
                    )
                )
            note = "每件物品一行，是现在的状态。视觉列和口述列分开，口述不会盖掉视觉。"
        elif kind == "plans":
            rows = [dict(row) for row in conn.execute("SELECT * FROM plans ORDER BY id")]
            payload = []
            for row in rows:
                state = "已由本人确认" if row.get("done") else "还没确认"
                payload.append(
                    row_payload(
                        f"{row.get('title') or '未命名'}，{speak_time(row.get('due_ts'))}，{state}。",
                        pack_fields(row, ["id", "title", "due_ts", "done", "done_source", "confirm_ts"]),
                    )
                )
            note = "到点提醒。看见药盒不会自动变成已经吃过。确认来源只有 self_report。"
        elif kind == "recall":
            if table_count(conn, "recall_log") is None:
                return {"note": "这只库没有输出日志表。", "rows": [], "count": 0}
            rows = [dict(row) for row in conn.execute("SELECT * FROM recall_log ORDER BY id")]
            payload = []
            for row in rows:
                payload.append(
                    row_payload(
                        f"问「{row.get('question')}」，走{row.get('path') or '—'}，物品是{object_name(row.get('object_id'))}。",
                        pack_fields(row, ["id", "ts", "question", "path", "object_id"]),
                    )
                )
            note = "输出记的是问句和路径，不是模型生成的一段话。回答本身由模板填出来，不另存一列。"
        elif kind == "session":
            rows = [dict(row) for row in conn.execute("SELECT * FROM session ORDER BY key")]
            payload = [
                row_payload(
                    f"{row.get('key')} = {object_name(row.get('value')) if row.get('key') == 'last_object' else row.get('value')}",
                    pack_fields(row, ["key", "value"]),
                )
                for row in rows
            ]
            note = "session 只记上一件问过的物品，给「它在哪」用。它不是放置记录。"
        elif kind == "aliases":
            if not table_count(conn, "alias_vec"):
                return {"note": "这只库里没有别名向量，或行数是 0。", "rows": [], "count": 0}
            rows = [dict(row) for row in conn.execute(
                "SELECT alias, object_id, embedding, updated_at FROM alias_vec ORDER BY object_id, alias"
            )]
            payload = []
            for row in rows:
                blob = row.get("embedding")
                length = len(blob) if isinstance(blob, bytes) else 0
                packed = pack_fields(
                    {
                        "alias": row.get("alias"),
                        "object_id": row.get("object_id"),
                        "embedding": blob,
                        "updated_at": row.get("updated_at"),
                    },
                    ["alias", "object_id", "embedding", "updated_at"],
                )
                for field in packed:
                    if field["label"] == "向量":
                        field["value"] = f"{length} 字节" if length else "—"
                payload.append(
                    row_payload(
                        f"别名「{row.get('alias')}」指向{object_name(row.get('object_id'))}。",
                        packed,
                    )
                )
            note = "别名向量给问句召回用。页面只显示指向哪件物品，不展开浮点数组。"
        else:
            return None
    finally:
        conn.close()
    return {"note": note, "rows": payload, "count": len(payload)}


def deletion_index():
    global _DELETIONS
    path = RESULTS / "trace.json"
    if not path.exists():
        return {}
    stamp = path.stat().st_mtime_ns
    if _DELETIONS and _DELETIONS.get("stamp") == stamp:
        return _DELETIONS["by_db"]
    trace = json.loads(path.read_text(encoding="utf-8"))
    last = {}
    last_main = None
    by_db = {}
    for item in trace:
        label = item.get("database") or ""
        snap = item.get("db")
        if not isinstance(snap, dict):
            continue
        events = snap.get("events") or []
        db_id = LABEL_TO_ID.get(label)
        previous = last.get(label)
        if previous and db_id:
            prev_ids = {row.get("id"): row for row in previous}
            now_ids = {row.get("id") for row in events}
            gone = [prev_ids[key] for key in prev_ids if key not in now_ids]
            if gone:
                by_db.setdefault(db_id, []).extend(
                    stamped(gone, item, "这一步的快照里，这些行从这只库消失了。")
                )
        last[label] = events
        if label == "主库":
            last_main = events
        if label == "清除后的副本" and not events and last_main:
            by_db.setdefault("deleted", []).extend(
                stamped(
                    last_main,
                    item,
                    "这些行先被复制进副本，再被整库清除。实验主库里的同一批行还在。",
                )
            )
    _DELETIONS = {"stamp": stamp, "by_db": by_db}
    return by_db


def stamped(rows, item, reason):
    stamped_rows = []
    for row in rows:
        stamped_rows.append(
            {
                "row": row,
                "step": item.get("id") or "",
                "title": item.get("title") or "",
                "reason": reason,
            }
        )
    return stamped_rows


def deleted_table(db_id):
    rows = deletion_index().get(db_id, [])
    payload = []
    for item in rows:
        row = item["row"]
        fields = event_fields(row)
        fields.append({"label": "删除发生在", "value": f"{item['step']} {item['title']}", "raw": item["step"]})
        payload.append(
            row_payload(
                event_sentence(row),
                fields,
                evidence_url(row.get("evidence_path")),
                stamp="已删除",
            )
        )
    notes = {
        "deleted": "文件里的事件表现在是空的。下面这些行来自清除前的快照，不是现在还能查到的记录。",
        "contract": "契约库在清除前有事件，清除后文件里是 0 行。超过 72 小时的那张 JPEG 是临时文件 old.jpg，内容是三个字节，不是现场照片，事件行没有因为这张图被删。",
        "memory": "正式记忆库没有整库清除记录。事件只追加，要等清除命令才会整表删掉。",
        "lab": "实验主库没有被清除。被清掉的是旁边那只副本。",
    }
    note = notes.get(db_id, "这只库的实验快照里没有消失的事件行。")
    if not payload and db_id == "deleted":
        note = "还没有从实验轨迹里还原出被清除的行。请先确认 实验\\results\\trace.json 还在。"
    return {"note": note, "rows": payload, "count": len(payload)}


def model_cards():
    cards = [
        ("bge-small-zh-v1.5", "生产用的嵌入模型", "问答的向量召回加载这一只。"),
        ("bge-small-zh-ft", "微调留下的模型", "没有换成生产模型。"),
        ("asr-paraformer-zh-en", "默认识别", "按键说话用的本地识别。"),
        ("asr-zipformer-zh-14m", "试过的识别", "合成语音回环更差，没有作为默认。"),
        ("tts-melo-zh_en", "朗读", "只朗读模板句。"),
    ]
    found = []
    for folder, role, note in cards:
        path = MODELS / folder
        size = 0
        if path.exists():
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        found.append(
            {
                "name": folder,
                "role": role,
                "note": note,
                "path": relative(path),
                "ready": path.exists(),
                "size": human_size(size) if path.exists() else "不在",
            }
        )
    for name in ("yolo26n.pt", "yoloe-26n-seg.pt", "mobileclip2_b.ts"):
        path = ROOT / "参考实现" / name
        found.append(
            {
                "name": name,
                "role": "检测权重" if name.endswith(".pt") else "文本编码器",
                "note": "闭集对照用 yolo26n。开放词汇还要 mobileclip，这次没有配齐。",
                "path": relative(path),
                "ready": path.exists(),
                "size": human_size(path.stat().st_size) if path.exists() else "不在",
            }
        )
    return found


def corpus_preview():
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    try:
        import sys

        code = ROOT / "参考实现"
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))
        import finetune_embed

        pairs = finetune_embed.build_corpus()
        alias_pairs = sum(1 for _, positive in pairs if "，" not in positive)
        _CORPUS = {
            "total": len(pairs),
            "unique_questions": len({question for question, _ in pairs}),
            "alias_pairs": alias_pairs,
            "sentence_pairs": len(pairs) - alias_pairs,
            "pairs": [{"question": question, "positive": positive} for question, positive in pairs[:24]],
            "error": None,
        }
    except Exception as exc:
        saved = read_json(RESULTS / "embed_finetune.json") or {}
        _CORPUS = {
            "total": saved.get("corpus_pairs") or 0,
            "unique_questions": None,
            "alias_pairs": None,
            "sentence_pairs": None,
            "pairs": [],
            "error": f"语料没有当场展开：{exc}",
        }
    return _CORPUS


def training_review():
    finetune = read_json(RESULTS / "embed_finetune.json") or {}
    closed = read_json(RESULTS / "training.json") or {}
    summary = read_json(RESULTS / "summary.json") or {}
    results = finetune.get("results") or {}
    base = results.get("base") or {}
    tuned = results.get("finetuned") or {}
    base_acc = (base.get("acceptance") or {})
    tuned_acc = (tuned.get("acceptance") or {})
    base_hold = (base.get("holdout") or {})
    tuned_hold = (tuned.get("holdout") or {})
    closed_set = closed.get("closed_set") or {}
    dataset = closed.get("dataset") or {}
    counts = dataset.get("counts") or {}
    note = ((summary.get("metrics") or {}).get("h2_finetune") or {}).get("note") or finetune.get("note") or ""
    diffs = []
    for split in ("acceptance", "holdout"):
        base_rows = {(row.get("question")): row for row in ((base.get(split) or {}).get("rows") or [])}
        for row in ((tuned.get(split) or {}).get("rows") or []):
            other = base_rows.get(row.get("question"))
            if not other:
                continue
            if other.get("verdict") != row.get("verdict") or other.get("resolved") != row.get("resolved"):
                diffs.append(
                    {
                        "split": "验收集" if split == "acceptance" else "留出集",
                        "question": row.get("question"),
                        "expected": object_name(row.get("expected")) if row.get("expected") else "不应命中",
                        "base": other.get("verdict"),
                        "tuned": row.get("verdict"),
                        "base_sim": other.get("top_sim"),
                        "tuned_sim": row.get("top_sim"),
                    }
                )
    images = real_gallery()
    if not images:
        images = []
        samples = [
            ("train", "b0_00_c0.jpg", "旧的合成训练集：椭圆被当成钥匙"),
            ("train", "b0_01_c1.jpg", "旧的合成训练集：两个圆被当成眼镜盒"),
            ("train", "b0_02_c2.jpg", "旧的合成训练集：方块被当成药盒"),
        ]
        for split, filename, caption in samples:
            path = CLOSED_ROOT / split / filename
            if path.exists():
                images.append({"url": f"/media/closed/{split}/{filename}", "caption": caption})
    real = dataset.get("kind") in {"lvis", "commons", "camera"}
    recipe = finetune.get("recipe") or {}
    corpus = corpus_preview()
    retrieval = finetune.get("retrieval") or {}
    base_ir = retrieval.get("base") or {}
    tuned_ir = retrieval.get("finetuned") or {}
    promoted = bool(recipe.get("promoted"))
    has_recipe = bool(recipe)

    def ir_text(block):
        if not block:
            return "还没有检索分数"
        return (
            f"nDCG@10 {block.get('ndcg@10', '—')}，"
            f"MRR@10 {block.get('mrr@10', '—')}，"
            f"Recall@10 {block.get('recall@10', '—')}"
        )

    if has_recipe and promoted:
        verdict = "检索有提升，已换上微调权重"
        lead = "嵌入模型按留出集 nDCG 重训过，并且生产配置改成了微调权重。语料仍是词表模板，不是真人问句。"
    elif has_recipe and real:
        verdict = "训完了，生产仍用基础模型"
        lead = (recipe.get("decision") or "嵌入微调没有换上生产模型。") + " 检测照片把实物放在更远的台面上，并加了轻微模糊。"
    elif has_recipe:
        verdict = "训完了，生产仍用基础模型"
        lead = recipe.get("decision") or "这一轮按检索微调做完了。留出 nDCG 没有拉开足够差距，问答仍加载基础模型。"
    else:
        verdict = "未达到训练级"
        lead = "嵌入微调还没有按检索指标重训。记忆库里的仿真验收不能写成模型已经训练好。"
    corpus = corpus_preview()
    map50 = closed_set.get("map50")
    map95 = closed_set.get("map50_95")
    return {
        "verdict": verdict,
        "lead": lead,
        "kept": [
            "验收集和留出问句没有写进训练问句。",
            "这一轮没有更换生产权重。要换，留出 nDCG@10 得至少高出 0.02，而且验收集和留出问句的正确数不能下降。"
            if has_recipe and not promoted
            else (
                "生产配置已改为微调权重。"
                if promoted
                else "上一轮没有把微调权重换成生产模型。"
            ),
            "闭集按图片切开，一张图不会同时出现在训练和验证。" if real else "闭集按生成批次切开，没有把同一批图随机拆进训练和验证。",
            "合成图上的 mAP 没有被写成这户台面找钥匙的准确率。",
        ],
        "embed": {
            "title": "嵌入微调",
            "status": "已换上" if promoted else ("训完未更换" if has_recipe else "未达到"),
            "facts": [
                {"label": "训练条数", "value": f"{finetune.get('corpus_pairs', corpus['total'])} 条模板问句"},
                {"label": "做法", "value": (
                    f"难负样本 {recipe.get('hard_negatives', '—')} 个，温度 {recipe.get('temperature', '—')}，"
                    f"{recipe.get('epochs', '—')} 轮，按 {recipe.get('selection', '—')}"
                ) if has_recipe else "只有批内负样本，训完直接保存最后一轮"},
                {"label": "用时", "value": f"{finetune.get('train_seconds', '—')} 秒，CPU"},
                {"label": "留出检索 · 基础", "value": ir_text(base_ir)},
                {"label": "留出检索 · 微调", "value": ir_text(tuned_ir)},
                {"label": "验收集", "value": f"基础 {base_acc.get('correct', '—')}/{base_acc.get('total', '—')}，微调 {tuned_acc.get('correct', '—')}/{tuned_acc.get('total', '—')}"},
                {"label": "留出问句", "value": f"基础 {base_hold.get('correct', '—')}/{base_hold.get('total', '—')}，微调 {tuned_hold.get('correct', '—')}/{tuned_hold.get('total', '—')}"},
                {"label": "生产模型", "value": recipe.get("decision") or note or "—"},
            ],
            "checks": [
                {
                    "name": "语料从哪来",
                    "have": f"{finetune.get('corpus_pairs', '—')} 条由词表别名和事件字段模板生成，没有真人问句。",
                    "bar": "领域对比学习通常从大约一千对到数万对真实问句。BGE 论文里的中文标注微调混合是 38.66 万对。",
                    "met": False,
                    "source": "BGE-M3 论文表 1",
                },
                {
                    "name": "负样本",
                    "have": f"每条问句 {recipe.get('hard_negatives', '—')} 个其他物品的文本，batch {recipe.get('batch_size', '—')}，采样 {recipe.get('batch_sampler', '—')}，温度 {recipe.get('temperature', '—')}。" if has_recipe else "只有批内负样本。",
                    "bar": "正样本之外要有难负样本，同一批避免重复句子，温度大约 0.01 到 0.1。",
                    "met": has_recipe,
                    "source": "FlagEmbedding 微调说明；sentence-transformers MultipleNegativesRankingLoss",
                },
                {
                    "name": "怎么打分",
                    "have": "留出问句对事件句和别名做 nDCG@10、MRR@10、Recall@10，并按 nDCG@10 留最好的一轮。" if has_recipe else "只看 20 条和 12 条有没有判对物品。",
                    "bar": "用 InformationRetrievalEvaluator 的 nDCG@10 在留出集上选检查点。",
                    "met": has_recipe,
                    "source": "sentence-transformers InformationRetrievalEvaluator",
                },
                {
                    "name": "有没有拉开基础模型",
                    "have": recipe.get("decision") or "还没有这一轮的对照。",
                    "bar": "nDCG@10 至少高出 0.02，并且验收集、留出集的正确数不下降、说错物品不增加，才更换生产权重。",
                    "met": promoted,
                    "source": "实验/results/embed_finetune.json",
                },
                {
                    "name": "留出是否干净",
                    "have": "训练问句排除了验收集的桥接词，也排除了留出问句本身。事件句会同时出现在训练正样本和检索文档里，因为上线时检索的就是这些句子。",
                    "bar": "评估问句不能出现在训练里。文档是生产索引本身，不另造一套。",
                    "met": True,
                    "source": "实验/finetune_embed.py 的 assert_no_leakage",
                },
            ],
        },
        "detect": {
            "title": "闭集检测",
            "status": "真实照片，规模不够" if real else "未达到",
            "facts": [
                {"label": "图像", "value": f"训练 {counts.get('train', '—')} 张，验证 {counts.get('val', '—')} 张，测试 0 张"},
                {"label": "内容", "value": dataset.get("what") or "程序画的几何图形"},
                {"label": "来源", "value": dataset.get("source") or "程序生成"},
                {"label": "mAP50", "value": f"{map50:.3f}" if isinstance(map50, float) else "—"},
                {"label": "mAP50-95", "value": f"{map95:.3f}" if isinstance(map95, float) else "—"},
                {"label": "设备", "value": f"{closed_set.get('device', '—')}，{closed_set.get('epochs', '—')} 个周期"},
            ],
            "checks": [
                {
                    "name": "图是什么",
                    "have": dataset.get("what") or "椭圆、两个圆、一个方块。",
                    "bar": "训练图应该是要检测的那件东西的真实照片，类别名和画面一致。",
                    "met": real,
                    "source": dataset.get("source") or "实验/data/closed_set",
                },
                {
                    "name": "数量",
                    "have": f"训练 {counts.get('train', '—')} 张，验证 {counts.get('val', '—')} 张。",
                    "bar": "Ultralytics 写明：试水大约每类几百个实例；要稳的真实场景，建议每类至少 1500 张图、10000 个标注实例。",
                    "met": False,
                    "source": "https://docs.ultralytics.com/guides/data-collection-and-annotation",
                },
                {
                    "name": "切分",
                    "have": dataset.get("split") or "按生成批次切开。没有测试集。",
                    "bar": "按图片切开，一张图不能同时进训练和验证。验证用来挑选，测试只看一次。",
                    "met": real,
                    "source": "Ultralytics Academy, Split the Dataset Correctly",
                },
                {
                    "name": "分数能不能外推",
                    "have": f"验证集 mAP50 {map50 if map50 is not None else '—'}。这是这批下载图片上的分数。",
                    "bar": "不能写成这户台面上找钥匙的准确率。这些照片来自网上，不是这户垫子上拍的。",
                    "met": False,
                    "source": "本项目 results/training.json",
                },
            ],
        },
        "diffs": diffs,
        "corpus": corpus,
        "samples": finetune.get("sample_rows") or [],
        "images": images,
        "models": model_cards(),
        "record_gap": (
            "这一轮的学习率 2e-5、warmup 10%、温度 0.05 和选模规则写在 results/embed_finetune.json。自动模型卡若仍显示另一套学习率，以这份 json 为准。"
            if has_recipe
            else "代码里的学习率是 2e-5，并设了大约 10% 的 warmup。微调目录里的自动模型卡写成学习率 5e-05、warmup 0。两份记录对不上，不能拿那张卡当作完整训练日志。"
        ),
    }


def real_gallery(limit_per_class=8):
    manifest_path = REAL_ROOT / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen = {}
    gallery = []
    for item in manifest.get("files") or []:
        if item.get("split") != "train":
            continue
        class_id = (item.get("class_ids") or ["other"])[0]
        if seen.get(class_id, 0) >= limit_per_class:
            continue
        seen[class_id] = seen.get(class_id, 0) + 1
        label = "、".join(item.get("classes") or [])
        gallery.append(
            {
                "url": f"/media/real/{item['split']}/{item['file']}",
                "caption": f"{label} · 训练照片",
            }
        )
    return gallery


def portrait(db_id):
    """系统对这户物品和问法的现有了解。不是身份档案。"""
    spec = db_by_id(db_id)
    if spec is None or not spec["path"].exists():
        return None
    conn = open_ro(spec["path"])
    try:
        states = [dict(row) for row in conn.execute("SELECT * FROM object_states ORDER BY object_id")]
        recalls = []
        if table_count(conn, "recall_log"):
            recalls = [dict(row) for row in conn.execute("SELECT * FROM recall_log ORDER BY id")]
        plans = [dict(row) for row in conn.execute("SELECT * FROM plans ORDER BY id")]
    finally:
        conn.close()
    objects = []
    for state in states:
        object_id = state.get("object_id")
        asked = [row for row in recalls if row.get("object_id") == object_id]
        speech = state.get("speech_action")
        if speech:
            spoken = f"{action_name(speech)} · {slot_name(state.get('speech_slot_id'))}"
            if state.get("speech_note"):
                spoken += f" · {state['speech_note']}"
        else:
            spoken = "没有口述"
        objects.append(
            {
                "id": object_id,
                "name": object_name(object_id),
                "photo": evidence_url(state.get("evidence_path")),
                "fields": [
                    {"label": "物品", "value": object_name(object_id), "raw": object_id},
                    {"label": "视觉动作", "value": action_name(state.get("visual_action")), "raw": state.get("visual_action") or "—"},
                    {"label": "视觉地点", "value": slot_name(state.get("visual_slot_id")), "raw": state.get("visual_slot_id") or "—"},
                    {"label": "视觉时间", "value": speak_time(state.get("visual_ts")), "raw": state.get("visual_ts") or "—"},
                    {"label": "置信度", "value": cell(state.get("visual_confidence")), "raw": cell(state.get("visual_confidence"))},
                    {"label": "口述", "value": spoken, "raw": state.get("speech_note") or "—"},
                    {"label": "问到这件", "value": f"{len(asked)} 次", "raw": str(len(asked))},
                ],
                "questions": [
                    {
                        "text": row.get("question"),
                        "path": row.get("path"),
                        "time": speak_time(row.get("ts")),
                    }
                    for row in asked
                ],
            }
        )
    path_counts = {}
    for row in recalls:
        path_counts[row.get("path") or "—"] = path_counts.get(row.get("path") or "—", 0) + 1
    path_text = "、".join(f"{key} {value} 次" for key, value in path_counts.items()) or "还没有问过"
    known = "、".join(item["name"] for item in objects) or "还没有物品"
    return {
        "title": spec["title"],
        "path": str(spec["path"]),
        "summary": f"这只库记得 {known}。问句 {len(recalls)} 条，其中 {path_text}。系统记的是物品和问法，不是这个人的身份。",
        "objects": objects,
        "questions": [
            {
                "text": row.get("question"),
                "path": row.get("path"),
                "object": object_name(row.get("object_id")),
                "time": speak_time(row.get("ts")),
            }
            for row in recalls
        ],
        "plans": [
            {
                "title": row.get("title"),
                "due": speak_time(row.get("due_ts")),
                "done": "已由本人确认" if row.get("done") else "还没确认",
                "source": row.get("done_source") or "—",
            }
            for row in plans
        ],
    }


def media_path(kind, name):
    if kind == "evidence":
        root_key, filename = name.split("/", 1) if "/" in name else ("", name)
        root = EVIDENCE_ROOTS.get(root_key)
        if root is None or not _safe_name(filename) or not filename.lower().endswith(".jpg"):
            return None
        path = (root / filename).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None
    if kind in {"closed", "real"}:
        split, filename = name.split("/", 1) if "/" in name else ("", name)
        root = CLOSED_ROOT if kind == "closed" else (REAL_ROOT / "images")
        if split not in {"train", "val"} or not _safe_name(filename) or not filename.lower().endswith(".jpg"):
            return None
        path = (root / split / filename).resolve()
        try:
            path.relative_to((root / split).resolve())
        except ValueError:
            return None
        return path if path.is_file() else None
    return None


def _safe_name(filename):
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        return False
    return all(ch.isalnum() or ch in "._-" for ch in filename)
