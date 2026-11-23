"""SQLite 记忆库。事件只追加。视觉列和口述列分开写。"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  object_id TEXT NOT NULL,
  slot_id TEXT NOT NULL,
  action TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL,
  cam_id TEXT,
  evidence_path TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS object_states (
  object_id TEXT PRIMARY KEY,
  visual_slot_id TEXT,
  visual_action TEXT,
  visual_ts TEXT,
  visual_confidence REAL,
  evidence_path TEXT,
  speech_slot_id TEXT,
  speech_action TEXT,
  speech_ts TEXT,
  speech_note TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  due_ts TEXT,
  done INTEGER NOT NULL DEFAULT 0,
  done_source TEXT,
  confirm_ts TEXT
);
CREATE TABLE IF NOT EXISTS session (
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS recall_log (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  question TEXT NOT NULL,
  path TEXT NOT NULL,
  object_id TEXT
);
"""

STATES_SQL = """
CREATE TABLE object_states (
  object_id TEXT PRIMARY KEY,
  visual_slot_id TEXT,
  visual_action TEXT,
  visual_ts TEXT,
  visual_confidence REAL,
  evidence_path TEXT,
  speech_slot_id TEXT,
  speech_action TEXT,
  speech_ts TEXT,
  speech_note TEXT,
  updated_at TEXT NOT NULL
)
"""

ACTIONS = {"placed", "removed", "occluded", "conflict"}
SOURCES = {"visual", "user_speech"}


def _columns(conn, table):
    found = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if found is None:
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _cell(row, key):
    if key not in row.keys() or row[key] is None:
        return None
    return row[key]


def _migrate(conn):
    event_cols = _columns(conn, "events")
    if event_cols and "source" not in event_cols:
        conn.execute("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'visual'")

    state_cols = _columns(conn, "object_states")
    if state_cols and "status" in state_cols:
        legacy_rows = list(conn.execute("SELECT * FROM object_states"))
        conn.execute("DROP TABLE IF EXISTS object_states_legacy")
        conn.execute("ALTER TABLE object_states RENAME TO object_states_legacy")
        conn.execute(STATES_SQL)
        for row in legacy_rows:
            visual_action = _cell(row, "visual_action") or _cell(row, "status")
            visual_slot = _cell(row, "visual_slot_id")
            if visual_slot is None:
                visual_slot = _cell(row, "slot_id")
            visual_ts = _cell(row, "visual_ts")
            if visual_ts is None:
                visual_ts = _cell(row, "last_seen_ts")
            visual_confidence = _cell(row, "visual_confidence")
            if visual_confidence is None:
                visual_confidence = _cell(row, "confidence")
            updated = _cell(row, "updated_at") or visual_ts or ""
            conn.execute(
                """INSERT INTO object_states
                   (object_id, visual_slot_id, visual_action, visual_ts, visual_confidence,
                    evidence_path, speech_slot_id, speech_action, speech_ts, speech_note, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["object_id"],
                    visual_slot,
                    visual_action,
                    visual_ts,
                    visual_confidence,
                    _cell(row, "evidence_path") or "",
                    _cell(row, "speech_slot_id"),
                    _cell(row, "speech_action"),
                    _cell(row, "speech_ts"),
                    _cell(row, "speech_note"),
                    updated,
                ),
            )
        conn.execute("DROP TABLE object_states_legacy")
    conn.commit()


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def apply_event(
    conn,
    ts,
    object_id,
    slot_id,
    action,
    confidence,
    cam_id,
    evidence_path,
    source="visual",
    speech_note="",
):
    if action not in ACTIONS:
        raise ValueError(action)
    if source not in SOURCES:
        raise ValueError(source)
    if slot_id is None:
        slot_id = ""
    cur = conn.execute(
        """INSERT INTO events
           (ts, object_id, slot_id, action, source, confidence, cam_id, evidence_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, object_id, slot_id, action, source, confidence, cam_id, evidence_path or "", ts),
    )
    event_id = cur.lastrowid
    conn.execute(
        """INSERT INTO object_states (object_id, updated_at) VALUES (?, ?)
           ON CONFLICT(object_id) DO NOTHING""",
        (object_id, ts),
    )
    if source == "visual":
        if evidence_path:
            conn.execute(
                """UPDATE object_states SET
                     visual_slot_id = ?, visual_action = ?, visual_ts = ?,
                     visual_confidence = ?, evidence_path = ?, updated_at = ?
                   WHERE object_id = ?""",
                (slot_id, action, ts, confidence, evidence_path, ts, object_id),
            )
        else:
            conn.execute(
                """UPDATE object_states SET
                     visual_slot_id = ?, visual_action = ?, visual_ts = ?,
                     visual_confidence = ?, updated_at = ?
                   WHERE object_id = ?""",
                (slot_id, action, ts, confidence, ts, object_id),
            )
    else:
        conn.execute(
            """UPDATE object_states SET
                 speech_slot_id = ?, speech_action = ?, speech_ts = ?,
                 speech_note = ?, updated_at = ?
               WHERE object_id = ?""",
            (slot_id, action, ts, speech_note or "", ts, object_id),
        )
    conn.commit()
    _index_vector(conn, event_id, ts, object_id, slot_id, action, source)


_VEC_WARNED = {"warned": False}


def _index_vector(conn, event_id, ts, object_id, slot_id, action, source):
    """插入成功后把规范句向量写进 vec_events（节点 H2）。
    嵌入模型或 sqlite-vec 不可用时只警告一次，事件行照常保留。"""
    try:
        from vec_memory import index_event_row

        index_event_row(conn, event_id, ts, object_id, slot_id, action, source)
    except Exception as exc:
        if not _VEC_WARNED["warned"]:
            _VEC_WARNED["warned"] = True
            print(f"向量索引未写入（{exc}）。问答仍可用，但召回降级为精确匹配。")



def recent_slots(conn, object_id, source="visual"):
    rows = conn.execute(
        """SELECT slot_id FROM events
           WHERE object_id = ? AND source = ? AND slot_id != ''
           ORDER BY id""",
        (object_id, source),
    ).fetchall()
    return [row["slot_id"] for row in rows]


def snapshot(conn):
    def rows(sql):
        return [dict(row) for row in conn.execute(sql)]

    return {
        "events": rows(
            """SELECT id, ts, object_id, slot_id, action, source, confidence,
                      cam_id, evidence_path, created_at
               FROM events ORDER BY id"""
        ),
        "object_states": rows(
            """SELECT object_id, visual_slot_id, visual_action, visual_ts, visual_confidence,
                      evidence_path, speech_slot_id, speech_action, speech_ts, speech_note, updated_at
               FROM object_states ORDER BY object_id"""
        ),
        "plans": rows(
            "SELECT id, title, due_ts, done, done_source, confirm_ts FROM plans ORDER BY id"
        ),
        "session": rows("SELECT key, value FROM session ORDER BY key"),
    }


def get_value(conn, key):
    row = conn.execute("SELECT value FROM session WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_value(conn, key, value):
    conn.execute(
        """INSERT INTO session (key, value) VALUES (?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        (key, value),
    )
    conn.commit()


def get_state(conn, object_id):
    row = conn.execute(
        "SELECT * FROM object_states WHERE object_id = ?", (object_id,)
    ).fetchone()
    return dict(row) if row else None


def log_recall(conn, question, path, object_id=None, ts=None):
    """节点 H2 的路径日志：精确命中还是向量召回。验收时两种路径分开计数。"""
    ts = ts or datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO recall_log (ts, question, path, object_id) VALUES (?, ?, ?, ?)",
        (ts, question, path, object_id),
    )
    conn.commit()


def add_plan(conn, title, due_ts):
    cur = conn.execute(
        "INSERT INTO plans (title, due_ts) VALUES (?, ?)", (title, due_ts)
    )
    conn.commit()
    return cur.lastrowid


def confirm_plan(conn, plan_id, ts):
    conn.execute(
        """UPDATE plans
           SET done = 1, done_source = 'self_report', confirm_ts = ?
           WHERE id = ?""",
        (ts, plan_id),
    )
    conn.commit()


def list_plans(conn):
    rows = conn.execute("SELECT * FROM plans ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def due_plans(conn, now_ts):
    """到点且尚未口述确认的计划。时间戳用同一套 ISO 字符串比较。"""
    rows = conn.execute(
        """SELECT * FROM plans
           WHERE done = 0 AND due_ts IS NOT NULL AND due_ts <= ?
           ORDER BY due_ts, id""",
        (now_ts,),
    ).fetchall()
    return [dict(row) for row in rows]


def delete_all(conn, evidence_dir):
    conn.execute("DELETE FROM events")
    conn.execute("DELETE FROM object_states")
    conn.execute("DELETE FROM plans")
    conn.execute("DELETE FROM session")
    conn.execute("DELETE FROM recall_log")
    conn.commit()
    # 清除时对应向量一起删（节点 H2）。sqlite-vec 不可用时静默跳过。
    try:
        from vec_memory import clear_vectors

        clear_vectors(conn)
    except Exception:
        pass
    if evidence_dir is None:
        return
    for path in Path(evidence_dir).glob("*.jpg"):
        path.unlink()


def purge_evidence(evidence_dir, hours=72, now=None):
    now = now or datetime.now()
    removed = 0
    root = Path(evidence_dir)
    if not root.exists():
        return 0
    for path in root.glob("*.jpg"):
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified < now - timedelta(hours=hours):
            path.unlink()
            removed += 1
    return removed
