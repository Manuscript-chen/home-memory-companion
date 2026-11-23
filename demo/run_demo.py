"""不接摄像头，检查配置、状态机、触发、数据库和问答。"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
if str(_repo / "common") not in sys.path:
    sys.path.insert(0, str(_repo / "common"))
import nodepath  # noqa: E402
from paths import REPO_ROOT as ROOT  # noqa: E402

from answer import answer, find_object  # noqa: E402
from check_config import errors_in_config  # noqa: E402
from memory import (  # noqa: E402
    add_plan,
    apply_event,
    confirm_plan,
    connect,
    delete_all,
    due_plans,
    get_state,
    get_value,
    purge_evidence,
    set_value,
)
from pixel_budget import evaluate  # noqa: E402
from state_machine import Track, judge, update_tracks  # noqa: E402
from trigger import Trigger  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

W, H = 640, 480
TS = "2026-11-17T08:12:00"


def samples(start, end, x_at, y=240.0, dt=0.2):
    points = []
    t = start
    while t <= end + 1e-9:
        x = x_at(t) if callable(x_at) else x_at
        points.append((round(t, 3), x, y))
        t += dt
    return points


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def test_pixels():
    bad = evaluate(2560, 640, 0.06, 1.5, 70)
    expect(not bad["pass"] and abs(bad["effective_px"] - 18.3) < 0.2, "原稿条件应判不合格")
    keys = evaluate(2560, 1280, 0.02, 0.6, 60)
    expect(keys["pass"] and abs(keys["effective_px"] - 37.0) < 0.2, "钥匙机位应约 37 px")
    large = evaluate(2560, 640, 0.05, 0.8, 60)
    expect(large["pass"] and abs(large["effective_px"] - 34.6) < 0.2, "大件机位应约 35 px")
    wrong_cam = evaluate(2560, 640, 0.02, 0.8, 60)
    expect(not wrong_cam["pass"], "钥匙不能放在大件机位")


def test_states():
    placed = judge(samples(0, 3.0, 320), 3.0, W, H)
    expect(placed == "placed", f"停稳应是放下，实际 {placed}")
    short = judge(samples(0, 1.5, 320), 1.5, W, H)
    expect(short == "unknown", "1.5 秒不能写成放下")
    jitter = samples(0, 3.0, lambda t: 320 if int(t * 5) % 2 == 0 else 323)
    expect(judge(jitter, 3.0, W, H) == "placed", "3 px 抖动仍应放下")

    moving = samples(0, 1.0, lambda t: 200 + 300 * t)
    expect(judge(moving, 3.0, W, H) == "removed", "拿走后应是 removed")
    covered = samples(0, 2.0, 320)
    expect(judge(covered, 4.0, W, H) == "occluded", "静止后消失应是挡住")
    ambiguous = samples(0, 1.0, lambda t: 200 + 64 * t)
    expect(judge(ambiguous, 3.0, W, H) == "unknown", "中间速度不能硬判")
    border = samples(0, 1.0, 10)
    expect(judge(border, 3.0, W, H) == "removed", "进入边缘应是拿走")

    hole = samples(0, 0.4, 320) + samples(2.0, 3.0, 320)
    expect(judge(hole, 3.0, W, H) == "unknown", "轨迹中断不能写成放下")


def test_emit_once():
    tracks = []
    emitted = []
    t = 0.0
    while t <= 4.0 + 1e-9:
        emitted += update_tracks(tracks, [("keys", 320, 240, 0.9)], round(t, 3), W, H)
        t += 0.2
    placed = [item for item in emitted if item[1] == "placed"]
    expect(len(placed) == 1, f"放下只能写一次，实际 {len(placed)}")
    t = 4.2
    while t <= 5.0 + 1e-9:
        x = 320 + (t - 4.2) * 400
        emitted += update_tracks(tracks, [("keys", x, 240, 0.9)], round(t, 3), W, H)
        t += 0.2
    emitted += update_tracks(tracks, [], 7.0, W, H)
    removed = [item for item in emitted if item[1] == "removed"]
    expect(len(removed) == 1, "拿走只能写一次")
    boxes = [("keys", 100, 240, 0.9), ("keys", 500, 240, 0.5)]
    bucket = []
    conflict = update_tracks(bucket, boxes, 0, W, H)
    expect(conflict[0][1] == "conflict", "两个同类框应冲突")
    expect(abs(conflict[0][2] - 0.5) < 1e-9, "冲突分数取较低的一只框")
    again = update_tracks(bucket, boxes, 0.2, W, H)
    expect(again == [], "同一次冲突不能每帧再写一行")
    update_tracks(bucket, [("keys", 320, 240, 0.9)], 0.4, W, H)
    second = update_tracks(bucket, boxes, 0.6, W, H)
    expect(len(second) == 1 and second[0][1] == "conflict", "冲突解除后再次出现应再写一次")
    expect(isinstance(tracks[0], Track), "轨迹对象应保留")


def test_trigger():
    trig = Trigger()
    flags = [trig.update(i * 0.2, 0.02) for i in range(3)]
    expect(flags == [False, False, True], f"连续 3 帧才触发，实际 {flags}")
    expect(trig.update(0.8, 0.0) is False, "低于退出阈值后应结束")
    expect(trig.update(1.0, 0.02) is False, "冷却期内不能立刻再触发")
    jumped = Trigger()
    expect(jumped.observe_brightness(0, 100) is False, "第一帧亮度不触发冻结")
    expect(jumped.observe_brightness(0.2, 140) is True, "亮度跳变 40% 应冻结")
    expect(jumped.update(1.0, 0.05) is False, "冻结期间不触发")


def test_memory_and_answer():
    vocab_holder = {}
    import yaml

    with (ROOT / "config" / "vocab.yaml").open(encoding="utf-8") as handle:
        vocab_holder["vocab"] = yaml.safe_load(handle)
    vocab = vocab_holder["vocab"]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        conn = connect(root / "memory.db")
        apply_event(conn, TS, "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
        text, last = answer("钥匙在哪", vocab, get_state(conn, "keys"), None)
        expect(text == "摄像头看到钥匙在玄关柜。", text)
        home, _ = answer("家门钥匙在哪", vocab, get_state(conn, "keys"), None)
        expect(home == "摄像头看到钥匙在玄关柜。", home)
        follow, _ = answer("它在哪", vocab, get_state(conn, "keys"), last)
        expect(follow == "摄像头看到钥匙在玄关柜。", follow)
        missing, _ = answer("它在哪", vocab, None, None)
        expect(missing == "请说出物品的名字。", missing)
        fuzzy, _ = answer("钥迟在哪", vocab, get_state(conn, "keys"), None)
        expect(fuzzy == "摄像头看到钥匙在玄关柜。", fuzzy)
        none, _ = answer("雨伞在哪", vocab, None, None)
        expect(none == "我没有这个物品的记录。", none)

        apply_event(conn, TS, "keys", "entry.cabinet.top", "removed", 0.2, "cam_entry", "")
        low, _ = answer("钥匙在哪", vocab, get_state(conn, "keys"), None)
        expect(low == "我不能确定。请查看该时刻保存的照片。", low)
        apply_event(conn, TS, "keys", "entry.cabinet.top", "removed", 0.8, "cam_entry", "")
        gone, _ = answer("钥匙在哪", vocab, get_state(conn, "keys"), None)
        expect(gone == "钥匙最后出现在玄关柜，11月17日 08:12 被拿走了。", gone)
        apply_event(conn, TS, "keys", "entry.cabinet.top", "occluded", 0.8, "cam_entry", "")
        covered, _ = answer("钥匙在哪", vocab, get_state(conn, "keys"), None)
        expect("被挡住了" in covered, covered)

        plan_id = add_plan(conn, "吃药", TS)
        expect(any(item["id"] == plan_id for item in due_plans(conn, TS)), "到点且未确认的计划应被列出")
        confirm_plan(conn, plan_id, TS)
        source = conn.execute("SELECT done_source FROM plans WHERE id=?", (plan_id,)).fetchone()[0]
        expect(source == "self_report", "服药确认必须标记为口述")
        expect(all(item["id"] != plan_id for item in due_plans(conn, TS)), "口述确认后不再提醒")

        apply_event(conn, TS, "keys", "entry.cabinet.top", "conflict", 0.8, "cam_entry", "")
        both, _ = answer("钥匙在哪", vocab, get_state(conn, "keys"), None)
        expect("两个" in both, both)

        mixed = {
            "objects": [
                {"id": "a", "canonical_name": "药盒", "aliases": ["药盒", "药"]},
                {"id": "b", "canonical_name": "家门钥匙", "aliases": ["家门钥匙"]},
            ],
            "slots": [],
        }
        hit, early = find_object("家门钥匙", mixed, None)
        expect(early is None and hit == "b", "长别名必须先于其中包含的短别名")
        _, which = find_object("药合", {"objects": [
            {"id": "a", "canonical_name": "药盒", "aliases": ["药盒"]},
            {"id": "b", "canonical_name": "药箱", "aliases": ["药箱"]},
        ], "slots": []}, None)
        expect(which.startswith("你说的是哪一个"), which)

        evidence = root / "evidence"
        evidence.mkdir()
        old = evidence / "old.jpg"
        old.write_bytes(b"x")
        old_time = (datetime.now() - timedelta(hours=80)).timestamp()
        import os
        os.utime(old, (old_time, old_time))
        expect(purge_evidence(evidence, now=datetime.now()) == 1, "超过 72 小时的照片应删除")
        set_value(conn, "last_object", "keys")
        delete_all(conn, evidence)
        expect(get_state(conn, "keys") is None, "清除后不能再查到状态")
        expect(get_value(conn, "last_object") is None, "清除后不应留下上一件物品")
        conn.close()


def test_sources():
    """旧库能迁到分列；口述不覆盖视觉列。"""
    import sqlite3

    import yaml

    with (ROOT / "config" / "vocab.yaml").open(encoding="utf-8") as handle:
        vocab = yaml.safe_load(handle)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        legacy = sqlite3.connect(root / "old.db")
        legacy.row_factory = sqlite3.Row
        legacy.executescript(
            """
            CREATE TABLE events (
              id INTEGER PRIMARY KEY,
              ts TEXT NOT NULL,
              object_id TEXT NOT NULL,
              slot_id TEXT NOT NULL,
              action TEXT NOT NULL,
              confidence REAL,
              cam_id TEXT,
              evidence_path TEXT,
              created_at TEXT NOT NULL
            );
            CREATE TABLE object_states (
              object_id TEXT PRIMARY KEY,
              slot_id TEXT,
              status TEXT NOT NULL,
              last_seen_ts TEXT,
              confidence REAL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE plans (
              id INTEGER PRIMARY KEY,
              title TEXT NOT NULL,
              due_ts TEXT,
              done INTEGER NOT NULL DEFAULT 0,
              done_source TEXT,
              confirm_ts TEXT
            );
            CREATE TABLE session (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        legacy.execute(
            """INSERT INTO object_states
               (object_id, slot_id, status, last_seen_ts, confidence, updated_at)
               VALUES ('keys', 'entry.cabinet.top', 'placed', ?, 0.91, ?)""",
            (TS, TS),
        )
        legacy.execute(
            """INSERT INTO events
               (ts, object_id, slot_id, action, confidence, cam_id, evidence_path, created_at)
               VALUES (?, 'keys', 'entry.cabinet.top', 'placed', 0.91, 'cam_entry', '', ?)""",
            (TS, TS),
        )
        legacy.commit()
        legacy.close()
        conn = connect(root / "old.db")
        state = get_state(conn, "keys")
        expect(state["visual_action"] == "placed", "旧库的 status 应迁到视觉列")
        expect(state["visual_slot_id"] == "entry.cabinet.top", "旧库槽位应保留")
        expect(conn.execute("SELECT source FROM events").fetchone()[0] == "visual", "旧事件来源应为 visual")
        apply_event(
            conn, TS, "keys", "", "removed", None, "", "",
            source="user_speech", speech_note="已经拿走",
        )
        state = get_state(conn, "keys")
        expect(state["visual_action"] == "placed", "写入口述后视觉动作应保持放下")
        expect(state["speech_action"] == "removed", "口述动作写在口述列")
        spoken, _ = answer("钥匙在哪", vocab, state, None)
        expect(spoken == "摄像头看到钥匙在玄关柜。您后来说已经拿走。", spoken)
        conn.close()

        conn = connect(root / "fresh.db")
        apply_event(conn, TS, "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
        apply_event(
            conn, TS, "keys", "", "removed", None, "", "",
            source="user_speech", speech_note="已经拿走",
        )
        apply_event(
            conn, "2026-11-17T09:00:00", "keys", "living.coffee.top", "placed", 0.88, "cam_coffee", "",
        )
        state = get_state(conn, "keys")
        expect(state["visual_slot_id"] == "living.coffee.top", "视觉列应是后一次放置的槽位")
        expect(state["visual_action"] == "placed", "后一次视觉放置不应被口述改掉")
        expect(state["speech_action"] == "removed", "后一次视觉放置不应抹掉口述")
        expect(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3, "事件应有三行")
        later, _ = answer("钥匙在哪", vocab, state, None)
        expect("摄像头看到钥匙在茶几垫。" in later and "您后来说已经拿走。" in later, later)
        expect(conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0, "看见物品不能生成服药计划")
        conn.close()


def main():
    hard = [item for item in errors_in_config() if "占位账号" not in item]
    expect(hard == [], "配置检查失败：" + "；".join(hard))
    test_pixels()
    test_states()
    test_emit_once()
    test_trigger()
    test_memory_and_answer()
    test_sources()
    print("演示通过。配置、像素、状态机、触发、数据库和问答均符合说明书。")
    print("真机前修改 config/sites.yaml 里的 RTSP 账号，再运行 python config/check_config.py。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print("演示失败：" + str(exc))
        sys.exit(1)
