"""节点 H2 的验收（说明书 10.5）。

用临时库跑四组案例：
1. 20 条别名完全匹配的问法，行为与节点 H 相同（句子一致，路径是精确命中）。
2. 20 条不在别名表里的说法，筛后物品正确的比例单独报告；说错位置不超过 1 条。
3. 第二道筛选：故意写出“在桌子附近”的句子必须被丢掉，改念模板。
4. 时间词收成区间后的召回，以及清除命令之后向量同步删除。

通过时打印“向量召回通过”。
"""

import sys
import tempfile
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from answer import answer, find_object  # noqa: E402
from ask import ask_once  # noqa: E402
from memory import apply_event, connect, delete_all, get_state, log_recall  # noqa: E402
import vec_memory as vec  # noqa: E402

ROOT = Path(__file__).resolve().parent

with (ROOT / "config" / "vocab.yaml").open(encoding="utf-8") as handle:
    VOCAB = yaml.safe_load(handle)

FAILURES = []
PATH_COUNTS = {}


def expect(condition, message):
    if not condition:
        FAILURES.append(message)
        print(f"  ✗ {message}")
    return condition


def build_scene(conn):
    """与 replay_demo 同一幕：钥匙放玄关又拿走，眼镜盒被挡住，药盒低分放下。"""
    apply_event(conn, "2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
    apply_event(conn, "2026-11-18T09:05:00", "glasses_case", "living.coffee.top", "placed", 0.80, "cam_coffee", "")
    apply_event(conn, "2026-11-18T09:06:00", "glasses_case", "living.coffee.top", "occluded", 0.80, "cam_coffee", "")
    apply_event(conn, "2026-11-18T11:00:00", "medicine_box", "dining.table.top", "placed", 0.30, "cam_dining", "")
    apply_event(conn, "2026-11-18T13:20:00", "keys", "entry.cabinet.top", "removed", 0.85, "cam_entry", "")
    apply_event(
        conn, "2026-11-18T13:21:00", "keys", "", "placed", None, "", "",
        source="user_speech", speech_note="已经放进抽屉",
    )


def run_question(conn, text, last=None):
    reply, new_last, path = ask_once(conn, VOCAB, text, last)
    PATH_COUNTS[path.split("（")[0]] = PATH_COUNTS.get(path.split("（")[0], 0) + 1
    return reply, new_last, path


def test_alias_questions(conn):
    """第一组：别名完全匹配的 20 条，行为与节点 H 相同。"""
    print("一、别名完全匹配的 20 条问法")
    last = None
    for index, text in enumerate([
        "钥匙在哪", "钥匙串放哪了", "家门钥匙在哪里",
        "眼镜盒呢", "老花镜盒子在哪", "眼镜盒子放哪了",
        "药盒在哪", "药箱呢", "吃药的盒子放哪了",
        "遥控器在哪", "遥控板呢", "电视遥控器放哪了",
        "医保卡在哪", "社保卡呢", "医疗卡放哪了",
        "钥匙放哪了", "钥匙在哪里", "钥迟放哪了",
        "它在哪", "那个呢",
    ], start=1):
        if text in {"它在哪", "那个呢"}:
            last = "keys"
        reply, new_last, path = run_question(conn, text, last)
        object_id, early = find_object(text, VOCAB, last)
        if early:
            node_h = early
        else:
            node_h, _ = answer(text, VOCAB, get_state(conn, object_id), last)
        same = reply == node_h
        exact = path == "精确命中"
        expect(same, f"[{index}] {text}：H2 给 {reply!r}，节点 H 给 {node_h!r}")
        expect(exact, f"[{index}] {text}：别名问法的路径应是精确命中，实际 {path}")
        if new_last:
            last = new_last
    print(f"  路径计数：{PATH_COUNTS}")


# (问法, 期望物品 id)。None 表示期望没有任何物品被召回。
NON_ALIAS = [
    ("看报纸用的那个", "glasses_case"),
    ("读报的眼镜", "glasses_case"),
    ("老花镜放哪了", "glasses_case"),
    ("桌上那副眼镜", "glasses_case"),
    ("茶几上的盒子", "glasses_case"),
    ("开门用的那串", "keys"),
    ("门口那串", "keys"),
    ("家门那串东西", "keys"),
    ("装药的那个", "medicine_box"),
    ("吃的药在哪里", "medicine_box"),
    ("药片盒", "medicine_box"),
    ("找一下我的药", "medicine_box"),
    ("电视的那个板子", "remote_control"),
    ("换台的板子", "remote_control"),
    ("挂号用的卡片", "insurance_card"),
    ("看病的那张卡", "insurance_card"),
    ("充电的那个呢", None),
    ("雨伞在哪", None),
    ("我的外套呢", None),
    ("保温杯在哪", None),
]

SLOT_NAMES = [slot["canonical_name"] for slot in VOCAB["slots"]]


def test_non_alias_questions(conn):
    """第二组：不在别名表里的说法。物品正确比例单独报告，说错位置不超过 1 条。"""
    print("二、不在别名表里的 20 条说法")
    correct = 0
    unresolved = 0
    wrong_object = []
    wrong_position = []
    for text, expected in NON_ALIAS:
        reply, resolved, path = run_question(conn, text)
        note = f"{text!r} → {reply!r}（resolved={resolved}，path={path}）"
        if expected is None:
            if resolved is None:
                unresolved += 1
                print(f"  · {note} 未召回，符合预期")
            else:
                wrong_object.append(note)
                print(f"  ✗ {note} 不该召回 {resolved}")
            if any(name in reply for name in SLOT_NAMES):
                wrong_position.append(note)
            continue
        if resolved == expected:
            correct += 1
            print(f"  ✓ {note}")
        elif resolved is None:
            unresolved += 1
            print(f"  · {note} 未召回")
        else:
            wrong_object.append(note)
            print(f"  ✗ {note} 召回成 {resolved}")
        state = get_state(conn, expected)
        allowed = set()
        if state and state.get("visual_slot_id"):
            for slot in VOCAB["slots"]:
                if slot["id"] == state["visual_slot_id"]:
                    allowed.add(slot["canonical_name"])
        if resolved == expected:
            # 召回对了物品却说了别的槽位，才算说错位置。
            if any(name in reply and name not in allowed for name in SLOT_NAMES):
                wrong_position.append(note)
        else:
            if any(name in reply for name in SLOT_NAMES):
                wrong_position.append(note)
    ratio = correct / len(NON_ALIAS)
    print(
        f"  小计：物品正确 {correct}/{len(NON_ALIAS)}（{ratio:.0%}），"
        f"未召回 {unresolved}，召回成别的物品 {len(wrong_object)}，"
        f"说错位置 {len(wrong_position)}"
    )
    expect(len(wrong_position) <= 1, f"说错位置 {len(wrong_position)} 条，超过 1 条：{wrong_position}")
    print(f"  筛后物品正确的比例单独报告：{correct}/{len(NON_ALIAS)}")
    return correct, len(NON_ALIAS)


def test_second_filter(conn):
    """第三组：第二道筛选。生成句说不出词表内的地点就整句丢弃，改念模板。"""
    print("三、第二道筛选")
    rows = [
        (
            "event",
            0.80,
            {
                "object_id": "keys",
                "slot_id": "entry.cabinet.top",
                "action": "removed",
                "ts": "2026-11-18T13:20:00",
                "source": "visual",
                "confidence": 0.85,
            },
        )
    ]
    bad = "钥匙大概在桌子附近。"
    ok, reasons = vec.filter_generated(bad, rows)
    expect(not ok, f"“{bad}”应被丢弃，实际通过")
    print(f"  丢弃理由：{reasons}")
    template, _ = answer("钥匙在哪", VOCAB, get_state(conn, "keys"), None)
    fallback = template if not ok else bad
    expect(fallback == template, "生成句被丢弃后应改念模板")
    print(f"  改念模板：{fallback}")

    good = "钥匙最后出现在玄关柜，11月18日 13:20 被拿走了。"
    ok, reasons = vec.filter_generated(good, rows)
    expect(ok, f"只说行内字段的句子不应被丢弃：{reasons}")

    cross = "药盒在茶几垫。"
    ok, reasons = vec.filter_generated(cross, rows)
    expect(not ok, "物品和槽位都不在行内的句子应被丢弃")

    speech_claim = "您说过钥匙已经放进抽屉。"
    ok, reasons = vec.filter_generated(speech_claim, rows)
    expect(not ok, "行里没有口述来源时，说“您说过”应被丢弃")


def test_time_words(conn):
    """第四组：时间词先收成区间，再在区间里检索。"""
    print("四、时间词")
    cases = [
        ("钥匙早上放哪了", "摄像头看到钥匙在玄关柜。"),
        ("早上眼镜搁哪了", "眼镜盒最后出现在茶几垫，后来被挡住了，我不能确定还在不在。"),
        (
            "中午钥匙呢",
            "钥匙最后出现在玄关柜，11月18日 13:20 被拿走了。您在11月18日 13:21说过钥匙放下了。",
        ),
        ("昨天放的那个药", "我没有药盒在这个时间段的记录。"),
        ("晚上钥匙在哪", "我没有钥匙在这个时间段的记录。"),
    ]
    for text, expected in cases:
        reply, resolved, path = run_question(conn, text)
        expect(reply == expected, f"{text}：期望 {expected!r}，得到 {reply!r}")
        expect(path == "向量召回", f"{text}：带时间词应走向量召回，实际 {path}")
        print(f"  ✓ {text} → {reply}")
    overlap = vec.parse_time_range("昨天早上")
    expect(overlap is not None and overlap[0].endswith("05:00:00"), f"“昨天早上”应收成昨天 05:00 起，实际 {overlap}")
    none = vec.parse_time_range("钥匙在哪")
    expect(none is None, "没有时间词时不应产出区间")


def test_clear_sync():
    """清除命令之后，向量跟着事件一起删。"""
    print("五、清除同步")
    with tempfile.TemporaryDirectory() as folder:
        conn = connect(Path(folder) / "clear.db")
        apply_event(conn, "2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
        vec.prepare(conn)
        before = conn.execute("SELECT COUNT(*) FROM vec_events").fetchone()[0]
        delete_all(conn, Path(folder) / "evidence")
        after = conn.execute("SELECT COUNT(*) FROM vec_events").fetchone()[0]
        expect(before == 1 and after == 0, f"清除后向量应删光：清前 {before}，清后 {after}")
        reply, _, _ = ask_once(conn, VOCAB, "钥匙在哪", None)
        expect(reply == "我没有钥匙的记录。", f"清除后应没有记录，实际 {reply!r}")
        conn.close()


def main():
    with tempfile.TemporaryDirectory() as folder:
        conn = connect(Path(folder) / "h2.db")
        build_scene(conn)
        test_alias_questions(conn)
        test_non_alias_questions(conn)
        test_second_filter(conn)
        test_time_words(conn)
        conn.close()
    test_clear_sync()
    print(f"路径计数（不含清除用例）：{PATH_COUNTS}")
    if FAILURES:
        print(f"向量召回未通过：{len(FAILURES)} 条")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("向量召回通过。别名问法与节点 H 一致，词表外说法按比例报告，两道筛选和清除同步均符合说明书。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
