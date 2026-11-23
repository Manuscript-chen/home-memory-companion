"""记忆线束。纯软件把处理单元串起来，写入同一份 SQLite，再调用问答和提醒。

不打开摄像头，不加载 YOLOE。画面是合成的，背景减除用和 run_camera.py 相同的 MOG2。
检测框标在画出来的矩形中心上，分数由脚本给出。状态机、触发器、数据库和问句模板都是正式代码。
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from background import create_background
from memory import apply_event, connect, delete_all, get_state
from state_machine import update_tracks
from trigger import Trigger

ROOT = Path(__file__).resolve().parent
BASE = datetime(2026, 11, 18, 8, 0, 0)
W, H = 640, 480
DT = 0.2


def load(name):
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iso(t):
    return (BASE + timedelta(seconds=t)).isoformat(timespec="seconds")


def mog2():
    return create_background("mog2")


class Lane:
    """一路摄像头的处理单元：缩放、MOG2、触发、状态机、入库。"""

    def __init__(self, cam, clock, conn, evidence):
        self.cam = cam
        self.clock = clock
        self.conn = conn
        self.evidence = evidence
        self.background = mog2()
        self.trigger = Trigger()
        self.tracks = []
        self.written = []
        self.areas = []

    def step(self, frame, detections):
        t = self.clock
        small = cv2.resize(frame, (640, 360))
        if self.trigger.observe_brightness(t, float(small.mean())):
            self.background = mog2()
        learning = 0 if t < self.trigger.freeze_until else -1
        mask = self.background.apply(small, learningRate=learning)
        area = float(cv2.countNonZero(mask)) / mask.size
        self.areas.append(area)
        active = self.trigger.update(t, area)
        if active:
            events = update_tracks(self.tracks, detections, t, frame.shape[1], frame.shape[0])
            for object_id, action, score in events:
                path = ""
                when = iso(t)
                if action in {"placed", "removed"}:
                    filename = f"{self.cam['id']}-{object_id}-{action}-{when.replace(':', '')}.jpg"
                    target = self.evidence / filename
                    cv2.imwrite(str(target), frame)
                    path = str(target)
                apply_event(
                    self.conn, when, object_id, self.cam["slot_id"], action,
                    float(score), self.cam["id"], path,
                )
                self.written.append((when, object_id, action, float(score)))
                print(f"  入库 {when} {self.cam['id']} {object_id} {action} {score:.2f}")
        return active


def blank(level=180):
    return np.full((H, W, 3), level, np.uint8)


def put_box(frame, cx, cy, bw=48, bh=36, color=(25, 25, 25)):
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), color, -1)


def put_disturbance(frame, step):
    """边缘上一块每帧都变的前景，避免静止物体过早被背景模型吃掉、触发提前关闭。"""
    rng = np.random.default_rng(step + 1)
    patch = rng.integers(0, 70, (80, 70, 3), dtype=np.uint8)
    x = 24 + (step % 3) * 6
    frame[36:116, x:x + 70] = patch


def det(object_id, cx, cy, score):
    return [(object_id, float(cx), float(cy), float(score))]


def run_frames(lane, count, make):
    triggered = 0
    for index in range(count):
        frame, detections = make(index)
        if lane.step(frame, detections):
            triggered += 1
        lane.clock += DT
    return triggered


def empty_frame(_index):
    return blank(), []


def main():
    sites = load("sites.yaml")
    cameras = {item["id"]: item for item in sites["cameras"]}
    evidence = ROOT / "data" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    conn = connect(ROOT / "data" / "memory.db")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    delete_all(conn, evidence)
    failures = []

    def expect(condition, message):
        if not condition:
            failures.append(message)
            print("  未通过：" + message)
        else:
            print("  通过：" + message)

    print(f"记忆库 {ROOT / 'data' / 'memory.db'} 日志模式 {mode}")
    print("处理单元：MOG2 前景 → Trigger → 仿真检测框 → state_machine → memory → ask")
    entry = Lane(cameras["cam_entry"], 0.0, conn, evidence)

    print("\n[玄关] 空垫 120 秒。真机验收是 10 分钟，这里先看合成空画面会不会写出放下。")
    run_frames(entry, 600, empty_frame)
    expect(entry.written == [], "空垫没有写入事件")
    expect(max(entry.areas[-50:]) < 0.006, "空垫末段前景低于退出阈值")

    print("\n[玄关] 亮度跳变后的 10 秒内，即使垫上有方块也不检测。")
    before = len(entry.written)

    def bright_object(index):
        frame = blank(255)
        put_box(frame, 320, 240)
        return frame, det("keys", 320, 240, 0.95)

    run_frames(entry, 55, bright_object)
    expect(len(entry.written) == before, "冻结的 10 秒内没有入库")
    expect(entry.trigger.freeze_until > 0, "亮度跳变已经把触发器冻住")

    print("\n[玄关] 等冻结结束，再用空垫把新的背景模型喂稳。")
    while entry.clock < entry.trigger.freeze_until:
        entry.step(*empty_frame(0))
        entry.clock += DT
    run_frames(entry, 120, empty_frame)

    print("\n[玄关] 手挥一下：速度落在 0.05 和 0.15 之间，离开后不写事件。")
    before = len(entry.written)

    def wave(index):
        cx = 220 + index * 12.8
        frame = blank()
        put_box(frame, cx, 240)
        put_disturbance(frame, index)
        return frame, det("keys", cx, 240, 0.9)

    run_frames(entry, 6, wave)

    def hand_only(index):
        frame = blank()
        put_disturbance(frame, index + 100)
        return frame, []

    run_frames(entry, 12, hand_only)
    expect(len(entry.written) == before, "没停稳的挥手没有入库")

    print("\n[玄关] 两个同类框同时在内区。冲突只写一次。")
    before = len(entry.written)

    def two_boxes(index):
        frame = blank()
        put_box(frame, 100, 240)
        put_box(frame, 500, 240)
        put_disturbance(frame, index + 200)
        return frame, det("keys", 100, 240, 0.9) + det("keys", 500, 240, 0.5)

    run_frames(entry, 4, two_boxes)
    conflicts = [item for item in entry.written[before:] if item[2] == "conflict"]
    expect(len(conflicts) == 1, "同一次冲突只入库一行")
    state = get_state(conn, "keys")
    expect(state is not None and state["visual_action"] == "conflict", "当前状态是 conflict")

    print("\n[玄关] 方块停在中央超过 2.6 秒，写一次放下。")
    before = len(entry.written)

    def still(index, score=0.91):
        frame = blank()
        put_box(frame, 320, 240)
        put_disturbance(frame, index + 300)
        return frame, det("keys", 320, 240, score)

    run_frames(entry, 25, still)
    placed = [item for item in entry.written[before:] if item[2] == "placed"]
    expect(len(placed) == 1, "放下只入库一次")
    expect(placed and placed[0][3] >= 0.60, "放下置信度不低于 0.60")
    shot = evidence / f"cam_entry-keys-placed-{placed[0][0].replace(':', '')}.jpg"
    expect(shot.is_file(), "放下存了 1 张 JPEG")

    print("\n[玄关] 方块快速移开，手还在画面里，约 1.5 秒后写拿走。")
    before = len(entry.written)

    def leaving(index):
        cx = 320 + index * 64
        frame = blank()
        put_box(frame, cx, 240)
        put_disturbance(frame, index + 400)
        return frame, det("keys", cx, 240, 0.85)

    run_frames(entry, 6, leaving)
    run_frames(entry, 12, hand_only)
    removed = [item for item in entry.written[before:] if item[2] == "removed"]
    expect(len(removed) == 1, "拿走只入库一次")
    expect(get_state(conn, "keys")["visual_action"] == "removed", "钥匙的当前状态是 removed")

    print("\n[茶几] 眼镜盒停稳后被纸盖住，应是挡住，不能是拿走。")
    coffee = Lane(cameras["cam_coffee"], entry.clock, conn, evidence)
    run_frames(coffee, 120, empty_frame)

    def glasses(index):
        frame = blank()
        put_box(frame, 320, 240, 70, 40, (60, 50, 40))
        put_disturbance(frame, index + 500)
        return frame, det("glasses_case", 320, 240, 0.80)

    run_frames(coffee, 25, glasses)

    def paper(index):
        frame = blank()
        cv2.rectangle(frame, (180, 120), (460, 360), (230, 230, 230), -1)
        put_disturbance(frame, index + 600)
        return frame, []

    before = len(coffee.written)
    run_frames(coffee, 12, paper)
    kinds = [item[2] for item in coffee.written]
    expect(kinds.count("placed") == 1 and kinds.count("occluded") == 1, "眼镜盒先放下，再被挡住")
    expect("removed" not in kinds, "盖住没有写成拿走")
    occluded_path = conn.execute(
        "SELECT evidence_path FROM events WHERE object_id = 'glasses_case' AND action = 'occluded'"
    ).fetchone()[0]
    expect(occluded_path == "", "挡住不另存照片")
    occluded = coffee.written[-1]
    expect(occluded[2] == "occluded" and occluded[3] >= 0.40, "挡住的置信度不低于 0.40")

    print("\n[餐桌] 药盒置信度 0.20。记录可以入库，回答必须说不能确定。")
    dining = Lane(cameras["cam_dining"], coffee.clock, conn, evidence)
    run_frames(dining, 120, empty_frame)

    def medicine(index):
        frame = blank()
        put_box(frame, 320, 240, 80, 50, (20, 40, 80))
        put_disturbance(frame, index + 700)
        return frame, det("medicine_box", 320, 240, 0.20)

    run_frames(dining, 25, medicine)
    med = get_state(conn, "medicine_box")
    expect(med is not None and med["visual_action"] == "placed", "低置信度的放下仍然入库")
    expect(med["visual_confidence"] < 0.40, "药盒置信度低于 0.40")

    print("\n[库] events 只追加。三路当前状态：")
    rows = conn.execute(
        "SELECT ts, cam_id, object_id, action, ROUND(confidence, 2) FROM events ORDER BY id"
    ).fetchall()
    for row in rows:
        print(" ", " | ".join(str(item) for item in row))
    states = conn.execute(
        """SELECT object_id, visual_slot_id, visual_action, visual_ts,
                  ROUND(visual_confidence, 2), speech_action
           FROM object_states ORDER BY object_id"""
    ).fetchall()
    for row in states:
        print("  状态", " | ".join(str(item) for item in row))
    expect(len(rows) == len(entry.written) + len(coffee.written) + len(dining.written), "事件行数和入库次数一致")
    videos = list(evidence.glob("*.mp4")) + list(evidence.glob("*.avi")) + list(evidence.glob("*.mkv"))
    expect(videos == [], "证据目录里没有视频文件")

    print("\n[问答] 调用 ask.py，读的是刚才这只库。")
    questions = {
        "钥匙在哪": "被拿走了",
        "它在哪": "被拿走了",
        "眼镜盒在哪": "被挡住了",
        "药盒在哪": "我不能确定",
        "雨伞在哪": "我没有这个物品的记录",
    }
    for text, piece in questions.items():
        result = subprocess.run(
            [sys.executable, "ask.py", text],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        )
        reply = result.stdout.strip()
        print(f"  问：{text}")
        print(f"  答：{reply}")
        expect(piece in reply, f"“{text}”的回答里有“{piece}”")

    print("\n[提醒] 调用 remind.py。确认来源必须是口述。")
    subprocess.run(
        [sys.executable, "remind.py", "add", "吃药", "2026-11-17T08:00:00"],
        cwd=ROOT, check=True,
    )
    subprocess.run([sys.executable, "remind.py", "due"], cwd=ROOT, check=True)
    plan_id = conn.execute("SELECT id FROM plans WHERE title = '吃药'").fetchone()["id"]
    subprocess.run(
        [sys.executable, "remind.py", "confirm", str(plan_id)],
        cwd=ROOT, check=True,
    )
    source = conn.execute("SELECT done_source FROM plans WHERE id = ?", (plan_id,)).fetchone()[0]
    expect(source == "self_report", "服药确认写成 self_report")
    conn.close()

    print()
    if failures:
        print(f"线束失败：{len(failures)} 项")
        return 1
    print("线束通过。触发、状态机、记忆库、问答和到点确认已在纯软件仿真里跑完。")
    print("检测框是合成的，不是 YOLOE 的输出。真摄像头仍用 python run_camera.py。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.exit(main())
