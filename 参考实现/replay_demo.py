"""没有摄像头时，用状态机把三条例子写进本机数据库。

位置和动作仍由 state_machine.py 判定，不接受手写“东西在哪”。
写出的 JPEG 是占位图，不是拍摄画面。真机记录请用 run_camera.py。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from memory import apply_event, connect, delete_all
from state_machine import update_tracks

ROOT = Path(__file__).resolve().parent
W, H = 640, 480
PLACEHOLDER = ROOT / "assets" / "offline.jpg"

SCENES = (
    ("keys", 0.91, "placed", "2026-11-17T08:12:00"),
    ("glasses_case", 0.80, "occluded", "2026-11-17T09:05:00"),
    ("medicine_box", 0.80, "removed", "2026-11-17T12:30:00"),
)


def load(name):
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ticks(start, end, step=0.2):
    values = []
    t = start
    while t <= end + 1e-9:
        values.append(round(t, 3))
        t += step
    return values


def play(frames):
    tracks = []
    events = []
    for moment, detections in frames:
        events += update_tracks(tracks, detections, moment, W, H)
    return events


def scene_frames(object_id, score, final):
    still = [(t, [(object_id, 320.0, 240.0, score)]) for t in ticks(0, 3.0)]
    if final == "placed":
        return still
    if final == "occluded":
        return still + [(4.6, [])]
    frames = list(still)
    x = 200.0
    for moment in ticks(3.2, 4.2):
        frames.append((moment, [(object_id, x, 240.0, score)]))
        x += 60.0
    frames.append((5.8, []))
    return frames


def earlier(ts, seconds):
    moment = datetime.fromisoformat(ts) - timedelta(seconds=seconds)
    return moment.isoformat(timespec="seconds")


def locate(sites, object_id):
    for cam in sites["cameras"]:
        if object_id in cam["objects"]:
            return cam["id"], cam["slot_id"]
    raise SystemExit(f"sites.yaml 里没有 {object_id}")


def write_scene(conn, evidence, sites, object_id, score, final, ts):
    events = play(scene_frames(object_id, score, final))
    if not events or events[-1][1] != final:
        raise SystemExit(f"{object_id} 回放没有得到 {final}，实际 {events}")
    cam_id, slot_id = locate(sites, object_id)
    for index, (cls, action, confidence) in enumerate(events):
        when = ts if index == len(events) - 1 else earlier(ts, (len(events) - index) * 60)
        path = ""
        if action in {"placed", "removed"}:
            filename = f"{cam_id}-{cls}-{action}-{when.replace(':', '')}.jpg"
            target = evidence / filename
            target.write_bytes(PLACEHOLDER.read_bytes())
            path = str(target)
        apply_event(conn, when, cls, slot_id, action, confidence, cam_id, path)
        print(f"写入 {cls} {action} {confidence:.2f} {when}")


def main():
    if not PLACEHOLDER.is_file():
        raise SystemExit(f"缺少占位图 {PLACEHOLDER}")
    reset = "--reset" in sys.argv[1:]
    sites = load("sites.yaml")
    evidence = ROOT / "data" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    conn = connect(ROOT / "data" / "memory.db")
    if reset:
        delete_all(conn, evidence)
        print("已清空数据库和证据照片，再写入回放。")
    else:
        print("在现有数据库上追加回放。若要先清空，改用 python replay_demo.py --reset")
    for object_id, score, final, ts in SCENES:
        write_scene(conn, evidence, sites, object_id, score, final, ts)
    conn.close()
    print("回放写完。这不是摄像头看见的结果。可以提问：")
    print("  python ask.py 钥匙在哪")
    print("  python ask.py 眼镜盒在哪")
    print("  python ask.py 药盒在哪")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
