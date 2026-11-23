"""单路摄像头。

默认是采用方案：子码流一直解码，触发后才打开主码流，运动结束后再读 3 秒。
`--main-always` 是备用方案：主码流一直连接，运动检测用缩到 640×360 的副本。
真机跟踪用 Ultralytics ByteTrack。ByteTrack 不可用时才退回最近邻，并在日志写明。
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import yaml

from background import create_background
from check_config import errors_in_config
from memory import apply_event, connect, purge_evidence
from state_machine import update_tracks
from streams import next_delay
from streams import MainWindow
from trigger import Trigger

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class LatestFrame:
    """只保留最新帧，避免 RTSP 缓冲把动作推迟几秒。"""

    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.lock = threading.Lock()
        self.frame = None
        self.alive = False
        self.stop = False
        self.reconnects = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        delay = 1.0
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.alive = False
                if self.stop:
                    break
                time.sleep(delay)
                delay = next_delay(delay)
                self.reconnects += 1
                self.cap.release()
                if self.stop:
                    break
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            delay = 1.0
            with self.lock:
                self.frame = frame
                self.alive = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def close(self):
        self.stop = True
        try:
            self.cap.release()
        except Exception:
            pass


def camera_by_id(sites, camera_id):
    for cam in sites["cameras"]:
        if cam["id"] == camera_id:
            return cam
    raise SystemExit(f"sites.yaml 里没有 {camera_id}")


def prompts_for(cam, vocab):
    wanted = set(cam["objects"])
    return [item["prompt"] for item in vocab["objects"] if item["id"] in wanted]


def class_names(cam, vocab):
    wanted = set(cam["objects"])
    return [item["id"] for item in vocab["objects"] if item["id"] in wanted]


def reset_ultralytics_tracker(model):
    if getattr(model, "predictor", None) is not None:
        model.predictor = None


def boxes_to_detections(result, names):
    found = []
    boxes = result.boxes
    if boxes is None:
        return found
    ids = boxes.id
    for index, box in enumerate(boxes):
        cls_index = int(box.cls)
        if cls_index >= len(names):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf)
        center = ((x1 + x2) / 2, (y1 + y2) / 2, score)
        if ids is None:
            found.append((names[cls_index], center[0], center[1], center[2]))
        else:
            found.append((names[cls_index], center[0], center[1], center[2], int(ids[index])))
    return found


def remember(conn, evidence, camera_id, slot_id, frame, events, now):
    for object_id, action, score in events:
        path = ""
        if action in {"placed", "removed"}:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = str(evidence / f"{camera_id}-{object_id}-{action}-{stamp}.jpg")
            cv2.imwrite(path, frame)
        ts = datetime.fromtimestamp(now).isoformat(timespec="seconds")
        apply_event(
            conn, ts, object_id, slot_id, action, float(score), camera_id, path, source="visual",
        )
        print(ts, object_id, action, f"{score:.2f}")


def main():
    flags = {item for item in sys.argv[1:] if item.startswith("-")}
    args = [item for item in sys.argv[1:] if not item.startswith("-")]
    camera_id = args[0] if args else "cam_entry"
    main_always = "--main-always" in flags
    kind = "knn" if "--knn" in flags else "mog2"
    if kind == "knn" and "--knn" in flags:
        print("备用背景：KNN。四条触发规则不变。只有空台面和手部面积分布重叠、并且反光处理之后仍然重叠，才应使用这一项。")

    problems = errors_in_config()
    if any("占位账号" in item or "有效像素" in item or "卧室" in item or "子码流" in item for item in problems):
        print("\n".join(problems))
        raise SystemExit("配置没有通过，不打开摄像头")

    sites = load("sites.yaml")
    vocab = load("vocab.yaml")
    track_cfg = load("bytetrack.yaml")
    cam = camera_by_id(sites, camera_id)
    names = class_names(cam, vocab)
    prompts = prompts_for(cam, vocab)
    expected_buffer = round(1.5 * cam["fps"])
    if int(track_cfg["track_buffer"]) != expected_buffer:
        print(
            f"bytetrack.yaml 的 track_buffer 是 {track_cfg['track_buffer']}，"
            f"按这一路 {cam['fps']} fps、丢失 1.5 秒应为 {expected_buffer}。"
        )

    from ultralytics import YOLOE

    model = YOLOE("yoloe-26n-seg.pt")
    model.set_classes(prompts)
    shown = list(model.names.values())
    if shown[: len(prompts)] != prompts:
        raise SystemExit(f"类别顺序和词表不一致：{shown[:len(prompts)]}")

    # 节点 H2 的嵌入模型在开摄像头前预载，避免第一次事件落在识别循环里卡几秒。
    try:
        import vec_memory

        vec_memory.embedder().load()
        print("向量索引已就绪（节点 H2）。")
    except Exception as exc:
        print(f"向量索引不可用（{exc}）。事件照常入库，召回降级为精确匹配。")

    if main_always:
        print("备用采集：主码流常开。命令行指定了 --main-always，空闲时仍会解码主码流。")
        sub_reader = LatestFrame(cam["rtsp_main"])
    else:
        print("采用采集：空闲只解码子码流，触发后才连接主码流。")
        sub_reader = LatestFrame(cam["rtsp_sub"])
    print("采用跟踪：ByteTrack。若本轮退回最近邻，下面会另起一行写明备用跟踪。")

    background = create_background(kind)
    trigger = Trigger()
    window = MainWindow(tail_s=3.0)
    tracks = []
    main_reader = None
    byte_failed = False
    conn = connect(ROOT / "data" / "memory.db")
    evidence = ROOT / "data" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    slot_id = cam["slot_id"]
    period = 1.0 / cam["fps"]
    tracker_file = str(ROOT / "config" / "bytetrack.yaml")
    announced = False
    print(f"正在读取 {camera_id}。检测输入 {cam['imgsz']}，每 {period:.1f} 秒处理一帧。")

    try:
        while True:
            started = time.time()
            ok, sub_frame = sub_reader.read()
            if not ok:
                time.sleep(0.2)
                continue
            small = cv2.resize(sub_frame, (640, 360))
            if trigger.observe_brightness(started, float(small.mean())):
                background = create_background(kind)
            frozen = started < trigger.freeze_until
            mask = background.apply(small, learningRate=0 if frozen else -1)
            area = float(cv2.countNonZero(mask)) / float(mask.size)
            active = False if frozen else trigger.update(started, area)
            detect_frame = None
            if frozen and main_reader is not None and not main_always:
                main_reader.close()
                main_reader = None
                window.shut()
                tracks.clear()
                reset_ultralytics_tracker(model)
            if main_always:
                if active and not announced:
                    print(f"打开检测 面积比例 {area:.4f} 连续命中 {trigger.hits} 冻结 {frozen}")
                announced = active
                if active:
                    detect_frame = sub_frame
            else:
                read_main, started_window = window.step(started, active)
                if read_main:
                    if main_reader is None:
                        main_reader = LatestFrame(cam["rtsp_main"])
                        reset_ultralytics_tracker(model)
                        tracks.clear()
                        print(
                            f"主码流窗口开始 面积比例 {area:.4f} "
                            f"连续命中 {trigger.hits} 冻结 {frozen}"
                        )
                    ok_main, main_frame = main_reader.read()
                    if ok_main:
                        detect_frame = main_frame
                elif main_reader is not None:
                    main_reader.close()
                    main_reader = None
                    reset_ultralytics_tracker(model)
                    tracks.clear()
                    print("主码流窗口结束")
            if detect_frame is not None:
                detections = []
                if not byte_failed:
                    try:
                        result = model.track(
                            detect_frame,
                            imgsz=cam["imgsz"],
                            conf=0.25,
                            persist=True,
                            tracker=tracker_file,
                            verbose=False,
                        )[0]
                        detections = boxes_to_detections(result, names)
                    except Exception as exc:
                        byte_failed = True
                        print(f"备用跟踪：ByteTrack 不可用（{exc}）。这一路改用最近邻。")
                if byte_failed:
                    result = model.predict(
                        detect_frame, imgsz=cam["imgsz"], conf=0.25, verbose=False
                    )[0]
                    detections = boxes_to_detections(result, names)
                events = update_tracks(
                    tracks, detections, started, detect_frame.shape[1], detect_frame.shape[0]
                )
                remember(conn, evidence, camera_id, slot_id, detect_frame, events, started)
            purge_evidence(evidence)
            time.sleep(max(0.0, period - (time.time() - started)))
    except KeyboardInterrupt:
        print("已停止")
    finally:
        sub_reader.close()
        if main_reader is not None:
            main_reader.close()
        conn.close()


if __name__ == "__main__":
    main()
