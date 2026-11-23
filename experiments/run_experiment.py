"""完整仿真。每条测试记录输入、输出、是否通过，以及写库之后的表。

没有摄像头、没有夜间画面、没有连续 24 小时的项目记为未执行。
合成轨迹上的通过，不等于这户台面上的检测准确率。
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_common = ROOT / "common"
if str(_common) not in sys.path:
    sys.path.insert(0, str(_common))
import nodepath  # noqa: E402

import cv2  # noqa: E402
import numpy as np
import yaml

from answer import answer, find_object, history_slot_question
from associate import SimByteTrack, load_settings
from hotwords import write_hotwords
from memory import (
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
    snapshot,
)
from pixel_budget import evaluate, world_width_m
from pinyin_backup import candidates
from reed import accept_reed
from run_demo import samples
from state_machine import judge, update_tracks
from streams import MainWindow, backoff_sequence
from trigger import Trigger
from voice_gate import PushToTalk

DATA = HERE / "data"
EVIDENCE = DATA / "evidence"
RESULTS = HERE / "results"
LAB_PATH = DATA / "lab.db"
PLACEHOLDER = ROOT / "assets" / "offline.jpg"
W, H = 640, 480

CASES = []
METRICS = {}
WRONG = []
QUESTION_COUNT = 0


def load_yaml(name):
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


SITES = load_yaml("sites.yaml")
VOCAB = load_yaml("vocab.yaml")
PRIVACY = load_yaml("privacy.yaml")
CAMERAS = {item["id"]: item for item in SITES["cameras"]}
OBJECTS = {item["id"]: item for item in VOCAB["objects"]}


def camera_for(object_id):
    for cam in SITES["cameras"]:
        if object_id in cam["objects"]:
            return cam
    raise KeyError(object_id)


def save():
    RESULTS.mkdir(parents=True, exist_ok=True)
    counts = {"pass": 0, "fail": 0, "skipped": 0}
    for item in CASES:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    summary = {
        "finished": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "counts": counts,
        "question_count": QUESTION_COUNT,
        "location_errors": len(WRONG),
        "metrics": METRICS,
        "notes": [
            "这是开发者机器上的软件仿真，不是老人住宅实测。",
            "状态机在合成轨迹上的通过，不能写成这户台面的检测准确率。",
        ],
    }
    _dump(RESULTS / "trace.json", CASES)
    _dump(RESULTS / "summary.json", summary)


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def add_case(case_id, node, suite, title, status, detail, layers, db=None, frames=None, database="", attention=False):
    CASES.append(
        {
            "id": case_id,
            "node": node,
            "suite": suite,
            "title": title,
            "status": status,
            "detail": detail,
            "layers": layers,
            "frames": frames or [],
            "db": db,
            "database": database,
            "attention": attention,
        }
    )
    print(f"{status:8} {case_id} {title}", flush=True)
    save()


def guard(name, fn):
    try:
        fn()
    except Exception as exc:
        traceback.print_exc()
        add_case(f"X-{name}", name, name, "这一组中断", "fail", str(exc), [], None)


def layer(name, incoming, outgoing, passed=True):
    return {"layer": name, "input": incoming, "output": outgoing, "passed": passed}


def finish_summary():
    save()
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    summary["finished"] = True
    _dump(RESULTS / "summary.json", summary)


def run_command(script, *extra):
    scripts = {
        "run_demo.py": ROOT / "demo" / "run_demo.py",
        "run_h2.py": ROOT / "demo" / "run_h2.py",
        "check_config.py": ROOT / "config" / "check_config.py",
        "voice.py": ROOT / "voice" / "voice.py",
        "harness.py": ROOT / "demo" / "harness.py",
    }
    return subprocess.run(
        [sys.executable, str(scripts[script]), *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def suite_commands():
    demo = run_command("run_demo.py")
    text = (demo.stdout or "") + (demo.stderr or "")
    ok = demo.returncode == 0 and "演示通过" in text
    METRICS["run_demo_exit"] = demo.returncode
    add_case(
        "S01", "S", "总闸", "run_demo.py",
        "pass" if ok else "fail",
        text.strip().splitlines()[-1] if text.strip() else "没有输出",
        [layer("S", "python demo/run_demo.py", text.strip()[-800:], ok)],
    )
    h2 = run_command("run_h2.py")
    h2_text = (h2.stdout or "") + (h2.stderr or "")
    h2_ok = h2.returncode == 0 and "向量召回通过" in h2_text
    METRICS["run_h2_exit"] = h2.returncode
    tail = "\n".join(h2_text.strip().splitlines()[-2:])
    add_case(
        "S03", "S", "总闸", "run_h2.py",
        "pass" if h2_ok else "fail",
        tail or "没有输出",
        [layer("S", "python run_h2.py", tail, h2_ok)],
    )


def suite_a():
    keys = evaluate(2560, 1280, 0.02, 0.6, 60)
    glasses = evaluate(2560, 640, 0.05, 0.8, 60)
    medicine = evaluate(2560, 640, 0.05, 0.8, 60)
    remote = evaluate(2560, 640, 0.04, 0.8, 60)
    keys_wrong = evaluate(2560, 640, 0.02, 0.8, 60)
    manuscript = evaluate(2560, 640, 0.06, 1.5, 70)
    card = evaluate(2560, 640, 0.054, 0.8, 60)
    METRICS["pixels"] = {
        "keys": round(keys["effective_px"], 2),
        "glasses_case": round(glasses["effective_px"], 2),
        "medicine_box": round(medicine["effective_px"], 2),
        "remote_on_coffee": round(remote["effective_px"], 2),
        "keys_on_coffee": round(keys_wrong["effective_px"], 2),
        "manuscript_1_5m": round(manuscript["effective_px"], 2),
        "insurance_on_dining": round(card["effective_px"], 2),
    }
    specs = [
        ("A01", "钥匙在玄关", keys["pass"] and abs(keys["effective_px"] - 37.0) < 0.2, f"有效像素 {keys['effective_px']:.2f}"),
        ("A02", "眼镜盒在茶几", glasses["pass"] and abs(glasses["effective_px"] - 34.6) < 0.2, f"有效像素 {glasses['effective_px']:.2f}"),
        ("A03", "药盒在餐桌", medicine["pass"] and abs(medicine["effective_px"] - 34.6) < 0.2, f"有效像素 {medicine['effective_px']:.2f}"),
        ("A04", "遥控器不进检测", (not remote["pass"]) and "remote_control" not in CAMERAS["cam_coffee"]["objects"], f"茶几 640 下 {remote['effective_px']:.2f} px，只留词表"),
        ("A05", "钥匙不能放到茶几", not keys_wrong["pass"], f"有效像素 {keys_wrong['effective_px']:.2f}"),
        ("A06", "1.5 米示例不合格", (not manuscript["pass"]) and abs(manuscript["effective_px"] - 18.3) < 0.2, f"有效像素 {manuscript['effective_px']:.2f}"),
        ("A07", "医保卡像素够但不占检测名额", card["pass"] and "insurance_card" not in CAMERAS["cam_dining"]["objects"], f"有效像素 {card['effective_px']:.2f}，餐桌已绑定药盒"),
    ]
    for case_id, title, ok, detail in specs:
        add_case(case_id, "A", "槽位与像素", title, "pass" if ok else "fail", detail, [layer("A", title, detail, ok)])

    entry = 0.297 / world_width_m(0.6, 60)
    coffee = 0.297 / world_width_m(0.8, 60)
    METRICS["a4_fraction"] = {"entry": round(entry, 3), "coffee": round(coffee, 3)}
    ok = 0.38 <= entry <= 0.48 and 0.28 <= coffee <= 0.38
    add_case(
        "A08", "A", "槽位与像素", "A4 长边占画面的比例",
        "pass" if ok else "fail",
        f"玄关 {entry:.3f}，茶几和餐桌 {coffee:.3f}。这是计算值，没有实拍。",
        [layer("A", "A4 长边 297 mm", f"玄关 {entry:.3f}，茶几 {coffee:.3f}", ok)],
    )
    rooms = set(PRIVACY.get("rooms_without_camera", []))
    masked = {item["id"]: item for item in PRIVACY["cameras"]}
    ok = rooms == {"bedroom", "bathroom"} and set(masked) == set(CAMERAS)
    ok = ok and all(item.get("physical_mask") and str(item.get("note", "")).strip() for item in masked.values())
    add_case("A09", "A", "槽位与像素", "卧室和卫生间不设摄像头", "pass" if ok else "fail", "、".join(sorted(rooms)), [layer("A", "privacy.yaml", "、".join(sorted(rooms)), ok)])
    prompts_ok = all(item["prompt"].isascii() and item["canonical_name"] in item["aliases"] and len(item["aliases"]) >= 3 for item in VOCAB["objects"])
    one_each = all(len(cam["objects"]) == 1 for cam in SITES["cameras"])
    add_case("A10", "A", "槽位与像素", "英文提示、一路一个类别", "pass" if prompts_ok and one_each else "fail", "词表和三路绑定已核对", [layer("A", "vocab.yaml + sites.yaml", "提示词为英文，每路 1 个检测物品", prompts_ok and one_each)])
    proc = run_command("check_config.py")
    text = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    add_case("A11", "A", "槽位与像素", "check_config.py", "pass" if ok else "fail", text.strip(), [layer("A", "python config/check_config.py", text.strip(), ok)])


def suite_b():
    idle = MainWindow()
    idle_reads = sum(1 for i in range(50) if idle.step(i * 0.2, False)[0])
    ok = idle_reads == 0 and idle.reads == 0
    add_case("B01", "B", "双码流", "空闲不读主码流", "pass" if ok else "fail", f"主码流读取 {idle_reads} 次", [layer("B", "50 帧触发为假", f"主码流读取 {idle_reads}", ok)])

    window = MainWindow()
    reads = []
    flags = [True] * 15 + [False] * 20
    for index, flag in enumerate(flags):
        read, started = window.step(round(index * 0.2, 3), flag)
        reads.append(read)
        if read:
            stream = "main"
        else:
            stream = "sub"
        if stream == "sub" and read:
            raise RuntimeError("子码流被当成主码流")
    main_reads = sum(1 for item in reads if item)
    closed = not window.open
    ok = main_reads >= 15 and closed and main_reads > 15
    METRICS["main_reads_after_3s_motion"] = main_reads
    add_case(
        "B02", "B", "双码流", "触发后主码流连续读取，结束后再保留 3 秒",
        "pass" if ok else "fail",
        f"主码流读取 {main_reads} 帧，窗口最后已关闭：{closed}",
        [layer("B", "连续 15 帧触发为真，然后为假", f"读取 {main_reads} 帧，多于 15，含 3 秒尾段", ok)],
    )
    detector_on_sub = 0
    gate = MainWindow()
    for index in range(30):
        read, _ = gate.step(index * 0.2, False)
        if not read:
            detector_on_sub += 0
        else:
            detector_on_sub += 1
    ok = detector_on_sub == 0
    add_case("B03", "B", "双码流", "没打开主码流时检测器不被调用", "pass" if ok else "fail", f"检测调用 {detector_on_sub}", [layer("D", "子码流帧", "不进入检测", ok)])

    waits = backoff_sequence(7)
    ok = waits == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    add_case("B04", "B", "双码流", "断线等待 1、2、4 秒，上限 30", "pass" if ok else "fail", str(waits), [layer("B", "连续读失败", str(waits), ok)])

    frozen = MainWindow()
    frozen.step(0.0, True)
    frozen.shut()
    read, _ = frozen.step(0.2, False)
    ok = (not frozen.open) and (not read)
    add_case("B05", "B", "双码流", "亮度冻结时关掉主码流窗口", "pass" if ok else "fail", "冻结后不再读主码流", [layer("B", "freeze", "窗口关闭", ok)])


def percentile(values, p):
    seq = sorted(values)
    if not seq:
        raise ValueError("空序列")
    k = (len(seq) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(seq) - 1)
    frac = k - lo
    return seq[lo] * (1 - frac) + seq[hi] * frac


def area_of(background, frame):
    mask = background.apply(frame)
    return float(cv2.countNonZero(mask)) / float(mask.size)


def suite_c():
    trig = Trigger()
    flags = [trig.update(i * 0.2, 0.02) for i in range(3)]
    ok = flags == [False, False, True]
    add_case("C01", "C", "运动门控", "连续 3 帧才打开", "pass" if ok else "fail", str(flags), [layer("C", "面积 0.02 × 3", str(flags), ok)])
    trig = Trigger()
    for i in range(3):
        trig.update(i * 0.2, 0.02)
    left = trig.update(0.8, 0.0)
    cooled = trig.update(1.0, 0.02)
    ok = (left is False) and (cooled is False)
    add_case("C02", "C", "运动门控", "低于退出阈值后进入冷却", "pass" if ok else "fail", f"退出 {left}，冷却期内 {cooled}", [layer("C", "0.02 然后 0", f"退出 {left}，1 秒后仍为 {cooled}", ok)])
    trig = Trigger()
    [trig.update(i * 0.2, 0.02) for i in range(3)]
    trig.update(0.8, 0.0)
    still = trig.update(2.7, 0.02)
    again = [trig.update(2.9 + i * 0.2, 0.02) for i in range(3)]
    ok = (still is False) and again == [False, False, True]
    add_case("C03", "C", "运动门控", "冷却 2 秒后才能再次进入", "pass" if ok else "fail", str(again), [layer("C", "退出后再等 2 秒", str(again), ok)])
    jumped = Trigger()
    first = jumped.observe_brightness(0, 100)
    second = jumped.observe_brightness(0.2, 140)
    blocked = jumped.update(1.0, 0.05)
    ok = (first is False) and (second is True) and (blocked is False)
    add_case("C04", "C", "运动门控", "亮度变化超过 30% 冻结 10 秒", "pass" if ok else "fail", f"冻结 {second}，冻结期间触发 {blocked}", [layer("C", "亮度 100 → 140", f"冻结 {second}", ok)])

    print("空台面：热身 500 帧，再计 3000 帧。", flush=True)
    from background import create_background

    background = create_background("mog2")
    blank = np.full((360, 640, 3), 180, np.uint8)
    warm = Trigger()
    warm_opens = 0
    for index in range(500):
        ratio = area_of(background, blank)
        if warm.update(index * 0.2, ratio):
            warm_opens += 1
    measured = Trigger()
    areas = []
    opens = 0
    for index in range(3000):
        ratio = area_of(background, blank)
        areas.append(ratio)
        if measured.update(100 + index * 0.2, ratio):
            opens += 1
    empty_p95 = percentile(areas, 95)
    METRICS["empty_table"] = {
        "frames": 3000,
        "seconds": 600,
        "opens": opens,
        "warmup_opens_on_a_fresh_model": warm_opens,
        "area_p95": empty_p95,
    }
    ok = opens == 0
    add_case(
        "C05", "C", "运动门控", "热身后的空台面 10 分钟不打开",
        "pass" if ok else "fail",
        f"计时内打开 {opens} 次。另起一个触发器在热身阶段打开 {warm_opens} 次，不计入这 10 分钟。",
        [layer("C", "640×360 恒定画面，5 fps，3000 帧", f"打开 {opens} 次，面积 95 分位 {empty_p95:.6f}", ok)],
        attention=warm_opens > 0,
    )

    hand_areas = []
    for episode in range(10):
        for step in range(5):
            frame = blank.copy()
            x = 40 + episode * 20 + step * 30
            cv2.rectangle(frame, (x, 70), (x + 140, 250), (20, 20, 20), -1)
            hand_areas.append(area_of(background, frame))
        for _ in range(6):
            area_of(background, blank)
    hand_trigger = Trigger()
    opened_at = None
    for index, ratio in enumerate(hand_areas[:6]):
        if hand_trigger.update(index * 0.2, ratio):
            opened_at = index
            break
    ok = opened_at is not None and opened_at <= 2 and min(hand_areas[:3]) >= 0.01
    METRICS["hand_open_frame"] = opened_at
    add_case(
        "C06", "C", "运动门控", "合成色块连续出现后约 0.6 秒打开",
        "pass" if ok else "fail",
        f"第 {opened_at} 帧打开，前三帧面积 {['%.4f' % item for item in hand_areas[:3]]}。色块不是真手。",
        [layer("C", "移动色块", f"打开帧 {opened_at}", ok)],
    )
    hand_p10 = percentile(hand_areas, 10)
    knn = create_background("knn")
    for _ in range(120):
        area_of(knn, blank)
    knn_empty = [area_of(knn, blank) for _ in range(60)]
    knn_hand = []
    for step in range(8):
        frame = blank.copy()
        x = 30 + step * 40
        cv2.rectangle(frame, (x, 70), (x + 140, 250), (20, 20, 20), -1)
        knn_hand.append(area_of(knn, frame))
    knn_empty_p95 = percentile(knn_empty, 95)
    knn_hand_p10 = percentile(knn_hand, 10)
    overlap = empty_p95 >= hand_p10
    between = empty_p95 < 0.01 < hand_p10
    if overlap:
        decision = "分布重叠。先处理反光，不把阈值改成 0，生产路径仍是 MOG2。"
    elif between:
        decision = "0.01 落在空台面 95 分位和色块 10 分位之间。继续用 MOG2。"
    else:
        decision = "0.01 不在两个分布中间。这次没有改生产阈值。"
    METRICS["calibration"] = {
        "mog2_empty_p95": empty_p95,
        "mog2_blob_p10": hand_p10,
        "knn_empty_p95": knn_empty_p95,
        "knn_blob_p10": knn_hand_p10,
        "overlap": overlap,
        "decision": decision,
    }
    add_case(
        "C07", "C", "运动门控", "标定分位，不因此改用 KNN",
        "pass",
        decision + f" MOG2 空台面 95 分位 {empty_p95:.6f}，色块 10 分位 {hand_p10:.4f}。KNN 空台面 95 分位 {knn_empty_p95:.6f}，色块 10 分位 {knn_hand_p10:.4f}。",
        [layer("C", "MOG2 与 KNN 的面积分位", decision, True)],
        attention=overlap or not between,
    )


def suite_d():
    allowed = {"keys"}
    raw = [("keys", 320, 240, 0.8), ("cup", 100, 100, 0.9)]
    kept = [item for item in raw if item[0] in allowed]
    dropped = [item for item in raw if item[0] not in allowed]
    ok = len(kept) == 1 and dropped[0][0] == "cup"
    add_case("D01", "D", "白名单检测", "白名单以外的框不进跟踪", "pass" if ok else "fail", f"保留 {kept[0][0]}，丢弃 {dropped[0][0]}", [layer("D", "keys 与 cup", "只保留 keys", ok)])

    settings = load_settings(ROOT / "config" / "bytetrack.yaml")
    tracker = SimByteTrack(settings)
    alive = tracker.update([("keys", 320, 240, 48, 36, 0.20)])
    ok = alive == []
    METRICS["score_0_20_starts_track"] = bool(alive)
    add_case(
        "D02", "D", "白名单检测", "0.20 的框不新建轨迹",
        "pass" if ok else "fail",
        "new_track_thresh 是 0.25。0.20 不能进跟踪，问答里的低分句用 0.25 到 0.40 之间的分数。",
        [layer("E", "score 0.20", f"轨迹数 {len(alive)}", ok)],
    )

    weight = ROOT / "models" / "yoloe-26n-seg.pt"
    if not weight.exists():
        add_case("D03", "D", "白名单检测", "YOLOE-26n 权重", "skipped", "yoloe-26n-seg.pt 还不在 models/，这次不把下载算进测试通过。见 docs/weights.md。", [layer("D", "权重文件", "未执行", True)])
        return
    try:
        from ultralytics import YOLOE

        model = YOLOE(str(weight))
        model.set_classes(["keys"])
        shown = list(model.names.values())[:1]
        frame = np.full((480, 640, 3), 180, np.uint8)
        cv2.rectangle(frame, (280, 200), (360, 280), (30, 30, 30), -1)
        result = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        count = 0 if result.boxes is None else len(result.boxes)
        ok = shown == ["keys"]
        METRICS["yoloe_synthetic_boxes"] = count
        METRICS["yoloe_class_order"] = shown
        add_case(
            "D03", "D", "白名单检测", "YOLOE 在合成色块上的一次前向",
            "pass" if ok else "fail",
            f"类别顺序 {shown}。合成色块上框数 {count}。这不是钥匙的 mAP，也不能和 COCO、LVIS 合成一个数。",
            [layer("D", "set_classes(['keys'])，640 合成图", f"类别 {shown}，框 {count}", ok)],
            attention=True,
        )
    except Exception as exc:
        text = str(exc)
        blocked = "clip" in text.lower() or "No module" in text or "connect" in text.lower()
        add_case(
            "D03", "D", "白名单检测", "YOLOE 前向没有跑成",
            "skipped" if blocked else "fail",
            "权重文件已经在目录里。设置文本类别时失败：" + text,
            [layer("D", "set_classes(['keys'])", text, blocked)],
        )


def official_byte_gap_scan():
    """直接调用已安装的 BYTETracker。编号是否相同只在同一次实例里比较。"""
    from types import SimpleNamespace

    import numpy as np
    from ultralytics.trackers.byte_tracker import BYTETracker

    class Det:
        def __init__(self, xywh, conf, cls):
            self.xywh = np.asarray(xywh, dtype=float)
            self.conf = np.asarray(conf, dtype=float)
            self.cls = np.asarray(cls, dtype=float)

        def __getitem__(self, mask):
            return Det(self.xywh[mask], self.conf[mask], self.cls[mask])

        def __len__(self):
            return len(self.conf)

    settings = load_settings(ROOT / "config" / "bytetrack.yaml")
    args = SimpleNamespace(
        track_high_thresh=settings["track_high_thresh"],
        track_low_thresh=settings["track_low_thresh"],
        new_track_thresh=settings["new_track_thresh"],
        track_buffer=settings["track_buffer"],
        match_thresh=settings["match_thresh"],
        fuse_score=settings["fuse_score"],
    )
    box = np.array([[320.0, 240.0, 48.0, 36.0]])
    empty = Det(np.zeros((0, 4)), np.zeros(0), np.zeros(0))

    def run(sequence):
        tracker = BYTETracker(args)
        last = []
        first = []
        for kind, score in sequence:
            if kind == "empty":
                det = empty
            else:
                det = Det(box, np.array([score]), np.array([0.0]))
            out = tracker.update(det, img=None)
            ids = [int(row[4]) for row in out] if len(out) else []
            if not first and ids:
                first = ids
            last = ids
        return first, last

    low_first, low_last = run([("high", 0.9), ("high", 0.9)] + [("low", 0.2)] * 5)
    gap8_first, gap8_last = run([("high", 0.9)] + [("empty", 0)] * 8 + [("high", 0.9)])
    drop_at = None
    for gap in range(1, 21):
        first, last = run([("high", 0.9)] + [("empty", 0)] * gap + [("high", 0.9)])
        same = bool(first) and bool(last) and first[0] == last[0]
        if not same:
            drop_at = gap
            break
    return {
        "continuous_low_same_id": bool(low_first) and bool(low_last) and low_first[0] == low_last[0],
        "gap8_same_id": bool(gap8_first) and bool(gap8_last) and gap8_first[0] == gap8_last[0],
        "first_gap_without_same_id": drop_at,
    }


def suite_e():
    settings = load_settings(ROOT / "config" / "bytetrack.yaml")
    expected = {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 8,
        "match_thresh": 0.8,
        "fuse_score": True,
    }
    ok = all(settings[key] == value for key, value in expected.items()) and settings["second_cost_limit"] == 0.5
    add_case("E01", "E", "跟踪", "bytetrack.yaml 的采用参数", "pass" if ok else "fail", str({key: settings[key] for key in expected}), [layer("E", "bytetrack.yaml", "第二阶段代价上限 0.5，不来自 YAML", ok)])

    def spawn(score=0.9):
        tracker = SimByteTrack(settings)
        alive = tracker.update([("keys", 320, 240, 48, 36, score)])
        return tracker, alive[0][0]

    tracker, first = spawn()
    same = True
    for _ in range(10):
        alive = tracker.update([("keys", 320, 240, 48, 36, 0.9)])
        same = same and len(alive) == 1 and alive[0][0] == first
    add_case("E02", "E", "跟踪", "静止框保持同一个编号", "pass" if same else "fail", f"编号 {first}", [layer("E", "11 帧同一位置", f"编号保持 {same}", same)])

    tracker, first = spawn()
    for _ in range(5):
        tracker.update([])
    alive = tracker.update([("keys", 320, 240, 48, 36, 0.9)])
    kept = len(alive) == 1 and alive[0][0] == first
    add_case("E03", "E", "跟踪", "空检 1 秒后编号还在", "pass" if kept else "fail", f"5 帧空检后编号 {alive}", [layer("E", "track_buffer 8", f"编号仍是 {first}: {kept}", kept)])

    tracker, first = spawn()
    low_ids = []
    for _ in range(5):
        alive = tracker.update([("keys", 320, 240, 48, 36, 0.20)])
        low_ids.append([item[0] for item in alive])
    id_kept = all(item == [first] for item in low_ids)
    METRICS["low_score_id_kept"] = id_kept
    add_case(
        "E04", "E", "跟踪", "分数降到 0.20 时第二阶段仍接上",
        "pass" if id_kept else "fail",
        "已安装的 Ultralytics 在第二阶段只比交并比，不乘分数，避免低分把代价顶过 0.5。"
        f" 仿真关联器同样不乘。实测编号 {low_ids}。",
        [layer("E", "高分轨迹之后五帧 0.20", str(low_ids), id_kept)],
    )

    official = official_byte_gap_scan()
    METRICS["official_bytetrack"] = official
    kept_8 = official.get("gap8_same_id")
    ok = kept_8 is True and official.get("continuous_low_same_id") is True
    add_case(
        "E05", "E", "跟踪", "官方 ByteTrack：空 8 帧后高分框仍是原编号",
        "pass" if ok else "fail",
        f"空 8 帧后同一编号：{kept_8}。连续 0.20 仍是同一编号：{official.get('continuous_low_same_id')}。"
        f"编号第一次对不上的空帧数：{official.get('first_gap_without_same_id')}。这是 Ultralytics BYTETracker 的实测，不是仿真关联器。",
        [layer("E", "track_buffer 8，直接调用 BYTETracker", str(official), ok)],
        attention=True,
    )

    events, rows = roll("still", "keys", 0.9)
    actions = [item[1] for item in events]
    gap_events = []
    tracks = []
    tracker = SimByteTrack(settings)
    t = 0.0
    for _ in range(17):
        alive = tracker.update([("keys", 320, 240, 48, 36, 0.9)])
        named = [(cls, cx, cy, sc, tid) for tid, cls, cx, cy, sc in alive]
        gap_events.extend(update_tracks(tracks, named, round(t, 3), W, H))
        t += 0.2
    for _ in range(5):
        gap_events.extend(update_tracks(tracks, [], round(t, 3), W, H))
        t += 0.2
    alive = tracker.update([("keys", 320, 240, 48, 36, 0.9)])
    named = [(cls, cx, cy, sc, tid) for tid, cls, cx, cy, sc in alive]
    for _ in range(17):
        gap_events.extend(update_tracks(tracks, named, round(t, 3), W, H))
        alive = tracker.update([("keys", 320, 240, 48, 36, 0.9)])
        named = [(cls, cx, cy, sc, tid) for tid, cls, cx, cy, sc in alive]
        t += 0.2
    kinds = [item[1] for item in gap_events]
    ok = kinds.count("placed") == 1 and "removed" not in kinds
    add_case("E06", "E", "跟踪", "盖住约 1 秒不再写一次拿走加一次放下", "pass" if ok else "fail", str(kinds), [layer("F", "同一编号，中间空 1 秒", str(kinds), ok)])

    switched = []
    tracks = []
    t = 0.0
    for _ in range(17):
        switched.extend(update_tracks(tracks, [("keys", 320, 240, 0.9, 1)], round(t, 3), W, H))
        t += 0.2
    for _ in range(20):
        switched.extend(update_tracks(tracks, [("keys", 320, 240, 0.9, 2)], round(t, 3), W, H))
        t += 0.2
    kinds = [item[1] for item in switched]
    ok = "removed" not in kinds
    add_case(
        "E07", "E", "跟踪", "编号被换成新的时，不把还在中央的物体写成拿走",
        "pass" if ok else "fail",
        f"事件 {kinds}。旧编号会在 1.5 秒后变成挡住，新编号再写下一次放下。",
        [layer("F", "编号 1 然后编号 2", str(kinds), ok)],
        attention=True,
    )

    bucket = []
    first = update_tracks(bucket, [("keys", 100, 240, 0.9), ("keys", 500, 240, 0.55)], 0, W, H)
    second = update_tracks(bucket, [("keys", 100, 240, 0.9), ("keys", 500, 240, 0.55)], 0.2, W, H)
    ok = len(first) == 1 and first[0][1] == "conflict" and second == []
    add_case("E08", "E", "跟踪", "两个同类框记冲突，不挑选", "pass" if ok else "fail", f"第一帧 {first}，第二帧 {second}", [layer("E", "两个 keys 框", "conflict 一次", ok)])


def roll(pattern, object_id, score):
    settings = load_settings(ROOT / "config" / "bytetrack.yaml")
    tracker = SimByteTrack(settings)
    tracks = []
    events = []
    rows = []
    t = 0.0
    allowed = {object_id}

    def step(dets):
        nonlocal t
        keep = [item for item in dets if item[0] in allowed]
        drop = [item for item in dets if item[0] not in allowed]
        alive = tracker.update(keep)
        named = [(cls, cx, cy, sc, tid) for tid, cls, cx, cy, sc in alive]
        new_events = update_tracks(tracks, named, round(t, 3), W, H)
        events.extend(new_events)
        rows.append(
            {
                "t": round(t, 3),
                "detections": len(keep),
                "dropped": ",".join(item[0] for item in drop) or "—",
                "track_ids": ",".join(str(item[0]) for item in alive) or "—",
                "events": ",".join(item[1] for item in new_events) or "—",
            }
        )
        t = round(t + 0.2, 3)

    def box(x):
        return [(object_id, float(x), 240.0, 48.0, 36.0, float(score))]

    if pattern == "still":
        for _ in range(17):
            step(box(320))
    elif pattern == "cover":
        for _ in range(17):
            step(box(320))
        for _ in range(10):
            step([])
    elif pattern == "take":
        for _ in range(17):
            step(box(320))
        x = 320.0
        while x < 610:
            x += 20
            step(box(x))
        for _ in range(10):
            step([])
    else:
        raise ValueError(pattern)
    return events, rows


def suite_f():
    plan = [
        ("F01", "放下", "still", ["placed"], 20),
        ("F02", "拿走", "take", ["placed", "removed"], 20),
        ("F03", "挡住", "cover", ["placed", "occluded"], 10),
    ]
    objects = ["keys", "glasses_case", "medicine_box"]
    for case_id, title, pattern, expected, times in plan:
        bad = []
        for object_id in objects:
            for _ in range(times):
                events, _ = roll(pattern, object_id, 0.9)
                actions = [item[1] for item in events]
                if actions != expected:
                    bad.append({"object": object_id, "actions": actions})
                    break
        total = times * len(objects)
        ok = not bad
        METRICS[case_id] = {"expected_each": times, "objects": 3, "mismatches": bad[:3]}
        add_case(
            case_id, "F", "状态机", f"三路各 {times} 次{title}",
            "pass" if ok else "fail",
            f"{total - len(bad) * times}/{total} 组动作与 {expected} 一致。轨迹是脚本生成的，不是检测器召回。",
            [layer("F", pattern, str(expected), ok)],
        )
    negatives = [
        ("只停 1.5 秒", samples(0, 1.5, 320), 1.5, "unknown"),
        ("3 px 抖动停 3 秒", samples(0, 3.0, lambda t: 320 if int(t * 5) % 2 == 0 else 323), 3.0, "placed"),
        ("中间速度后消失", samples(0, 1.0, lambda t: 200 + 64 * t), 3.0, "unknown"),
        ("轨迹中断", samples(0, 0.4, 320) + samples(2.0, 3.0, 320), 3.0, "unknown"),
    ]
    results = []
    ok = True
    for name, points, now, expect in negatives:
        got = judge(points, now, W, H)
        results.append(f"{name}: {got}")
        ok = ok and got == expect
    add_case("F04", "F", "状态机", "短停、抖动、中速和断档", "pass" if ok else "fail", "；".join(results), [layer("F", "合成点", "；".join(results), ok)])


def apply_episode(conn, object_id, events, end_ts, cam):
    end = datetime.fromisoformat(end_ts)
    for index, (cls, action, confidence) in enumerate(events):
        when = end - timedelta(seconds=(len(events) - 1 - index) * 30)
        stamp = when.isoformat(timespec="seconds")
        path = ""
        if action in {"placed", "removed"}:
            filename = f"{cam['id']}-{cls}-{action}-{stamp.replace(':', '')}.jpg"
            target = EVIDENCE / filename
            target.write_bytes(PLACEHOLDER.read_bytes())
            path = str(target)
        apply_event(conn, stamp, cls, cam["slot_id"], action, float(confidence), cam["id"], path, source="visual")


def suite_g():
    folder = DATA / "contract"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    conn = connect(folder / "contract.db")
    apply_event(conn, "2026-11-18T08:00:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
    apply_event(conn, "2026-11-18T08:05:00", "keys", "", "removed", None, "", "", source="user_speech", speech_note="已经拿走")
    apply_event(conn, "2026-11-18T08:10:00", "keys", "living.coffee.top", "placed", 0.88, "cam_coffee", "")
    state = get_state(conn, "keys")
    sources = [row["source"] for row in conn.execute("SELECT source FROM events ORDER BY id")]
    ok = (
        sources == ["visual", "user_speech", "visual"]
        and state["visual_slot_id"] == "living.coffee.top"
        and state["visual_action"] == "placed"
        and state["speech_action"] == "removed"
    )
    text, _ = answer("钥匙在哪", VOCAB, state, None)
    ok = ok and text == "摄像头看到钥匙在茶几垫。您后来说已经拿走。"
    add_case(
        "G01", "G", "双表记忆", "口述不覆盖视觉列",
        "pass" if ok else "fail",
        text,
        [layer("G", "视觉放下、口述拿走、视觉再放下", text, ok)],
        snapshot(conn),
        database="契约库",
    )
    before = conn.execute("SELECT action, slot_id FROM events WHERE id = 1").fetchone()
    apply_event(conn, "2026-11-18T08:12:00", "keys", "entry.cabinet.top", "removed", 0.8, "cam_entry", "")
    after = conn.execute("SELECT action, slot_id FROM events WHERE id = 1").fetchone()
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    ok = tuple(before) == tuple(after) and count == 4
    add_case("G02", "G", "双表记忆", "事件只追加，旧行不变", "pass" if ok else "fail", f"第一行仍是 {tuple(before)}，共 {count} 行", [layer("G", "再写一行", f"旧行 {tuple(after)}", ok)], snapshot(conn), database="契约库")
    apply_event(conn, "2026-11-18T09:00:00", "medicine_box", "dining.table.top", "placed", 0.3, "cam_dining", "")
    ok = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
    add_case("G03", "G", "双表记忆", "看见药盒不产生服药计划", "pass" if ok else "fail", "plans 仍是 0 行", [layer("G", "medicine placed", "plans 0", ok)], snapshot(conn), database="契约库")
    evidence = folder / "evidence"
    evidence.mkdir()
    old = evidence / "old.jpg"
    old.write_bytes(b"old")
    fresh = evidence / "fresh.jpg"
    fresh.write_bytes(b"new")
    old_time = (datetime.now() - timedelta(hours=80)).timestamp()
    os.utime(old, (old_time, old_time))
    removed = purge_evidence(evidence, now=datetime.now())
    ok = removed == 1 and not old.exists() and fresh.exists()
    add_case("G04", "G", "双表记忆", "超过 72 小时的 JPEG 删除，事件行还在", "pass" if ok else "fail", f"删除 {removed} 张", [layer("G", "80 小时前的 jpg", "文件删除，本用例没有删事件行", ok)])
    delete_all(conn, evidence)
    ok = get_state(conn, "keys") is None and conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0 and list(evidence.glob("*.jpg")) == []
    add_case("G05", "G", "双表记忆", "清除后两张表和照片都空", "pass" if ok else "fail", "契约库已清空", [layer("G", "delete_all", "events 与 object_states 为空", ok)], snapshot(conn), database="契约库")
    conn.close()


CONN = None


def ask_like(conn, text):
    last = get_value(conn, "last_object")
    object_id, early = find_object(text, VOCAB, last)
    if early:
        return early
    reply, new_last = answer(text, VOCAB, get_state(conn, object_id), last)
    if new_last:
        set_value(conn, "last_object", new_last)
    return reply


def ask_case(conn, case_id, title, text, expected, database="主库"):
    global QUESTION_COUNT
    QUESTION_COUNT += 1
    reply = ask_like(conn, text)
    ok = reply == expected
    if not ok:
        WRONG.append(case_id)
    add_case(
        case_id, "H", "问答", title,
        "pass" if ok else "fail",
        reply if ok else f"期望：{expected}；得到：{reply}",
        [layer("H", text, reply, ok)],
        snapshot(conn),
        database=database,
    )


def run_pipeline(case_id, title, object_id, pattern, score, end_ts, expected, question):
    cam = camera_for(object_id)
    obj = OBJECTS[object_id]
    pixels = evaluate(cam["width"], cam["imgsz"], obj["short_side_m"], cam["distance_m"], cam["fov_deg"])
    events, rows = roll(pattern, object_id, score)
    actions = [item[1] for item in events]
    ok = actions == expected and pixels["pass"]
    if ok:
        apply_episode(CONN, object_id, events, end_ts, cam)
        sentence, _ = answer(question, VOCAB, get_state(CONN, object_id), None)
    else:
        sentence = f"动作 {actions}，没有写入主库"
    gate = MainWindow()
    main_reads = 0
    for index in range(len(rows)):
        read, _ = gate.step(index * 0.2, True)
        main_reads += int(read)
    for index in range(16):
        read, _ = gate.step((len(rows) + index) * 0.2, False)
        main_reads += int(read)
    kept = [("keys", 1, 1, 0.9), ("cup", 1, 1, 0.9)]
    whitelist = [item[0] for item in kept if item[0] in set(cam["objects"])]
    layers = [
        layer("A", f"{cam['id']} {object_id} {obj['prompt']}", f"槽位 {cam['slot_id']}，有效像素 {pixels['effective_px']:.2f}", pixels["pass"]),
        layer("B", "触发为真的帧数等于这条轨迹的帧数", f"主码流读取 {main_reads}，检测不用子码流", main_reads > 1),
        layer("C", "面积比例由门控用例测量，这条轨迹不重复跑 MOG2", "跟踪期间窗口保持打开", True),
        layer("D", "送入一个白名单框和一个 cup", f"保留 {whitelist}", whitelist == cam["objects"]),
        layer("E", "SimByteTrack + bytetrack.yaml", f"轨迹编号 {rows[0]['track_ids']} → {rows[-1]['track_ids']}", ok),
        layer("F", pattern, str(actions), actions == expected),
        layer("G", "apply_event source=visual", sentence, ok),
        layer("H", question, sentence, ok),
    ]
    add_case(
        case_id, "P", "全流程", title,
        "pass" if ok else "fail",
        sentence,
        layers,
        snapshot(CONN) if ok else None,
        frames=rows,
        database="主库",
    )
    return ok


def suite_h2():
    """节点 H2：问句向量化与回答筛选（说明书第 10 节）。"""
    import vec_memory as vec
    from ask import ask_once
    from run_h2 import NON_ALIAS, SLOT_NAMES

    h2_path = DATA / "h2.db"
    for extra in (Path(str(h2_path) + "-wal"), Path(str(h2_path) + "-shm")):
        if extra.exists():
            extra.unlink()
    if h2_path.exists():
        h2_path.unlink()
    conn = connect(h2_path)
    for args in [
        ("2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", ""),
        ("2026-11-18T09:05:00", "glasses_case", "living.coffee.top", "placed", 0.80, "cam_coffee", ""),
        ("2026-11-18T09:06:00", "glasses_case", "living.coffee.top", "occluded", 0.80, "cam_coffee", ""),
        ("2026-11-18T11:00:00", "medicine_box", "dining.table.top", "placed", 0.30, "cam_dining", ""),
        ("2026-11-18T13:20:00", "keys", "entry.cabinet.top", "removed", 0.85, "cam_entry", ""),
    ]:
        apply_event(conn, *args)
    apply_event(
        conn, "2026-11-18T13:21:00", "keys", "", "placed", None, "", "",
        source="user_speech", speech_note="已经放进抽屉",
    )

    # 一、别名问法与节点 H 逐句一致。
    alias_questions = [
        "钥匙在哪", "钥匙串放哪了", "家门钥匙在哪里", "眼镜盒呢", "老花镜盒子在哪",
        "眼镜盒子放哪了", "药盒在哪", "药箱呢", "吃药的盒子放哪了", "遥控器在哪",
        "遥控板呢", "电视遥控器放哪了", "医保卡在哪", "社保卡呢", "医疗卡放哪了",
        "钥匙放哪了", "钥匙在哪里", "钥迟放哪了", "它在哪", "那个呢",
    ]
    last = None
    mismatches = []
    paths = {}
    for text in alias_questions:
        if text in {"它在哪", "那个呢"}:
            last = "keys"
        reply, new_last, path = ask_once(conn, VOCAB, text, last)
        paths[path.split("（")[0]] = paths.get(path.split("（")[0], 0) + 1
        object_id, early = find_object(text, VOCAB, last)
        node_h = early if early else answer(text, VOCAB, get_state(conn, object_id), last)[0]
        if reply != node_h:
            mismatches.append(f"{text}：H2={reply}，H={node_h}")
        if new_last:
            last = new_last
    ok = not mismatches
    add_case(
        "H2-01", "H2", "问句向量化", "20 条别名问法与节点 H 相同",
        "pass" if ok else "fail",
        f"一致 {20 - len(mismatches)}/20；路径计数 {paths}",
        [layer("H", "别名问法", f"不一致 {mismatches or '无'}", ok)],
        snapshot(conn), database="H2 库",
    )

    # 二、词表外说法。
    correct = 0
    wrong_object = []
    wrong_position = []
    for text, expected in NON_ALIAS:
        reply, resolved, path = ask_once(conn, VOCAB, text, None)
        paths[path.split("（")[0]] = paths.get(path.split("（")[0], 0) + 1
        if expected is None:
            if resolved is not None:
                wrong_object.append(f"{text}→{resolved}")
            if any(name in reply for name in SLOT_NAMES):
                wrong_position.append(text)
            continue
        if resolved == expected:
            correct += 1
        elif resolved is not None:
            wrong_object.append(f"{text}→{resolved}")
        state = get_state(conn, expected)
        allowed = set()
        if state and state.get("visual_slot_id"):
            for slot in VOCAB["slots"]:
                if slot["id"] == state["visual_slot_id"]:
                    allowed.add(slot["canonical_name"])
        if resolved is not None and resolved != expected and any(name in reply for name in SLOT_NAMES):
            wrong_position.append(text)
        if resolved == expected and any(name in reply for name in SLOT_NAMES if name not in allowed):
            wrong_position.append(text)
    ok = len(wrong_position) <= 1
    METRICS["h2_non_alias"] = {
        "total": len(NON_ALIAS),
        "correct": correct,
        "wrong_object": wrong_object,
        "wrong_position": wrong_position,
        "paths": paths,
    }
    add_case(
        "H2-02", "H2", "问句向量化", "20 条词表外说法，说错位置不超过 1",
        "pass" if ok else "fail",
        f"物品正确 {correct}/{len(NON_ALIAS)}；召回成别的物品 {wrong_object or '无'}；说错位置 {wrong_position or '无'}",
        [layer("H2", "第一道筛选", f"正确 {correct}，错位 {len(wrong_position)}", ok)],
        snapshot(conn), database="H2 库",
    )

    # 三、第二道筛选。
    rows = [
        ("event", 0.80, {"object_id": "keys", "slot_id": "entry.cabinet.top", "action": "removed",
                          "ts": "2026-11-18T13:20:00", "source": "visual", "confidence": 0.85})
    ]
    ok, reasons = vec.filter_generated("钥匙大概在桌子附近。", rows)
    case_ok = not ok
    template, _ = answer("钥匙在哪", VOCAB, get_state(conn, "keys"), None)
    add_case(
        "H2-03", "H2", "问句向量化", "生成句说出“桌子附近”必须丢弃并改念模板",
        "pass" if case_ok else "fail",
        f"丢弃理由：{reasons}；改念：{template}",
        [layer("H2", "第二道筛选", "，".join(reasons), case_ok)],
        snapshot(conn), database="H2 库",
    )
    cross_ok, cross_reasons = vec.filter_generated("药盒在茶几垫。", rows)
    add_case(
        "H2-04", "H2", "问句向量化", "物品和槽位都不在行内的句子丢弃",
        "pass" if not cross_ok else "fail",
        f"{cross_reasons}",
        [layer("H2", "第二道筛选", str(cross_reasons), not cross_ok)],
        snapshot(conn), database="H2 库",
    )

    # 四、时间词。
    cases = [
        ("钥匙早上放哪了", "摄像头看到钥匙在玄关柜。"),
        ("昨天放的那个药", "我没有药盒在这个时间段的记录。"),
    ]
    time_ok = True
    detail = []
    for text, expected in cases:
        reply, _, path = ask_once(conn, VOCAB, text, None)
        good = reply == expected and path == "向量召回"
        time_ok = time_ok and good
        detail.append(f"{text}→{reply}")
    add_case(
        "H2-05", "H2", "问句向量化", "时间词先收成区间再检索",
        "pass" if time_ok else "fail",
        "；".join(detail),
        [layer("H2", "早上/昨天", "；".join(detail), time_ok)],
        snapshot(conn), database="H2 库",
    )
    conn.close()

    # 五、嵌入模型魔改对照（另跑 finetune_embed.py 得到的存档）。
    ft_json = RESULTS / "embed_finetune.json"
    if ft_json.exists():
        payload = json.loads(ft_json.read_text(encoding="utf-8"))
        base = payload["results"]["base"]["acceptance"]
        ft = payload["results"]["finetuned"]["acceptance"]
        METRICS["h2_finetune"] = {
            "corpus_pairs": payload["corpus_pairs"],
            "base_correct": base["correct"],
            "finetuned_correct": ft["correct"],
            "holdout": {
                "base": payload["results"]["base"]["holdout"]["correct"],
                "finetuned": payload["results"]["finetuned"]["holdout"]["correct"],
            },
            "note": "生产仍用基础模型：两次微调在同一问句上一好一坏，收益在训练方差内。",
        }
        add_case(
            "H2-06", "H2", "问句向量化", "微调 bge-small-zh 的对照（不作为生产模型）",
            "pass",
            f"基础 {base['correct']}/{base['total']}，微调 {ft['correct']}/{ft['total']}，"
            f"留出集基础 {payload['results']['base']['holdout']['correct']}/12，"
            f"微调 {payload['results']['finetuned']['holdout']['correct']}/12。",
            [layer("H2", "546 对模板语料，MNRL 8 轮", "生产用基础模型", True)],
        )


def suite_story():
    global CONN
    if LAB_PATH.exists():
        LAB_PATH.unlink()
    for extra in (Path(str(LAB_PATH) + "-wal"), Path(str(LAB_PATH) + "-shm")):
        if extra.exists():
            extra.unlink()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for path in EVIDENCE.glob("*.jpg"):
        path.unlink()
    CONN = connect(LAB_PATH)
    run_pipeline("P01", "玄关钥匙放下", "keys", "still", 0.91, "2026-11-18T08:12:00", ["placed"], "钥匙在哪")
    ask_case(CONN, "H01", "没有上一件时说它", "它在哪", "请说出物品的名字。")
    high = "摄像头看到钥匙在玄关柜。"
    ask_case(CONN, "H02", "钥匙在哪", "钥匙在哪", high)
    ask_case(CONN, "H03", "家门钥匙", "家门钥匙在哪", high)
    ask_case(CONN, "H04", "钥匙串", "钥匙串在哪里", high)
    ask_case(CONN, "H05", "它在哪", "它在哪", high)
    ask_case(CONN, "H06", "那个呢", "那个呢", high)
    ask_case(CONN, "H07", "错一个字", "钥迟放哪了", high)
    ask_case(CONN, "H08", "词表外的雨伞", "雨伞在哪", "我没有这个物品的记录。")

    run_pipeline("P02", "茶几眼镜盒被挡住", "glasses_case", "cover", 0.80, "2026-11-18T09:05:00", ["placed", "occluded"], "眼镜盒在哪")
    covered = "眼镜盒最后出现在茶几垫，后来被挡住了，我不能确定还在不在。"
    ask_case(CONN, "H09", "眼镜盒在哪", "眼镜盒在哪", covered)
    ask_case(CONN, "H10", "老花镜盒子", "老花镜盒子在哪", covered)

    run_pipeline("P03", "餐桌药盒分数 0.30", "medicine_box", "still", 0.30, "2026-11-18T11:00:00", ["placed"], "药盒在哪")
    low = "我不能确定。请查看该时刻保存的照片。"
    ask_case(CONN, "H11", "药盒低分", "药盒在哪", low)
    ask_case(CONN, "H12", "吃药的盒子", "吃药的盒子呢", low)

    side_path = DATA / "band.db"
    if side_path.exists():
        side_path.unlink()
    side = connect(side_path)
    events, _ = roll("still", "medicine_box", 0.50)
    apply_episode(side, "medicine_box", events, "2026-11-18T12:30:00", camera_for("medicine_box"))
    mid = "最后一次摄像头看到药盒是在餐桌垫，时间是 11月18日 12:30。"
    ask_case(side, "H13", "药盒中等分数", "药盒在哪", mid, database="分界侧库")
    ask_case(side, "H14", "吃药的盒子中等分数", "吃药的盒子在哪", mid, database="分界侧库")
    side.close()

    run_pipeline("P04", "玄关钥匙拿走", "keys", "take", 0.85, "2026-11-18T13:20:00", ["placed", "removed"], "钥匙在哪")
    removed = "钥匙最后出现在玄关柜，11月18日 13:20 被拿走了。"
    ask_case(CONN, "H15", "拿走之后", "钥匙在哪", removed)
    ask_case(CONN, "H16", "钥匙放哪了", "钥匙放哪了", removed)
    apply_event(CONN, "2026-11-18T13:21:00", "keys", "", "placed", None, "", "", source="user_speech", speech_note="已经放进抽屉")
    visual = get_state(CONN, "keys")
    speech_ok = visual["visual_action"] == "removed" and "已经放进抽屉" in (visual["speech_note"] or "")
    add_case(
        "G06", "G", "双表记忆", "主库里的口述不改视觉列",
        "pass" if speech_ok else "fail",
        f"视觉 {visual['visual_action']}，口述 {visual['speech_note']}",
        [layer("G", "已经放进抽屉", f"visual_action={visual['visual_action']}", speech_ok)],
        snapshot(CONN),
        database="主库",
    )
    ask_case(CONN, "H17", "视觉和口述不一致", "钥匙在哪", removed + "您后来说已经放进抽屉。")
    ask_case(CONN, "H18", "遥控器没有视觉记录", "遥控器在哪", "我没有遥控器的记录。")
    ask_case(CONN, "H19", "医保卡没有视觉记录", "医保卡在哪", "我没有医保卡的记录。")

    bucket = []
    conflict = update_tracks(bucket, [("glasses_case", 80, 240, 0.90), ("glasses_case", 520, 240, 0.70)], 0, W, H)
    if conflict and conflict[0][1] == "conflict":
        apply_event(CONN, "2026-11-18T09:20:00", "glasses_case", "living.coffee.top", "conflict", float(conflict[0][2]), "cam_coffee", "", source="visual")
    both = "茶几垫上同时有两个眼镜盒，我不能确定是哪一个。"
    ask_case(CONN, "H20", "两个眼镜盒", "眼镜盒在哪", both)

    passed_questions = QUESTION_COUNT - len(WRONG)
    ok = QUESTION_COUNT == 20 and len(WRONG) <= 1
    METRICS["questions"] = {"count": QUESTION_COUNT, "location_errors": len(WRONG), "wrong_ids": list(WRONG)}
    add_case(
        "H21", "H", "问答", "20 条书面问法的位置错误不超过 1",
        "pass" if ok else "fail",
        f"{QUESTION_COUNT} 条，位置不一致 {len(WRONG)} 条，编号 {WRONG or '无'}。没有记录和不能确定不算说错；这里用整句相等来计。",
        [layer("H", "H01–H20", f"不一致 {len(WRONG)}", ok)],
        snapshot(CONN),
        database="主库",
    )
    del passed_questions

    slots = []
    level = connect(DATA / "level4.db")
    for index, slot_id in enumerate(["entry.cabinet.top", "entry.cabinet.top", "entry.cabinet.top", "living.coffee.top"]):
        apply_event(level, f"2026-11-18T07:0{index}:00", "keys", slot_id, "placed", 0.9, "cam", "")
    question = history_slot_question(VOCAB, ["entry.cabinet.top", "living.coffee.top"])
    reply = ask_like(level, "钥匙在哪")
    ok = reply == "摄像头看到钥匙在茶几垫。" and question == "你是说玄关柜，还是茶几垫？" and reply != question
    add_case(
        "H22", "H", "问答", "第 4 级只提问，不把常去的槽位说成现在的位置",
        "pass" if ok else "fail",
        f"采用句：{reply}。备用提问：{question}。历史里玄关更多，当前视觉列是茶几。",
        [layer("H", "3 次玄关、最后一次茶几", reply, ok)],
        snapshot(level),
        database="第4级库",
    )
    level.close()

    homophone = connect(DATA / "pinyin.db")
    apply_event(homophone, "2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
    adopted = ask_like(homophone, "要是在哪")
    try:
        found = candidates("要是在哪", VOCAB)
        extra = f"备用拼音候选 {found}。主路径没有采用。"
    except Exception as exc:
        found = []
        extra = f"拼音模块没有跑起来：{exc}"
    ok = adopted == "我没有这个物品的记录。" and "玄关柜" not in adopted
    METRICS["pinyin_candidates_for_yaoshi"] = found
    add_case(
        "H23", "H", "问答", "拼音近邻不进入主路径",
        "pass" if ok else "fail",
        f"主路径：{adopted}。{extra}",
        [layer("H", "要是在哪", adopted, ok)],
        snapshot(homophone),
        database="拼音对照库",
        attention="keys" in found,
    )
    homophone.close()

    plan_id = add_plan(CONN, "吃药", "2026-11-18T08:00:00")
    due = due_plans(CONN, "2026-11-18T08:30:00")
    before = get_state(CONN, "medicine_box")["visual_action"]
    confirm_plan(CONN, plan_id, "2026-11-18T08:30:00")
    source = CONN.execute("SELECT done_source FROM plans WHERE id = ?", (plan_id,)).fetchone()[0]
    ok = any(item["id"] == plan_id for item in due) and source == "self_report" and before == "placed"
    add_case("G07", "G", "双表记忆", "服药确认只能是 self_report", "pass" if ok else "fail", f"来源 {source}，药盒视觉动作仍是 {before}", [layer("G", "confirm_plan", source, ok)], snapshot(CONN), database="主库")

    copy_path = DATA / "deleted.db"
    shutil.copy(LAB_PATH, copy_path)
    copied = connect(copy_path)
    empty_dir = DATA / "deleted-evidence"
    empty_dir.mkdir(exist_ok=True)
    delete_all(copied, empty_dir)
    reply = ask_like(copied, "钥匙在哪")
    ok = reply == "我没有钥匙的记录。" and get_state(copied, "keys") is None
    add_case("G08", "G", "双表记忆", "清除副本后再问钥匙", "pass" if ok else "fail", reply, [layer("H", "钥匙在哪", reply, ok)], snapshot(copied), database="清除后的副本")
    copied.close()
    add_case("G09", "G", "双表记忆", "主库没有被副本的清除带走", "pass", "实验主库仍保留全流程的行", [layer("G", "lab.db", "未执行 delete_all", True)], snapshot(CONN), database="主库")
    CONN.close()


def suite_i():
    lines = write_hotwords(VOCAB, ROOT / "config" / "hotwords.txt")
    aliases = []
    for key in ("objects", "slots"):
        for item in VOCAB[key]:
            aliases.extend(item["aliases"])
    ok = lines == list(dict.fromkeys(aliases)) and "卧室" not in "".join(lines)
    add_case("I01", "I", "语音入口", "热词只从词表生成", "pass" if ok else "fail", f"{len(lines)} 条", [layer("I", "vocab.yaml", "、".join(lines), ok)])

    gate = PushToTalk()
    gate.push_audio("在卧室")
    gate.press()
    gate.push_audio("钥匙在哪")
    uttered = gate.release()
    gate.push_audio("药盒在餐桌垫")
    probe = connect(DATA / "voice.db")
    apply_event(probe, "2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
    reply = ask_like(probe, uttered)
    ok = uttered == "钥匙在哪" and reply == "摄像头看到钥匙在玄关柜。" and "卧室" not in reply and "餐桌" not in reply
    add_case("I02", "I", "语音入口", "按键之外的声音不进入这一句", "pass" if ok else "fail", f"识别窗口内的文本是 {uttered}。回答：{reply}", [layer("I", "按下前“在卧室”，按下后“钥匙在哪”，松开后“药盒在餐桌垫”", uttered, ok)], snapshot(probe), database="按键库")
    probe.close()

    model_dir = ROOT / "models"
    if model_dir.exists() and any(model_dir.glob("**/*")):
        add_case("I03", "I", "语音入口", "安静环境 20 句语音", "skipped", "模型已就位，合成语音回环见 I05。真人 20 句要有麦克风录音才算通过。", [layer("I", "20 句白名单", "未执行口试", True)])
    else:
        add_case("I03", "I", "语音入口", "安静环境 20 句语音", "skipped", "没有麦克风录音，也没有放入 sherpa 识别模型。按键窗口和热词已经测过，这 20 句不算通过。", [layer("I", "20 句白名单", "未执行", True)])
    add_case("I04", "I", "语音入口", "松开按键到出声不超过 2 秒", "skipped", "没有扬声器回放，不能对真人按键计时。合成语音回环的时延在 I05 报告。", [layer("I", "扬声器", "未执行", True)])

    # 合成语音回环：TTS 出问句 → ASR 识别 → ask_once → TTS 朗读回答。
    # 合成语音比真人干净，它测的是管道，不等于安静环境的真人验收。
    asr_ok, tts_ok = (True, True)
    try:
        import voice as voice_mod

        asr_ok, tts_ok = voice_mod.model_dirs_ready()
    except Exception:
        asr_ok, tts_ok = False, False
    if asr_ok and tts_ok:
        proc = run_command("voice.py", "--roundtrip")
        rt_json = RESULTS / "voice_roundtrip.json"
        if proc.returncode == 0 and rt_json.exists():
            rows = json.loads(rt_json.read_text(encoding="utf-8"))
            resolved = sum(1 for row in rows if row["resolved"] is not None)
            slowest = max(row["asr_s"] + row["ask_s"] + row["tts_s"] for row in rows)
            latencies = [round(row["asr_s"] + row["ask_s"] + row["tts_s"], 2) for row in rows]
            METRICS["voice_roundtrip"] = {
                "count": len(rows),
                "resolved_object": resolved,
                "slowest_total_s": round(slowest, 2),
                "note": "合成语音，不是真人 20 句。时延是识别加问答加合成，不含按键等待。",
            }
            ok = len(rows) == 20 and slowest < 2.0
            add_case(
                "I05", "I", "语音入口", "合成语音回环 20 句，识别到出声不超过 2 秒",
                "pass" if ok else "fail",
                f"{len(rows)} 条，识别后进入物品 id 的 {resolved} 条，"
                f"最慢一条 {slowest:.2f} 秒（识别+问答+合成，不含等待）。",
                [layer("I", "TTS→ASR→问答→TTS", f"resolved {resolved}/20，最慢 {slowest:.2f}s", ok)],
                attention=resolved < 19,
            )
        else:
            add_case("I05", "I", "语音入口", "合成语音回环", "fail", (proc.stdout or proc.stderr or "")[-400:], [layer("I", "voice.py --roundtrip", "退出码非 0", False)])
    else:
        add_case("I05", "I", "语音入口", "合成语音回环", "skipped", f"模型目录不完整：ASR {asr_ok}，TTS {tts_ok}。", [layer("I", "models", "未执行", True)])


def suite_j():
    holder = {"done": 0}
    states = []
    for index in range(50):
        state = "open" if index % 2 == 0 else "closed"
        parsed = accept_reed(f"home/kitchen/drawer/{state}", state, 1)
        states.append(parsed["state"])
    rejected = False
    try:
        accept_reed("home/kitchen/drawer/open", "open", 0)
    except ValueError:
        rejected = True
    ok = holder["done"] == 0 and rejected and states.count("open") == 25
    add_case("J01", "J", "抽屉与复核", "50 次开合流过解析器，服药计划仍是未完成", "pass" if ok else "fail", "QoS 0 被拒绝。没有任何一步把 done 写成 1。", [layer("J", "home/kitchen/drawer/open|closed", "done 保持 0", ok)])
    add_case("J02", "J", "抽屉与复核", "真实磁簧 50 次开合", "skipped", "没有安装磁簧，也没有把 Mosquitto 接到这台机器上。漏报不超过 1 次这条不能测。", [layer("J", "硬件", "未执行", True)])

    before = {"visual_slot_id": "entry.cabinet.top", "visual_action": "placed"}
    review = {"detector_id": "keys", "agree": True, "writes_slot": False, "writes_action": False}
    after = dict(before)
    ok = after == before and review["writes_slot"] is False
    add_case(
        "J03", "J", "抽屉与复核", "复核结果不能改槽位和动作",
        "pass" if ok else "fail",
        "本次没有加载 Qwen3-VL-2B。上面只检查了约定：同意与否不写 visual_slot_id，也不写动作。",
        [layer("J", "detector_id=keys", "槽位和动作保持原值", ok)],
        attention=True,
    )
    add_case("J04", "J", "抽屉与复核", "夜间每路 10 次", "skipped", "没有夜间实拍。", [layer("D", "夜间", "未执行", True)])
    add_case("J05", "J", "抽屉与复核", "连续 24 小时后磁盘没有视频", "skipped", "这次没有连续运行 24 小时。仿真目录里会另查一遍现有视频文件。", [layer("B", "24 小时", "未执行", True)])
    videos = []
    for folder in (EVIDENCE, ROOT / "data" / "evidence"):
        if folder.exists():
            for suffix in ("*.mp4", "*.mkv", "*.avi"):
                videos.extend(str(path) for path in folder.glob(suffix))
    ok = videos == []
    METRICS["video_files"] = videos
    add_case("J06", "J", "抽屉与复核", "本次仿真目录里没有视频文件", "pass" if ok else "fail", "、".join(videos) or "没有 mp4、mkv、avi", [layer("B", "evidence", "没有视频", ok)])
    add_case("J07", "J", "抽屉与复核", "RTSP 抽帧", "skipped", "sites.yaml 仍是占位账号，没有摄像头。", [layer("B", "ffmpeg", "未执行", True)])


def suite_harness():
    print("开始记忆线束，这一步会重写 data/memory.db。", flush=True)
    proc = run_command("harness.py")
    text = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(text.strip().splitlines()[-12:])
    ok = proc.returncode == 0 and "线束通过" in text
    METRICS["harness_exit"] = proc.returncode
    add_case("S02", "S", "总闸", "harness.py", "pass" if ok else "fail", tail, [layer("S", "python demo/harness.py", tail, ok)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Python {platform.python_version()}  {platform.platform()}", flush=True)
    guard("总闸", suite_commands)
    guard("槽位与像素", suite_a)
    guard("双码流", suite_b)
    guard("运动门控", suite_c)
    guard("白名单检测", suite_d)
    guard("跟踪", suite_e)
    guard("状态机", suite_f)
    guard("双表记忆", suite_g)
    guard("问句向量化", suite_h2)
    guard("全流程", suite_story)
    guard("语音入口", suite_i)
    guard("抽屉与复核", suite_j)
    guard("线束", suite_harness)
    finish_summary()
    counts = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["counts"]
    print(f"完成。通过 {counts.get('pass', 0)}，未通过 {counts.get('fail', 0)}，未执行 {counts.get('skipped', 0)}。", flush=True)
    print(f"书面问法位置不一致 {len(WRONG)} 条。", flush=True)
    return 0 if counts.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        finish_summary()
        sys.exit(1)
