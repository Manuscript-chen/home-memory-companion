"""闭集对照。只在合成图上训练，数字不能写成这户台面的准确率。

训练集和验证集按生成批次切开，不把同一段连续帧随机拆开。
开放词汇另记一次，不和闭集 mAP 合成一个百分数。
"""

import json
import os
import random
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"
RUNS = HERE / "runs"
DATA = HERE / "data" / "closed_set"
RESULTS = HERE / "results"

CLASSES = ["keys", "eyeglasses_case", "medicine_box"]
PROMPTS = ["keys", "eyeglasses case", "medicine box"]


def draw(kind, rng):
    image = np.full((480, 640, 3), int(rng.integers(150, 210)), np.uint8)
    if kind == 0:
        cx, cy = int(rng.integers(80, 560)), int(rng.integers(80, 400))
        w, h = int(rng.integers(36, 70)), int(rng.integers(16, 28))
        cv2.ellipse(image, (cx, cy), (w // 2, h // 2), 0, 0, 360, (20, 20, 20), -1)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    elif kind == 1:
        cx, cy = int(rng.integers(100, 520)), int(rng.integers(100, 380))
        r = int(rng.integers(16, 28))
        cv2.circle(image, (cx - r, cy), r, (40, 40, 90), -1)
        cv2.circle(image, (cx + r, cy), r, (40, 40, 90), -1)
        box = (cx - 2 * r, cy - r, cx + 2 * r, cy + r)
    else:
        cx, cy = int(rng.integers(100, 500)), int(rng.integers(90, 360))
        w, h = int(rng.integers(70, 120)), int(rng.integers(46, 80))
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 70, 160), -1)
        cv2.rectangle(image, (x1, cy - 4), (x2, cy + 4), (230, 230, 230), -1)
        box = (x1, y1, x2, y2)
    return image, box


def yolo_line(kind, box, width=640, height=480):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{kind} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_dataset():
    if DATA.exists():
        shutil.rmtree(DATA)
    rng = np.random.default_rng(7)
    splits = {"train": (0, 1), "val": (2,)}
    counts = {"train": 0, "val": 0}
    for split, batches in splits.items():
        (DATA / "images" / split).mkdir(parents=True)
        (DATA / "labels" / split).mkdir(parents=True)
        for batch in batches:
            for index in range(16):
                kind = int((batch * 16 + index) % 3)
                image, box = draw(kind, rng)
                stem = f"b{batch}_{index:02d}_c{kind}"
                cv2.imwrite(str(DATA / "images" / split / f"{stem}.jpg"), image)
                (DATA / "labels" / split / f"{stem}.txt").write_text(yolo_line(kind, box), encoding="utf-8")
                counts[split] += 1
    data_yaml = {
        "path": str(DATA),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASSES)},
    }
    (DATA / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")
    return counts


def train_once(device, epochs):
    from ultralytics import YOLO

    model = YOLO(str(MODELS / "yolo26n.pt"))
    result = model.train(
        data=str(DATA / "data.yaml"),
        epochs=epochs,
        imgsz=320,
        batch=8 if device != "cpu" else 4,
        device=device,
        workers=0,
        project=str(RUNS),
        name="closed_set",
        seed=7,
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    metrics = getattr(result, "results_dict", None) or {}
    box = getattr(result, "box", None)
    summary = {
        "device": str(device),
        "epochs": epochs,
        "map50": float(getattr(box, "map50", metrics.get("metrics/mAP50(B)", 0.0)) or 0.0),
        "map50_95": float(getattr(box, "map", metrics.get("metrics/mAP50-95(B)", 0.0)) or 0.0),
    }
    return summary


def open_vocab_hits():
    weight = MODELS / "yoloe-26n-seg.pt"
    clip = MODELS / "mobileclip2_b.ts"
    if not weight.exists() or not clip.exists():
        return {"ran": False, "reason": "缺少 YOLOE 权重或 mobileclip2_b.ts"}
    from ultralytics import YOLOE

    model = YOLOE(str(weight))
    model.set_classes(PROMPTS)
    shown = list(model.names.values())[: len(PROMPTS)]
    hits = 0
    total = 0
    per_class = {name: {"hit": 0, "n": 0} for name in CLASSES}
    for image_path in sorted((DATA / "images" / "val").glob("*.jpg")):
        label = (DATA / "labels" / "val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").split()
        kind = int(label[0])
        cx, cy, w, h = map(float, label[1:])
        box = ((cx - w / 2) * 640, (cy - h / 2) * 480, (cx + w / 2) * 640, (cy + h / 2) * 480)
        result = model.predict(str(image_path), imgsz=640, conf=0.25, verbose=False)[0]
        matched = False
        if result.boxes is not None:
            for pred in result.boxes:
                if int(pred.cls) != kind:
                    continue
                x1, y1, x2, y2 = pred.xyxy[0].tolist()
                if iou(box, (x1, y1, x2, y2)) >= 0.5:
                    matched = True
                    break
        total += 1
        per_class[CLASSES[kind]]["n"] += 1
        if matched:
            hits += 1
            per_class[CLASSES[kind]]["hit"] += 1
    return {
        "ran": True,
        "class_order": shown,
        "images": total,
        "iou50_hits": hits,
        "per_class": per_class,
        "note": "这是合成图上的提示命中次数，不是 LVIS AP，也不和闭集 mAP 相加。",
    }


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    os.chdir(MODELS)
    random.seed(7)
    np.random.seed(7)
    counts = build_dataset()
    report = {
        "ready": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "counts": counts,
            "split": "batch 0 和 batch 1 训练，batch 2 验证。不是从同一段视频里随机抽帧。",
            "what": "程序画的椭圆、双圆和方块。不是钥匙、眼镜盒或药盒的照片。",
        },
    }
    try:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        device = "cpu"
        report["torch_error"] = str(exc)
    epochs = 20
    try:
        report["closed_set"] = train_once(device, epochs)
    except Exception as exc:
        report["closed_set_error"] = str(exc)
        if device != "cpu":
            try:
                report["closed_set"] = train_once("cpu", 4)
                report["closed_set"]["fallback"] = "GPU 训练失败后改在 CPU 上跑了 4 个周期。"
            except Exception as second:
                report["cpu_error"] = str(second)
                report["traceback"] = traceback.format_exc()
    try:
        report["open_vocab"] = open_vocab_hits()
    except Exception as exc:
        report["open_vocab"] = {"ran": False, "reason": str(exc)}
    report["ready"] = "closed_set" in report
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = RESULTS / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["training"] = {
            "ready": report["ready"],
            "closed_set": report.get("closed_set"),
            "open_vocab_hits": (report.get("open_vocab") or {}).get("iou50_hits"),
            "open_vocab_images": (report.get("open_vocab") or {}).get("images"),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "traceback"}, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
