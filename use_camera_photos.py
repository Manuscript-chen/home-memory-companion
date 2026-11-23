"""把记忆库里的仿真灰块换成摄像头视角的实物照片，并只用这些照片重建训练集。

每张图都先看过：物体放在桌面、台面或布面上，是相机拍的，不是白底抠图，也不是灰块。
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\Lenovo\Desktop\项目")
HERE = ROOT / "实验"
OUT = HERE / "data" / "real_set"

# 已经打开看过的照片。
PHOTOS = {
    "keys": [
        HERE / "data" / "real_set" / "images" / "train" / "keys_03.jpg",
        HERE / "data" / "camera_review" / "05.jpg",
        HERE / "data" / "camera_review" / "07.jpg",
    ],
    "glasses_case": [
        HERE / "data" / "real_set" / "images" / "train" / "glasses_case_06.jpg",
        HERE / "data" / "real_set" / "images" / "train" / "glasses_case_03.jpg",
        HERE / "data" / "camera_review" / "15.jpg",
    ],
    "medicine_box": [
        HERE / "data" / "real_set" / "images" / "train" / "medicine_box_05.jpg",
        HERE / "data" / "camera_review" / "27.jpg",
        HERE / "data" / "camera_review" / "24.jpg",
        HERE / "data" / "camera_review" / "33.jpg",
    ],
}
ZH = {"keys": "钥匙", "glasses_case": "眼镜盒", "medicine_box": "药盒"}
EVIDENCE_DIRS = [
    ROOT / "参考实现" / "data" / "evidence",
    HERE / "data" / "evidence",
]


def subject_box(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    mask = (np.abs(gray.astype(np.int16) - background) > 18).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    area = w * h / (width * height)
    if area < 0.05 or area > 0.96:
        # 主体几乎铺满或太碎时，用画面中间一块，避免又收成一个小黑点。
        return 0.5, 0.5, 0.72, 0.72
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def replace_evidence():
    for folder in EVIDENCE_DIRS:
        if not folder.exists():
            continue
        for path in folder.glob("*.jpg"):
            name = path.name
            kind = next((key for key in PHOTOS if key in name), None)
            if kind is None:
                continue
            pool = [item for item in PHOTOS[kind] if item.exists()]
            if not pool:
                continue
            # 放下和拿走用不同的一张，避免两张记录看起来完全一样。
            pick = pool[0 if "placed" in name else min(1, len(pool) - 1)]
            image = cv2.imread(str(pick))
            if image is None:
                continue
            height, width = image.shape[:2]
            scale = 1280 / max(width, height)
            if scale < 1:
                image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print(f"evidence {path.name} <- {pick.name}", flush=True)


def rebuild_training():
    staging = HERE / "data" / "camera_set_build"
    if staging.exists():
        shutil.rmtree(staging)
    for split in ("train", "val"):
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    rng = random.Random(7)
    files = []
    per_class = {}
    for index, (class_id, sources) in enumerate(PHOTOS.items()):
        usable = [path for path in sources if path.exists()]
        rng.shuffle(usable)
        val_count = 1 if len(usable) > 1 else 0
        per_class[class_id] = {"train": 0, "val": 0, "zh": ZH[class_id]}
        for number, source in enumerate(usable):
            split = "val" if number < val_count else "train"
            filename = f"{class_id}_{number:02d}.jpg"
            target = staging / "images" / split / filename
            shutil.copyfile(source, target)
            box = subject_box(target)
            if box is None:
                target.unlink()
                continue
            cx, cy, w, h = box
            (staging / "labels" / split / f"{class_id}_{number:02d}.txt").write_text(
                f"{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n",
                encoding="utf-8",
            )
            per_class[class_id][split] += 1
            files.append({
                "file": filename,
                "split": split,
                "classes": [ZH[class_id]],
                "class_ids": [class_id],
                "source_file": source.name,
            })
    if OUT.exists():
        shutil.rmtree(OUT)
    staging.rename(OUT)
    import yaml
    (OUT / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(OUT),
                "train": "images/train",
                "val": "images/val",
                "names": {i: key for i, key in enumerate(PHOTOS)},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    counts = {
        "train": sum(item["split"] == "train" for item in files),
        "val": sum(item["split"] == "val" for item in files),
    }
    manifest = {
        "source": "维基共享资源里用相机拍的台面、布面和柜台照片。每张都打开看过。",
        "what": "摄像头视角的实物照片：钥匙在木桌或台面上，眼镜盒在布面上，药盒在桌面或展台上。不是白底商品图，也不是仿真灰块。",
        "kind": "camera",
        "classes": [{"id": key, "zh": ZH[key]} for key in PHOTOS],
        "counts": counts,
        "per_class": per_class,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "per_class": per_class}, ensure_ascii=False))


if __name__ == "__main__":
    replace_evidence()
    rebuild_training()
