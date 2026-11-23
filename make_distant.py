"""把核对过的实物照片放到更远的台面上，并加一点摄像头的模糊。

台面取自原照片四周没有物体的部分，再铺满画面，避免中间一块被涂掉的方斑。
物体大约只占画面宽度的一成到两成。
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVIEW = HERE / "data" / "camera_review"
OUT = HERE / "data" / "real_set"
EVIDENCE = [
    ROOT / "参考实现" / "data" / "evidence",
    HERE / "data" / "evidence",
]
WIDTH, HEIGHT = 1280, 720

# split, class index, file. 验证用的原图不进训练。
SOURCES = [
    ("train", 0, "12.jpg"),
    ("val", 0, "07.jpg"),
    ("train", 1, "15.jpg"),
    ("val", 1, "20.jpg"),
    ("train", 2, "27.jpg"),
    ("val", 2, "24.jpg"),
]
NAMES = ["keys", "glasses_case", "medicine_box"]
ZH = ["钥匙", "眼镜盒", "药盒"]


def foreground_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[:6, :].ravel(), gray[-6:, :].ravel(), gray[:, :6].ravel(), gray[:, -6:].ravel()])
    background = float(np.median(border))
    mask = (np.abs(gray.astype(np.int16) - background) > 16).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    keep = np.zeros_like(mask)
    cv2.drawContours(keep, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return cv2.GaussianBlur(keep, (7, 7), 0)


def table_background(image):
    height, width = image.shape[:2]
    band = max(16, min(height, width) // 12)
    candidates = [image[:band, :], image[-band:, :], image[:, :band], image[:, -band:]]
    texture = min(candidates, key=lambda part: float(part.var()))
    return cv2.resize(texture, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def place(image, rng):
    mask = foreground_mask(image)
    if mask is None:
        return None
    ys, xs = np.where(mask > 40)
    if len(xs) < 30:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = image[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    table = table_background(image)
    scale = rng.uniform(0.18, 0.30)
    target_w = int(WIDTH * scale)
    target_h = max(10, int(crop.shape[0] * target_w / max(1, crop.shape[1])))
    if target_h > int(HEIGHT * 0.32):
        target_h = int(HEIGHT * 0.32)
        target_w = max(10, int(crop.shape[1] * target_h / max(1, crop.shape[0])))
    crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)[..., None]
    x = int(rng.uniform(WIDTH * 0.15, max(WIDTH * 0.16, WIDTH * 0.85 - target_w)))
    y = int(rng.uniform(HEIGHT * 0.30, max(HEIGHT * 0.31, HEIGHT * 0.75 - target_h)))
    patch = table[y:y + target_h, x:x + target_w].astype(np.float32)
    pasted = crop.astype(np.float32) * alpha + patch * (1 - alpha)
    table[y:y + target_h, x:x + target_w] = pasted.astype(np.uint8)
    table = cv2.GaussianBlur(table, (0, 0), rng.uniform(0.9, 1.6))
    cx = (x + target_w / 2) / WIDTH
    cy = (y + target_h / 2) / HEIGHT
    return table, (cx, cy, target_w / WIDTH, target_h / HEIGHT)


def main():
    rng = random.Random(11)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
    files = []
    per_class = {name: {"train": 0, "val": 0, "zh": zh} for name, zh in zip(NAMES, ZH)}
    shots = {name: [] for name in NAMES}
    for split, class_index, filename in SOURCES:
        image = cv2.imread(str(REVIEW / filename))
        if image is None:
            print("missing", filename)
            continue
        copies = 6 if split == "train" else 3
        for copy_index in range(copies):
            made = place(image, rng)
            if made is None:
                print("skip", filename)
                break
            frame, box = made
            name = NAMES[class_index]
            out_name = f"{name}_{filename[:-4]}_{copy_index}.jpg"
            cv2.imwrite(str(OUT / "images" / split / out_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            cx, cy, w, h = box
            (OUT / "labels" / split / f"{out_name[:-4]}.txt").write_text(
                f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8"
            )
            per_class[name][split] += 1
            files.append({"file": out_name, "split": split, "classes": [ZH[class_index]], "class_ids": [name]})
            if len(shots[name]) < 2:
                shots[name].append(frame)
            print(split, out_name, flush=True)

    for folder in EVIDENCE:
        if not folder.exists():
            continue
        for path in folder.glob("*.jpg"):
            kind = next((name for name in NAMES if name in path.name), None)
            if kind and shots[kind]:
                shot = shots[kind][0 if "placed" in path.name else -1]
                cv2.imwrite(str(path), shot, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    import yaml
    (OUT / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(OUT), "train": "images/train", "val": "images/val", "names": {i: n for i, n in enumerate(NAMES)}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    counts = {"train": sum(i["split"] == "train" for i in files), "val": sum(i["split"] == "val" for i in files)}
    manifest = {
        "kind": "camera",
        "source": "实物照片缩小后放在原图的台面纹理上，并加了轻微模糊。Roboflow 的 Home item detection 有 707 张钥匙、眼镜盒和药盒，需要密钥，这次没有下载。",
        "what": "家用摄像头距离：物体约占画面宽度的两成到三成，整体略糊。不是贴脸近景，也不是仿真灰块。",
        "classes": [{"id": name, "zh": zh} for name, zh in zip(NAMES, ZH)],
        "counts": counts,
        "per_class": per_class,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
