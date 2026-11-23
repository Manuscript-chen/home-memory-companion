"""从维基共享资源下载钥匙、眼镜盒、药盒的实物照片，并按画面主体生成框。

只收标题就能确定是这三件东西的文件。框取相对背景明显的那一块，
不是手画的椭圆。框不成立的照片不进训练集。
"""

import json
import random
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "real_set"

ITEMS = [
    ("keys", "钥匙", "开锁用的钥匙", [
        "File:1970s Mister Minit duplicate key for Boda locks cut on Dominion Lock Co B41B blank.jpg",
        "File:1980s or 1990s wafer lock key with unidentified manufacturer airplane logo.jpg",
        "File:4-way utility key.JPG",
        "File:A hand holds a set of keys.jpg",
        "File:A key is inserted in a door lock.jpg",
        "File:A key is shown inserted into a lock.jpg",
        "File:Ancient rusty key.jpg",
        "File:Bathroom key with pole attached for identification.jpg",
        "File:Bit Key by Russell & Erwin Mfg.jpg",
        "File:Bossnøkkel.jpg",
        "File:Broken key IMG 7122.JPG",
        "File:A key of vintage wooden wall clock of Smiths (from circa 1930s).jpg",
    ]),
    ("glasses_case", "眼镜盒", "眼镜盒，不是戴在脸上的眼镜", [
        "File:2023 Okulary korekcyjne w etui.jpg",
        "File:Brillenetui Reinhold-Kühn 01.jpg",
        "File:Brillenetui Reinhold-Kühn 02.jpg",
        "File:Brillenetui Reinhold-Kühn 03.jpg",
        "File:Eyeglass case.jpg",
        "File:Eyeglasses and case (AM 71510-1).jpg",
        "File:Glasögonfodral av näver - XLM.00863.jpg",
        "File:Spectacle-case-kuehn0 hg.jpg",
        "File:Spectacle-case-kuehn2 hg.jpg",
        "File:Spectacle-case-kuehn5 hg.JPG",
        "File:Spectacle-case-kuehn7 hg.jpg",
        "File:Spectacle-case-optal hg.jpg",
    ]),
    ("medicine_box", "药盒", "药盒、分药盒或药瓶盒，不是医院输液架", [
        "File:2023 Kasetka z lekami.jpg",
        "File:Pill Dispenser (daily).jpg",
        "File:Pill Organizer With Vitamins And Medicines.jpg",
        "File:Pill box with pills.JPG",
        "File:Pilulier semainier.JPG",
        "File:Ronde pillendoos met schroefdeksel - INDUS V09679.JPG",
        "File:Pillendoos voor pastillen tegen hoesten - INDUS V09650.JPG",
        "File:Ronde pillendoos met schroefdeksel voor de pijnstiller Saridon - INDUS V09682.JPG",
    ]),
]


def api(params):
    query = urlencode(params)
    request = Request(
        "https://commons.wikimedia.org/w/api.php?" + query,
        headers={"User-Agent": "HomeMemoryResearch/1.0 (local training photos)"},
    )
    with urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def lookup(titles):
    found = {}
    for start in range(0, len(titles), 20):
        chunk = titles[start:start + 20]
        data = api({
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": "960",
            "format": "json",
        })
        for page in data["query"]["pages"].values():
            info = (page.get("imageinfo") or [None])[0]
            if not info or not str(info.get("mime", "")).startswith("image/"):
                continue
            meta = info.get("extmetadata") or {}
            license_name = (meta.get("LicenseShortName") or {}).get("value") or "见文件页"
            found[page["title"]] = {
                "url": info.get("thumburl") or info.get("url"),
                "license": license_name,
            }
    return found


def download(url, dest):
    request = Request(url, headers={"User-Agent": "HomeMemoryResearch/1.0"})
    with urlopen(request, timeout=40) as response:
        dest.write_bytes(response.read())


def subject_box(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    mask = (np.abs(gray.astype(np.int16) - background) > 22).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h / (width * height)
    if area < 0.04 or area > 0.92 or w < 12 or h < 12:
        return None
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def main():
    titles = [title for _class_id, _zh, _meaning, group in ITEMS for title in group]
    info = lookup(titles)
    if OUT.exists():
        for path in list(OUT.rglob("*")):
            if path.is_file():
                path.unlink()
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(7)
    files = []
    skipped = []
    per_class = {}
    for class_index, (class_id, zh, meaning, group) in enumerate(ITEMS):
        usable = []
        for title in group:
            meta = info.get(title)
            if meta is None:
                skipped.append({"title": title, "reason": "没有下载地址"})
                continue
            usable.append((title, meta))
        rng.shuffle(usable)
        val_count = max(2, round(len(usable) * 0.25)) if len(usable) > 3 else 1
        per_class[class_id] = {"train": 0, "val": 0, "zh": zh, "meaning": meaning}
        for index, (title, meta) in enumerate(usable):
            split = "val" if index < val_count else "train"
            filename = f"{class_id}_{index:02d}.jpg"
            image_path = OUT / "images" / split / filename
            try:
                time.sleep(1.2)
                download(meta["url"], image_path)
            except Exception as exc:
                skipped.append({"title": title, "reason": str(exc)})
                continue
            box = subject_box(image_path)
            if box is None:
                image_path.unlink(missing_ok=True)
                skipped.append({"title": title, "reason": "主体框不成立"})
                continue
            cx, cy, w, h = box
            (OUT / "labels" / split / f"{class_id}_{index:02d}.txt").write_text(
                f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n",
                encoding="utf-8",
            )
            per_class[class_id][split] += 1
            files.append({
                "file": filename,
                "split": split,
                "classes": [zh],
                "class_ids": [class_id],
                "title": title,
                "license": meta["license"],
                "source_url": meta["url"],
            })
            print(f"{split} {zh} {title}", flush=True)

    import yaml
    yaml_body = {
        "path": str(OUT),
        "train": "images/train",
        "val": "images/val",
        "names": {index: class_id for index, (class_id, _zh, _meaning, _group) in enumerate(ITEMS)},
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(yaml_body, allow_unicode=True), encoding="utf-8")
    counts = {
        "train": sum(1 for item in files if item["split"] == "train"),
        "val": sum(1 for item in files if item["split"] == "val"),
    }
    manifest = {
        "source": "Wikimedia Commons。每张照片页面上的许可不同，清单里按文件记录。",
        "what": "维基共享资源上的实物照片：钥匙、眼镜盒、药盒。框是相对背景算出的主体，不是程序画的几何图形。",
        "classes": [
            {"id": class_id, "zh": zh, "meaning": meaning}
            for class_id, zh, meaning, _group in ITEMS
        ],
        "counts": counts,
        "per_class": per_class,
        "skipped": skipped,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "per_class": {k: v for k, v in per_class.items()}, "skipped": len(skipped)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
