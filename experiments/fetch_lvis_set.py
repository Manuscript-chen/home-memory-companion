"""从 LVIS v1 验证集取出带框的真实照片，做成 YOLO 训练集。

钥匙用 LVIS 的 key。眼镜用 spectacles（镜框和镜片），公开框数据里没有眼镜盒。
药品用 medicine。标注来自 LVIS，图片来自 COCO val2017。
一张图只进训练或验证其中一边。
"""

import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ZIP_PATH = HERE / "data" / "openimages" / "lvis_v1_val.json.zip"
OUT = HERE / "data" / "real_set"

CLASSES = [
    {
        "lvis_id": 605,
        "id": "keys",
        "zh": "钥匙",
        "lvis_name": "key",
        "meaning": "开锁的金属钥匙",
        "cap": 43,
        "min_area": 0.001,
        "val_min": 8,
    },
    {
        "lvis_id": 995,
        "id": "spectacles",
        "zh": "眼镜",
        "lvis_name": "spectacles",
        "meaning": "矫正视力的眼镜。不是眼镜盒，LVIS 没有眼镜盒这一类。",
        "cap": 60,
        "min_area": 0.01,
        "val_min": 12,
    },
    {
        "lvis_id": 683,
        "id": "medicine",
        "zh": "药品",
        "lvis_name": "medicine",
        "meaning": "LVIS 标成药品的实物。验证集里数量少，不是药盒的专用采集。",
        "cap": 13,
        "min_area": 0.004,
        "val_min": 3,
    },
]

LICENSES = {
    1: "CC BY-NC 2.0",
    2: "CC BY-NC-SA 2.0",
    3: "CC BY-NC-ND 2.0",
    4: "CC BY 2.0",
    5: "CC BY-SA 2.0",
    6: "CC BY-ND 2.0",
    7: "No known copyright",
    8: "United States Government Work",
}


def load_lvis():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        name = archive.namelist()[0]
        return json.loads(archive.read(name))


def box_area_ratio(bbox, width, height):
    _x, _y, w, h = bbox
    if width <= 0 or height <= 0:
        return 0.0
    return max(0.0, w) * max(0.0, h) / (width * height)


def choose_images(data):
    images = {item["id"]: item for item in data["images"]}
    by_class = {item["lvis_id"]: [] for item in CLASSES}
    for ann in data["annotations"]:
        if ann["category_id"] not in by_class:
            continue
        image = images.get(ann["image_id"])
        if image is None:
            continue
        ratio = box_area_ratio(ann["bbox"], image["width"], image["height"])
        by_class[ann["category_id"]].append((ann["image_id"], ratio))

    chosen = {}
    rng = random.Random(7)
    for spec in CLASSES:
        best = {}
        for image_id, ratio in by_class[spec["lvis_id"]]:
            if ratio < spec["min_area"]:
                continue
            best[image_id] = max(best.get(image_id, 0.0), ratio)
        ranked = sorted(best, key=lambda image_id: best[image_id], reverse=True)
        rng.shuffle(ranked[spec["cap"]:])
        chosen[spec["lvis_id"]] = ranked[: spec["cap"]]
    return images, chosen


def split_ids(chosen):
    val = set()
    rng = random.Random(7)
    for spec in CLASSES:
        ids = list(chosen[spec["lvis_id"]])
        rng.shuffle(ids)
        count = max(spec["val_min"], round(len(ids) * 0.2))
        count = min(count, max(1, len(ids) - 1))
        val.update(ids[:count])
    return val


def yolo_lines(annotations, image, class_index):
    width = image["width"]
    height = image["height"]
    lines = []
    for ann in annotations:
        index = class_index.get(ann["category_id"])
        if index is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        lines.append(
            f"{index} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}\n"
        )
    return lines


def download(url, dest):
    request = Request(url, headers={"User-Agent": "HomeMemoryResearch/1.0"})
    with urlopen(request, timeout=40) as response:
        dest.write_bytes(response.read())


def main():
    data = load_lvis()
    images, chosen = choose_images(data)
    val_ids = split_ids(chosen)
    class_index = {spec["lvis_id"]: index for index, spec in enumerate(CLASSES)}
    anns_by_image = {}
    for ann in data["annotations"]:
        if ann["category_id"] in class_index:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    selected = set()
    for ids in chosen.values():
        selected.update(ids)

    if OUT.exists():
        for path in OUT.rglob("*"):
            if path.is_file():
                path.unlink()
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs = []
    for image_id in sorted(selected):
        image = images[image_id]
        split = "val" if image_id in val_ids else "train"
        dest = OUT / "images" / split / f"{image_id}.jpg"
        jobs.append((image, split, dest))

    failed = []

    def fetch(job):
        image, split, dest = job
        url = image.get("coco_url") or image.get("flickr_url")
        try:
            download(url, dest)
            if dest.stat().st_size < 2000:
                dest.unlink(missing_ok=True)
                return image["id"], split, "too small"
        except Exception as exc:
            dest.unlink(missing_ok=True)
            return image["id"], split, str(exc)
        return image["id"], split, None

    done = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            image_id, split, error = future.result()
            done[image_id] = (split, error)
            if error:
                failed.append({"id": image_id, "error": error})
            print(f"{image_id} {split} {error or 'ok'}", flush=True)

    files = []
    counts = {"train": 0, "val": 0}
    per_class = {spec["id"]: {"train": 0, "val": 0} for spec in CLASSES}
    for image_id, (split, error) in done.items():
        if error:
            continue
        image = images[image_id]
        lines = yolo_lines(anns_by_image.get(image_id, []), image, class_index)
        if not lines:
            continue
        (OUT / "labels" / split / f"{image_id}.txt").write_text("".join(lines), encoding="utf-8")
        counts[split] += 1
        present = sorted({int(line.split()[0]) for line in lines})
        for index in present:
            per_class[CLASSES[index]["id"]][split] += 1
        files.append(
            {
                "file": f"{image_id}.jpg",
                "split": split,
                "image_id": image_id,
                "width": image["width"],
                "height": image["height"],
                "license": LICENSES.get(image.get("license"), str(image.get("license"))),
                "coco_url": image.get("coco_url"),
                "classes": [CLASSES[index]["zh"] for index in present],
                "class_ids": [CLASSES[index]["id"] for index in present],
            }
        )

    yaml = {
        "path": str(OUT),
        "train": "images/train",
        "val": "images/val",
        "names": {index: spec["id"] for index, spec in enumerate(CLASSES)},
    }
    import yaml as yaml_lib

    (OUT / "data.yaml").write_text(yaml_lib.safe_dump(yaml, allow_unicode=True), encoding="utf-8")
    manifest = {
        "source": "LVIS v1 validation annotations, images from COCO val2017",
        "annotation": "LVIS annotations are CC BY 4.0. Each photo keeps its own COCO license.",
        "what": "真实照片。钥匙是 key，眼镜是 spectacles（不是眼镜盒），药品是 medicine。",
        "classes": [
            {"id": spec["id"], "zh": spec["zh"], "lvis": spec["lvis_name"], "meaning": spec["meaning"]}
            for spec in CLASSES
        ],
        "counts": counts,
        "per_class": per_class,
        "failed": failed,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "per_class": per_class, "failed": len(failed)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
