"""用 LVIS 真实照片训练闭集检测。不重新生成几何图形。"""

import json
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "参考实现"
OUT = HERE / "data" / "real_set"
RUNS = HERE / "runs"
RESULTS = HERE / "results"


def train_once(device, epochs):
    from ultralytics import YOLO

    model = YOLO(str(CODE / "yolo26n.pt"))
    result = model.train(
        data=str(OUT / "data.yaml"),
        epochs=epochs,
        imgsz=416,
        batch=4,
        device=device,
        workers=0,
        project=str(RUNS),
        name="real_set",
        seed=7,
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    metrics = getattr(result, "results_dict", None) or {}
    box = getattr(result, "box", None)
    return {
        "device": str(device),
        "epochs": epochs,
        "map50": float(getattr(box, "map50", metrics.get("metrics/mAP50(B)", 0.0)) or 0.0),
        "map50_95": float(getattr(box, "map", metrics.get("metrics/mAP50-95(B)", 0.0)) or 0.0),
    }


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "ready": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "kind": manifest.get("kind") or "commons",
            "counts": manifest.get("counts") or {},
            "per_class": manifest.get("per_class") or {},
            "split": "按图片切开。一张图只在训练或验证其中一边。",
            "what": manifest.get("what"),
            "source": manifest.get("source"),
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
    epochs = 15 if device == "cpu" else 30
    try:
        report["closed_set"] = train_once(device, epochs)
    except Exception as exc:
        report["closed_set_error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    report["open_vocab"] = {"ran": False, "reason": "这一轮只训练带框的闭集照片，没有再跑开放词汇。"}
    report["ready"] = "closed_set" in report
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in report if k != "traceback"}, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
