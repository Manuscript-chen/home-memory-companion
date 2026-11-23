"""检查三份配置能否一起使用。有错误就退出，不启动摄像头。"""

import sys
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
if str(_repo / "common") not in sys.path:
    sys.path.insert(0, str(_repo / "common"))
import nodepath  # noqa: E402
from paths import REPO_ROOT as ROOT  # noqa: E402

import yaml  # noqa: E402

from pixel_budget import evaluate  # noqa: E402


def load(name):
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def errors_in_config():
    sites = load("sites.yaml")
    vocab = load("vocab.yaml")
    privacy = load("privacy.yaml")
    errors = []
    objects = {item["id"]: item for item in vocab["objects"]}
    slots = {item["id"]: item for item in vocab["slots"]}

    if set(privacy.get("rooms_without_camera", [])) != {"bedroom", "bathroom"}:
        errors.append("卧室和卫生间必须写在禁止安装列表里")

    camera_ids = []
    for cam in sites["cameras"]:
        camera_ids.append(cam["id"])
        if cam["id"] not in {"cam_entry", "cam_coffee", "cam_dining"}:
            errors.append(f"{cam['id']} 不在首版三路之内")
        if any(word in cam["slot_id"] for word in ("bed", "bath", "toilet")):
            errors.append(f"{cam['id']} 把槽位放在了卧室或卫生间")
        if cam["slot_id"] not in slots:
            errors.append(f"{cam['id']} 的槽位不在词表中")
        if cam.get("slot_fill", 1.0) != 1.0:
            errors.append(f"{cam['id']} 首版要求垫子铺满画面，slot_fill 只能是 1")
        if cam.get("width", 0) < 2560:
            errors.append(f"{cam['id']} 主码流宽度必须不小于 2560")
        if not 60 <= cam.get("fov_deg", 0) <= 70:
            errors.append(f"{cam['id']} 水平视场角必须在 60 到 70 度")
        if len(cam["objects"]) != 1:
            errors.append(f"{cam['id']} 首版只能绑定 1 个物品")
        else:
            only = cam["objects"][0]
            expect_imgsz = 1280 if only == "keys" else 640
            if cam.get("imgsz") != expect_imgsz:
                errors.append(f"{cam['id']} 的检测输入应为 {expect_imgsz}")
        for object_id in cam["objects"]:
            if object_id not in objects:
                errors.append(f"{cam['id']} 使用了词表里没有的物品 {object_id}")
                continue
            result = evaluate(
                main_width=cam["width"],
                imgsz=cam["imgsz"],
                object_m=objects[object_id]["short_side_m"],
                distance_m=cam["distance_m"],
                fov_deg=cam["fov_deg"],
                slot_fill=cam.get("slot_fill", 1.0),
            )
            if not result["pass"]:
                errors.append(
                    f"{cam['id']} 上的 {object_id} 有效像素为 {result['effective_px']:.1f}，低于 32"
                )
        if "USER:PASS" in cam.get("rtsp_main", ""):
            errors.append(f"{cam['id']} 的主码流 RTSP 仍是占位账号，真机运行前要改成摄像头上的账号")
        sub = cam.get("rtsp_sub", "")
        if not sub:
            errors.append(f"{cam['id']} 缺少子码流 rtsp_sub")
        elif "USER:PASS" in sub:
            errors.append(f"{cam['id']} 的子码流 RTSP 仍是占位账号，真机运行前要改成摄像头上的账号")

    masked = {item["id"] for item in privacy["cameras"]}
    if set(camera_ids) != {"cam_entry", "cam_coffee", "cam_dining"}:
        errors.append("首版必须正好是 cam_entry、cam_coffee、cam_dining 三路")
    if set(camera_ids) != masked:
        errors.append("privacy.yaml 必须给每一路摄像头写物理遮挡，且不能多写")
    for item in privacy["cameras"]:
        if not item.get("physical_mask"):
            errors.append(f"{item['id']} 没有物理遮挡记录")
        if not str(item.get("note", "")).strip():
            errors.append(f"{item['id']} 没有写明胶带挡住了画面的哪一部分")

    for obj in vocab["objects"]:
        if obj["canonical_name"] not in obj["aliases"]:
            errors.append(f"{obj['id']} 的别名列表缺少规范名")
        if len(obj["aliases"]) < 3:
            errors.append(f"{obj['id']} 的别名少于 3 条")
        if not obj["prompt"].isascii() or not obj["prompt"].strip():
            errors.append(f"{obj['id']} 的检测提示必须是英文")
    for slot in vocab["slots"]:
        if slot["canonical_name"] not in slot["aliases"]:
            errors.append(f"{slot['id']} 的别名列表缺少规范名")
    return errors


def main():
    found = errors_in_config()
    hard = [item for item in found if "占位账号" not in item]
    for item in found:
        print(item)
    if hard:
        print(f"配置未通过：{len(hard)} 项")
        return 1
    if found:
        print("像素、词表和隐私检查通过。RTSP 仍是示例账号，演示可以继续，真机前必须修改。")
        return 0
    print("配置通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
