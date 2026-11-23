"""有效像素计算。放大不算达标。"""

import math

PASS_PX = 32


def world_width_m(distance_m, fov_deg):
    if distance_m <= 0 or fov_deg <= 0 or fov_deg >= 180:
        raise ValueError("距离和视场角超出范围")
    return 2 * distance_m * math.tan(math.radians(fov_deg) / 2)


def evaluate(main_width, imgsz, object_m, distance_m, fov_deg, slot_fill=1.0):
    """垫子铺满画面时 slot_fill=1。只占画面宽度的一部分时，填实际占比。

    检测像素按裁剪后的槽位宽度计算。有效像素不超过主码流上的原始像素，
    因此把裁剪图放大不会让不合格的物品变成合格。
    """
    if not 0 < slot_fill <= 1:
        raise ValueError("slot_fill 必须在 (0, 1] 内")
    world = world_width_m(distance_m, fov_deg)
    native = main_width * object_m / world
    slot_w = world * slot_fill
    network = imgsz * object_m / slot_w
    effective = min(native, network)
    return {
        "world_m": world,
        "slot_m": slot_w,
        "native_px": native,
        "network_px": network,
        "effective_px": effective,
        "pass": effective >= PASS_PX,
    }
