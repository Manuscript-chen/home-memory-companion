"""按 bytetrack.yaml 的阈值做两阶段交并比关联。

这是仿真用的关联器，对照已安装的 Ultralytics BYTETracker：
高分先配，低分再配。第一阶段可以乘上分数。第二阶段只比交并比，代价上限固定 0.5，
因为低分再乘一次会让代价超过 0.5。它不加载外观模型，也不做卡尔曼预测。
真机上 run_camera.py 调用 Ultralytics 自己的 BYTETracker。
"""

from pathlib import Path

import yaml


def load_settings(path):
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return {
        "track_high_thresh": float(raw["track_high_thresh"]),
        "track_low_thresh": float(raw["track_low_thresh"]),
        "new_track_thresh": float(raw["new_track_thresh"]),
        "track_buffer": int(raw["track_buffer"]),
        "match_thresh": float(raw["match_thresh"]),
        "fuse_score": bool(raw.get("fuse_score", True)),
        "second_cost_limit": 0.5,
    }


def iou(a, b):
    """a、b 是 (cx, cy, w, h)。"""
    ax1, ay1 = a[0] - a[2] / 2.0, a[1] - a[3] / 2.0
    ax2, ay2 = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bx1, by1 = b[0] - b[2] / 2.0, b[1] - b[3] / 2.0
    bx2, by2 = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return inter / union


def _cost(track, det, fuse):
    overlap = iou(track["box"], det["box"])
    if fuse:
        return 1.0 - overlap * det["score"]
    return 1.0 - overlap


def _greedy(tracks, dets, limit, fuse):
    pairs = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(dets):
            if track["cls"] != det["cls"]:
                continue
            cost = _cost(track, det, fuse)
            if cost <= limit:
                pairs.append((cost, ti, di))
    pairs.sort()
    used_t, used_d = set(), set()
    matched = []
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((ti, di))
    return matched


class SimByteTrack:
    def __init__(self, settings):
        self.settings = settings
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """detections: (cls, cx, cy, w, h, score)。返回本帧还在的 (id, cls, cx, cy, score)。"""
        buffer = self.settings["track_buffer"]
        self.tracks = [track for track in self.tracks if track["lost"] <= buffer]
        high_i, low_i = [], []
        parsed = []
        for cls, cx, cy, w, h, score in detections:
            item = {"cls": cls, "box": (cx, cy, w, h), "score": float(score)}
            parsed.append(item)
            if score >= self.settings["track_high_thresh"]:
                high_i.append(len(parsed) - 1)
            elif score > self.settings["track_low_thresh"]:
                low_i.append(len(parsed) - 1)
        fuse = self.settings["fuse_score"]
        first = _greedy(
            self.tracks,
            [parsed[i] for i in high_i],
            self.settings["match_thresh"],
            fuse,
        )
        taken = set()
        matched_tracks = set()
        for ti, di in first:
            det = parsed[high_i[di]]
            self._take(ti, det)
            taken.add(high_i[di])
            matched_tracks.add(ti)
        left_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        second = _greedy(
            [self.tracks[i] for i in left_tracks],
            [parsed[i] for i in low_i],
            self.settings["second_cost_limit"],
            False,
        )
        for ti, di in second:
            det = parsed[low_i[di]]
            self._take(left_tracks[ti], det)
            taken.add(low_i[di])
            matched_tracks.add(left_tracks[ti])
        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track["lost"] += 1
        for index, det in enumerate(parsed):
            if index in taken:
                continue
            if det["score"] < self.settings["new_track_thresh"]:
                continue
            self.tracks.append(
                {
                    "id": self.next_id,
                    "cls": det["cls"],
                    "box": det["box"],
                    "score": det["score"],
                    "lost": 0,
                }
            )
            self.next_id += 1
        alive = []
        for track in self.tracks:
            if track["lost"] == 0:
                cx, cy = track["box"][0], track["box"][1]
                alive.append((track["id"], track["cls"], cx, cy, track["score"]))
        return alive

    def _take(self, index, det):
        track = self.tracks[index]
        track["box"] = det["box"]
        track["score"] = det["score"]
        track["lost"] = 0
