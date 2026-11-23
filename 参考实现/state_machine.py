"""轨迹到事件。纯函数，不读数据库，不调用模型。"""

import math
from dataclasses import dataclass, field
from statistics import median

BORDER = 0.08
STILL = 0.05
MOVING = 0.15
PLACE_S = 2.6
LOST_S = 1.5
SPEED_WINDOW_S = 0.4
MATCH_DIST = 0.20


def in_inner(cx, cy, frame_w, frame_h):
    return (
        BORDER * frame_w <= cx <= (1 - BORDER) * frame_w
        and BORDER * frame_h <= cy <= (1 - BORDER) * frame_h
    )


def speeds(points, frame_w):
    values = []
    for a, b in zip(points, points[1:]):
        dt = b[0] - a[0]
        if dt <= 1e-6:
            continue
        values.append(math.hypot(b[1] - a[1], b[2] - a[2]) / dt / frame_w)
    return values


def judge(points, now, frame_w, frame_h):
    """points 是 (t, cx, cy)。返回 placed、removed、occluded 或 unknown。"""
    if len(points) < 2:
        return "unknown"
    last_t, cx, cy = points[-1]
    gap = now - last_t
    recent = [p for p in points if p[0] >= last_t - SPEED_WINDOW_S]
    recent_speeds = speeds(recent, frame_w)
    speed = median(recent_speeds) if recent_speeds else 0.0
    inner = in_inner(cx, cy, frame_w, frame_h)

    if gap >= LOST_S:
        if (not inner) or speed >= MOVING:
            return "removed"
        if inner and speed < STILL:
            return "occluded"
        return "unknown"

    window = [p for p in points if p[0] >= now - PLACE_S]
    if not window or window[0][0] > now - PLACE_S + 0.25:
        return "unknown"
    if any(b[0] - a[0] > 0.6 for a, b in zip(window, window[1:])):
        return "unknown"
    if any(not in_inner(p[1], p[2], frame_w, frame_h) for p in window):
        return "unknown"
    window_speeds = speeds(window, frame_w)
    if not window_speeds:
        return "unknown"
    if median(window_speeds) >= STILL or any(item >= MOVING for item in window_speeds):
        return "unknown"
    return "placed"


def find_conflict(detections, frame_w, frame_h):
    """同一类有两个框心都在内区，且相距超过画面宽度的 25%。

    返回 (类别, 较低的那只框的分数)。分数留给问答：低于 0.40 仍说不确定。
    """
    grouped = {}
    for cls, cx, cy, score in detections:
        if in_inner(cx, cy, frame_w, frame_h):
            grouped.setdefault(cls, []).append((cx, cy, score))
    for cls, centers in grouped.items():
        if len(centers) < 2:
            continue
        distance = math.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1])
        if distance > 0.25 * frame_w:
            return cls, min(centers[0][2], centers[1][2])
    return None


@dataclass
class Track:
    cls: str
    points: list = field(default_factory=list)
    last_action: str | None = None
    closed: bool = False
    track_id: int | None = None


def _parse_detection(item):
    if len(item) == 5:
        cls, cx, cy, score, track_id = item
        return cls, float(cx), float(cy), float(score), track_id
    cls, cx, cy, score = item
    return cls, float(cx), float(cy), float(score), None


def update_tracks(tracks, detections, now, frame_w, frame_h):
    """detections 是 (cls, cx, cy, score)，或再加 track_id。

    没有 track_id 时用最近邻。有 track_id 时按编号续上，这是 ByteTrack 的输出进入状态机的方式。
    返回新事件列表 (cls, action, score)。
    """
    parsed = [_parse_detection(item) for item in detections]
    conflict = find_conflict([(item[0], item[1], item[2], item[3]) for item in parsed], frame_w, frame_h)
    events = []
    if conflict is not None:
        conflict_cls, conflict_score = conflict
        # 冲突期间不给这一类续轨迹。标记只用来避免每一帧都再写一行。
        marker = next(
            (
                track
                for track in tracks
                if track.cls == conflict_cls and track.last_action == "conflict" and not track.points
            ),
            None,
        )
        if marker is None:
            events.append((conflict_cls, "conflict", conflict_score))
            tracks.append(Track(cls=conflict_cls, points=[], last_action="conflict"))
        parsed = [item for item in parsed if item[0] != conflict_cls]
    else:
        tracks[:] = [
            track
            for track in tracks
            if not (track.last_action == "conflict" and not track.points)
        ]

    for cls, cx, cy, score, track_id in parsed:
        best = None
        if track_id is not None:
            for track in tracks:
                if track.closed or track.cls != cls or track.track_id != track_id:
                    continue
                best = track
                break
        else:
            best_d = MATCH_DIST * frame_w
            for track in tracks:
                if track.closed or track.cls != cls or not track.points or track.track_id is not None:
                    continue
                if now - track.points[-1][0] > LOST_S:
                    continue
                distance = math.hypot(track.points[-1][1] - cx, track.points[-1][2] - cy)
                if distance < best_d:
                    best, best_d = track, distance
        if best is None:
            tracks.append(Track(cls=cls, points=[(now, cx, cy, score)], track_id=track_id))
        else:
            best.points.append((now, cx, cy, score))

    for track in tracks:
        if track.closed:
            continue
        action = judge([(p[0], p[1], p[2]) for p in track.points], now, frame_w, frame_h)
        if action == "unknown" or action == track.last_action:
            continue
        track.last_action = action
        if action in {"removed", "occluded"}:
            track.closed = True
        sample = [p[3] for p in track.points[-5:]]
        events.append((track.cls, action, median(sample)))
    return events
