"""抽屉磁簧消息。只表示开或关，不能写成已经服过药。"""


def accept_reed(topic, payload, qos):
    parts = str(topic).split("/")
    if len(parts) != 4 or parts[0] != "home" or any(not item for item in parts):
        raise ValueError("主题必须是 home/{房间}/{设备}/{状态}")
    if parts[3] not in {"open", "closed"}:
        raise ValueError("状态只能是 open 或 closed")
    if int(qos) != 1:
        raise ValueError("QoS 必须是 1")
    if payload != parts[3]:
        raise ValueError("消息体必须和主题里的状态一致")
    return {"room": parts[1], "device": parts[2], "state": parts[3]}
