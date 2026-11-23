# 节点 E：跟踪

检测器每帧单独出框。跟踪给同一件东西一条轨迹，状态机才能区分「停住了」和「新出现了一个」。

| 文件 | 作用 |
| --- | --- |
| `associate.py` | 仿真用的两阶段关联，对照 Ultralytics 的 ByteTrack。不加载外观模型，也不做卡尔曼预测 |
| `config/bytetrack.yaml` | 真机参数：`track_buffer` 为 8（5 fps 下约 1.5 秒），`match_thresh` 为 0.8 |

真机默认走 Ultralytics ByteTrack，调用在 `demo/run_camera.py`。每次主码流窗口开始时清空，不把上一次事件的 id 带到下一次。

`state_machine.py` 里的最近邻留给 `demo/run_demo.py`。真机若暂时仍走最近邻，日志第一行要写明「备用跟踪」。

两个同类框的中心都在内区、相距超过画面宽度的 25% 时记为 conflict，不挑选其中一只。
