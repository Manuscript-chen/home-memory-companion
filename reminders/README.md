# 节点 J 里的提醒，以及到点提问

放置记录和键盘问答不依赖这里。未安装时，节点 B 到 H 仍然成立。

| 文件 | 作用 |
| --- | --- |
| `remind.py` | 服药、锁门、关窗的计划。到点在屏幕上提问。确认来源固定是 `self_report` |
| `reed.py` | 抽屉磁簧的 MQTT 主题检查：`home/{房间}/{设备}/{状态}`，QoS 1。只表示开或关，不能写成已经服过药 |

```bash
python reminders/remind.py add 吃药 2026-11-18T08:00:00
python reminders/remind.py due
```

还没有写进仓库的部分（计划书里的备用，未安装时不用做）：Qwen3-VL 只核对物品 id、Mosquitto 收磁簧、塑料钥匙扣上的 NFC。不要用视觉语言模型代替 YOLOE 或状态机。
