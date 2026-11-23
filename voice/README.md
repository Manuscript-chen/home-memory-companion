# 节点 I：语音入口

物理按键按下到松开之间的采样才送去识别。本节点不决定槽位，不写入 `object_states`。合成器只朗读节点 H 已经生成的句子。

| 文件 | 作用 |
| --- | --- |
| `voice_gate.py` | 按键窗口。按下开始攒采样，松开得到这一句 |
| `voice.py` | sherpa-onnx 识别和合成。空格键演示按键；`--roundtrip` 用合成语音回环，不需要麦克风 |
| `hotwords.py` | 热词只从 `vocab.yaml` 生成 |

模型目录见 `config/voice.yaml` 和 `docs/weights.md`。节点 H 的 20 条书面问法通过之后再接麦克风。

计划书采用 sherpa-onnx 2.0.1。Silero VAD、唤醒词和云端实时语音都是备用，首版不接。
