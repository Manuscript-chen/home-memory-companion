# 节点 A：槽位标定与词表

安装时把「哪颗摄像头对着哪一块台面、老人怎么叫这件东西」写成 YAML。检测提示、问答别名和语音热词都从这里来，不要在代码里再维护一份名单。

| 文件 | 作用 |
| --- | --- |
| `sites.yaml` | 每路摄像头：RTSP、距离、视场角、检测输入边长、只绑一个物品 |
| `vocab.yaml` | 槽位和物品的 id、英文 prompt、中文规范名、别名 |
| `privacy.yaml` | 卧室和卫生间不装摄像头，以及每路胶带遮挡 |
| `voice.yaml` | 节点 I 的识别 / 合成模型目录 |
| `recall.yaml` | 节点 H2 向量召回的备用参数 |
| `bytetrack.yaml` / `botsort.yaml` | 节点 E 的跟踪参数。BoT-SORT 是备用，默认不启用 |
| `hotwords.txt` | 由词表生成的热词，改词表后要重新生成 |
| `pixel_budget.py` | 有效像素。不足 32 的物品不能进该路检测名单 |
| `check_config.py` | 启动前检查。像素不够、槽位落在卧室或卫生间、一路多个检测物品、提示词不是英文，就拒绝 |

```bash
python config/check_config.py
```

占位账号 `USER:PASS` 可以留着跑演示。真的打开摄像头之前必须改成相机上的账号。

这个目录里的 YAML 全项目共用。几个人一起改之前先说一声，避免两个人同时改 `vocab.yaml`。
