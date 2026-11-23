# 演示和线束

不接摄像头就能跑的检查，以及真机入口。计划书要求每一步通过了再进下一步。

| 文件 | 作用 |
| --- | --- |
| `run_demo.py` | 配置、像素、状态机、触发、数据库和问答。通过时打印「演示通过」 |
| `run_h2.py` | 向量召回的备用验收。需要 `models/bge-small-zh-v1.5` |
| `replay_demo.py` | 用状态机把三条例子写入本机数据库。JPEG 是 `assets/offline.jpg` 占位图 |
| `harness.py` | 把触发、状态机、记忆、问答和到点确认串起来。检测框是合成的 |
| `run_camera.py` | 真机。`python demo/run_camera.py cam_entry` |

先运行：

```bash
python demo/run_demo.py
```
