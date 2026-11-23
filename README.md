# 家庭视觉记忆与语音陪伴系统

固定摄像头被动记下钥匙、老花镜、药盒放在哪一块台面上；老人按下按键，用语音问「我的 X 放哪了」。原始视频不保留。位置来自安装时写进 YAML 的槽位，动作来自可测试的状态机，不由模型编造地点。

## 目录对应计划书的哪一节

几个人可以分头改不同目录。配置 YAML 全项目共用，改之前先说一声。

| 目录 | 计划书节点 | 里面有什么 |
| --- | --- | --- |
| `config/` | A 槽位与词表 | `sites.yaml`、`vocab.yaml`、`privacy.yaml`，以及像素检查 |
| `capture/` | B 双码流采集 | 主码流窗口和断线等待 |
| `gating/` | C 运动门控 | MOG2 和四条触发规则 |
| `detection/` | D 白名单检测 | 说明。YOLOE 调用仍在 `demo/run_camera.py` |
| `tracking/` | E 跟踪 | 仿真用的 ByteTrack 关联；真机参数在 `config/bytetrack.yaml` |
| `state_machine/` | F 状态机 | 放下、拿走、挡住、未知 |
| `memory/` | G 双表记忆 | `events` 只追加，视觉列和口述列分开 |
| `qa/` | H 消歧与回答 | `answer.py`、`ask.py`。向量召回是备用 |
| `voice/` | I 语音入口 | 按键窗口、sherpa-onnx 识别和合成 |
| `reminders/` | 到点提问，以及 J 的磁簧解析 | 服药计划。确认只能是口述 |
| `demo/` | 不接摄像头的验收，以及真机入口 | `run_demo.py` 通过时打印「演示通过」 |
| `experiments/` | 闭集对照和实验记录 | 不是每天要跑的程序 |
| `docs/` | 说明和画面核对照片 | 权重怎么下、7 张缩略图 |
| `tools/` | 外部工具备忘 | go2rtc 只在摄像头连接数满了才用 |
| `assets/` | 占位图 | `offline.jpg`，回放演示用，不是拍摄画面 |
| `models/` | 本机权重 | 不进 Git。见 `docs/weights.md` |
| `common/` | 启动路径 | 让各目录里的 `import` 仍按文件名查找 |
| `data/` | 运行时库和证据照片 | 程序自己创建，不进 Git |

各目录里的 `README.md` 写了这个节点的输入输出和该用哪条命令。

原来所有程序都在 `参考实现/` 一个文件夹里，实验脚本散在仓库根目录。现在按节点拆开了。看某个文件的旧历史：

```bash
git log --follow -- qa/answer.py
```

## 环境

用 Python **3.11、3.12 或 3.13**。不要和 Home Assistant 2026.3 的 Python 3.14 装在同一个环境里。

```bash
python -m venv .venv
```

Windows：

```bat
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

然后：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 里，sherpa-onnx 按计划书写成 2.0.1。其余库计划书只点了名字，没有点版本，所以不锁死版本号。`ultralytics` 会再装 PyTorch，下载比较大。

没有麦克风、没有摄像头时，先跑：

```bash
python demo/run_demo.py
```

通过时最后一行附近有「演示通过」。配置检查：

```bash
python config/check_config.py
```

`sites.yaml` 里的 `USER:PASS` 是占位账号。演示可以继续；真机前改成摄像头上的账号，再检查一次。

权重、语音模型和 go2rtc 压缩包的下载位置见 [docs/weights.md](docs/weights.md) 和 [tools/README.md](tools/README.md)。不要把 `.pt`、`.onnx`、`.zip` 或 `data/` 提交进仓库。

## 常用命令

在仓库根目录执行：

```bash
python config/check_config.py
python demo/run_demo.py
python demo/replay_demo.py
python qa/ask.py 钥匙在哪
python demo/run_camera.py cam_entry
python reminders/remind.py list
python voice/voice.py
```

`run_camera.py` 会找 `models/yoloe-26n-seg.pt`。没有这个文件时会直接退出并指出下载说明。

清除本机记忆库（会删掉事件、当前状态和证据 jpg）：

```bash
python -c "from pathlib import Path; import sys; sys.path[:0]=['common']; import nodepath; from memory import connect, delete_all; c=connect('data/memory.db'); delete_all(c,'data/evidence'); c.close(); print('已清除')"
```

## 怎么把剩下的代码加进来

一个人负责一个目录，例如你改问答，就只动 `qa/`。新文件放进对应节点的目录，**文件名不要和别的目录重复**（程序靠文件名查找模块）。不要在这些目录里加 `__init__.py`。

新的可执行脚本，把 `demo/run_demo.py` 开头查找 `common/paths.py` 的那几行抄上去，再写自己的 `import`。查找配置和数据库时用仓库根下的 `config/` 和 `data/`，不要用「这个文件旁边」。

下面是一套从零开始的 Git 命令。把尖括号换成自己的分支名和说明。分支名用小写英文或拼音，中间用短横线。

```bash
git status
git pull
git checkout -b qa-wrong-one-character
```

改完 `qa/` 之后先看一眼将要提交什么。列表里如果出现 `.pt`、`.zip`、`.db`、`__pycache__` 或 `data/`，不要加进去。

```bash
git status
git add qa/
git commit -m "问答：错一个字时仍能对上钥匙"
git push -u origin qa-wrong-one-character
```

然后在 GitHub 网页上打开 Pull Request，合并进 `main`。不要在 `main` 上直接改。

第二次以及以后，别人已经合并过新代码时：

```bash
git checkout main
git pull
git checkout -b memory-speech-column
```

只想看自己改了哪些文件：`git status` 和 `git diff`。提交信息写「改了什么」，不要写「update」。

这次整理之前，第一次上传的历史里仍然留着权重、缓存和安装包。克隆仓库时那些旧文件还会被下载一次，但它们已经不在当前文件树里。不要把它们再 `git add` 回来。若以后要给仓库瘦身，需要单独改写历史，那是另一件事，做之前先和大家说好。
