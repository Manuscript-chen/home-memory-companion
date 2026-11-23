# 模型权重放哪里

权重不进 Git。体积大，而且换一家就能重新下载。请放到仓库根目录的 `models/`，并保持下面的文件名。程序只认这些路径。

## 日常检测要用的（节点 D，采用方案）

| 文件 | 作用 |
| --- | --- |
| `models/yoloe-26n-seg.pt` | YOLOE-26n 文本提示权重。`demo/run_camera.py` 读这一只 |
| MobileCLIP2（约 254 MB 的 `mobileclip2_b.ts`） | 第一次 `set_classes` 时由 Ultralytics 下载，用来把英文提示变成向量 |

说明与模型卡：<https://docs.ultralytics.com/models/yoloe>

在已经安装 `requirements.txt` 的虚拟环境里，于仓库根目录执行：

```bash
mkdir -p models
cd models
python -c "from ultralytics import YOLOE; YOLOE('yoloe-26n-seg.pt')"
cd ..
```

若提示文件下到了别的目录，把它移动或复制为 `models/yoloe-26n-seg.pt`。不要改名，也不要提交。

`set_classes` 还会再下载 MobileCLIP2。闭集对照脚本额外在 `models/mobileclip2_b.ts` 找它。若自动下载落在 Ultralytics 自己的缓存里，把该文件复制到 `models/mobileclip2_b.ts` 即可。

## 只做闭集对照时才要的（节点 D 的备用实验）

| 文件 | 作用 |
| --- | --- |
| `models/yolo26n.pt` | 闭集微调的起点。`experiments/train_closed.py` 和 `experiments/train_real.py` 使用 |

这不是每户交付时的检测器。开放词汇和闭集数字要分开记，不能合成一个准确率。

```bash
cd models
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
cd ..
```

YOLO26 的说明见 Ultralytics 文档（与 YOLOE 同一文档站的 YOLO26 模型页）。

## 语音（节点 I，接麦克风之前再准备）

库的版本是 sherpa-onnx 2.0.1：<https://github.com/k2-fsa/sherpa-onnx/releases/tag/v2.0.1>

预训练模型列表：<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>

按 `config/voice.yaml` 解压成下面三个目录（目录名不要改）：

| 目录 | 里面至少要有 |
| --- | --- |
| `models/asr-paraformer-zh-en/` | `encoder.int8.onnx`、`decoder.int8.onnx`、`tokens.txt`。这是默认识别器 |
| `models/asr-zipformer-zh-14m/` | `encoder-epoch*int8.onnx`、对应的 decoder 与 joiner、`tokens.txt`。只有把 `asr_type` 改成 `transducer` 才用，这种才吃热词 |
| `models/tts-melo-zh_en/` | 一个大于 1 KB 的 `*.onnx`、`lexicon.txt`、`tokens.txt`。压缩包里 133 字节的 `model.int8.onnx` 是占位指针，不能当模型 |

`config/hotwords.txt` 由 `vocab.yaml` 的别名生成，不要手写第二份名单。

## 向量召回（节点 H2，备用）

`models/bge-small-zh-v1.5/`：中文嵌入模型 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)。`config/recall.yaml` 的 `model_dir` 指向这里。不跑 `demo/run_h2.py` 或 `experiments/finetune_embed.py` 就不用下。

## 以前提交过、现已从当前版本拿掉的文件

这些文件还留在最早那次提交的历史里，克隆时仍会下载一次，但工作区里不应再出现，也不要重新 `git add`：

- `参考实现/yolo26n.pt`
- `参考实现/yoloe-26n-seg.pt`
- `参考实现/data/` 下的数据库、证据 jpg、试听 wav
- `downloads/go2rtc_win64.zip` 和 `工具/downloads/go2rtc_win64.zip`
- 全部 `__pycache__/*.pyc`
