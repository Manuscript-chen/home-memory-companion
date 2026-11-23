# 节点 D：白名单检测

采用 YOLOE-26n 的英文文本提示，权重是 `models/yoloe-26n-seg.pt`。每路只传这一路 `vocab.yaml` 里的 prompt。白名单以外的框丢掉，不进跟踪。分割掩码不参与槽位判断。

检测调用目前写在 `demo/run_camera.py` 里（构造 `YOLOE`、`set_classes`、`predict`）。这里还没有单独的模块。要把检测从摄像头脚本里拆出来时，新文件放本目录，文件名不要和别的目录重复。

不要把 `.pt`、`.onnx` 或 `mobileclip2_b.ts` 放进 Git。下载步骤见 `docs/weights.md`。

闭集微调是对照实验，脚本在 `experiments/`，不作为每户交付的前置条件。
