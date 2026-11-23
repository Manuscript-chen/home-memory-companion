# 实验脚本

这些文件原来堆在仓库根目录，路径是按「上一级目录里有参考实现和工具」来写的。现在脚本在 `experiments/`，仓库根就是原来的上一级。

它们做闭集对照、语料下载、嵌入微调和实验记录，**不是**老人家里每天要跑的程序。日常演示用 `demo/run_demo.py`。

| 文件 | 在做什么 |
| --- | --- |
| `run_experiment.py` | 按节点把仿真用例跑一遍，结果写到 `experiments/results/` |
| `train_closed.py` | 用合成图做闭集对照，起点权重是 `models/yolo26n.pt` |
| `train_real.py` | 用 LVIS 真实照片做闭集训练 |
| `fetch_lvis_set.py` | 下载 LVIS / COCO 图片，做成 YOLO 训练集 |
| `fetch_commons_set.py` | 从维基共享资源下载钥匙、眼镜盒、药盒照片 |
| `make_distant.py` | 把核对过的照片放到更远的台面上 |
| `use_camera_photos.py` | 用实物照片替换仿真灰块。原先写死了 `C:\Users\Lenovo\Desktop\项目`，现已改成仓库相对路径 |
| `finetune_embed.py` | 微调 `bge-small-zh`。只有留出集明显更好才改 `config/recall.yaml` |
| `library_view.py` | 只读查看已有的库和证据，不写业务数据 |
| `serve.py` | 实验页面。页面文件应在 `experiments/web/`（这次上传里没有） |
| `write_paper.py` | 把实验结果写成 Word，输出到仓库根的 `论文/` |

输出目录（`experiments/data/`、`experiments/results/`、`experiments/runs/`、`论文/`）已忽略，不要提交。

这次**没有**一起上传的实验材料：

- LVIS / COCO 压缩包和整理后的 `experiments/data/real_set/`
- 闭集合成图 `experiments/data/closed_set/`
- 脚本里点名的更多核对照片（`07.jpg`、`12.jpg`、`15.jpg`、`20.jpg`、`24.jpg`、`27.jpg`、`33.jpg` 等）。仓库里现有的 7 张在 `docs/camera_review/`，是另一批维基共享资源缩略图，对不上这些文件名
- `experiments/web/`、已跑完的 `results/` 和 `runs/`
- `models/` 下的权重（见 `docs/weights.md`）
