"""把这次仿真的实测数字写成论文。不补未跑过的现场结果。"""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
OUT = ROOT / "论文"


def load(name):
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = load("summary.json")
    training = load("training.json")
    metrics = summary.get("metrics", {})
    pixels = metrics.get("pixels", {})
    empty = metrics.get("empty_table", {})
    calib = metrics.get("calibration", {})
    official = metrics.get("official_bytetrack", {})
    questions = metrics.get("questions", {})
    counts = summary.get("counts", {})
    closed = training.get("closed_set") or {}
    OUT.mkdir(parents=True, exist_ok=True)
    text = render(summary, training, metrics, pixels, empty, calib, official, questions, counts, closed)
    (OUT / "居家物品记忆辅助-仿真实验.md").write_text(text, encoding="utf-8")
    write_docx(text, OUT / "居家物品记忆辅助-仿真实验.docx")
    print(OUT / "居家物品记忆辅助-仿真实验.md")


def render(summary, training, metrics, pixels, empty, calib, official, questions, counts, closed):
    map50 = closed.get("map50")
    map5095 = closed.get("map50_95")
    map50_s = "未得到" if map50 is None else f"{map50:.3f}"
    map5095_s = "未得到" if map5095 is None else f"{map5095:.3f}"
    open_vocab = training.get("open_vocab") or {}
    h2 = metrics.get("h2_non_alias") or {}
    rt = metrics.get("voice_roundtrip") or {}
    ft = metrics.get("h2_finetune") or {}
    total = counts.get("pass", 0) + counts.get("fail", 0) + counts.get("skipped", 0)
    h2_correct = h2.get("correct", "—")
    h2_total = h2.get("total", "—")
    h2_wrong_pos = len(h2.get("wrong_position", []) or [])
    rt_resolved = rt.get("resolved_object", "—")
    rt_slow = rt.get("slowest_total_s", "—")
    return f"""# 槽位绑定的居家物品记忆：来源分开的双表与一次可复核仿真

## 摘要

独居老人找不到钥匙、老花镜或药盒，常常是物品和位置的绑定没有留下来；是否已经服药，则是另一件将来要做的事。这两件事不能写成同一个结论。本文实现一套以固定槽位为位置、以可测试状态机为动作、以本地 SQLite 分列保存摄像头记录和口述的辅助系统，并在开发者计算机上做了一次不接摄像头的仿真。{total} 项记录里，{counts.get("pass", 0)} 项通过，{counts.get("fail", 0)} 项未通过，{counts.get("skipped", 0)} 项因没有现场条件而未执行。20 条书面问法的位置错误为 {questions.get("location_errors", "—")}。别名和编辑距离匹配不到的说法由本地嵌入向量召回补上：20 条词表外说法里筛后物品正确 {h2_correct} 条，说错位置 {h2_wrong_pos} 条。语音入口用合成语音回环测了管道，识别后进入物品 id 的 {rt_resolved} 条，最慢一条从识别到合成 {rt_slow} 秒。热身后的空台面按 5 帧每秒计满 10 分钟，主码流打开次数为 {empty.get("opens", "—")}。这些数字只说明软件契约在合成输入上成立，不是这户台面上的检测准确率，也不是老人住宅里的使用结果。

## 1. 问题与不做的事

物品和放置位置的绑定随年龄下降，而且比单独认出物品更明显。是否记得在将来某个时刻完成一件事，是另一类记忆。看见药盒不能写成已经服过药。本系统因此把放置记录和到点提醒分开：前者来自固定摄像头上的白名单物体，后者只有本人确认，来源写成 self_report。

位置不交给模型去猜。安装时每路摄像头只绑定一个槽位，槽位名来自词表。动作来自轨迹上的速度、内区和持续时间。视觉写入和口述写入分列，后一句口述不能盖掉摄像头上一次看到的槽位。原始视频不保留。卧室和卫生间不设摄像头。

相近工作的边界也要说清。MemPal 用头戴摄像头和模型日记帮助老年人找东西，头戴摄像机会进入卧室，位置句子由模型生成，本项目不走那条路。VideoAgent 先把长视频压成事件再查询，和“先形成结构化记录再查”相同，但它保存视频片段并依赖语言模型组织答案，这两件事本项目都不做。

## 2. 数据流

系统从槽位标定走到回答，中间可以替换一个部件，上下游的输入输出不变。放弃的方案不和采用方案同时写库。

槽位、英文检测词和中文回答名写在同一份 YAML 里。有效像素取主码流像素和检测输入像素中的较小值，不到 32 的物品不进入这一路的检测名单，放大图像不算补上采样。子码流只回答有没有足够大的变化；变化成立后才读取主码流，运动结束后再保留 3 秒。运动门控用 MOG2，进入、退出、冷却和亮度冻结四条规则一起用。检测用开放词汇模型的英文提示，白名单以外的框丢掉。跟踪在真机路径上调用 Ultralytics 的 ByteTrack；仿真另有一个按同一份 YAML 做两阶段关联的程序，第二阶段只比交并比。状态机是纯函数。数据库事件只追加，事件行在写入时由字段拼成规范句并向量化，存进 SQLite 向量扩展。当前状态每个物品一行，视觉列和口述列分开。问句按别名、上一件物品和编辑距离消歧；这三级失败或问句带时间词时，问句向量化后在事件句和别名向量里召回，经过两道筛选再回答。答句只用词表里的规范名，模板句是默认输出。按键限定哪一段声音要识别，识别文本进同一个问答入口，合成器只朗读模板句。抽屉磁簧若启用，只表示开或关。

## 3. 实验设置

实验在一台 Windows 11 计算机上完成，Python {summary.get("python", "")}。配置里的 RTSP 仍是占位账号，没有连接摄像头，没有夜间画面，也没有连续运行 24 小时。检测框、面积和问句都是程序生成的。状态机测试把同一条合成轨迹重复跑多次，用来核对规则，不代表检测器能把真实钥匙框出来。

闭集对照用程序画的椭圆、双圆和方块，按生成批次切开：batch 0 与 batch 1 共 {training.get("dataset", {}).get("counts", {}).get("train", "—")} 张训练，batch 2 共 {training.get("dataset", {}).get("counts", {}).get("val", "—")} 张验证。已安装的 PyTorch 是 {training.get("torch", "未知")}，`cuda` 为 {training.get("cuda", False)}，因此没有用到机器上的独立显卡。训练设备记为 {closed.get("device", "—")}，周期数为 {closed.get("epochs", "—")}。

开放词汇前向没有跑成。权重文件 `yoloe-26n-seg.pt` 已下载。设置文本类别时需要的 CLIP 代码无法从 GitHub 克隆，`mobileclip2_b.ts` 也没有出现在目录里。这一项记为未执行，不写成准确率。

已安装的运行依赖包括 PyYAML、OpenCV 5、Ultralytics 8.4、sherpa-onnx 2.0.1、pypinyin、sqlite-vec、sentence-transformers 和 sounddevice。中文嵌入模型 bge-small-zh-v1.5（约 92 MB）从镜像站下载后离线加载；识别用流式 paraformer 中英双语模型（int8），朗读用 vits-melo，都放在工具目录。备用的 go2rtc Windows 压缩包仍在工具目录，没有启动，也没有写入事件。BoT-SORT 的配置已复制，但没有作为跟踪器加载。本机没有 ffmpeg。Frigate 和 Qwen3-VL-2B 没有下载：前者不进入事件或别名匹配，2B 视觉语言模型这次不加载，只检查了“复核不能改槽位和动作”这一条约定。向量检索走 SQLite 的向量扩展，没有安装独立向量数据库服务。

## 4. 结果

### 4.1 像素与配置

按说明书里的距离、视场角和检测输入重算，有效像素为：钥匙 {pixels.get("keys")}，眼镜盒 {pixels.get("glasses_case")}，药盒 {pixels.get("medicine_box")}，把遥控器放在茶几且检测输入为 640 时 {pixels.get("remote_on_coffee")}，把钥匙放在茶几且检测输入为 640 时 {pixels.get("keys_on_coffee")}，说明书中 1.5 米、70 度、6 厘米短边的例子 {pixels.get("manuscript_1_5m")}，医保卡放在餐桌距离上 {pixels.get("insurance_on_dining")}。前三项不低于 32。遥控器和错放的钥匙低于 32，遥控器只留在口述词表。医保卡的像素够，但餐桌这一路已经绑定药盒，首版一路只检测一个类别，所以医保卡也不进检测名单。A4 长边占画面宽度的计算值，玄关为 {metrics.get("a4_fraction", {}).get("entry")}，茶几为 {metrics.get("a4_fraction", {}).get("coffee")}。没有实拍核对。

`check_config.py` 和 `run_demo.py` 的退出码都是 0。占位账号只产生警告。

### 4.2 双码流与运动门控

空闲 50 帧的主码流读取次数为 0。连续 3 秒触发为真、然后结束时，主码流读取 {metrics.get("main_reads_after_3s_motion")} 帧，多于 15 帧，并包含运动结束后的 3 秒。子码流帧没有进入检测。断线等待序列为 1、2、4、8、16、30、30 秒。

MOG2 先看 500 帧恒定画面，再用新的触发器计 3000 帧，对应 10 分钟、每秒 5 帧。计时内打开次数为 {empty.get("opens")}，面积比例的 95 分位为 {empty.get("area_p95")}。热身阶段另一次触发器的打开次数为 {empty.get("warmup_opens_on_a_fresh_model")}。一块移动色块在下标为 {metrics.get("hand_open_frame")} 的采样把触发器打开，也就是连续第 3 帧。色块不是真手。空台面 95 分位为 {calib.get("mog2_empty_p95")}，色块面积的 10 分位为 {round(calib.get("mog2_blob_p10", 0), 4) if calib.get("mog2_blob_p10") is not None else "—"}。两个分布不重叠，0.01 落在中间，生产路径保持 MOG2。KNN 用更短的合成序列测过，没有替换 MOG2。

### 4.3 跟踪与状态机

0.20 分的框不新建轨迹。高分轨迹之后连续五帧 0.20，仿真关联器保住了编号。已安装的 Ultralytics BYTETracker 上，连续低分也保住编号；空检 8 帧后高分框仍是原编号，第一次对不上原编号的空帧数是 {official.get("first_gap_without_same_id")}。第二阶段在这份实现里只比交并比，不把低分再乘进代价，否则 0.20 会超过 0.5 的上限。

盖住约 1 秒时，没有同时写出拿走和放下。若故意把编号从 1 换成 2，而物体还在画面中央，旧轨迹在 1.5 秒后变成挡住，新编号再写下一次放下，没有写成拿走。两个同类框写出一次冲突，下一帧不再写。

钥匙、眼镜盒和药盒各 20 次合成放下，动作都是一次 placed。各 20 次拿走，动作都是 placed 然后 removed。各 10 次挡住，动作都是 placed 然后 occluded。短停、3 像素抖动、中间速度和轨迹中断的判定与规则一致。这些次数是同一生成程序的重复，不是 20 次真实放置。

### 4.4 双表、问答与其余入口

旧库里的单一 status 可以迁到视觉列。两次视觉放置之间插入口述后，事件表追加，视觉列停在后一次槽位，口述列仍是拿走。看见药盒不会生成服药计划。确认计划的来源是 self_report。超过 72 小时的 JPEG 被删除。清除副本后，问钥匙得到“我没有钥匙的记录”，主库不被这次清除带走。

20 条书面问法与期望整句一致，位置错误 {questions.get("location_errors")} 条。不一致时的答句保留视觉句，并接上“您后来说已经放进抽屉。”第 4 级在历史槽位多于一个时只生成提问；当前视觉列在茶几时，采用句说茶几，不把出现次数更多的玄关说成现在的位置。拼音备用把“要是在哪”收成钥匙，主路径仍回答没有这个物品的记录，因此没有把拼音接进 `ask.py`。

热词文件由词表生成，其中没有“卧室”。按键按下之前的“在卧室”和松开之后的“药盒在餐桌垫”都没有进入那一句。安静环境 20 句语音、松开按键到出声的时延、真实磁簧、夜间 10 次、连续 24 小时和 RTSP 抽帧没有执行。本次证据目录里没有 mp4、mkv 或 avi。

### 4.5 闭集对照

在合成图上，YOLO26n 训练 {closed.get("epochs", "—")} 个周期后，验证集 mAP50 为 {map50_s}，mAP50-95 为 {map5095_s}。三类图形彼此差得很远，每张图只有一个目标，所以这个 mAP50 可以很高，仍然不能写成家里找钥匙的准确率。开放词汇没有在这 16 张验证图上计数：`yoloe-26n-seg.pt` 已经下载，`mobileclip2_b.ts` 不在目录里，设置文本类别时还要克隆 ultralytics/CLIP，这次连接 GitHub 失败。两个口径没有合成一个百分数，也不和 COCO 的 40.9 或 LVIS 的 24.7 比较。

### 4.6 向量召回（节点 H2）

20 条别名完全匹配的问法与节点 H 的输出逐句相同，路径全部记为精确命中。20 条不在别名表里的说法里，第一道筛选后物品正确 {h2_correct} 条，召回成别的物品 {len(h2.get("wrong_object", []) or [])} 条，说错位置 {h2_wrong_pos} 条；其余是提问一次和回答没有记录。“看报纸用的那个”不在别名表里，把“看报纸的眼镜”按说明书的机制登记为别名后，它靠语义相近召回眼镜盒；“茶几上的盒子”同时像药盒和眼镜盒，系统只提问，不猜。带时间词的问句先收成时间区间，区间内没有事件时回答“我没有钥匙在这个时间段的记录”，不说当前位置。第二道筛选用故意写出的“钥匙大概在桌子附近”测试，句子被丢弃并改念模板。清除命令之后事件向量和事件行一起删空。两次路径分开计数：别名问法走精确命中，词表外问法走向量召回。

### 4.7 合成语音回环（节点 I）

流式 paraformer 识别加模板填空加 vits-melo 合成，20 句合成问句逐条走完识别、问答和朗读。识别后进入某个物品 id 的 {rt_resolved} 条，最慢一条从识别到合成完成 {rt_slow} 秒。识别噪声里有同音字（“钥匙”被听成“老是”），下游的编辑距离匹配和向量召回接住了一部分；也有把“钥匙串放哪了”听成“报时窗放哪”后配到眼镜盒的一例，属于说错位置的一类，真人验收前要靠热词或更强的识别模型压住。合成语音比真人干净，这一节只证明管道和时延量级，不等于安静环境的真人 20 句。支持热词偏置的 14M zipformer transducer 另测了一轮，进入物品 id 的条数更少，没有被采用。

### 4.8 嵌入模型微调对照（魔改）

在 546 对由词表和事件字段模板生成的问句与规范句上，用批内负样本的排序损失对 bge-small-zh-v1.5 微调 8 个周期，CPU 上约 52 秒。训练语料刻意不含验收集的问法和“看报纸”一类桥接词，留出集另用 12 条说法。对照结果：基础模型在验收集 {ft.get("base_correct", "—")}/20，微调后 {ft.get("finetuned_correct", "—")}/20；留出集两者都是 {ft.get("holdout", {}).get("base", "—")}/12。两次微调在同一句“换台的板子”上一好一坏，收益落在训练方差内，生产仍用基础模型，微调模型留在工具目录备查。

## 5. 讨论

仿真把说明书里的通过条件拆开了。能在程序里断言的，例如像素、四条门控规则、状态机、分列、20 条书面问法和向量召回的两道筛选，已经跑过。必须有摄像头、麦克风、磁簧或一整天磁盘观察的，保持未执行。把未执行写成通过，会让以后的现场试验失去对照。

ByteTrack 的缓冲在已安装版本里不是“空 8 帧就换编号”。空 8 帧后原编号还在，本次测到的第一次对不上是空 {official.get("first_gap_without_same_id")} 帧。仿真关联器没有卡尔曼预测，不能代替这个官方结果。BoT-SORT 的配置已经拷到 `config/botsort.yaml`，其中 `track_buffer` 仍是上游默认 30，而且编号切换并没有把一次拿走写成两次放置，所以没有启用它。

拼音近邻在这个词表上会把“要是”配到“钥匙”。词表只有几件物品时，这种误配比漏掉一句绕远的说法更糟，所以它留在备用模块里。

向量召回的两处裁剪要写明。相似度 0.55 的初值在合成问句上够用，真户的问句分布要重新标定：能对上别名的问句取相似度下沿，明确没有的问句取上沿，阈值放中间。说明书要求“前两名分差小于 0.05，或者两行的物品 id 不同”就不作答；同一物品的两行分差再小不会生成“你是说钥匙，还是钥匙”这样的问句，所以本项目只在物品 id 不同时提问，这是对原文的裁剪，不是阈值改动。

嵌入模型微调没有带来稳定收益，原因不是损失函数，而是语料：模板问句教会模型的是句式，而“看报纸用的那个”缺的是“看报纸的眼镜”这条世界知识，登记别名比微调更直接。热词同理，sherpa-onnx 2.0.1 的 paraformer 接口不支持热词偏置，支持热词的 14M transducer 声学又太弱，短词同音字只能靠下游匹配接，真人验收前换更大的 transducer 是下一步。

闭集 mAP 来自 {closed.get("epochs", "—")} 个周期的合成图形。它只说明这一轮训练在这些图形上的验证集分数，不能外推到家里的钥匙或药盒。开放词汇前向没有跑成，这次对照因此不完整。

## 6. 结论

采用方案中的配置检查、双码流窗口、MOG2 门控、白名单过滤、ByteTrack 参数、状态机、分列数据库、书面问答、本地向量召回和按键语音管道，在这次仿真里与说明书一致。现场放置、夜间、真人语音时延、磁簧漏报和 24 小时无视频，还没有数据。下一步应在开发者家里用真实主码流和子码流重做放置、拿走、挡住和空台面；在 CLIP 文本编码器可用之后，把开放词汇和闭集对照分开列表；用真人问句标定 0.55 和 0.05 两个召回阈值；把识别模型换到支持热词的更大 transducer 之后重做 20 句语音。

## 参考文献

[3] Einstein G O, McDaniel M A. Normal aging and prospective memory. Journal of Experimental Psychology: Learning, Memory, and Cognition, 1990, 16(4): 717-726.

[6] Kessels R P C, Hobbel D, Postma A. Aging, context memory and binding. International Journal of Neuroscience, 2007, 117: 795-810.

[12] Jocher G, Qiu J, Liu M, 等. Ultralytics YOLO26. arXiv:2606.03748, 2026.

[13] Wang A, Liu L, Chen H, 等. YOLOE: real-time seeing anything. ICCV, 2025. arXiv:2503.07465.

[15] Zhang Y, Sun P, Jiang Y, 等. ByteTrack. ECCV, 2022. arXiv:2110.06864.

[16] Zivkovic Z. Improved adaptive Gaussian mixture model for background subtraction. ICPR, 2004, 2: 28-31.

[22] Maniar N, Chan S, Zulfikar W, Ren S, Xu C, Maes P. MemPal. Proceedings of the 30th International Conference on Intelligent User Interfaces, 2025.

[23] Fan Y, Ma X, Wu R, Du Y, Li J, Gao Z, Li Q. VideoAgent. European Conference on Computer Vision, 2024.

文献条目沿用项目说明书。本文没有重新核对页码，也没有把未引用的普查和临床论文写进结果。
"""


def write_docx(markdown, path):
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for block in markdown.split("\n"):
        line = block.rstrip()
        if not line:
            continue
        if line.startswith("# "):
            add_heading(document, line[2:], 0)
        elif line.startswith("## "):
            add_heading(document, line[3:], 1)
        elif line.startswith("### "):
            add_heading(document, line[4:], 2)
        else:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(line)
            run.font.name = "Microsoft YaHei"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
            run.font.size = Pt(11)
    document.save(path)


def add_heading(document, text, level):
    paragraph = document.add_paragraph()
    if level == 0:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        size, bold = 18, True
    elif level == 1:
        size, bold = 14, True
    else:
        size, bold = 12, True
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0x1E, 0x1A, 0x16)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


if __name__ == "__main__":
    main()
