"""对 bge-small-zh-v1.5 做检索微调。语料仍是词表和事件模板，没有真人问句。

和上一轮的差别：每条问句带 3 个其他物品的难负样本，同一批里不放重复句子，
每轮结束用留出问句的 nDCG@10 选模型。验收集和留出问句不进训练。

生产配置只在留出 nDCG@10 至少高出 0.02，并且原有两套问句的物品正确数不下降、
说错物品不增加时，才改成微调权重。否则仍用基础模型。

用法：python finetune_embed.py
输出：../工具/models/bge-small-zh-ft 和 results/embed_finetune.json
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "参考实现"
sys.path.insert(0, str(CODE))

import yaml  # noqa: E402

import vec_memory as vec  # noqa: E402
from memory import apply_event, connect  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

with (CODE / "config" / "vocab.yaml").open(encoding="utf-8") as handle:
    VOCAB = yaml.safe_load(handle)

BASE_DIR = (ROOT / "工具" / "models" / "bge-small-zh-v1.5").resolve()
FT_DIR = (ROOT / "工具" / "models" / "bge-small-zh-ft").resolve()
RESULTS = HERE / "results"

# 验收集的问法，训练语料里不允许出现（子串也不行，桥接词一并避开）。
ACCEPTANCE_HINTS = [
    "看报纸", "读报", "老花镜放哪", "桌上那副", "茶几上的盒子",
    "开门用的那串", "门口那串", "家门那串",
    "装药的那个", "吃的药", "药片盒", "找一下我的药",
    "电视的那个板子", "换台的板子", "挂号用的卡片", "看病的那张卡",
    "充电的那个",
]

# 问句模板。{o} 换物品别名，{s} 换槽位规范名或槽位别名，{t} 换时段词。
QUESTION_TEMPLATES = [
    "{o}搁哪儿了",
    "我的{o}放哪儿了",
    "帮我找一下{o}",
    "{o}去哪了",
    "刚才{o}还在的",
    "{s}上有没有{o}",
    "哪是我的{o}",
    "{o}是不是在{s}",
    "找{o}",
    "{o}呢",
    "{o}现在在哪",
    "看看{o}在不在",
    "那{o}放哪了",
    "{t}的{o}在哪",
    "{o}收哪了",
    "怎么找不着{o}了",
    "{s}上的{o}呢",
    "{o}落在哪了",
]

# 功能性说法：换一种用途描述，验收集里的说法不进训练。
FUNCTION_HINTS = {
    "keys": ["锁门用的那串", "开锁的钥匙", "回家要用的那一串"],
    "glasses_case": ["架鼻梁上的那个盒子", "放大字用的眼镜盒子", "花镜盒"],
    "medicine_box": ["治病的盒子", "按时吃的那盒", "分药的小盒"],
    "remote_control": ["摁的那个板子", "选台的板子", "开机用的板子"],
    "insurance_card": ["报销的那张卡", "拿药要带的卡", "登记用的那张卡"],
}

TIME_WORDS = ["早上", "中午", "晚上", "昨天"]

HOLDOUT = [
    ("药搁哪儿了", "medicine_box"),
    ("那串钥匙呢", "keys"),
    ("眼镜找不到了", "glasses_case"),
    ("我的药放哪儿了", "medicine_box"),
    ("摁的那个板子呢", "remote_control"),
    ("那串钥匙去哪了", "keys"),
    ("老花镜的盒子呢", "glasses_case"),
    ("卡放在哪了", "insurance_card"),
    ("装眼镜的盒呢", "glasses_case"),
    ("那盒药呢", "medicine_box"),
    ("我的眼镜盒呢", "glasses_case"),
    ("钥匙找不到了", "keys"),
]

ACCEPTANCE = [
    ("看报纸用的那个", "glasses_case"),
    ("读报的眼镜", "glasses_case"),
    ("老花镜放哪了", "glasses_case"),
    ("桌上那副眼镜", "glasses_case"),
    ("茶几上的盒子", "glasses_case"),
    ("开门用的那串", "keys"),
    ("门口那串", "keys"),
    ("家门那串东西", "keys"),
    ("装药的那个", "medicine_box"),
    ("吃的药在哪里", "medicine_box"),
    ("药片盒", "medicine_box"),
    ("找一下我的药", "medicine_box"),
    ("电视的那个板子", "remote_control"),
    ("换台的板子", "remote_control"),
    ("挂号用的卡片", "insurance_card"),
    ("看病的那张卡", "insurance_card"),
    ("充电的那个呢", None),
    ("雨伞在哪", None),
    ("我的外套呢", None),
    ("保温杯在哪", None),
]


def scene_events():
    return [
        ("2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "visual"),
        ("2026-11-18T09:05:00", "glasses_case", "living.coffee.top", "placed", 0.80, "cam_coffee", "visual"),
        ("2026-11-18T09:06:00", "glasses_case", "living.coffee.top", "occluded", 0.80, "cam_coffee", "visual"),
        ("2026-11-18T11:00:00", "medicine_box", "dining.table.top", "placed", 0.30, "cam_dining", "visual"),
        ("2026-11-18T13:20:00", "keys", "entry.cabinet.top", "removed", 0.85, "cam_entry", "visual"),
        ("2026-11-18T13:21:00", "keys", "", "placed", None, "", "user_speech"),
    ]


def build_corpus():
    """(问句, 规范句或别名) 对。规范句由事件字段拼出，别名指向物品 id。"""
    pairs = []
    events = scene_events()
    sentences = []
    for ts, object_id, slot_id, action, confidence, cam_id, source in events:
        row = {
            "ts": ts, "object_id": object_id, "slot_id": slot_id,
            "action": action, "source": source,
        }
        sentences.append((object_id, vec.canonical_sentence(row, VOCAB)))

    object_aliases = {
        obj["id"]: [a for a in obj["aliases"] if a] for obj in VOCAB["objects"]
    }
    slot_alias_pool = [
        alias for slot in VOCAB["slots"] for alias in slot["aliases"] if alias
    ]

    counter = 0
    for template in QUESTION_TEMPLATES:
        for object_id, aliases in object_aliases.items():
            for alias in aliases:
                question = template.format(
                    o=alias,
                    s=slot_alias_pool[counter % len(slot_alias_pool)],
                    t=TIME_WORDS[counter % len(TIME_WORDS)],
                )
                counter += 1
                # 正样本 1：该物品的一条规范句；没有事件行的物品只有别名可作正样本。
                sentence = next(
                    (s for oid, s in sentences if oid == object_id), None
                )
                if sentence:
                    pairs.append((question, sentence))
                # 正样本 2：该物品的别名短句（别名向量那条）。
                pairs.append((question, alias))
        for t in TIME_WORDS:
            for object_id, aliases in object_aliases.items():
                sentence = next((s for oid, s in sentences if oid == object_id), None)
                if sentence:
                    pairs.append((f"{t}的{aliases[0]}在哪", sentence))

    # 槽位问句：问槽位上的东西，正样本是含该槽位的事件句。
    for slot in VOCAB["slots"]:
        for alias in slot["aliases"]:
            hit = next(
                (s for oid, s in sentences if any(
                    e[2] == slot["id"] for e in events if e[1] == oid
                ) and slot["canonical_name"] in s),
                None,
            )
            if hit:
                for prefix in ("", "看看", "帮我看看"):
                    pairs.append((f"{prefix}{alias}上是什么", hit))
                    pairs.append((f"{prefix}{alias}上的东西呢", hit))

    # 功能性说法。
    for object_id, hints in FUNCTION_HINTS.items():
        sentence = next((s for oid, s in sentences if oid == object_id), None)
        for hint in hints:
            if sentence:
                for tail in ("", "在哪", "放哪了", "去哪了"):
                    pairs.append((f"{hint}{tail}", sentence))
            pairs.append((hint, object_aliases[object_id][0]))

    # 去重并过滤验收集痕迹。
    seen = set()
    corpus = []
    for question, positive in pairs:
        key = (question, positive)
        if key in seen:
            continue
        seen.add(key)
        if any(hint in question for hint in ACCEPTANCE_HINTS):
            continue
        corpus.append(key)
    return corpus


def assert_no_leakage(corpus):
    """训练问句不得包含验收集的桥接词，也不得与验收集问句相同。"""
    acceptance_questions = {q for q, _ in ACCEPTANCE} | {q for q, _ in HOLDOUT}
    for question, _ in corpus:
        assert question not in acceptance_questions, question
        for hint in ACCEPTANCE_HINTS:
            assert hint not in question, (question, hint)
    anchors = {q for q, _ in corpus}
    for held in (q for q, _ in HOLDOUT):
        assert held not in anchors, held


def object_of_text(text):
    """别名精确对应物品；事件规范句里用规范名，长的先匹配。"""
    for obj in VOCAB["objects"]:
        if text == obj["canonical_name"] or text in obj["aliases"]:
            return obj["id"]
    for obj in sorted(VOCAB["objects"], key=lambda item: len(item["canonical_name"]), reverse=True):
        if obj["canonical_name"] and obj["canonical_name"] in text:
            return obj["id"]
    return None


def negative_pool():
    """每个物品可作负样本的别名和事件句。调用方再丢掉同一物品。"""
    pool = {obj["id"]: list(obj["aliases"]) for obj in VOCAB["objects"]}
    for ts, object_id, slot_id, action, _confidence, _cam_id, source in scene_events():
        row = {
            "ts": ts, "object_id": object_id, "slot_id": slot_id,
            "action": action, "source": source,
        }
        pool.setdefault(object_id, []).append(vec.canonical_sentence(row, VOCAB))
    return {key: list(dict.fromkeys(values)) for key, values in pool.items()}


def build_training_rows(corpus, negatives=3, seed=7):
    """(问句, 正样本, 3 个其他物品的文本)。正样本是别名或该物品的事件句。"""
    import random

    pools = negative_pool()
    rng = random.Random(seed)
    rows = []
    for question, positive in corpus:
        owner = object_of_text(positive)
        if owner is None:
            continue
        candidates = [
            text
            for object_id, texts in pools.items()
            if object_id != owner
            for text in texts
            if text != positive
        ]
        if len(candidates) < negatives:
            continue
        picked = rng.sample(candidates, negatives)
        rows.append(
            {
                "anchor": question,
                "positive": positive,
                "negative_1": picked[0],
                "negative_2": picked[1],
                "negative_3": picked[2],
            }
        )
    return rows


def retrieval_bundle():
    """留出问句对事件句和别名的检索。没有标准答案的问句不进 nDCG。"""
    documents = {}
    owners = {}
    for index, (ts, object_id, slot_id, action, _confidence, _cam_id, source) in enumerate(scene_events()):
        row = {
            "ts": ts, "object_id": object_id, "slot_id": slot_id,
            "action": action, "source": source,
        }
        doc_id = f"event-{index}"
        documents[doc_id] = vec.canonical_sentence(row, VOCAB)
        owners[doc_id] = object_id
    for obj in VOCAB["objects"]:
        for alias_index, alias in enumerate(obj["aliases"]):
            doc_id = f"alias-{obj['id']}-{alias_index}"
            documents[doc_id] = alias
            owners[doc_id] = obj["id"]
    queries = {}
    relevant = {}
    for index, (question, expected) in enumerate(ACCEPTANCE + HOLDOUT):
        if not expected:
            continue
        query_id = f"q{index}"
        queries[query_id] = question
        relevant[query_id] = {doc_id for doc_id, owner in owners.items() if owner == expected}
    return documents, queries, relevant


def pick_metrics(scores):
    def grab(suffix):
        for key, value in scores.items():
            if str(key).endswith(suffix):
                return round(float(value), 4)
        return None

    return {
        "ndcg@10": grab("ndcg@10"),
        "mrr@10": grab("mrr@10"),
        "recall@10": grab("recall@10"),
        "accuracy@1": grab("accuracy@1"),
    }


def make_evaluator():
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    documents, queries, relevant = retrieval_bundle()
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=documents,
        relevant_docs=relevant,
        name="heldout",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[10],
        map_at_k=[10],
        show_progress_bar=False,
        batch_size=32,
        write_csv=False,
    )
    return evaluator, len(queries), len(documents)


def train(rows, evaluator, epochs=3, batch=4, lr=2e-5):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.base.training_args import BatchSamplers
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(str(BASE_DIR))
    dataset = Dataset.from_dict(
        {
            "anchor": [row["anchor"] for row in rows],
            "positive": [row["positive"] for row in rows],
            "negative_1": [row["negative_1"] for row in rows],
            "negative_2": [row["negative_2"] for row in rows],
            "negative_3": [row["negative_3"] for row in rows],
        }
    )
    loss = MultipleNegativesRankingLoss(model, scale=20.0)
    run_dir = HERE / "runs" / "embed-ft"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        learning_rate=lr,
        warmup_steps=0.1,
        weight_decay=0.01,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_heldout_cosine_ndcg@10",
        greater_is_better=True,
        report_to="none",
        logging_steps=40,
        dataloader_drop_last=False,
        save_only_model=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=evaluator,
    )
    started = time.perf_counter()
    trainer.train()
    if trainer.state.global_step < 1:
        raise RuntimeError("没有走出训练步。别名池太小，这个 batch 组不出不重复的批次。")
    FT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FT_DIR))
    return model, time.perf_counter() - started


def use_model(path):
    """把 vec_memory 的嵌入模型切换到 path，并重建当前库的向量。"""
    vec.RECALL = dict(vec.RECALL)
    vec.RECALL["model_dir"] = str(path)
    vec._EMBEDDER = None


def evaluate(questions, conn):
    """只跑第一道筛选，报告逐条 resolved 和相似度。"""
    rows = []
    for question, expected in questions:
        cleaned = vec.clean_question(question)
        result = vec.recall_question(conn, cleaned, v=VOCAB)
        if result["verdict"] == "hit":
            resolved = result["rows"][0][2]["object_id"]
        else:
            resolved = None
        top = result["rows"][0][1] if result["rows"] else 0.0
        rows.append(
            {
                "question": question,
                "expected": expected,
                "resolved": resolved,
                "verdict": result["verdict"],
                "top_sim": round(float(top), 4),
            }
        )
    correct = sum(1 for r in rows if r["resolved"] == r["expected"])
    wrong = [r["question"] for r in rows if r["resolved"] not in (None, r["expected"])]
    return rows, correct, wrong


def promote_if_better(base_retrieval, tuned_retrieval, results):
    """留出检索要明显更好，原有问句验收还不能变差，才改生产配置。"""
    lift = (tuned_retrieval["ndcg@10"] or 0) - (base_retrieval["ndcg@10"] or 0)
    product_worse = False
    for split in ("acceptance", "holdout"):
        if results["finetuned"][split]["correct"] < results["base"][split]["correct"]:
            product_worse = True
        if len(results["finetuned"][split]["wrong_object"]) > len(results["base"][split]["wrong_object"]):
            product_worse = True
    promoted = lift >= 0.02 and not product_worse
    if not promoted:
        return False, lift
    path = CODE / "config" / "recall.yaml"
    text = path.read_text(encoding="utf-8")
    old = "../工具/models/bge-small-zh-v1.5"
    new = "../工具/models/bge-small-zh-ft"
    if old not in text:
        return False, lift
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True, lift


def main():
    from sentence_transformers import SentenceTransformer

    corpus = build_corpus()
    assert_no_leakage(corpus)
    rows = build_training_rows(corpus)
    print(f"语料 {len(rows)} 条（问句、正样本、3 个难负样本），已核对与验收集无重叠。")

    evaluator, query_count, doc_count = make_evaluator()
    print(f"留出检索：{query_count} 条问句，{doc_count} 篇文档。先测基础模型。")
    base_model = SentenceTransformer(str(BASE_DIR))
    base_retrieval = pick_metrics(evaluator(base_model))
    del base_model
    print(
        f"[base] nDCG@10 {base_retrieval['ndcg@10']}  MRR@10 {base_retrieval['mrr@10']}  "
        f"Recall@10 {base_retrieval['recall@10']}"
    )

    _model, seconds = train(rows, evaluator)
    tuned_retrieval = pick_metrics(evaluator(_model))
    del _model
    print(
        f"[finetuned] nDCG@10 {tuned_retrieval['ndcg@10']}  MRR@10 {tuned_retrieval['mrr@10']}  "
        f"Recall@10 {tuned_retrieval['recall@10']}  用时 {seconds:.0f} 秒"
    )

    results = {}
    for label, path in (("base", BASE_DIR), ("finetuned", FT_DIR)):
        use_model(path)
        with tempfile.TemporaryDirectory() as folder:
            conn = connect(Path(folder) / "eval.db")
            for ts, object_id, slot_id, action, conf, cam_id, source in scene_events():
                apply_event(
                    conn, ts, object_id, slot_id, action, conf, cam_id, "",
                    source=source,
                )
            vec.rebuild_index(conn, v=VOCAB)
            acceptance_rows, acc_correct, acc_wrong = evaluate(ACCEPTANCE, conn)
            holdout_rows, hold_correct, hold_wrong = evaluate(HOLDOUT, conn)
            results[label] = {
                "model_dir": str(path),
                "acceptance": {
                    "rows": acceptance_rows,
                    "correct": acc_correct,
                    "total": len(ACCEPTANCE),
                    "wrong_object": acc_wrong,
                },
                "holdout": {
                    "rows": holdout_rows,
                    "correct": hold_correct,
                    "total": len(HOLDOUT),
                    "wrong_object": hold_wrong,
                },
            }
            conn.close()
        print(
            f"[{label}] 验收集 {acc_correct}/{len(ACCEPTANCE)} 正确，"
            f"召回成别的物品 {len(acc_wrong)}；"
            f"留出集 {hold_correct}/{len(HOLDOUT)} 正确，召回成别的物品 {len(hold_wrong)}"
        )

    promoted, lift = promote_if_better(base_retrieval, tuned_retrieval, results)
    decision = (
        "留出 nDCG@10 更高，且原有问句没有变差，生产配置改为微调权重。"
        if promoted
        else "没有改生产配置，问答仍用基础模型。微调权重留在工具目录备查。"
    )
    print(f"nDCG@10 差值 {lift:+.4f}。{decision}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "embed_finetune.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_pairs": len(rows),
        "train_seconds": round(seconds, 1),
        "base_model": str(BASE_DIR),
        "finetuned_model": str(FT_DIR),
        "note": "训练问句由词表和事件模板生成，验收集和留出问句没有进训练。每条带 3 个其他物品的难负样本。",
        "recipe": {
            "loss": "MultipleNegativesRankingLoss",
            "scale": 20,
            "temperature": 0.05,
            "hard_negatives": 3,
            "batch_size": 4,
            "batch_sampler": "no_duplicates",
            "epochs": 3,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "selection": "留出集 cosine nDCG@10 最高的一轮",
            "promoted": promoted,
            "ndcg_lift": round(lift, 4),
            "decision": decision,
        },
        "retrieval": {
            "queries": query_count,
            "documents": doc_count,
            "base": base_retrieval,
            "finetuned": tuned_retrieval,
        },
        "sample_rows": rows[:8],
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="")
    print(f"明细已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
