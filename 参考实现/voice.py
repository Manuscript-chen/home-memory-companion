"""节点 I：按键说话（说明书第 11 节）。

按键限定哪一段采样要送去识别。识别文本交给 ask.ask_once 的同一入口，
不另写一套位置句；合成器只朗读节点 H / H2 已经填好的模板句。
热词文件由 vocab.yaml 生成，不手写第二份名单。

交互模式（需要麦克风）：空格开始，再按空格结束。物理按键是“按下到松开”，
软件演示用切换键替代，识别窗口的语义不变：两次按键之间的采样才进入识别。

回环模式（不需要麦克风）：TTS 合成问句 → ASR 识别 → 问答 → TTS 朗读回答。
合成语音比真人干净，它测的是这一段管道的通过能力，不等于安静环境的真人 20 句。
"""

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

with (ROOT / "config" / "voice.yaml").open(encoding="utf-8") as handle:
    VOICE = yaml.safe_load(handle)
with (ROOT / "config" / "vocab.yaml").open(encoding="utf-8") as handle:
    VOCAB = yaml.safe_load(handle)

PROJECT = ROOT.parent
ASR_DIR = (ROOT / VOICE["asr_dir"]).resolve()
ASR_TYPE = str(VOICE.get("asr_type", "paraformer"))
ASR_TRANSDUCER_DIR = (ROOT / VOICE.get("asr_transducer_dir", "")).resolve() if VOICE.get("asr_transducer_dir") else None
TTS_DIR = (ROOT / VOICE["tts_dir"]).resolve()
SAMPLE_RATE = int(VOICE.get("sample_rate", 16000))


def model_dirs_ready():
    base = ASR_TRANSDUCER_DIR if ASR_TYPE == "transducer" else ASR_DIR
    asr_ok = (base / "tokens.txt").exists()
    tts_ok = TTS_DIR.exists() and any(TTS_DIR.glob("*.onnx"))
    return asr_ok, tts_ok


def build_recognizer():
    """paraformer 识别更准但 sherpa-onnx 2.0.1 的 paraformer 接口不吃热词偏置；
    transducer（zipformer-zh）吃热词。热词文件两种情况都由 vocab.yaml 生成。"""
    import sherpa_onnx

    from hotwords import write_hotwords

    hotwords = write_hotwords(VOCAB, ROOT / "config" / "hotwords.txt")
    quant = VOICE.get("asr_quant") or ""
    suffix = "int8." if quant == "int8" else ""
    if ASR_TYPE == "transducer" and ASR_TRANSDUCER_DIR is not None:
        names = sorted(ASR_TRANSDUCER_DIR.glob(f"encoder-epoch*{suffix}onnx"))
        encoder = names[0]
        stem = encoder.name.split("encoder-")[1]
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(ASR_TRANSDUCER_DIR / f"decoder-{stem}"),
            joiner=str(ASR_TRANSDUCER_DIR / f"joiner-{stem}"),
            tokens=str(ASR_TRANSDUCER_DIR / "tokens.txt"),
            hotwords_file=str(ROOT / "config" / "hotwords.txt") if hotwords else "",
            hotwords_score=float(VOICE.get("hotwords_score", 1.5)),
            decoding_method="modified_beam_search",
            num_threads=int(VOICE.get("num_threads", 2)),
        )
    encoder = ASR_DIR / ("encoder.int8.onnx" if quant == "int8" else "encoder.onnx")
    decoder = ASR_DIR / ("decoder.int8.onnx" if quant == "int8" else "decoder.onnx")
    return sherpa_onnx.OnlineRecognizer.from_paraformer(
        encoder=str(encoder),
        decoder=str(decoder),
        tokens=str(ASR_DIR / "tokens.txt"),
        num_threads=int(VOICE.get("num_threads", 2)),
    )


def transcribe(recognizer, samples):
    """整段采样一次送入。句尾由按键决定，不用端点检测。"""
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    stream.accept_waveform(SAMPLE_RATE, np_zeros(int(0.4 * SAMPLE_RATE)))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return (recognizer.get_result(stream) or "").strip()


def np_zeros(n):
    import numpy as np

    return np.zeros(n, dtype=np.float32)


def resample(samples, from_rate, to_rate=SAMPLE_RATE):
    """识别模型要 16 kHz。合成器是 44.1 kHz，回环时先转采样率。"""
    if int(from_rate) == int(to_rate):
        return samples
    import math

    import numpy as np
    from scipy.signal import resample_poly

    gcd = math.gcd(int(to_rate), int(from_rate))
    out = resample_poly(np.asarray(samples, dtype=np.float32), int(to_rate) // gcd, int(from_rate) // gcd)
    return out.astype(np.float32)


def build_tts():
    import sherpa_onnx

    # 只认 1 KB 以上的 onnx。压缩包里的 model.int8.onnx 是 133 字节的占位指针，不能用。
    candidates = [p for p in TTS_DIR.glob("*.onnx") if p.stat().st_size > 1024]
    model = max(candidates, key=lambda p: p.stat().st_size)
    lexicon = TTS_DIR / "lexicon.txt"
    tokens = TTS_DIR / "tokens.txt"
    dict_dir = TTS_DIR / "dict"
    rule_fsts = ",".join(
        str(TTS_DIR / name)
        for name in ("number.fst", "date.fst", "new_heteronym.fst")
        if (TTS_DIR / name).exists()
    )
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model),
        lexicon=str(lexicon) if lexicon.exists() else "",
        tokens=str(tokens) if tokens.exists() else "",
        data_dir="",
        dict_dir=str(dict_dir) if dict_dir.exists() else "",
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits),
        rule_fsts=rule_fsts,
    )
    return sherpa_onnx.OfflineTts(config)


def speak(tts, text):
    """合成并播放。返回 (采样, 采样率)。没有扬声器时写 wav 落盘。"""
    audio = tts.generate(str(text))
    try:
        import sounddevice as sd

        sd.play(audio.samples, audio.sample_rate)
        return audio.samples, audio.sample_rate
    except Exception:
        path = ROOT / "data" / "reply.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(path, audio.samples, audio.sample_rate)
        print(f"没有扬声器，回答已写入 {path}")
        return audio.samples, audio.sample_rate


def write_wav(path, samples, sample_rate):
    import wave

    import numpy as np

    data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def answer_text(conn, text):
    from ask import ask_once
    from memory import log_recall

    reply, new_last, path = ask_once(conn, VOCAB, text, None)
    log_recall(conn, text, path, new_last)
    return reply, new_last, path


def interactive():
    """空格切换录音。松开后到出声的时延在每轮打印。"""
    import sounddevice as sd

    from memory import connect

    asr_ok, tts_ok = model_dirs_ready()
    if not (asr_ok and tts_ok):
        raise SystemExit(f"模型目录不完整：ASR {asr_ok}，TTS {tts_ok}。先放到 工具/models 下。")
    recognizer = build_recognizer()
    tts = build_tts()
    conn = connect(ROOT / "data" / "memory.db")
    state = {"recording": False, "chunks": []}

    def callback(indata, frames, time_info, status):
        if state["recording"]:
            state["chunks"].append(indata[:, 0].copy())

    print("空格开始/结束说话，q 退出。")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=1024, callback=callback):
            while True:
                import msvcrt

                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"q", b"\x03"):
                        break
                    if key == b" ":
                        if not state["recording"]:
                            state["recording"] = True
                            state["chunks"] = []
                            print("…在听")
                        else:
                            state["recording"] = False
                            release = time.perf_counter()
                            import numpy as np

                            samples = np.concatenate(state["chunks"]) if state["chunks"] else np.zeros(0, dtype=np.float32)
                            text = transcribe(recognizer, samples)
                            recognized = time.perf_counter()
                            if not text:
                                print("这一段没有听清。")
                                continue
                            print(f"听到：{text}")
                            reply, _, path = answer_text(conn, text)
                            answered = time.perf_counter()
                            print(f"回答：{reply}")
                            print(f"路径：{path}")
                            speak(tts, reply)
                            spoken = time.perf_counter()
                            print(
                                f"时延：识别 {recognized - release:.2f} 秒，"
                                f"问答 {answered - recognized:.2f} 秒，"
                                f"合成到开始播放 {spoken - answered:.2f} 秒，"
                                f"合计 {spoken - release:.2f} 秒（目标 1.5，验收 2.0）。"
                            )
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass


ROUNDTRIP_QUESTIONS = [
    "钥匙在哪",
    "钥匙串放哪了",
    "家门钥匙在哪里",
    "眼镜盒呢",
    "老花镜盒子在哪",
    "眼镜盒子放哪了",
    "药盒在哪",
    "药箱呢",
    "吃药的盒子放哪了",
    "遥控器在哪",
    "遥控板呢",
    "电视遥控器放哪了",
    "医保卡在哪",
    "社保卡呢",
    "医疗卡放哪了",
    "钥匙放哪了",
    "钥匙在哪里",
    "开门用的那串在哪",
    "装药的那个放哪了",
    "看报纸用的那个在哪",
]


def roundtrip(questions=None):
    """合成语音 → 识别 → 问答 → 合成回答。逐条报告。"""
    import numpy as np

    from memory import apply_event, connect, delete_all

    asr_ok, tts_ok = model_dirs_ready()
    if not (asr_ok and tts_ok):
        raise SystemExit(f"模型目录不完整：ASR {asr_ok}，TTS {tts_ok}。先放到 工具/models 下。")
    recognizer = build_recognizer()
    tts = build_tts()
    conn = connect(ROOT / "data" / "voice_roundtrip.db")
    evidence = ROOT / "data" / "voice_roundtrip_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    delete_all(conn, evidence)
    apply_event(conn, "2026-11-18T08:12:00", "keys", "entry.cabinet.top", "placed", 0.91, "cam_entry", "")
    apply_event(conn, "2026-11-18T09:05:00", "glasses_case", "living.coffee.top", "placed", 0.80, "cam_coffee", "")
    apply_event(conn, "2026-11-18T11:00:00", "medicine_box", "dining.table.top", "placed", 0.60, "cam_dining", "")

    rows = []
    for question in questions or ROUNDTRIP_QUESTIONS:
        t0 = time.perf_counter()
        spoken = tts.generate(question)
        synth_s = time.perf_counter() - t0
        samples = resample(spoken.samples, spoken.sample_rate)
        t1 = time.perf_counter()
        text = transcribe(recognizer, samples)
        asr_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        reply, resolved, path = answer_text(conn, text if text else question)
        ask_s = time.perf_counter() - t2
        t3 = time.perf_counter()
        reply_audio = tts.generate(reply)
        tts_s = time.perf_counter() - t3
        rows.append(
            {
                "question": question,
                "heard": text,
                "reply": reply,
                "resolved": resolved,
                "path": path,
                "synth_s": round(synth_s, 3),
                "asr_s": round(asr_s, 3),
                "ask_s": round(ask_s, 3),
                "tts_s": round(tts_s, 3),
            }
        )
        print(
            f"问「{question}」→ 听到「{text}」→ {reply} "
            f"（识别 {asr_s:.2f}s，问答 {ask_s:.2f}s，合成 {tts_s:.2f}s）"
        )
    correct = sum(1 for row in rows if row["resolved"] is not None)
    print(f"回环小计：{len(rows)} 条，识别后进入某个物品 id 的 {correct} 条。")
    conn.close()
    return rows


def main():
    args = sys.argv[1:]
    if args and args[0] == "--roundtrip":
        custom = [item for item in args[1:] if not item.startswith("-")]
        rows = roundtrip(custom or None)
        import json

        out = ROOT.parent / "实验" / "results" / "voice_roundtrip.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
        )
        print(f"明细已写入 {out}")
        return 0
    interactive()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
