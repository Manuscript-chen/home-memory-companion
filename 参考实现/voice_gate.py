"""按键限定哪一段采样要送去识别。识别文本再交给 answer.py。"""


class PushToTalk:
    def __init__(self):
        self.holding = False
        self._chunks = []

    def press(self):
        self.holding = True
        self._chunks = []

    def push_audio(self, payload):
        if self.holding:
            self._chunks.append(payload)

    def release(self):
        self.holding = False
        chunks = self._chunks
        self._chunks = []
        if not chunks:
            return ""
        if isinstance(chunks[0], str):
            return "".join(chunks).strip()
        return b"".join(chunks)
