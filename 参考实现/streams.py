"""双码流窗口和断线等待。这里不打开网络。"""


def next_delay(delay, cap=30.0):
    """读失败后的下一次等待。第一次等待由调用方从 1 秒开始。"""
    return min(float(delay) * 2.0, float(cap))


def backoff_sequence(times, cap=30.0):
    delay = 1.0
    values = []
    for _ in range(times):
        values.append(delay)
        delay = next_delay(delay, cap)
    return values


class MainWindow:
    """触发为真时读主码流；变为假之后再读 tail_s 秒，然后关掉。"""

    def __init__(self, tail_s=3.0):
        self.tail_s = tail_s
        self.open = False
        self.tail_until = None
        self.reads = 0

    def shut(self):
        self.open = False
        self.tail_until = None

    def step(self, now, trigger_on):
        """返回 (是否读取主码流, 是否刚打开窗口)。"""
        if trigger_on:
            started = not self.open
            self.open = True
            self.tail_until = None
            self.reads += 1
            return True, started
        if not self.open:
            return False, False
        if self.tail_until is None:
            self.tail_until = now + self.tail_s
        if now <= self.tail_until + 1e-9:
            self.reads += 1
            return True, False
        self.shut()
        return False, False
