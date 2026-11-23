"""运动触发。面积用占画面的比例，时间用秒。"""


class Trigger:
    def __init__(self, enter=0.01, exit_ratio=0.6, hits=3, cooldown_s=2.0):
        self.enter = enter
        self.exit = enter * exit_ratio
        self.hits_need = hits
        self.cooldown_s = cooldown_s
        self.hits = 0
        self.on = False
        self.cool_until = -1.0
        self.freeze_until = -1.0
        self.prev_brightness = None

    def freeze(self, now, seconds=10.0):
        self.freeze_until = now + seconds
        self.hits = 0
        self.on = False

    def observe_brightness(self, now, brightness):
        if self.prev_brightness is not None:
            base = max(self.prev_brightness, 1.0)
            if abs(brightness - self.prev_brightness) / base > 0.30:
                self.freeze(now, 10.0)
                self.prev_brightness = brightness
                return True
        self.prev_brightness = brightness
        return False

    def update(self, now, area_ratio):
        if now < self.freeze_until:
            self.hits = 0
            self.on = False
            return False
        if self.on:
            if area_ratio < self.exit:
                self.on = False
                self.hits = 0
                self.cool_until = now + self.cooldown_s
            return self.on
        if now < self.cool_until:
            self.hits = 0
            return False
        if area_ratio >= self.enter:
            self.hits += 1
        else:
            self.hits = 0
        if self.hits >= self.hits_need:
            self.on = True
        return self.on
