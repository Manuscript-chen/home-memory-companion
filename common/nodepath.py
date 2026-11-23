"""让各节点目录里的模块仍用原来的写法：from memory import connect。

每个节点一个目录，方便几个人分头改。这些目录被加进 sys.path，
所以模块名就是文件名，不带目录前缀。因此全仓库的 .py 文件名不能重名。

入口脚本在 import 本项目其他模块之前，先把 common 放进 sys.path，再 import nodepath。
"""

import sys

from paths import REPO_ROOT

# 检测节点目前没有单独的 .py，调用写在 demo/run_camera.py 里。
NODE_DIRS = (
    "config",
    "capture",
    "gating",
    "tracking",
    "state_machine",
    "memory",
    "qa",
    "voice",
    "reminders",
    "demo",
)


def install():
    for name in NODE_DIRS:
        path = str(REPO_ROOT / name)
        if path not in sys.path:
            sys.path.append(path)


install()
