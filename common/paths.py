"""仓库根目录上的共用路径。

节点里的 .py 不要再用 Path(__file__).parent 去找 config 或 data。
文件搬进子目录之后，那个 parent 不再是仓库根。
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
ASSETS_DIR = REPO_ROOT / "assets"
MODELS_DIR = REPO_ROOT / "models"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
