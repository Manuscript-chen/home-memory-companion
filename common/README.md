# 启动时用的两份路径代码

| 文件 | 作用 |
| --- | --- |
| `paths.py` | 仓库根、`config/`、`data/`、`models/` 的位置 |
| `nodepath.py` | 把各节点目录加进 `sys.path`，这样原来的 `from memory import connect` 不用改成带点的包名 |

入口脚本都会先找到 `common/paths.py`，再 `import nodepath`。新的可执行脚本照抄 `demo/run_demo.py` 开头那几行。

这些目录**不是** Python 包：不要加 `__init__.py`。模块名就是文件名，所以两个目录不能有同名的 `.py`。
