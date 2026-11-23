# 节点 G：双表记忆

SQLite，WAL。`events` 只追加。`object_states` 每个物品一行，视觉列和口述列分开，后写的口述不能抹掉摄像头看到的槽位。

| 文件 | 作用 |
| --- | --- |
| `memory.py` | 建库、写入、查询、72 小时后的证据清理、整库清除 |

数据库在 `data/memory.db`，由程序创建，不进 Git。证据 JPEG 在 `data/evidence/`，只在视觉上的放下和拿走时各存一张。

服药、锁门、关窗不写这两列，它们走 `reminders/remind.py` 的计划。看见药盒不能写成已经服过药。

向量是节点 H2 的备用，代码在 `qa/vec_memory.py`，不在本目录。
