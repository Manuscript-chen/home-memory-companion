"""到点在屏幕上提问。确认只能记成口述，不能从摄像头推断已经做完。"""

import sys
from datetime import datetime
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
if str(_repo / "common") not in sys.path:
    sys.path.insert(0, str(_repo / "common"))
import nodepath  # noqa: E402
from paths import REPO_ROOT as ROOT  # noqa: E402

from memory import add_plan, confirm_plan, connect, due_plans, list_plans  # noqa: E402

USAGE = """用法：
  python reminders/remind.py add 吃药 2026-11-18T08:00:00
  python reminders/remind.py list
  python reminders/remind.py due
  python reminders/remind.py confirm 1
时间写成 2026-11-18T08:00:00 这种形式。确认来源固定是口述。"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        raise SystemExit(USAGE)
    command = sys.argv[1]
    conn = connect(ROOT / "data" / "memory.db")
    now = datetime.now().isoformat(timespec="seconds")
    if command == "add":
        if len(sys.argv) != 4:
            raise SystemExit(USAGE)
        title, due = sys.argv[2], sys.argv[3]
        datetime.fromisoformat(due)
        plan_id = add_plan(conn, title, due)
        print(f"已记下 #{plan_id} {title}，到 {due} 提问。")
    elif command == "list":
        rows = list_plans(conn)
        if not rows:
            print("还没有计划。")
        for row in rows:
            state = "已口述确认" if row["done"] else "未确认"
            print(f"#{row['id']} {row['title']} 到点 {row['due_ts']} {state}")
    elif command == "due":
        rows = due_plans(conn, now)
        if not rows:
            print("现在没有到点的事。")
        for row in rows:
            print(f"到点了：{row['title']}。做了吗？做了请运行 python reminders/remind.py confirm {row['id']}")
    elif command == "confirm":
        if len(sys.argv) != 3:
            raise SystemExit(USAGE)
        confirm_plan(conn, int(sys.argv[2]), now)
        print(f"#{sys.argv[2]} 已记为口述确认。这不是摄像头看见的结果。")
    else:
        raise SystemExit(USAGE)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except ValueError as exc:
        raise SystemExit(f"参数不对：{exc}") from exc
