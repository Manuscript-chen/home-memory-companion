"""实验页面和资料室。只读结果文件和仿真库，不改业务数据。"""

import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = HERE / "web"
RESULTS = HERE / "results"
_common = ROOT / "common"
if str(_common) not in sys.path:
    sys.path.insert(0, str(_common))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nodepath  # noqa: E402

import library_view  # noqa: E402


def read_json(path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def live_db():
    path = HERE / "data" / "lab.db"
    if not path.exists():
        return {"ready": False}
    from memory import connect, snapshot

    conn = connect(path)
    data = snapshot(conn)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    data["journal_mode"] = mode
    data["ready"] = True
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, body, content_type, status=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload, status=200):
        self._send(json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html"}:
            self._send((WEB / "index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
            return
        if path in {"/library", "/library/"}:
            self._send((WEB / "library.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/summary":
            self._json(read_json(RESULTS / "summary.json", {"finished": False, "counts": {}}))
            return
        if path == "/api/trace":
            self._json(read_json(RESULTS / "trace.json", []))
            return
        if path == "/api/training":
            self._json(read_json(RESULTS / "training.json", {"ready": False}))
            return
        if path == "/api/live":
            self._json(live_db())
            return
        if path == "/api/library":
            self._json(library_view.catalog())
            return
        if path == "/api/library/table":
            db_id = (query.get("db") or ["memory"])[0]
            kind = (query.get("kind") or ["events"])[0]
            try:
                payload = library_view.fetch_table(db_id, kind)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
                return
            if payload is None:
                self._json({"error": "没有这只库或这种视图"}, 404)
                return
            self._json(payload)
            return
        if path == "/api/library/training":
            try:
                self._json(library_view.training_review())
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
            return
        if path == "/api/library/portrait":
            db_id = (query.get("db") or ["memory"])[0]
            try:
                payload = library_view.portrait(db_id)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
                return
            if payload is None:
                self._json({"error": "没有这只库"}, 404)
                return
            self._json(payload)
            return
        if path.startswith("/media/evidence/"):
            self._file(library_view.media_path("evidence", path[len("/media/evidence/"):]))
            return
        if path.startswith("/media/closed/"):
            self._file(library_view.media_path("closed", path[len("/media/closed/"):]))
            return
        if path.startswith("/media/real/"):
            self._file(library_view.media_path("real", path[len("/media/real/"):]))
            return
        self._send("not found", "text/plain; charset=utf-8", 404)

    def _file(self, path):
        if path is None:
            self._send("not found", "text/plain; charset=utf-8", 404)
            return
        self._send(path.read_bytes(), "image/jpeg")


def port_is_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main():
    server = None
    port = None
    for candidate in range(8765, 8776):
        if port_is_open(candidate):
            print(f"{candidate} 已有程序在听，跳过。", flush=True)
            continue
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            print(f"{candidate} 不可用：{exc}", flush=True)
            continue
    if server is None:
        raise SystemExit("8765 到 8775 都被占用，资料室没有启动。")
    print(f"实验台 http://127.0.0.1:{port}", flush=True)
    print(f"资料室 http://127.0.0.1:{port}/library", flush=True)
    if port != 8765:
        print("8765 已被别的程序占用，这一次改用上面的端口。", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
