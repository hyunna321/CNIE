"""MTSR 시뮬레이션 뷰어 로컬 서버.

사용법 (저장소 루트에서):
    python -m viewer.server            # http://127.0.0.1:8765 를 브라우저로 연다
    python -m viewer.server --port 9000

/api/run은 시뮬레이션을 끝까지 돌린 기록을 돌려주고, 재생은 브라우저에서 한다.
/api/export는 실행 하나를 단독 HTML 파일로 내려준다.
"""
import argparse
import json
import os
import sys
import threading
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viewer import runner  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
RUNS: "OrderedDict[str, dict]" = OrderedDict()
MAX_RUNS = 8
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[viewer] " + fmt % args + "\n")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, open(STATIC, "rb").read(), "text/html; charset=utf-8")
        elif u.path == "/api/meta":
            self._json(runner.meta())
        elif u.path == "/api/export":
            rid = parse_qs(u.query).get("id", [""])[0]
            with _lock:
                res = RUNS.get(rid)
            if res is None:
                return self._json({"error": "실행 기록이 없습니다 (서버를 다시 켰다면 다시 실행하세요)"}, 404)
            html = open(STATIC, encoding="utf-8").read()
            data = json.dumps(res, ensure_ascii=False).replace("</", "<\\/")
            html = html.replace('<script id="embedded-run" type="application/json">null</script>',
                                f'<script id="embedded-run" type="application/json">{data}</script>')
            c = res["config"]
            name = f"mtsr_replay_{c.get('policy')}_M{c.get('num_robots')}_K{c.get('num_workstations')}_lam{c.get('lam_hr')}_seed{c.get('seed')}.html"
            self._send(200, html.encode(), "text/html; charset=utf-8",
                       {"Content-Disposition": f'attachment; filename="{name}"'})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/run":
                res = runner.run(body)
                rid = uuid.uuid4().hex[:10]
                res["id"] = rid
                with _lock:
                    RUNS[rid] = res
                    while len(RUNS) > MAX_RUNS:
                        RUNS.popitem(last=False)
                self._json(res)
            elif u.path == "/api/checkpoint":
                self._json(runner.read_checkpoint_meta(body["path"]))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    ap = argparse.ArgumentParser(description="MTSR 시뮬레이션 뷰어")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MTSR viewer: http://{args.host}:{args.port}  (Ctrl+C로 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
