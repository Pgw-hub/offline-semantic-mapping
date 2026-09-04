#!/usr/bin/env python3
"""rerun 웹뷰어용 .rrd 파일 HTTP 서버 (CORS 허용).
사용: python tools/rrd_http.py [서빙 디렉터리=이 저장소] [포트=9092]
뷰어 링크: http://localhost:9091?url=http://localhost:9092/<디렉터리 기준 상대경로>.rrd
"""
import http.server, functools, os, sys

root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9092

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, fmt, *args):
        pass

print(f"serving {root} on :{port}", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", port), functools.partial(H, directory=root)).serve_forever()
