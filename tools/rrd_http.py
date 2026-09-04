#!/usr/bin/env python3
"""rerun 웹뷰어용 .rrd 파일 HTTP 서버 (CORS 허용). 포트 9092."""
import http.server, functools

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

http.server.ThreadingHTTPServer(
    ("0.0.0.0", 9092),
    functools.partial(H, directory="/data2/cg"),
).serve_forever()
