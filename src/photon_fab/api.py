"""用于离线验收的无依赖 JSON HTTP API。"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import PhotonService


class Handler(BaseHTTPRequestHandler):
    service = PhotonService()

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok", "service": "photon-fab"})
        if self.path.startswith("/lots/"):
            try:
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                return self._json(200, self.service.get_lot(token, self.path.split("/", 2)[2]))
            except PermissionError as exc:
                return self._json(403, {"error": str(exc)})
            except KeyError as exc:
                return self._json(404, {"error": f"lot not found: {exc.args[0]}"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                return self._json(400, {"error": f"invalid JSON body: {exc.msg}"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "request body must be a JSON object"})
            if self.path == "/login":
                try:
                    return self._json(200, {"token": self.service.auth.login(body["user_id"], body["password"])})
                except KeyError:
                    return self._json(400, {"error": "user_id and password are required"})
            token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if self.path == "/lots":
                try:
                    return self._json(201, self.service.create_lot(token, body["lot_id"], body["product"], body["process_rev"], body["wafer_count"]))
                except KeyError as exc:
                    return self._json(400, {"error": f"missing required field: {exc.args[0]}"})
            if self.path.startswith("/lots/") and self.path.endswith("/measurements"):
                lot_id = self.path.split("/")[2]
                required = ("wavelength_nm", "response", "instrument")
                missing = [name for name in required if name not in body]
                if missing:
                    return self._json(400, {"error": f"missing required field: {missing[0]}"})
                return self._json(201, self.service.add_measurement(token, lot_id, body["wavelength_nm"], body["response"], body.get("noise", 0.0), body["instrument"]))
            if self.path.startswith("/lots/") and self.path.endswith("/analysis"):
                return self._json(200, self.service.analyze(token, self.path.split("/")[2]))
            return self._json(404, {"error": "not found"})
        except PermissionError as exc:
            return self._json(403, {"error": str(exc)})
        except KeyError as exc:
            return self._json(404, {"error": f"lot not found: {exc.args[0]}"})
        except ValueError as exc:
            return self._json(422, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=":memory:")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    Handler.service = PhotonService(args.database)
    Handler.service.bootstrap_admin()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
