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

    def _read_json(self) -> dict:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        if not raw:
            raise ValueError("request body is required and must be a JSON object")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body must be valid JSON: {exc.msg}")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    @staticmethod
    def _field(body: dict, key: str, expected: type | tuple[type, ...] | None = None):
        if key not in body or body[key] is None:
            raise ValueError(f"missing required field: {key}")
        value = body[key]
        if expected is not None and not isinstance(value, expected):
            label = expected.__name__ if isinstance(expected, type) else " or ".join(t.__name__ for t in expected)
            raise ValueError(f"field {key} must be of type {label}")
        return value

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok", "service": "photon-fab"})
        if self.path.startswith("/lots/"):
            try:
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                return self._json(200, self.service.get_lot(token, self.path.split("/", 2)[2]))
            except PermissionError as exc:
                return self._json(403, {"error": str(exc)})
            except KeyError:
                return self._json(404, {"error": "lot not found"})
            except (TypeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/login":
                body = self._read_json()
                return self._json(200, {"token": self.service.auth.login(
                    self._field(body, "user_id", str), self._field(body, "password", str))})
            token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if self.path == "/lots":
                body = self._read_json()
                return self._json(201, self.service.create_lot(
                    token, self._field(body, "lot_id", str), self._field(body, "product", str),
                    self._field(body, "process_rev", str), self._field(body, "wafer_count", int)))
            if self.path.startswith("/lots/") and self.path.endswith("/measurements"):
                body = self._read_json()
                lot_id = self.path.split("/")[2]
                numeric = (int, float)
                return self._json(201, self.service.add_measurement(
                    token, lot_id, self._field(body, "wavelength_nm", numeric),
                    self._field(body, "response", numeric), body.get("noise", 0.0),
                    self._field(body, "instrument", str)))
            if self.path.startswith("/lots/") and self.path.endswith("/analysis"):
                return self._json(200, self.service.analyze(token, self.path.split("/")[2]))
            return self._json(404, {"error": "not found"})
        except PermissionError as exc:
            return self._json(403, {"error": str(exc)})
        except KeyError:
            return self._json(404, {"error": "lot not found"})
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})


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
