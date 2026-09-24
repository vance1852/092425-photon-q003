from __future__ import annotations

import http.client
import json
import math
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from photon_fab.analytics import summarize_spectrum
from photon_fab.api import Handler
from photon_fab.service import PhotonService


# (波长 nm, 响应) 中两端 0.744 在数学上恰好是峰值 0.93 的 80%。
SPECTRUM = [
    (400.0, 0.744),
    (450.0, 0.93),
    (500.0, 0.85),
    (550.0, 0.744),
    (600.0, 0.30),
]


class SpectrumStatsTests(unittest.TestCase):
    def test_threshold_endpoint_is_included(self) -> None:
        summary = summarize_spectrum([w for w, _ in SPECTRUM], [r for _, r in SPECTRUM])
        self.assertEqual(summary.peak_wavelength_nm, 450.0)
        self.assertEqual(summary.peak_response, 0.93)
        # 恰好等于 80% 峰值的两个端点必须包含，不能因 0.93*0.8=0.7440000000000001 被排除。
        self.assertEqual(summary.pass_band_nm, (400.0, 550.0))

    def test_results_invariant_to_sample_order(self) -> None:
        expected = summarize_spectrum([w for w, _ in SPECTRUM], [r for _, r in SPECTRUM])
        for ordering in (
            list(reversed(SPECTRUM)),
            [SPECTRUM[i] for i in (3, 0, 4, 1, 2)],
            [SPECTRUM[i] for i in (2, 4, 0, 3, 1)],
        ):
            shuffled = summarize_spectrum([w for w, _ in ordering], [r for _, r in ordering])
            self.assertEqual(shuffled, expected)

    def test_noise_rms_is_population_rms(self) -> None:
        summary = summarize_spectrum([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary.mean_response, 3.0)
        self.assertAlmostEqual(summary.noise_rms, math.sqrt(2.0))

    def test_duplicate_wavelength_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate wavelength"):
            summarize_spectrum([400.0, 450.0, 450.0], [0.7, 0.9, 0.8])

    def test_non_finite_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_spectrum([400.0, 450.0, 500.0], [0.7, float("nan"), 0.8])
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_spectrum([400.0, float("inf"), 500.0], [0.7, 0.9, 0.8])
        with self.assertRaisesRegex(ValueError, "numeric"):
            summarize_spectrum([400.0, 450.0, 500.0], [0.7, "hot", 0.8])

    def test_threshold_and_length_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            summarize_spectrum([1, 2, 3], [1.0, 2.0, 3.0], threshold=0.0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            summarize_spectrum([1, 2, 3], [1.0, 2.0, 3.0], threshold=1.5)
        with self.assertRaisesRegex(ValueError, "at least three"):
            summarize_spectrum([1, 2], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "same length"):
            summarize_spectrum([1, 2, 3], [1.0, 2.0])


class MeasurementValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PhotonService()
        self.service.bootstrap_admin()
        self.token = self.service.auth.login("admin", "photon-admin")
        self.service.create_lot(self.token, "LOT-X", "CMOS image sensor", "P3.2", 10)

    def _add(self, **overrides) -> None:
        fields = {"wavelength_nm": 450.0, "response": 0.9, "noise": 0.01, "instrument": "spec-1"}
        fields.update(overrides)
        self.service.add_measurement(self.token, "LOT-X", **fields)

    def test_negative_noise_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise"):
            self._add(noise=-0.001)

    def test_non_finite_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise"):
            self._add(noise=float("nan"))
        with self.assertRaisesRegex(ValueError, "response"):
            self._add(response=float("inf"))
        with self.assertRaisesRegex(ValueError, "wavelength"):
            self._add(wavelength_nm=-450.0)
        with self.assertRaisesRegex(ValueError, "wavelength"):
            self._add(wavelength_nm=0.0)

    def test_duplicate_wavelength_in_lot_is_rejected(self) -> None:
        self._add(wavelength_nm=450.0)
        with self.assertRaisesRegex(ValueError, "duplicate wavelength"):
            self._add(wavelength_nm=450.0, response=0.95)

    def test_invalid_measurement_does_not_persist(self) -> None:
        with self.assertRaises(ValueError):
            self._add(noise=-1.0)
        rows = self.service.db.execute(
            "SELECT COUNT(*) AS n FROM measurements WHERE lot_id='LOT-X'").fetchone()
        self.assertEqual(rows["n"], 0)

    def test_analysis_on_missing_lot_is_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.service.analyze(self.token, "NOPE")


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PhotonService()
        self.service.bootstrap_admin()

        class BoundHandler(Handler):
            pass

        BoundHandler.service = self.service
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), BoundHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.token = self.login("admin", "photon-admin")[1]["token"]

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(self, method: str, path: str, body=None, token: str | None = None, raw: bytes | None = None):
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if raw is not None:
            data = raw
            headers["Content-Type"] = "application/json"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        else:
            data = None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def login(self, user_id: str, password: str):
        return self.request("POST", "/login", {"user_id": user_id, "password": password})

    def create_lot(self, lot_id: str) -> None:
        status, _ = self.request("POST", "/lots", {
            "lot_id": lot_id, "product": "CMOS image sensor",
            "process_rev": "P3.2", "wafer_count": 10}, token=self.token)
        self.assertEqual(status, 201)

    def add_measurements(self, lot_id: str, points) -> None:
        for wavelength, response in points:
            status, body = self.request("POST", f"/lots/{lot_id}/measurements", {
                "wavelength_nm": wavelength, "response": response,
                "noise": 0.01, "instrument": "spectrometer-1"}, token=self.token)
            self.assertEqual(status, 201, body)

    def test_health(self) -> None:
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "photon-fab")

    def test_analysis_band_includes_threshold_endpoints(self) -> None:
        self.create_lot("LOT-A")
        self.add_measurements("LOT-A", SPECTRUM)
        status, body = self.request("POST", "/lots/LOT-A/analysis", token=self.token)
        self.assertEqual(status, 200, body)
        spectrum = body["spectrum"]
        self.assertEqual(spectrum["peak_wavelength_nm"], 450.0)
        self.assertEqual(spectrum["pass_band_nm"], [400.0, 550.0])

    def test_analysis_is_invariant_to_insertion_order(self) -> None:
        self.create_lot("LOT-FWD")
        self.add_measurements("LOT-FWD", SPECTRUM)
        self.create_lot("LOT-REV")
        self.add_measurements("LOT-REV", list(reversed(SPECTRUM)))
        _, forward = self.request("POST", "/lots/LOT-FWD/analysis", token=self.token)
        _, reverse = self.request("POST", "/lots/LOT-REV/analysis", token=self.token)
        self.assertEqual(forward["spectrum"], reverse["spectrum"])

    def test_negative_noise_returns_400(self) -> None:
        self.create_lot("LOT-B")
        status, body = self.request("POST", "/lots/LOT-B/measurements", {
            "wavelength_nm": 450.0, "response": 0.9, "noise": -0.5,
            "instrument": "spectrometer-1"}, token=self.token)
        self.assertEqual(status, 400)
        self.assertIn("noise", body["error"])

    def test_duplicate_wavelength_returns_400(self) -> None:
        self.create_lot("LOT-C")
        self.add_measurements("LOT-C", SPECTRUM[:3])
        status, body = self.request("POST", "/lots/LOT-C/measurements", {
            "wavelength_nm": 450.0, "response": 0.95, "noise": 0.01,
            "instrument": "spectrometer-1"}, token=self.token)
        self.assertEqual(status, 400)
        self.assertIn("duplicate wavelength", body["error"])

    def test_non_numeric_and_missing_fields_return_400(self) -> None:
        self.create_lot("LOT-D")
        status, body = self.request("POST", "/lots/LOT-D/measurements", {
            "wavelength_nm": "blue", "response": 0.9, "noise": 0.01,
            "instrument": "spectrometer-1"}, token=self.token)
        self.assertEqual(status, 400)
        self.assertIn("wavelength_nm", body["error"])
        status, body = self.request("POST", "/lots/LOT-D/measurements", {
            "response": 0.9, "instrument": "spectrometer-1"}, token=self.token)
        self.assertEqual(status, 400)
        self.assertIn("wavelength_nm", body["error"])

    def test_malformed_and_empty_body_return_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/lots/LOT-D/measurements", body="{not json",
                     headers={"Authorization": f"Bearer {self.token}",
                              "Content-Type": "application/json",
                              "Content-Length": "9"})
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn("valid JSON", json.loads(response.read())["error"])
        conn.close()
        status, body = self.request("POST", "/lots/LOT-D/measurements",
                                    body=None, token=self.token)
        # urllib 无 body 时不发送载荷，服务端应明确拒绝空请求体。
        self.assertEqual(status, 400)
        self.assertIn("body", body["error"])

    def test_missing_lot_returns_404(self) -> None:
        status, body = self.request("GET", "/lots/GHOST", token=self.token)
        self.assertEqual(status, 404)
        self.assertIn("not found", body["error"])
        status, body = self.request("POST", "/lots/GHOST/analysis", token=self.token)
        self.assertEqual(status, 404)

    def test_analysis_requires_three_points(self) -> None:
        self.create_lot("LOT-E")
        self.add_measurements("LOT-E", SPECTRUM[:2])
        status, body = self.request("POST", "/lots/LOT-E/analysis", token=self.token)
        self.assertEqual(status, 400)
        self.assertIn("three measurements", body["error"])

    def test_missing_or_bad_token_returns_403(self) -> None:
        self.create_lot("LOT-F")
        status, _ = self.request("POST", "/lots/LOT-F/measurements", {
            "wavelength_nm": 450.0, "response": 0.9, "noise": 0.01,
            "instrument": "spectrometer-1"})
        self.assertEqual(status, 403)
        status, _ = self.request("POST", "/lots/LOT-F/measurements", {
            "wavelength_nm": 450.0, "response": 0.9, "noise": 0.01,
            "instrument": "spectrometer-1"}, token="deadbeef")
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
