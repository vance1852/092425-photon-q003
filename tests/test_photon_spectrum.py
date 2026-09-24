from __future__ import annotations

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


class SpectrumStatsTests(unittest.TestCase):
    def test_threshold_endpoint_is_included_at_exactly_80_percent(self) -> None:
        # 数学上 0.93 * 0.8 == 0.744；旧实现因浮点舍入把 400 nm 端点排除。
        wavelengths = [400.0, 550.0, 700.0]
        response = [0.744, 0.93, 0.5]
        summary = summarize_spectrum(wavelengths, response)
        self.assertEqual(summary.peak_wavelength_nm, 550.0)
        self.assertEqual(summary.pass_band_nm, (400.0, 550.0))

    def test_both_threshold_endpoints_are_included(self) -> None:
        wavelengths = [400.0, 550.0, 700.0]
        response = [0.8, 1.0, 0.8]
        summary = summarize_spectrum(wavelengths, response)
        self.assertEqual(summary.pass_band_nm, (400.0, 700.0))

    def test_stats_are_invariant_to_sample_order(self) -> None:
        wavelengths = [400.0, 450.0, 550.0, 650.0, 700.0]
        response = [0.744, 0.81, 0.93, 0.86, 0.42]
        ordered = summarize_spectrum(wavelengths, response)
        shuffled = summarize_spectrum(list(reversed(wavelengths)), list(reversed(response)))
        self.assertEqual(ordered, shuffled)
        # 再用一次随机重排确认峰值、带宽、噪声完全一致。
        reordered = summarize_spectrum([650.0, 400.0, 700.0, 550.0, 450.0],
                                       [0.86, 0.744, 0.42, 0.93, 0.81])
        self.assertEqual(reordered.peak_wavelength_nm, ordered.peak_wavelength_nm)
        self.assertEqual(reordered.pass_band_nm, ordered.pass_band_nm)
        self.assertTrue(math.isclose(reordered.noise_rms, ordered.noise_rms, rel_tol=1e-15))

    def test_duplicate_wavelength_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate wavelength"):
            summarize_spectrum([400.0, 400.0, 550.0], [0.7, 0.8, 0.9])

    def test_non_finite_and_non_numeric_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_spectrum([400.0, float("nan"), 550.0], [0.7, 0.8, 0.9])
        with self.assertRaisesRegex(ValueError, "numeric"):
            summarize_spectrum([400.0, "x", 550.0], [0.7, 0.8, 0.9])

    def test_threshold_must_be_within_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            summarize_spectrum([400.0, 550.0, 700.0], [0.7, 0.9, 0.5], threshold=0.0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            summarize_spectrum([400.0, 550.0, 700.0], [0.7, 0.9, 0.5], threshold=1.5)


class ServiceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PhotonService()
        self.service.bootstrap_admin()
        self.token = self.service.auth.login("admin", "photon-admin")
        self.service.create_lot(self.token, "LOT-1", "sensor", "P1", 10)

    def _add(self, wavelength=550.0, response=0.9, noise=0.01, instrument="spec-1"):
        return self.service.add_measurement(self.token, "LOT-1", wavelength, response, noise, instrument)

    def test_negative_noise_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise"):
            self._add(noise=-0.01)

    def test_non_numeric_and_non_finite_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "wavelength_nm"):
            self._add(wavelength="blue")
        with self.assertRaisesRegex(ValueError, "response"):
            self._add(response=float("inf"))
        with self.assertRaisesRegex(ValueError, "noise"):
            self._add(noise=float("nan"))

    def test_duplicate_wavelength_in_same_lot_is_rejected(self) -> None:
        self._add(wavelength=550.0)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._add(wavelength=550.0)

    def test_analyze_missing_lot_is_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.service.analyze(self.token, "NOPE")


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Handler.service = PhotonService()
        Handler.service.bootstrap_admin()
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.token = self._post("/login", {"user_id": "admin", "password": "photon-admin"})["token"]
        self._post(
            "/lots",
            {"lot_id": "LOT-API", "product": "sensor", "process_rev": "P1", "wafer_count": 10},
            token=self.token,
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body=None, token: str | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _post(self, path: str, body: dict, token: str | None = None):
        status, payload = self._request("POST", path, body, token)
        self.assertEqual(status, 200 if path == "/login" else 201, payload)
        return payload

    def _measure(self, wavelength: float, response: float, noise: float = 0.01):
        return self._post(
            f"/lots/LOT-API/measurements",
            {"wavelength_nm": wavelength, "response": response, "noise": noise, "instrument": "spec-1"},
            token=self.token,
        )

    def test_analysis_endpoint_includes_threshold_endpoint(self) -> None:
        for wavelength, response in ((400.0, 0.744), (550.0, 0.93), (700.0, 0.5)):
            self._measure(wavelength, response)
        status, payload = self._request("POST", "/lots/LOT-API/analysis", {}, self.token)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["spectrum"]["pass_band_nm"], [400.0, 550.0])
        self.assertEqual(payload["spectrum"]["peak_wavelength_nm"], 550.0)

    def test_negative_noise_returns_422(self) -> None:
        status, payload = self._request(
            "POST", "/lots/LOT-API/measurements",
            {"wavelength_nm": 550.0, "response": 0.9, "noise": -0.05, "instrument": "spec-1"},
            self.token,
        )
        self.assertEqual(status, 422)
        self.assertIn("noise", payload["error"])

    def test_duplicate_wavelength_returns_422(self) -> None:
        self._measure(550.0, 0.9)
        status, payload = self._request(
            "POST", "/lots/LOT-API/measurements",
            {"wavelength_nm": 550.0, "response": 0.8, "noise": 0.01, "instrument": "spec-1"},
            self.token,
        )
        self.assertEqual(status, 422)
        self.assertIn("already exists", payload["error"])

    def test_malformed_json_returns_400(self) -> None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/lots", data=b"{not json", method="POST"
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.token}")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)

    def test_missing_lot_analysis_returns_404(self) -> None:
        status, payload = self._request("POST", "/lots/NOPE/analysis", {}, self.token)
        self.assertEqual(status, 404)
        self.assertIn("NOPE", payload["error"])

    def test_missing_required_field_returns_400(self) -> None:
        status, payload = self._request(
            "POST", "/lots/LOT-API/measurements",
            {"wavelength_nm": 550.0, "noise": 0.01},
            self.token,
        )
        self.assertEqual(status, 400)
        self.assertIn("response", payload["error"])


if __name__ == "__main__":
    unittest.main()
