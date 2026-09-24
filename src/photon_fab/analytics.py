"""光谱和良率测量的确定性科学计算。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


# 端点判据的相对容差。仪器按“峰值的 80%”上报的采样点经过浮点乘法后
# 可能落在阈值下方一个 ULP（例如 0.93*0.8 = 0.7440000000000001 > 0.744），
# 用容差把数学上恰好等于阈值的端点包含进合格带宽。
_BAND_REL_TOL = 1e-12


@dataclass(frozen=True)
class SpectrumSummary:
    count: int
    peak_wavelength_nm: float
    peak_response: float
    mean_response: float
    noise_rms: float
    pass_band_nm: tuple[float, float]


def _pairs(wavelengths: Sequence[float], response: Sequence[float]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    seen_wavelengths: set[float] = set()
    for w, r in zip(wavelengths, response):
        try:
            wf, rf = float(w), float(r)
        except (TypeError, ValueError):
            raise ValueError("wavelength and response must be numeric")
        # 有限性必须在排序之前校验：NaN 会让 sorted/max 的比较结果不可解释。
        if not math.isfinite(wf) or not math.isfinite(rf):
            raise ValueError("wavelength and response must be finite")
        if wf in seen_wavelengths:
            raise ValueError(f"duplicate wavelength: {wf:g} nm")
        seen_wavelengths.add(wf)
        pairs.append((wf, rf))
    if len(wavelengths) != len(response):
        raise ValueError("wavelength and response sequences must have the same length")
    if len(pairs) < 3:
        raise ValueError("at least three wavelength/response pairs are required")
    # 始终按波长重排，保证峰值、带宽和噪声统计与上报顺序无关。
    pairs.sort(key=lambda p: p[0])
    return pairs


def summarize_spectrum(wavelengths: Sequence[float], response: Sequence[float], threshold: float = 0.8) -> SpectrumSummary:
    pairs = _pairs(wavelengths, response)
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be within (0, 1]")
    # 对已按波长排序的序列取 max，等高峰值确定性地落在最短波长处。
    peak_w, peak_r = max(pairs, key=lambda p: p[1])
    values = [r for _, r in pairs]
    mean = statistics.fmean(values)
    noise = math.sqrt(statistics.fmean((r - mean) ** 2 for r in values))
    cutoff = peak_r * threshold
    tolerance = abs(cutoff) * _BAND_REL_TOL
    band = [w for w, r in pairs if r >= cutoff - tolerance]
    # 退化输入下（例如全负响应）兜底为峰值点，保证带端始终有定义。
    if not band:
        band = [peak_w]
    return SpectrumSummary(len(pairs), peak_w, peak_r, mean, noise, (band[0], band[-1]))


def confidence_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    data = [float(v) for v in values]
    if not data or not 0 < confidence < 1:
        raise ValueError("values and confidence are invalid")
    mean = statistics.fmean(data)
    if len(data) == 1:
        return mean, mean
    z = 1.96 if confidence >= 0.95 else 1.645
    margin = z * statistics.stdev(data) / math.sqrt(len(data))
    return mean - margin, mean + margin


def yield_rate(total: int, passed: int, rejected: int = 0) -> dict[str, float]:
    if total <= 0 or passed < 0 or rejected < 0 or passed + rejected > total:
        raise ValueError("inconsistent lot counts")
    return {"yield": passed / total, "reject_rate": rejected / total, "unknown_rate": (total - passed - rejected) / total}


def responsivity(current_ma: float, optical_power_mw: float) -> float:
    if optical_power_mw <= 0:
        raise ValueError("optical power must be positive")
    return current_ma / optical_power_mw
