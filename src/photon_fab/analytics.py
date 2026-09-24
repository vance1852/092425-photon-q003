"""光谱和良率测量的确定性科学计算。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SpectrumSummary:
    count: int
    peak_wavelength_nm: float
    peak_response: float
    mean_response: float
    noise_rms: float
    pass_band_nm: tuple[float, float]


# 端点恰好落在阈值上时，允许该量级的浮点舍入误差，避免 0.93*0.8=0.7440000000000001
# 这类误差把数学上恰好在阈值上的采样点排除掉。
_BAND_REL_TOL = 1e-12
_BAND_ABS_TOL = 1e-12


def _pairs(wavelengths: Sequence[float], response: Sequence[float]) -> list[tuple[float, float]]:
    if len(wavelengths) != len(response):
        raise ValueError("wavelengths and response must have equal length")
    if len(wavelengths) < 3:
        raise ValueError("at least three wavelength/response pairs are required")
    try:
        pairs = [(float(w), float(r)) for w, r in zip(wavelengths, response)]
    except (TypeError, ValueError) as exc:
        raise ValueError("wavelength and response must be numeric") from exc
    seen: set[float] = set()
    for w, r in pairs:
        if not math.isfinite(w) or not math.isfinite(r):
            raise ValueError("measurements must be finite")
        if w <= 0:
            raise ValueError("wavelength must be positive")
        if w in seen:
            raise ValueError(f"duplicate wavelength measurement: {w:g} nm")
        seen.add(w)
    # 始终按波长重排，使峰值、带宽、噪声等统计结果与采样上报顺序无关。
    pairs.sort(key=lambda p: p[0])
    return pairs


def summarize_spectrum(wavelengths: Sequence[float], response: Sequence[float], threshold: float = 0.8) -> SpectrumSummary:
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be numeric") from exc
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie within (0, 1]")
    pairs = _pairs(wavelengths, response)
    peak_w, peak_r = max(pairs, key=lambda p: p[1])
    values = [r for _, r in pairs]
    mean = statistics.fmean(values)
    noise = math.sqrt(statistics.fmean((r - mean) ** 2 for r in values))
    cutoff = peak_r * threshold
    # 端点包含规则：响应达到或超过阈值（含浮点容差）的采样点都属于合格带宽，
    # 阈值恰好等于峰值 80% 时不得排除边界采样点。
    band = [w for w, r in pairs if r >= cutoff or math.isclose(r, cutoff, rel_tol=_BAND_REL_TOL, abs_tol=_BAND_ABS_TOL)]
    if not band:
        raise ValueError("no samples fall within the pass band")
    return SpectrumSummary(len(pairs), peak_w, peak_r, mean, noise, (min(band), max(band)))


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
