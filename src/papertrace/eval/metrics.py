"""Signal-reconstruction metrics, following the scoring of the PhysioNet Challenge 2024.

``align_signals`` and ``compute_snr`` are typed re-implementations of the functions of the same
name in ``helper_code.py`` of https://github.com/physionetchallenges/evaluation-2024 so that SNR
values are comparable with the challenge. The upstream code is distributed under this licence:

    BSD 2-Clause License

    Copyright (c) 2024 George B. Moody PhysioNet Challenges
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Differences from upstream: type annotations; ``lead_snr`` bundles the shift limits that the
upstream evaluation script applies; a flat signal (no amplitude range) is treated as "cannot be
aligned" instead of dividing by zero; and the horizontal shift for negative offsets is corrected.
Upstream builds that case as ``x_est[:-offset_hz]``, which keeps the first ``|offset|`` samples
instead of dropping the last ones, so the shifted signal is only ``2 * |offset|`` samples long and
the coverage penalty of ``compute_snr`` turns the lead's score into about 0 dB. Pass
``upstream_quirk=True`` to reproduce the upstream behaviour exactly (checked against the upstream
code in the tests, which is skipped when that code is not available).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

NUM_QUANT_LEVELS = 2**8
MAX_VERTICAL_SHIFT_MV = 1.0


def _convert_signal(
    x: NDArray[np.float64], levels: int, min_amp: float, max_amp: float, max_t: int
) -> NDArray[np.float64]:
    """Quantise a 1-D signal into a binary ``(levels, max_t)`` image."""
    finite = np.isfinite(x)
    t = np.arange(1, x.size + 1)[finite]
    y = np.round((levels - 1) * (x[finite] - min_amp) / (max_amp - min_amp) + 1).astype(int)
    y = np.clip(y, 1, levels)
    image = np.zeros((levels, max_t))
    image[y - 1, t - 1] = 1
    return image


def _fft_correlate(a_ref: NDArray[np.float64], a_est: NDArray[np.float64]) -> NDArray[np.float64]:
    flipped = np.flip(np.flip(a_est, axis=0), axis=1)
    return np.asarray(fftconvolve(a_ref, flipped, mode="full"))


def estimate_offsets(
    x_ref: NDArray[np.float64],
    x_est: NDArray[np.float64],
    num_quant_levels: int = NUM_QUANT_LEVELS,
    smooth: bool = True,
    sigma: float = 0.5,
) -> tuple[int, float]:
    """Horizontal (samples) and vertical (signal units) offset of ``x_est`` against ``x_ref``.

    Upstream also correlates the reference with itself to find the zero-lag position; that peak
    is always at the centre of the correlation, so it is not computed here.
    """
    x_ref = np.asarray(x_ref, dtype=np.float64)
    x_est = np.asarray(x_est, dtype=np.float64)
    if not (np.isfinite(x_ref).any() and np.isfinite(x_est).any()):
        return 0, 0.0
    min_amp = float(min(np.nanmin(x_ref), np.nanmin(x_est)))
    max_amp = float(max(np.nanmax(x_ref), np.nanmax(x_est)))
    if max_amp == min_amp:
        return 0, 0.0
    max_t = max(x_ref.size, x_est.size)
    a_ref = _convert_signal(x_ref, num_quant_levels, min_amp, max_amp, max_t)
    a_est = _convert_signal(x_est, num_quant_levels, min_amp, max_amp, max_t)
    if smooth:
        a_ref = gaussian_filter(a_ref, sigma)
        a_est = gaussian_filter(a_est, sigma)
    cross = _fft_correlate(a_ref, a_est)
    idx_cross = np.unravel_index(np.argmax(cross), cross.shape)
    offset_hz = int(a_ref.shape[1] - 1 - idx_cross[1])
    offset_vt = float(
        (a_ref.shape[0] - 1 - idx_cross[0]) / (num_quant_levels - 1) * (max_amp - min_amp)
    )
    return offset_hz, offset_vt


def shift_signal(
    x_est: NDArray[np.float64], offset_hz: int, offset_vt: float, upstream_quirk: bool = False
) -> NDArray[np.float64]:
    """Apply offsets from :func:`estimate_offsets` (NaN where samples moved out of range).

    Args:
        x_est: Estimated signal.
        offset_hz: Horizontal offset in samples.
        offset_vt: Vertical offset in signal units.
        upstream_quirk: Reproduce the upstream handling of negative horizontal offsets (see the
            module docstring) instead of the corrected one.
    """
    x_est = np.asarray(x_est, dtype=np.float64)
    if offset_hz < 0:
        kept = x_est[:-offset_hz] if upstream_quirk else x_est[:offset_hz]
        shifted = np.concatenate((np.full(-offset_hz, np.nan), kept))
    else:
        shifted = np.concatenate((x_est[offset_hz:], np.full(offset_hz, np.nan)))
    return shifted - offset_vt


def align_signals(
    x_ref: NDArray[np.float64],
    x_est: NDArray[np.float64],
    num_quant_levels: int = NUM_QUANT_LEVELS,
    smooth: bool = True,
    sigma: float = 0.5,
    upstream_quirk: bool = False,
) -> tuple[NDArray[np.float64], int, float]:
    """Estimate and remove the horizontal and vertical offset of ``x_est`` against ``x_ref``.

    Returns:
        The shifted estimate, the horizontal offset in samples and the vertical offset in the
        units of the signals. Arguments as in :func:`estimate_offsets` and :func:`shift_signal`.
    """
    offset_hz, offset_vt = estimate_offsets(x_ref, x_est, num_quant_levels, smooth, sigma)
    if offset_hz == 0 and offset_vt == 0.0:
        return np.asarray(x_est, dtype=np.float64).copy(), 0, 0.0
    return shift_signal(x_est, offset_hz, offset_vt, upstream_quirk), offset_hz, offset_vt


def compute_snr(
    x_ref: NDArray[np.float64],
    x_est: NDArray[np.float64],
    keep_nans: bool = True,
    noise_median: bool = False,
) -> tuple[float, float, float]:
    """Signal-to-noise ratio of an estimate in dB, as defined by the challenge.

    Only samples that are finite in both signals are compared, and the result is scaled by the
    share of reference samples that the estimate covers, so missing samples are penalised. A
    perfect reconstruction has zero noise power and yields NaN, exactly as upstream.

    Returns:
        ``(snr_db, signal_power, noise_power)``.
    """
    ref = np.asarray(x_ref, dtype=np.float64).copy()
    est = np.asarray(x_est, dtype=np.float64).copy()
    if ref.ndim != 1 or est.ndim != 1:
        raise ValueError("signals must be one-dimensional")
    if est.size < ref.size:
        est = np.concatenate((est, np.full(ref.size - est.size, np.nan)))
    elif est.size > ref.size:
        ref = np.concatenate((ref, np.full(est.size - ref.size, np.nan)))
    finite_ref, finite_est = np.isfinite(ref), np.isfinite(est)
    if keep_nans:
        keep = finite_ref & finite_est
    else:
        est[~finite_est] = 0
        keep = finite_ref
    ref, est = ref[keep], est[keep]
    noise = ref - est
    p_signal = float(np.mean(ref**2)) if ref.size else float("nan")
    if noise.size:
        p_noise = float(np.median(noise**2) if noise_median else np.mean(noise**2))
    else:
        p_noise = float("nan")
    if p_signal > 0 and p_noise > 0:
        snr = 10 * np.log10(p_signal / p_noise)
    elif keep.sum() == 0 and finite_ref.sum() > 0:
        snr = 0.0
    else:
        snr = float("nan")
    if keep_nans:
        snr *= keep.sum() / finite_ref.sum() if finite_ref.sum() else float("nan")
    return float(snr), p_signal, p_noise


def _gated_estimate(
    x_ref: NDArray[np.float64],
    x_est: NDArray[np.float64],
    offsets: tuple[int, float],
    sampling_frequency: float,
    upstream_quirk: bool,
) -> NDArray[np.float64]:
    offset_hz, offset_vt = offsets
    within = (
        abs(offset_hz) <= np.round(0.5 * sampling_frequency)
        and abs(offset_vt) <= MAX_VERTICAL_SHIFT_MV
    )
    if not within or (offset_hz == 0 and offset_vt == 0.0):
        return np.asarray(x_est, dtype=np.float64)
    return shift_signal(x_est, offset_hz, offset_vt, upstream_quirk)


def aligned_estimate(
    x_ref: NDArray[np.float64],
    x_est: NDArray[np.float64],
    sampling_frequency: float = 500.0,
    upstream_quirk: bool = False,
) -> NDArray[np.float64]:
    """The estimate shifted as the challenge does, or unshifted if the shift is out of limits.

    The shift is used only if the horizontal shift is at most half a second and the vertical shift
    at most 1 mV.
    """
    offsets = estimate_offsets(x_ref, x_est)
    return _gated_estimate(x_ref, x_est, offsets, sampling_frequency, upstream_quirk)


def lead_snr(
    x_ref: NDArray[np.float64],
    x_est: NDArray[np.float64],
    sampling_frequency: float = 500.0,
    upstream_quirk: bool = False,
) -> float:
    """SNR in dB of one lead after the challenge's alignment (see :func:`aligned_estimate`)."""
    estimate = aligned_estimate(x_ref, x_est, sampling_frequency, upstream_quirk)
    return compute_snr(x_ref, estimate)[0]


def lead_snr_both(
    x_ref: NDArray[np.float64], x_est: NDArray[np.float64], sampling_frequency: float = 500.0
) -> tuple[float, float]:
    """SNR with the corrected alignment and with the upstream behaviour, from one alignment."""
    offsets = estimate_offsets(x_ref, x_est)
    corrected = _gated_estimate(x_ref, x_est, offsets, sampling_frequency, upstream_quirk=False)
    upstream = _gated_estimate(x_ref, x_est, offsets, sampling_frequency, upstream_quirk=True)
    return compute_snr(x_ref, corrected)[0], compute_snr(x_ref, upstream)[0]


def pearson(x_ref: NDArray[np.float64], x_est: NDArray[np.float64]) -> float:
    """Pearson correlation over the samples that are finite in both signals."""
    keep = np.isfinite(x_ref) & np.isfinite(x_est)
    if keep.sum() < 2 or np.std(x_ref[keep]) == 0 or np.std(x_est[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x_ref[keep], x_est[keep])[0, 1])


def rmse(x_ref: NDArray[np.float64], x_est: NDArray[np.float64]) -> float:
    """Root-mean-square error over the samples that are finite in both signals."""
    keep = np.isfinite(x_ref) & np.isfinite(x_est)
    if not keep.any():
        return float("nan")
    return float(np.sqrt(np.mean((x_ref[keep] - x_est[keep]) ** 2)))


def summarize(values: list[float]) -> dict[str, float]:
    """Median, quartiles, mean and count of the finite ``values``."""
    finite = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        nan = float("nan")
        return {"n": 0, "median": nan, "q1": nan, "q3": nan, "mean": nan}
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return {
        "n": int(finite.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(finite.mean()),
    }
