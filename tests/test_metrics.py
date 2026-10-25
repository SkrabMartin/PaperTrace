import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from papertrace.eval.metrics import (
    align_signals,
    compute_snr,
    lead_snr,
    pearson,
    rmse,
    summarize,
)

OFFICIAL = Path(__file__).resolve().parent.parent / "data/external/evaluation-2024/helper_code.py"


def beats(n: int = 5000, seed: int = 0) -> NDArray[np.float64]:
    """A crude ECG-like signal: a few R-wave and T-wave bumps plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.zeros(n)
    for centre in range(250, n, 400):
        x += np.exp(-0.5 * ((t - centre) / 6) ** 2) * 1.2
        x += np.exp(-0.5 * ((t - centre - 110) / 25) ** 2) * 0.3
    return x + rng.normal(0, 0.01, n)


def test_snr_matches_the_definition() -> None:
    ref = beats()
    noise = np.random.default_rng(1).normal(0, 0.1, ref.size)
    snr, p_signal, p_noise = compute_snr(ref, ref + noise)
    assert p_signal == pytest.approx(np.mean(ref**2))
    assert p_noise == pytest.approx(np.mean(noise**2))
    assert snr == pytest.approx(10 * np.log10(p_signal / p_noise))


def test_snr_penalises_missing_samples() -> None:
    ref = beats()
    est = ref + np.random.default_rng(2).normal(0, 0.1, ref.size)
    full, _, _ = compute_snr(ref, est)
    est[: ref.size // 2] = np.nan
    half, _, _ = compute_snr(ref, est)
    assert half < full
    assert half == pytest.approx(
        compute_snr(ref[ref.size // 2 :], est[ref.size // 2 :])[0] * 0.5, rel=0.05
    )


def test_snr_edge_cases_follow_the_challenge() -> None:
    ref = beats()
    assert np.isnan(compute_snr(ref, ref)[0])  # perfect reconstruction: zero noise power
    assert compute_snr(ref, np.full_like(ref, np.nan))[0] == 0.0  # nothing digitised
    assert compute_snr(ref, ref[:100] + 0.1)[2] > 0  # shorter estimate is padded with NaN


def test_snr_rejects_multidimensional_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_snr(np.zeros((3, 3)), np.zeros((3, 3)))


def test_alignment_recovers_a_shift_and_an_offset() -> None:
    ref = beats()
    est = np.concatenate((np.full(10, np.nan), ref[:-10])) + 0.3
    shifted, offset_hz, offset_vt = align_signals(ref, est)
    assert abs(offset_hz) == 10
    assert abs(offset_vt) == pytest.approx(0.3, abs=0.02)
    assert compute_snr(ref, shifted)[0] > compute_snr(ref, est)[0] + 10


def test_negative_offsets_are_shifted_correctly_unlike_upstream() -> None:
    ref = beats()
    noise = np.random.default_rng(3).normal(0, 0.02, ref.size)
    est = np.concatenate(((ref + noise)[10:], np.full(10, np.nan)))  # runs 10 samples early
    shifted, offset_hz, _ = align_signals(ref, est)
    assert offset_hz == -10
    assert shifted.size == ref.size
    assert compute_snr(ref, shifted)[0] > compute_snr(ref, est)[0] + 10
    quirk, _, _ = align_signals(ref, est, upstream_quirk=True)
    assert quirk.size == 2 * 10  # upstream keeps only the first |offset| samples
    assert lead_snr(ref, est) > lead_snr(ref, est, upstream_quirk=True) + 10


def test_lead_snr_ignores_shifts_beyond_the_limits() -> None:
    ref = beats()
    est = ref + 3.0  # a 3 mV offset is more than the 1 mV the challenge allows to correct
    assert lead_snr(ref, est) == pytest.approx(compute_snr(ref, est)[0])


def test_flat_signals_cannot_be_aligned() -> None:
    flat = np.zeros(100)
    shifted, offset_hz, offset_vt = align_signals(flat, flat)
    assert (offset_hz, offset_vt) == (0, 0.0)
    np.testing.assert_array_equal(shifted, flat)


def test_pearson_rmse_and_nans() -> None:
    ref = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    est = np.array([0.0, 2.0, 4.0, 6.0, 1.0])
    assert pearson(ref, est) == pytest.approx(1.0)
    assert rmse(ref, est) == pytest.approx(np.sqrt(np.mean([0.0, 1.0, 4.0, 9.0])))
    assert np.isnan(pearson(ref, np.zeros(5)))
    assert np.isnan(rmse(np.full(3, np.nan), np.zeros(3)))


def test_summarize_ignores_nans() -> None:
    stats = summarize([1.0, 2.0, 3.0, 4.0, float("nan")])
    assert stats == {"n": 4, "median": 2.5, "q1": 1.75, "q3": 3.25, "mean": 2.5}
    assert summarize([float("nan")])["n"] == 0


def _official() -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_helper_code", OFFICIAL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not OFFICIAL.exists(), reason="official evaluation-2024 checkout not present")
@pytest.mark.parametrize("seed", range(6))
def test_matches_the_official_implementation(seed: int) -> None:
    official = _official()
    rng = np.random.default_rng(seed)
    ref = beats(seed=seed)
    est = (
        np.roll(ref, int(rng.integers(-40, 40)))
        + rng.normal(0, 0.05, ref.size)
        + rng.normal(0, 0.2)
    )
    est[rng.integers(0, ref.size, 200)] = np.nan
    ours = align_signals(ref, est, upstream_quirk=True)
    theirs = official.align_signals(ref, est, num_quant_levels=2**8)
    assert ours[1] == theirs[1]
    assert ours[2] == pytest.approx(theirs[2])
    np.testing.assert_allclose(ours[0], theirs[0], equal_nan=True)
    for keep_nans in (True, False):
        mine = compute_snr(ref, est, keep_nans=keep_nans)
        upstream = official.compute_snr(ref, est, keep_nans=keep_nans)
        np.testing.assert_allclose(mine, upstream, equal_nan=True)
    np.testing.assert_allclose(
        compute_snr(ref, est, noise_median=True), official.compute_snr(ref, est, noise_median=True)
    )
