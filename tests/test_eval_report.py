import numpy as np
import pytest

from papertrace.core.baseline import TOTAL_SAMPLES
from papertrace.core.layout import LEAD_NAMES
from papertrace.eval.report import ImageScore, build_report, masked_reference


def test_reference_is_masked_to_the_plotted_samples() -> None:
    names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal = np.arange(TOTAL_SAMPLES * 12, dtype=np.float64).reshape(TOTAL_SAMPLES, 12)
    plotted = [
        {"name": "I", "start_sample": 0, "end_sample": 1250},
        {"name": "aVR", "start_sample": 1250, "end_sample": 2500},
        {"name": "II", "start_sample": 0, "end_sample": 1250},
        {"name": "II", "start_sample": 0, "end_sample": TOTAL_SAMPLES},  # rhythm strip
    ]
    reference = masked_reference(signal, names, plotted)
    assert reference.shape == (TOTAL_SAMPLES, 12)
    i, avr, ii, v1 = (LEAD_NAMES.index(n) for n in ("I", "aVR", "II", "V1"))
    assert np.isfinite(reference[:1250, i]).all()
    assert np.isnan(reference[1250:, i]).all()
    assert np.isfinite(reference[1250:2500, avr]).all()
    assert np.isfinite(reference[:, ii]).all()  # the rhythm strip covers lead II completely
    assert np.isnan(reference[:, v1]).all()
    np.testing.assert_array_equal(reference[:1250, i], signal[:1250, 0])
    np.testing.assert_array_equal(
        reference[1250:2500, avr], signal[1250:2500, 3]
    )  # AVR is column 3


def _score(snr: float, layout: str, dpi: int, scale: float | None, ink: float = 0.05) -> ImageScore:
    n = len(LEAD_NAMES)
    return ImageScore(
        1, layout, dpi, [snr] * n, [snr - 1] * n, [0.9] * n, [0.1] * n, scale, 0.5, ink
    )


def test_report_groups_by_lead_layout_and_resolution() -> None:
    scores = [
        _score(10.0, "3x4", 200, 0.01, ink=0.03),
        _score(20.0, "3x4", 300, -0.03, ink=0.02),
        _score(4.0, "6x2", 200, None, ink=0.09),
    ]
    report = build_report(scores)
    assert report["n_images"] == 3
    assert report["overall"]["snr_db"]["median"] == pytest.approx(10.0)
    assert report["overall"]["snr_db_upstream_alignment"]["median"] == pytest.approx(9.0)
    assert set(report["per_lead"]) == set(LEAD_NAMES)
    assert report["by_layout"]["3x4"]["median"] == pytest.approx(15.0)
    assert report["by_resolution_dpi"]["200"]["n"] == 24
    assert report["scale_relative_error"]["n"] == 2
    terciles = report["by_ink_fraction_tercile"]
    assert terciles["low"]["snr_db"]["median"] == pytest.approx(20.0)
    assert terciles["high"]["snr_db"]["median"] == pytest.approx(4.0)
    assert terciles["high"]["ink_fraction_range"] == [0.09, 0.09]
