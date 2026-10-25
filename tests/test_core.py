from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from PIL import Image

from papertrace.core import image as image_module
from papertrace.core.baseline import FS, TOTAL_SAMPLES, digitize_baseline
from papertrace.core.calibrate import (
    mv_to_px,
    px_to_mv,
    px_to_seconds,
    resample_trace,
    seconds_to_px,
)
from papertrace.core.grid import estimate_grid_period, estimate_rotation_deg, rotate_image
from papertrace.core.image import load_image, resolution_warnings
from papertrace.core.layout import LEAD_NAMES, LeadCell, cells_from_entries, find_row_bands
from papertrace.core.vectorize import centroid_trace, ink_mask
from papertrace.eval.metrics import lead_snr
from papertrace.synth.masks import STANDARD_LEADS


def render_grid(width: int, height: int, mm_px: float) -> NDArray[np.uint8]:
    """White sheet with 1 mm pink and 5 mm red grid lines, one pixel wide."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for k in range(int(max(width, height) / mm_px) + 1):
        colour = (255, 0, 0) if k % 5 == 0 else (255, 190, 190)
        position = round(k * mm_px)
        if position < width:
            image[:, position] = colour
        if position < height:
            image[position, :] = colour
    return image


def beats(seed: int, n: int = TOTAL_SAMPLES) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.zeros(n)
    period = int(rng.integers(330, 450))
    for centre in range(200 + int(rng.integers(0, 60)), n, period):
        x += np.exp(-0.5 * ((t - centre) / 5) ** 2) * rng.uniform(0.6, 1.2)
        x += np.exp(-0.5 * ((t - centre - 100) / 22) ** 2) * 0.25
    return x


def test_lead_names_agree_across_layers() -> None:
    assert LEAD_NAMES == STANDARD_LEADS


@pytest.mark.parametrize("mm_px", [4.0, 5.9, 7.874, 9.5, 11.81])
def test_grid_period_is_recovered_within_one_percent(mm_px: float) -> None:
    estimate = estimate_grid_period(render_grid(1400, 1000, mm_px))
    assert estimate == pytest.approx(mm_px, rel=0.01)


@pytest.mark.parametrize("degrees", [-8.0, -3.3, 0.0, 2.1, 6.3])
def test_rotation_is_estimated(degrees: float) -> None:
    tilted = rotate_image(render_grid(1400, 1000, 7.874), degrees)
    assert estimate_rotation_deg(tilted) == pytest.approx(-degrees, abs=0.4)


def test_grid_period_survives_rotation_correction() -> None:
    tilted = rotate_image(render_grid(1600, 1200, 7.874), 5.0)
    upright = rotate_image(tilted, estimate_rotation_deg(tilted))
    assert estimate_grid_period(upright) == pytest.approx(7.874, rel=0.02)


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    source = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    path = tmp_path / "photo.jpg"
    source.save(path, exif=exif)
    assert load_image(path).shape == (40, 20, 3)


def test_load_image_rejects_decompression_bombs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.png"
    Image.new("RGB", (100, 100)).save(path)
    monkeypatch.setattr(image_module, "MAX_PIXELS", 5_000)
    with pytest.raises(ValueError, match="too large"):
        load_image(path)


def test_low_resolution_is_flagged() -> None:
    assert resolution_warnings(np.zeros((100, 400, 3), dtype=np.uint8))
    assert not resolution_warnings(np.zeros((100, 1200, 3), dtype=np.uint8))


finite = st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
scale = st.floats(min_value=2.0, max_value=20.0, allow_nan=False)


@given(st.lists(finite, min_size=1, max_size=20), finite, scale)
def test_millivolt_conversion_round_trips(values: list[float], baseline: float, ppm: float) -> None:
    mv = np.array(values)
    back = px_to_mv(mv_to_px(mv, baseline, ppm), baseline, ppm)
    np.testing.assert_allclose(back, mv, atol=1e-9)


@given(st.lists(finite, min_size=1, max_size=20), scale)
def test_time_conversion_round_trips(values: list[float], ppm: float) -> None:
    seconds = np.array(values)
    np.testing.assert_allclose(px_to_seconds(seconds_to_px(seconds, ppm), ppm), seconds, atol=1e-9)


def test_one_millivolt_is_ten_millimetres_and_one_second_is_25() -> None:
    assert px_to_mv(np.array([90.0]), 100.0, 1.0)[0] == pytest.approx(1.0)
    assert seconds_to_px(np.array([1.0]), 2.0)[0] == pytest.approx(50.0)


def test_resample_recovers_a_ramp_and_leaves_gaps() -> None:
    ppm = 8.0  # 200 px per second
    x = np.arange(1000, dtype=np.float64)
    y = x * 0.001  # 1 mV per 1000 px
    y[300:600] = np.nan  # 1.5 s gap, longer than the allowed 3 mm = 0.12 s
    out = resample_trace(x, y, ppm, n_samples=2600, fs=500)
    assert out[0] == pytest.approx(0.0)
    assert out[200] == pytest.approx(0.08, abs=1e-3)  # t = 0.4 s -> x = 80 px -> 0.08 mV
    assert np.isnan(out[1000])  # t = 2.0 s -> x = 400 px lies inside the gap
    assert np.isnan(out[2550])  # t = 5.1 s is beyond the observed 4.995 s


def test_resample_without_data_is_all_nan() -> None:
    assert np.isnan(resample_trace(np.arange(5.0), np.full(5, np.nan), 8.0, 10)).all()


def test_ink_excludes_coloured_grid() -> None:
    image = render_grid(200, 200, 8.0)
    image[100, 20:180] = (20, 20, 20)
    ink = ink_mask(image)
    assert ink[100, 50]
    assert ink.sum() == 160  # only the trace; red and pink grid lines are not ink


def test_centroid_trace_follows_a_line() -> None:
    ink = np.zeros((50, 30), dtype=bool)
    for x in range(30):
        ink[10 + x // 2 : 13 + x // 2, x] = True
    ink[:, 5] = False
    trace = centroid_trace(ink, y_offset=100.0)
    assert trace[0] == pytest.approx(111.0)
    assert trace[20] == pytest.approx(121.0)
    assert np.isnan(trace[5])


def _entries(names: list[str], samples: int, columns: int) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "start_sample": (index % columns) * samples,
            "end_sample": (index % columns + 1) * samples,
        }
        for index, name in enumerate(names)
    ]


def test_cells_are_laid_out_bottom_row_first() -> None:
    names = ["III", "aVF", "V3", "V6", "II", "aVL", "V2", "V5", "I", "aVR", "V1", "V4", "II"]
    cells = cells_from_entries(_entries(names, 1250, 4), columns=4)
    by_name = {c.name: c for c in cells if not c.rhythm}
    assert (by_name["I"].row, by_name["I"].col) == (0, 0)
    assert (by_name["V6"].row, by_name["V6"].col) == (2, 3)
    assert by_name["aVR"].start_sample == 1250
    rhythm = [c for c in cells if c.rhythm]
    assert len(rhythm) == 1
    assert rhythm[0].row == 3


def test_incomplete_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="do not fill rows"):
        cells_from_entries(_entries(["I", "II", "III"], 100, 2), columns=2)


def test_row_bands_split_the_sheet_between_rows() -> None:
    ink = np.zeros((400, 300), dtype=bool)
    for centre in (50, 150, 250, 350):
        ink[centre - 2 : centre + 2, 10:290] = True
    bands = find_row_bands(ink, 4)
    assert len(bands) == 4
    for (top, bottom), centre in zip(bands, (50, 150, 250, 350), strict=True):
        assert top < centre < bottom
    assert bands[0][1] == bands[1][0]


def _page(
    mm_px: float, columns: int, rotation: float, seed: int, frame: bool = False
) -> tuple[NDArray[np.uint8], list[LeadCell], NDArray[np.float64]]:
    """A synthetic ECG page with a known signal, grid scale and lead topology."""
    rows = 3 if columns == 4 else 6
    pitch = round(30 * mm_px)
    x0 = round(15 * mm_px)
    width = round(250 * mm_px) + 2 * x0
    height = (rows + 2) * pitch
    image = render_grid(width, height, mm_px)
    if frame:  # a dark border along the page edge, as in many scans and in the generator output
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (0, 0, 0), thickness=4)
    signals = {name: beats(seed * 100 + i) for i, name in enumerate(LEAD_NAMES)}
    names = list(reversed(LEAD_NAMES))
    samples = TOTAL_SAMPLES // columns
    cells = cells_from_entries(
        [*_entries(names, samples, columns), *_entries(["II"], 0, 1)], columns
    )
    cells = [c if not c.rhythm else LeadCell("II", c.row, 0, 0, TOTAL_SAMPLES, True) for c in cells]
    truth = np.full((TOTAL_SAMPLES, len(LEAD_NAMES)), np.nan)
    for cell in cells:
        index = LEAD_NAMES.index(cell.name)
        n = np.arange(cell.start_sample, cell.end_sample)
        x = x0 + n / FS * 25 * mm_px
        y = (cell.row + 1) * pitch - signals[cell.name][n] * 10 * mm_px
        points = np.rint(np.stack([x, y], axis=1)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], isClosed=False, color=(10, 10, 10), thickness=2)
        truth[n, index] = signals[cell.name][n]
    return (rotate_image(image, rotation) if rotation else image), cells, truth


GOLDEN_CASES = [
    # (pixels per mm, columns, rotation in degrees, minimum median SNR in dB)
    (7.874, 4, 0.0, 15.0),
    (5.9, 4, 3.0, 12.0),
    (11.81, 4, -4.0, 15.0),
    (7.874, 2, 0.0, 15.0),
    (7.874, 2, 5.0, 12.0),
]
FRAMED_CASE = (7.874, 4, 0.0, 10.0)


@pytest.mark.parametrize(("mm_px", "columns", "rotation", "minimum"), GOLDEN_CASES)
def test_baseline_reconstructs_a_known_signal(
    mm_px: float, columns: int, rotation: float, minimum: float
) -> None:
    image, cells, truth = _page(mm_px, columns, rotation, seed=1)
    output = digitize_baseline(image, cells, columns)
    assert output.px_per_mm == pytest.approx(mm_px, rel=0.02)
    assert output.rotation_deg == pytest.approx(-rotation, abs=0.5)
    assert output.result.scale_source == "grid"
    scores = [lead_snr(truth[:, j], output.signal[:, j]) for j in range(truth.shape[1])]
    assert np.nanmedian(scores) > minimum, scores


def test_a_dark_page_frame_does_not_shift_the_time_axis() -> None:
    mm_px, columns, rotation, minimum = FRAMED_CASE
    image, cells, truth = _page(mm_px, columns, rotation, seed=1, frame=True)
    output = digitize_baseline(image, cells, columns)
    scores = [lead_snr(truth[:, j], output.signal[:, j]) for j in range(truth.shape[1])]
    assert np.nanmedian(scores) > minimum, scores
