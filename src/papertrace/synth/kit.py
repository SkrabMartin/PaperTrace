"""Running ECG-Image-Kit as an external tool.

The generator pins old dependencies (it was written for Python below 3.11 and TensorFlow 2.14),
so it runs in its own environment and is driven through its command line. Parameters here follow
the generator's own documentation; unknown flags are not guessed.
"""

import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Layout = Literal["3x4", "6x2"]

LAYOUT_COLUMNS: dict[Layout, int] = {"3x4": 4, "6x2": 2}
RESOLUTIONS = (100, 150, 200, 250, 300)
ROTATIONS = (0, 5, 10)
NOISE_LEVELS = (0, 10, 20)
BATCH_SCRIPT = "gen_ecg_images_from_data_batch.py"
GENERATOR_SUBDIR = Path("codes") / "ecg-image-generator"


@dataclass(frozen=True)
class KitSetup:
    """Where the generator checkout and its Python interpreter live."""

    kit_dir: Path
    python: Path

    @property
    def script_dir(self) -> Path:
        """Directory the generator must be run from (it uses relative imports)."""
        return self.kit_dir / GENERATOR_SUBDIR


@dataclass(frozen=True)
class BatchSpec:
    """Generator settings shared by all records of one batch."""

    layout: Layout
    resolution: int
    wrinkles: bool
    rotate: int
    noise: int
    seed: int

    def to_args(self) -> list[str]:
        """Command-line flags for the generator (geometry is always stored for the masks).

        The generator's header printing is deliberately not used: it writes the local path of the
        input record onto the image.
        """
        args = [
            "-se", str(self.seed),
            "-r", str(self.resolution),
            "--num_columns", str(LAYOUT_COLUMNS[self.layout]),
            "--store_config", "2",
            "--lead_bbox",
            "--lead_name_bbox",
            "--calibration_pulse", "0.8",
            "--random_bw", "0.2",
            "--random_grid_color",
        ]  # fmt: skip
        if self.wrinkles:
            args.append("--wrinkles")
        if self.rotate or self.noise:
            args += ["--augment", "-rot", str(self.rotate), "-noise", str(self.noise)]
            args.append("--deterministic_noise")
        return args

    def to_dict(self) -> dict[str, object]:
        """Plain-dict form for the manifest."""
        return asdict(self)


def sample_spec(rng: random.Random, seed: int) -> BatchSpec:
    """Draw one batch specification (layout, resolution, wrinkles, rotation, noise)."""
    layouts: tuple[Layout, ...] = ("3x4", "6x2")
    return BatchSpec(
        layout=rng.choice(layouts),
        resolution=rng.choice(RESOLUTIONS),
        wrinkles=rng.random() < 0.3,
        rotate=rng.choice(ROTATIONS),
        noise=rng.choice(NOISE_LEVELS),
        seed=seed,
    )


def run_batch(setup: KitSetup, spec: BatchSpec, input_dir: Path, output_dir: Path) -> None:
    """Generate images for every WFDB record in ``input_dir`` into ``output_dir``.

    Raises:
        RuntimeError: If the generator exits with an error (its output is included).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(setup.python.resolve()),
        BATCH_SCRIPT,
        "-i", str(input_dir.resolve()),
        "-o", str(output_dir.resolve()),
        *spec.to_args(),
    ]  # fmt: skip
    result = subprocess.run(
        command, cwd=setup.script_dir, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        raise RuntimeError(f"ECG-Image-Kit failed (exit {result.returncode}):\n{tail}")
