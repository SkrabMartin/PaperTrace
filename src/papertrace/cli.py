"""Command-line interface: thin wrappers over the library, one subcommand per pipeline step."""

import json
import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from papertrace import __version__
from papertrace.config import load_settings
from papertrace.data.manifest import build_manifest, read_manifest, write_manifest
from papertrace.data.ptbxl import load_index
from papertrace.eval.report import evaluate_dataset
from papertrace.synth.generate import generate_dataset, select_records
from papertrace.synth.kit import KitSetup
from papertrace.synth.manifest import write_synth_manifest

NOT_IMPLEMENTED_EXIT_CODE = 2

app = typer.Typer(
    name="papertrace",
    help="Digitise paper and photographed ECGs into signals with confidence estimates.",
    no_args_is_help=True,
    add_completion=False,
)


def _not_implemented(command: str) -> NoReturn:
    typer.echo(f"papertrace {command}: not implemented yet", err=True)
    raise typer.Exit(code=NOT_IMPLEMENTED_EXIT_CODE)


def _version_callback(show: bool) -> None:
    if show:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """PaperTrace command-line tools."""


@app.command()
def digitize(
    image: Annotated[Path | None, typer.Argument(help="ECG image (JPG, PNG, PDF).")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory.")] = None,
) -> None:
    """Digitise an ECG image into a 12-lead signal with a quality report."""
    _not_implemented("digitize")


@app.command()
def manifest(
    root: Annotated[
        Path | None, typer.Option(help="PTB-XL directory. Default: <data_dir>/ptb-xl.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Parquet file to write.")
    ] = None,
) -> None:
    """Build the patient-level split manifest from PTB-XL metadata."""
    settings = load_settings()
    root = root or settings.paths.data_dir / "ptb-xl"
    output = output or settings.paths.output_dir / "split_manifest.parquet"
    try:
        built = build_manifest(load_index(root))
    except FileNotFoundError as error:
        typer.echo(f"papertrace manifest: {error}", err=True)
        raise typer.Exit(code=1) from error
    write_manifest(built, output)
    counts = built.groupby("split").agg(
        records=("ecg_id", "size"), patients=("patient_id", "nunique")
    )
    typer.echo(counts.to_string())
    typer.echo(f"wrote {len(built)} records to {output}")


@app.command()
def synth(
    split: Annotated[
        str, typer.Option(help="Split to draw records from: train, val or test.")
    ] = "train",
    count: Annotated[int, typer.Option("-n", "--count", help="Number of PTB-XL records.")] = 100,
    variants: Annotated[int, typer.Option(help="Rendered images per record.")] = 1,
    batch_size: Annotated[int, typer.Option(help="Records per generator run.")] = 10,
    seed: Annotated[int, typer.Option(help="Master seed.")] = 0,
    artefacts: Annotated[
        bool | None,
        typer.Option(
            "--artefacts/--no-artefacts",
            help="Extra phone-photo artefacts. Default: on for train, off otherwise.",
        ),
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest", help="Split manifest. Default: <output_dir>/split_manifest.parquet."
        ),
    ] = None,
    root: Annotated[
        Path | None, typer.Option(help="PTB-XL directory. Default: <data_dir>/ptb-xl.")
    ] = None,
    kit_dir: Annotated[
        Path | None,
        typer.Option(help="ECG-Image-Kit checkout. Default: <data_dir>/external/ecg-image-kit."),
    ] = None,
    kit_python: Annotated[
        Path | None,
        typer.Option(
            help="Python of the generator environment. Default: <data_dir>/external/venv-ecgkit."
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output directory. Default: <output_dir>/synth."),
    ] = None,
) -> None:
    """Generate synthetic ECG images, lead masks and a manifest for training."""
    settings = load_settings()
    data_dir, output_dir = settings.paths.data_dir, settings.paths.output_dir
    manifest_path = manifest_path or output_dir / "split_manifest.parquet"
    root = root or data_dir / "ptb-xl"
    kit_dir = kit_dir or data_dir / "external" / "ecg-image-kit"
    venv = data_dir / "external" / "venv-ecgkit"
    kit_python = kit_python or (
        venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    )
    output = output or output_dir / "synth"
    if split not in ("train", "val", "test"):
        typer.echo("papertrace synth: --split must be train, val or test", err=True)
        raise typer.Exit(code=1)
    for required in (manifest_path, kit_dir, kit_python):
        if not required.exists():
            typer.echo(f"papertrace synth: not found: {required}", err=True)
            raise typer.Exit(code=1)
    records = select_records(read_manifest(manifest_path), split, count, seed)
    result = generate_dataset(
        records,
        root,
        output,
        KitSetup(kit_dir=kit_dir, python=kit_python),
        variants=variants,
        batch_size=batch_size,
        seed=seed,
        artefacts=(split == "train") if artefacts is None else artefacts,
    )
    write_synth_manifest(result, output / "synth_manifest.parquet")
    typer.echo(f"wrote {len(result)} images to {output}")


@app.command()
def train() -> None:
    """Train the curve segmentation model."""
    _not_implemented("train")


@app.command()
def evaluate(
    synth_dir: Annotated[
        Path | None,
        typer.Option("--synth", help="Output of `synth`. Default: <output_dir>/synth_val."),
    ] = None,
    root: Annotated[
        Path | None, typer.Option(help="PTB-XL directory. Default: <data_dir>/ptb-xl.")
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest", help="Split manifest. Default: <output_dir>/split_manifest.parquet."
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "-o", "--output", help="JSON report. Default: <output_dir>/baseline_report.json."
        ),
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Evaluate only the first N images.")] = None,
    workers: Annotated[
        int | None, typer.Option(help="Parallel processes (default: CPUs, max 8).")
    ] = None,
) -> None:
    """Score the classical baseline on synthetic images against the PTB-XL signals."""
    settings = load_settings()
    data_dir, output_dir = settings.paths.data_dir, settings.paths.output_dir
    synth_dir = synth_dir or output_dir / "synth_val"
    root = root or data_dir / "ptb-xl"
    manifest_path = manifest_path or output_dir / "split_manifest.parquet"
    output = output or output_dir / "baseline_report.json"
    for required in (synth_dir / "synth_manifest.parquet", manifest_path):
        if not required.exists():
            typer.echo(f"papertrace evaluate: not found: {required}", err=True)
            raise typer.Exit(code=1)
    report, _ = evaluate_dataset(synth_dir, root, read_manifest(manifest_path), limit, workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    overall = report["overall"]["snr_db"]
    typer.echo(
        f"{report['n_images']} images, SNR median {overall['median']:.2f} dB "
        f"(IQR {overall['q1']:.2f} to {overall['q3']:.2f}), report: {output}"
    )


@app.command()
def serve() -> None:
    """Run the REST API."""
    _not_implemented("serve")
