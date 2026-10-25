"""Command-line interface: thin wrappers over the library, one subcommand per pipeline step."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from papertrace import __version__

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
def manifest() -> None:
    """Build the patient-level split manifest from PTB-XL metadata."""
    _not_implemented("manifest")


@app.command()
def synth() -> None:
    """Generate synthetic ECG images and masks for training."""
    _not_implemented("synth")


@app.command()
def train() -> None:
    """Train the curve segmentation model."""
    _not_implemented("train")


@app.command()
def evaluate() -> None:
    """Evaluate digitisation quality against ground-truth signals."""
    _not_implemented("evaluate")


@app.command()
def serve() -> None:
    """Run the REST API."""
    _not_implemented("serve")
