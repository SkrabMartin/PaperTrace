# PaperTrace

Turn photos and scans of paper ECGs into digital 12-lead signals, and tell the user how far
each part of the result can be trusted.

> **Status: Phase 0, project skeleton.** The repository structure, command-line placeholders
> and quality gates are in place. No digitisation logic exists yet.
>
> **Not a medical device.** PaperTrace is a research and teaching project. Its output must not
> be used for clinical decisions.

## Idea

Many ECGs exist only as printouts, PDF scans or phone photos, which makes them impossible to
analyse automatically. PaperTrace aims to reconstruct the underlying signal from such images
and, unlike a plain image-to-signal converter, to estimate per-lead and per-segment
confidence, so that an unreliable reconstruction can be flagged or rejected instead of
silently trusted. Planned outputs are WFDB, CSV/JSON and HL7 FHIR, plus clinical interval
measurements (RR, PR, QRS, QT/QTc) compared against the original signal.

## Project structure

```
src/papertrace/
  core/     digitisation pipeline as a pure library (no web or training dependencies)
  data/     PTB-XL access, patient-level splits, leakage checks
  synth/    synthetic ECG images and masks for training
  train/    model training and ONNX export
  eval/     metrics, robustness matrix, clinical interval error
  export/   WFDB, CSV/JSON and FHIR writers
  api/      REST API and background worker
  config.py settings: config/default.toml + PAPERTRACE_* environment variables
  cli.py    `papertrace` command line
frontend/   React + TypeScript + Vite web interface
config/     default settings
scripts/    developer tooling
tests/      unit, smoke and architecture tests
```

Dependencies between packages point one way only (`core` and `data` are leaves, `cli` is on
top). `tests/test_architecture.py` enforces this.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js (for the frontend).

```
uv sync
uv run papertrace --help

cd frontend
npm ci
```

All subcommands are placeholders for now and exit with code 2 and "not implemented yet".

Settings come from `config/default.toml`; any value can be overridden with an environment
variable, for example `PAPERTRACE_API__PORT=9000`. Secrets are read from the environment only.

## Running the checks

```
uv run python scripts/check.py     # every gate below, in CI order
```

Or individually:

```
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run pre-commit run --all-files

cd frontend
npm run lint && npm run format:check && npm run typecheck && npm test && npm run build
```

Install the git hooks once with `uv run pre-commit install`.

## Licence

MIT, see [LICENSE](LICENSE).
