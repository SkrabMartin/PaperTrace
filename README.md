# PaperTrace

Turn photos and scans of paper ECGs into digital 12-lead signals, and tell the user how far
each part of the result can be trusted.

> **Status: Phase 2, classical baseline.** PTB-XL loading, patient-level splits, synthetic training
> images with lead masks, and a naive non-learned digitiser with the challenge's scoring work. The
> learned model, confidence estimates and the web application are not built yet.
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
  core/     digitisation as a pure library (no web or training dependencies): grid, calibration, baseline
  data/     PTB-XL access, patient-level splits, leakage checks
  synth/    synthetic ECG images, lead masks and phone-photo artefacts
  train/    model training and ONNX export
  eval/     scoring (challenge SNR), baseline evaluation report
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

`manifest`, `synth` and `evaluate` work (see below); the other subcommands are still placeholders
that exit with code 2 and "not implemented yet".

### Data and synthetic images

The PTB-XL dataset (CC BY 4.0) is not part of the repository. Download it from PhysioNet into
`data/ptb-xl/` (the `data/` folder is git-ignored), then:

```
uv run papertrace manifest          # patient-level split manifest -> output/split_manifest.parquet
uv run papertrace synth -n 100      # synthetic images, lead masks, manifest -> output/synth/
uv run python scripts/preview_masks.py output/synth -n 5   # overlays for a visual check
```

`evaluate` runs the classical baseline (grid scale, rotation correction, centroid per column) on a
synthetic set and scores it against the original PTB-XL signals with the PhysioNet Challenge 2024
signal-to-noise ratio, per lead:

```
uv run papertrace synth --split val -n 100 -o output/synth_val
uv run papertrace evaluate          # writes output/baseline_report.json
```

The baseline is deliberately naive: it is the reference that the learned model has to beat, and it
is weak where the paper grid is as dark as the trace.

Synthetic images are rendered with [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit),
which needs its own Python 3.10 environment (it pins old TensorFlow and NumPy versions). By default
`synth` looks for the checkout in `data/external/ecg-image-kit` and its interpreter in
`data/external/venv-ecgkit`; both can be changed with `--kit-dir` and `--kit-python`.
Every synthetic image inherits the patient and fold of its source record, so test records never
leak into training.

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
