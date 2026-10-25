"""PaperTrace: digitise paper and photographed ECGs into signals with confidence estimates.

Layers (dependencies point downwards only, enforced by tests/test_architecture.py):
``core`` and ``data`` are leaves; ``synth``, ``eval``, ``export``, ``train`` and ``api`` build on
them; ``cli`` sits on top.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("papertrace")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
