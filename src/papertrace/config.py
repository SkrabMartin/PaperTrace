"""Settings loaded from a TOML file and overridden by ``PAPERTRACE_*`` environment variables.

Precedence (low to high): model defaults, ``config/default.toml``, environment variables.
Nested keys use a double underscore, e.g. ``PAPERTRACE_API__PORT=9000``. Secrets are read
from the environment only and are rejected if they appear in the TOML file.
"""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr

ENV_PREFIX = "PAPERTRACE_"
NESTED_DELIMITER = "__"
CONFIG_PATH_VAR = "PAPERTRACE_CONFIG"
DEFAULT_CONFIG_PATH = Path("config/default.toml")
SECRET_FIELDS = frozenset({"database_url", "storage_secret_key"})


class ConfigError(ValueError):
    """Raised when configuration is inconsistent or unsafe."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsSettings(_Section):
    """Local directories; all are git-ignored."""

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    output_dir: Path = Path("output")


class LoggingSettings(_Section):
    """Logging options."""

    level: str = "INFO"


class ApiSettings(_Section):
    """Network binding of the REST API."""

    host: str = "127.0.0.1"
    port: int = 8000


class Settings(_Section):
    """Top-level settings."""

    paths: PathsSettings = PathsSettings()
    logging: LoggingSettings = LoggingSettings()
    api: ApiSettings = ApiSettings()
    database_url: SecretStr | None = None
    storage_secret_key: SecretStr | None = None


def _set_nested(target: dict[str, Any], keys: list[str], value: str) -> None:
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigError(f"conflicting configuration for '{key}'")
        target = child
    target[keys[-1]] = value


def load_settings(path: Path | None = None, environ: Mapping[str, str] | None = None) -> Settings:
    """Build settings from a TOML file and environment overrides.

    Args:
        path: TOML file. Defaults to ``$PAPERTRACE_CONFIG`` or ``config/default.toml``; if that
            default is missing, built-in defaults are used. An explicit path must exist.
        environ: Environment mapping; defaults to ``os.environ``.

    Raises:
        ConfigError: If a secret is set in the file.
        FileNotFoundError: If an explicitly requested file does not exist.
    """
    env = os.environ if environ is None else environ
    explicit = path or (Path(env[CONFIG_PATH_VAR]) if CONFIG_PATH_VAR in env else None)
    file_path = explicit or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if explicit is not None or file_path.exists():
        with file_path.open("rb") as handle:
            data = tomllib.load(handle)
    leaked = SECRET_FIELDS & data.keys()
    if leaked:
        raise ConfigError(f"secrets must come from the environment, not the file: {sorted(leaked)}")
    for name, value in env.items():
        if name.startswith(ENV_PREFIX) and name != CONFIG_PATH_VAR:
            keys = name.removeprefix(ENV_PREFIX).lower().split(NESTED_DELIMITER)
            _set_nested(data, keys, value)
    return Settings.model_validate(data)
