from pathlib import Path

import pytest

from papertrace.config import ConfigError, load_settings

DEFAULT_TOML = Path(__file__).resolve().parent.parent / "config" / "default.toml"


def test_default_file_loads() -> None:
    settings = load_settings(DEFAULT_TOML, environ={})
    assert settings.api.port == 8000
    assert settings.database_url is None


def test_environment_overrides_file() -> None:
    settings = load_settings(DEFAULT_TOML, environ={"PAPERTRACE_API__PORT": "9000"})
    assert settings.api.port == 9000


def test_secret_comes_from_environment_and_is_masked() -> None:
    settings = load_settings(DEFAULT_TOML, environ={"PAPERTRACE_DATABASE_URL": "postgresql://x"})
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == "postgresql://x"
    assert "postgresql" not in repr(settings)


def test_secret_in_file_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('database_url = "postgresql://x"\n')
    with pytest.raises(ConfigError):
        load_settings(bad, environ={})


def test_missing_explicit_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml", environ={})
