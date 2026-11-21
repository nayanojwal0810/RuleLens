"""Offline unit tests for RuleLens configuration and package foundation."""

import importlib
import socket
import rulelens
import rulelens.config as config


def test_package_version():
    """Verify package version is defined as 0.1.0."""
    assert rulelens.__version__ == "0.1.0"


def test_model_identifiers():
    """Verify embedding and generation model identifier constants."""
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert config.GENERATION_MODEL == "openai/gpt-oss-20b"


def test_evidence_limit():
    """Verify generation evidence limit equals 4."""
    assert config.GENERATION_EVIDENCE_LIMIT == 4


def test_env_var_constants():
    """Verify environment variable names are correctly defined."""
    assert config.GROQ_API_KEY == "GROQ_API_KEY"
    assert config.QDRANT_URL == "QDRANT_URL"
    assert config.QDRANT_API_KEY == "QDRANT_API_KEY"


def test_get_env_var_reading(monkeypatch):
    """Verify environment variable reader returns set values and defaults."""
    monkeypatch.setenv("TEST_RULELENS_VAR", "test_value_123")
    assert config.get_env_var("TEST_RULELENS_VAR") == "test_value_123"
    assert config.get_env("TEST_RULELENS_VAR") == "test_value_123"

    monkeypatch.delenv("TEST_RULELENS_VAR", raising=False)
    assert config.get_env_var("TEST_RULELENS_VAR") is None
    assert config.get_env_var("TEST_RULELENS_VAR", "default_val") == "default_val"


def test_missing_optional_env_vars_allow_import(monkeypatch):
    """Verify missing optional environment variables do not prevent import."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    reloaded = importlib.reload(config)
    assert reloaded.GROQ_API_KEY == "GROQ_API_KEY"
    cfg = reloaded.get_config()
    assert cfg[reloaded.GROQ_API_KEY] is None
    assert cfg[reloaded.QDRANT_URL] is None
    assert cfg[reloaded.QDRANT_API_KEY] is None


def test_no_external_services_called(monkeypatch):
    """Verify no network or external socket connections are made during config access."""
    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline test execution")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # Reload config and read values with blocked network
    reloaded = importlib.reload(config)
    _ = reloaded.get_config()
    _ = reloaded.get_env_var("GROQ_API_KEY")
    assert reloaded.GENERATION_EVIDENCE_LIMIT == 4
