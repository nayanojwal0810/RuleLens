"""Keep the test suite independent of credentials, services and model downloads."""

import os
import socket

import pytest


for name in ("GROQ_API_KEY", "QDRANT_API_KEY", "QDRANT_URL"):
    os.environ.pop(name, None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["RULELENS_CACHE_ENABLED"] = "false"


_network_patch = pytest.MonkeyPatch()
_network_attempts = []


def deny_network(*args, **kwargs):
    _network_attempts.append("outbound_connection")
    raise AssertionError("Network access is disabled during offline tests.")


def pytest_configure(config):
    _network_patch.setattr(socket, "create_connection", deny_network)
    _network_patch.setattr(socket.socket, "connect", deny_network)
    _network_patch.setattr(socket.socket, "connect_ex", deny_network)


def pytest_sessionfinish(session, exitstatus):
    if _network_attempts:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.getplugin("terminalreporter")
        reporter.write_line(f"Unexpected network attempts: {len(_network_attempts)}")


def pytest_unconfigure(config):
    _network_patch.undo()
