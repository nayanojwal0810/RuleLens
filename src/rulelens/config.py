"""Configuration constants and environment variable helpers for RuleLens."""

import os
import sys
from pathlib import Path
from typing import Optional, Union

# Environment variable names
GROQ_API_KEY: str = "GROQ_API_KEY"
QDRANT_URL: str = "QDRANT_URL"
QDRANT_API_KEY: str = "QDRANT_API_KEY"

# Qdrant configuration
QDRANT_COLLECTION_NAME: str = "rulelens_policies"
QDRANT_CACHE_COLLECTION_NAME: str = "rulelens_answer_cache_v1"

# Semantic cache configuration
RULELENS_CACHE_ENABLED_VAR: str = "RULELENS_CACHE_ENABLED"
RULELENS_CACHE_THRESHOLD_VAR: str = "RULELENS_CACHE_THRESHOLD"
CACHE_DEFAULT_SIMILARITY_THRESHOLD: float = 0.90


def is_cache_enabled() -> bool:
    """Return True if semantic caching is explicitly enabled in the environment.

    Defaults to False until an evaluated similarity threshold is configured.
    """
    val = get_env_var(RULELENS_CACHE_ENABLED_VAR, "false")
    return bool(val and val.strip().lower() in ("true", "1", "yes", "on"))


def get_cache_similarity_threshold() -> Optional[float]:
    """Return the configured similarity threshold for semantic cache candidate lookup.

    Returns None if caching is disabled or no threshold is set.
    """
    if not is_cache_enabled():
        return None
    val = get_env_var(RULELENS_CACHE_THRESHOLD_VAR)
    if val is None or not val.strip():
        return CACHE_DEFAULT_SIMILARITY_THRESHOLD
    try:
        thresh = float(val.strip())
        if 0.0 <= thresh <= 1.0:
            return thresh
    except ValueError:
        pass
    return CACHE_DEFAULT_SIMILARITY_THRESHOLD

# Model identifiers
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL: str = "openai/gpt-oss-20b"

# Evidence limits
GENERATION_EVIDENCE_LIMIT: int = 4


def load_dotenv_if_present(env_path: Optional[Union[Path, str]] = None) -> None:
    """Load key-value pairs from a local .env file into os.environ if not already set.

    Preserves any explicitly configured environment variables (never overrides them).
    Silently ignores missing files. Never logs or prints credential values.
    """
    path = Path(env_path) if env_path else Path.cwd() / ".env"
    if not path.is_file():
        # Also check parent directory if cwd is inside tests or a subpackage
        parent_env = Path.cwd().parent / ".env"
        if parent_env.is_file():
            path = parent_env
        else:
            return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    key, val = stripped.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Strip surrounding quotes if present
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    if key and val and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        # Silently pass on read/encoding errors
        pass


# Automatically load local .env on import if not already set in environment and not under pytest
if "PYTEST_CURRENT_TEST" not in os.environ and "pytest" not in sys.modules:
    load_dotenv_if_present()


def is_invalid_qdrant_url(url: Optional[str]) -> bool:
    """Check if the provided URL is the Qdrant marketing website rather than an API cluster endpoint."""
    if not url:
        return False
    cleaned = url.strip().rstrip("/").lower()
    return cleaned in ("https://qdrant.io", "http://qdrant.io")


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable without validating or making external calls.

    Args:
        name: The name of the environment variable.
        default: Optional fallback value if the environment variable is not set.

    Returns:
        The value of the environment variable, or the default value.
    """
    return os.getenv(name, default)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Alias for get_env_var."""
    return get_env_var(name, default)


def get_config() -> dict[str, Optional[str]]:
    """Return core configuration variable values from the current environment.

    Does not validate credentials or make external network calls.
    """
    return {
        GROQ_API_KEY: get_env_var(GROQ_API_KEY),
        QDRANT_URL: get_env_var(QDRANT_URL),
        QDRANT_API_KEY: get_env_var(QDRANT_API_KEY),
    }
