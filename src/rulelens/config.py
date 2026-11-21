"""Configuration constants and environment variable helpers for RuleLens."""

import os
from typing import Optional

# Environment variable names
GROQ_API_KEY: str = "GROQ_API_KEY"
QDRANT_URL: str = "QDRANT_URL"
QDRANT_API_KEY: str = "QDRANT_API_KEY"

# Model identifiers
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL: str = "openai/gpt-oss-20b"

# Evidence limits
GENERATION_EVIDENCE_LIMIT: int = 4


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
