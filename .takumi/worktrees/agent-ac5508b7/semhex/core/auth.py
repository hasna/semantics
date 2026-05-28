"""Secrets-aware API key loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_api_key(var_name: str) -> str | None:
    """Load an API key from env or ~/.secrets env files."""
    env_value = os.environ.get(var_name)
    if env_value:
        return env_value

    secrets_dir = Path.home() / ".secrets"
    if not secrets_dir.is_dir():
        return None

    for path in sorted(secrets_dir.rglob("*.env")):
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("export "):
                stripped = stripped[len("export "):].strip()

            if not stripped.startswith(f"{var_name}="):
                continue

            return stripped.split("=", 1)[1].strip().strip('"').strip("'")

    return None
