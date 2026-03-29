"""Root pytest configuration for the llm_swap test suite.

Configures the anyio backend (asyncio) for all async tests and sets
common fixtures and markers.
"""

from __future__ import annotations

import pytest


# Make all async tests use asyncio via anyio
pytest_plugins = ("anyio",)
