"""
conftest.py — pytest configuration for the triage agent test suite.

Provides:
  - sys.path setup so imports work from tests/ directory
  - Session-level retriever fixture (expensive, loaded once)
  - Markers for test categories
"""

import sys
import pathlib

# Ensure code/ is importable from tests/
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks tests requiring ANTHROPIC_API_KEY")
    config.addinivalue_line("markers", "requires_output: marks tests requiring output.csv")
