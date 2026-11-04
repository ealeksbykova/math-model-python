"""Общие фикстуры для тестов лабораторных работ."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).resolve().parents[1]

# пакет labkit лежит рядом с тестами, поэтому достаточно добавить solution/ в путь
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))


@pytest.fixture(scope="session")
def solution_dir() -> Path:
    """Каталог solution/ (там же лежат ноутбуки)."""
    return SOLUTION_DIR


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Корень репозитория."""
    return SOLUTION_DIR.parent
