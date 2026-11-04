"""Сквозной тест: ноутбуки исполняются без ошибок с нуля.

Тест собирает ноутбуки заново (через скрипты ``tools/notebooks/build_*.py``) и
исполняет их в отдельном процессе Jupyter-ядра. Это гарантирует, что
поставляемые `.ipynb` воспроизводимы, а не только «застыли» в исполненном виде.

Тест помечен ``slow``: ``pytest -m "not slow"`` пропускает его.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")

BUILDERS = {
    "lab1_1": "build_lab1_1.py",
    "lab1_2": "build_lab1_2.py",
    "lab2": "build_lab2.py",
}

NOTEBOOKS = {
    "lab1_1": "Практическое занятие 1_1 - метод Либмана.ipynb",
    "lab1_2": "Практическое занятие 1_2 - метод конечных разностей.ipynb",
    "lab2": "Практическое занятие 2 - явная схема.ipynb",
}


@pytest.mark.parametrize("key", sorted(NOTEBOOKS))
def test_notebook_executes_from_scratch(key, solution_dir: Path, tmp_path: Path):
    builder = solution_dir / "tools" / "notebooks" / BUILDERS[key]
    assert builder.exists(), f"нет скрипта сборки {builder}"

    target = tmp_path / NOTEBOOKS[key]
    # собираем ноутбук во временный каталог
    subprocess.run([sys.executable, str(builder), str(target)],
                   check=True, cwd=builder.parent,
                   capture_output=True, text=True)
    assert target.exists()

    nb = nbformat.read(target, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        # ноутбук ищет labkit по родительским каталогам, поэтому запускаем из solution/
        resources={"metadata": {"path": str(solution_dir)}},
        allow_errors=False,
    )
    client.execute()

    errors = [(o.get("ename"), str(o.get("evalue"))[:200])
              for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, f"{key}: ошибки при исполнении {errors}"

    images = sum(1 for c in nb.cells if c.cell_type == "code"
                 for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    assert images >= 4, f"{key}: ожидались графики, найдено {images}"
