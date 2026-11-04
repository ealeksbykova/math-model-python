"""Общие помощники для сборки ноутбуков лабораторных работ."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

#: Каталог solution/ (родитель каталога tools/notebooks)
SOLUTION_DIR = Path(__file__).resolve().parents[2]
#: Каталог с исходными заданиями
TASK_DIR = SOLUTION_DIR.parent / "task"

KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

LANGUAGE_INFO = {
    "name": "python",
    "version": "3.9.6",
    "mimetype": "text/x-python",
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "codemirror_mode": {"name": "ipython", "version": 3},
}


def md(text: str):
    """Markdown-ячейка."""
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    """Ячейка с кодом."""
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(cells) -> nbf.NotebookNode:
    """Собрать ноутбук nbformat v4 из списка ячеек."""
    nb = nbf.v4.new_notebook()
    nb["cells"] = list(cells)
    nb["metadata"] = {
        "kernelspec": dict(KERNELSPEC),
        "language_info": dict(LANGUAGE_INFO),
    }
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    return nb


# Ячейка настройки: делает пакет labkit импортируемым независимо от того,
# откуда запущен ноутбук (из каталога solution/ или из корня репозитория).
SETUP_CELL = r'''
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- поиск пакета labkit (ноутбук может быть запущен из solution/ или из корня) ---
for _candidate in [Path.cwd(), *Path.cwd().parents]:
    if (_candidate / "labkit" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        PROJECT_ROOT = _candidate
        break
else:  # pragma: no cover
    raise RuntimeError("не найден пакет labkit рядом с ноутбуком")

import labkit as lk

np.set_printoptions(precision=4, suppress=True, linewidth=120)

plt.rcParams.update({
    "figure.dpi": 110,
    "figure.figsize": (7.5, 4.4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 11,
})

print("Python", sys.version.split()[0])
print("numpy", np.__version__, "| matplotlib", matplotlib.__version__)
# печатаем только имя каталога, чтобы в ноутбук не попадал абсолютный путь машины
print("пакет labkit найден в подкаталоге:", PROJECT_ROOT.name)
'''
