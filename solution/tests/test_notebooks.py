"""Тесты ноутбуков: структура, валидность nbformat и отсутствие ошибок исполнения.

Ноутбуки — основной результат лабораторных работ, поэтому проверяется:

* каждый ноутбук существует и валиден по схеме nbformat v4;
* все ячейки кода имеют сохранённый ``execution_count`` (ноутбук исполнен);
* в сохранённых выходах нет ошибок;
* есть графики (встроенные PNG) и markdown-описание с выводами;
* в ноутбуках нет абсолютных путей (артефакты переносимы).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS = {
    "lab1_1": "Практическое занятие 1_1 - метод Либмана.ipynb",
    "lab1_2": "Практическое занятие 1_2 - метод конечных разностей.ipynb",
    "lab2": "Практическое занятие 2 - явная схема.ipynb",
}


@pytest.fixture(scope="module", params=sorted(NOTEBOOKS))
def notebook(request, solution_dir: Path):
    path = solution_dir / NOTEBOOKS[request.param]
    if not path.exists():
        pytest.skip(f"ноутбук не собран: {path.name}")
    return request.param, path, nbformat.read(path, as_version=4)


def test_notebook_is_valid_nbformat(notebook):
    _, path, nb = notebook
    nbformat.validate(nb)          # бросит исключение, если структура неверна
    assert nb.nbformat == 4
    assert path.suffix == ".ipynb"


def test_notebook_metadata_has_python_kernel(notebook):
    _, _, nb = notebook
    assert nb.metadata["kernelspec"]["name"] == "python3"
    assert nb.metadata["language_info"]["name"] == "python"


def test_all_code_cells_are_executed(notebook):
    name, _, nb = notebook
    pending = [i for i, c in enumerate(nb.cells)
               if c.cell_type == "code" and c.execution_count is None]
    assert not pending, f"{name}: не исполнены ячейки {pending}"


def test_no_error_outputs(notebook):
    name, _, nb = notebook
    errors = [(i, o.get("ename"), str(o.get("evalue"))[:120])
              for i, c in enumerate(nb.cells) if c.cell_type == "code"
              for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, f"{name}: ошибки исполнения {errors}"


def test_no_stderr_warnings(notebook):
    """Предупреждения (например, overflow) означают, что модель работает нештатно."""
    name, _, nb = notebook
    stderr = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
                      for o in c.get("outputs", [])
                      if o.output_type == "stream" and o.get("name") == "stderr")
    assert stderr.strip() == "", f"{name}: предупреждения в stderr:\n{stderr[:500]}"


def test_notebook_contains_plots(notebook):
    _, _, nb = notebook
    images = sum(1 for c in nb.cells if c.cell_type == "code"
                 for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    assert images >= 4, f"ожидалось несколько графиков, найдено {images}"


def test_notebook_has_theory_and_conclusions(notebook):
    name, _, nb = notebook
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Постановка задачи" in text, name
    assert "Выводы" in text, name
    # должны быть формулы (LaTeX) и таблицы
    assert "$" in text
    assert "|" in text


def test_notebook_produces_text_output(notebook):
    _, _, nb = notebook
    stdout = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
                       for o in c.get("outputs", [])
                       if o.output_type == "stream" and o.get("name") == "stdout")
    assert len(stdout) > 500, "ноутбук должен выводить числовые результаты"


def test_no_absolute_paths_in_outputs(notebook):
    """Артефакты не должны содержать абсолютных путей конкретной машины."""
    name, _, nb = notebook
    raw = json.dumps(nb, ensure_ascii=False)
    # ищем домашние каталоги пользователей в любом месте (в т.ч. в тексте вывода)
    leaks = re.findall(r'(?:/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+)', raw)
    assert not leaks, f"{name}: в ноутбуке найдены абсолютные пути: {sorted(set(leaks))[:3]}"


def test_notebooks_use_labkit_only(notebook):
    """Численная логика должна жить в пакете, а не дублироваться в ноутбуке."""
    _, _, nb = notebook
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "import labkit as lk" in src
