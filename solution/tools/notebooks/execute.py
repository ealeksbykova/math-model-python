"""Исполнение ноутбуков лабораторных работ (встраивание выходных данных).

Запуск::

    python3 solution/tools/notebooks/execute.py            # все ноутбуки
    python3 solution/tools/notebooks/execute.py lab1_1     # выборочно

Скрипт выполняет ячейки «на месте» (in place), так что в файлах .ipynb
сохраняются графики, таблицы и текстовый вывод. Это позволяет проверять
корректность ноутбуков автоматическими тестами без Jupyter-сервера.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = {
    "lab1_1": "Практическое занятие 1_1 - метод Либмана.ipynb",
    "lab1_2": "Практическое занятие 1_2 - метод конечных разностей.ipynb",
    "lab2": "Практическое занятие 2 - явная схема.ipynb",
}

TIMEOUT = 900


def execute(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()

    # защита от «тихой» порчи артефакта: ноутбук обязан иметь выходные данные
    errors = [(c.source[:60], o.get("ename"))
              for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.output_type == "error"]
    if errors:
        raise RuntimeError(f"{path.name}: ошибки при исполнении: {errors}")
    if not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
        raise RuntimeError(f"{path.name}: исполнение не дало выходных данных")

    nbformat.write(nb, str(path))
    n_images = sum(1 for c in nb.cells if c.cell_type == "code"
                   for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    print(f"✔ выполнен: {path.name} (графиков: {n_images})")


def main(argv: list[str]) -> int:
    keys = [a for a in argv[1:] if not a.startswith("-")] or list(NOTEBOOKS)
    unknown = set(keys) - set(NOTEBOOKS)
    if unknown:
        print(f"неизвестные ноутбуки: {sorted(unknown)}", file=sys.stderr)
        return 2
    for key in keys:
        target = Path(__file__).resolve().parents[2] / NOTEBOOKS[key]
        if not target.exists():
            print(f"пропущен (не собран): {target.name}", file=sys.stderr)
            continue
        execute(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
