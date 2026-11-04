"""Уравнение Лапласа: метод Либмана и метод конечных разностей.

Практические занятия 1.1 (метод Либмана, ускоренный метод Либмана) и
1.2 (метод конечных разностей, формирование СЛАУ).

Постановка задачи (единичный квадрат, ``0 <= x <= 1``, ``0 < y < 1``)::

    d2u/dx2 + d2u/dy2 = 0

    u(0, y) = sin(pi*y)
    u(1, y) = 0
    u(x, 0) = 0
    u(x, 1) = 0

Аналитическое решение::

    u(x, y) = sin(pi*y) * sinh(pi*(1 - x)) / sinh(pi)
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "grid",
    "bc_lab1",
    "bc_straight_line",
    "bc_walls",
    "mesh",
    "laplace_analytic",
    "straight_line_analytic",
    "liebmann",
    "liebmann_traces",
    "verification_test",
    "laplace_ssau",
    "solve_laplace_direct",
    "jacobi_ssau",
    "iterations_vs_eps",
    "iterations_vs_omega",
    "best_omega",
    "max_error",
    "convergence_order",
]

BoundaryCondition = Callable[[np.ndarray, np.ndarray, np.ndarray], None]


# ---------------------------------------------------------------------------
# Сетка, граничные условия, аналитическое решение
# ---------------------------------------------------------------------------
def grid(n: int, m: int | None = None):
    """Равномерная сетка ``n`` узлов по ``x`` и ``m`` узлов по ``y`` на [0, 1]^2.

    Возвращает ``(x, y, hx, hy)``; индекс массива решения -- ``U[i, j]``,
    где ``i`` отвечает координате ``x``, ``j`` -- координате ``y``.
    """
    n = int(n)
    m = n if m is None else int(m)
    if n < 3 or m < 3:
        raise ValueError("для двумерной задачи нужно минимум 3x3 узла")
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, m)
    return x, y, float(x[1] - x[0]), float(y[1] - y[0])


def mesh(x: np.ndarray, y: np.ndarray):
    """Двумерные координатные сетки ``X[i, j] = x[i]``, ``Y[i, j] = y[j]``."""
    return np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="ij")


def bc_lab1(U: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Граничные условия исходной задачи практических занятий 1.1 / 1.2."""
    U[0, :] = np.sin(np.pi * y)
    U[-1, :] = 0.0
    U[:, 0] = 0.0
    U[:, -1] = 0.0


def bc_walls(left: float = 100.0, right: float = 40.0) -> BoundaryCondition:
    """Граничные условия тестовой задачи п. 1.2.3.2: ``u(0, y) = left``, ``u(1, y) = right``.

    Возвращает функцию ГУ для :func:`liebmann`. Если дополнительно включить
    ``adiabatic_y=True`` (отсутствие перетекания энергии через горизонтальные
    границы ``y = 0`` и ``y = 1``), решение задачи вырождается в одномерное::

        u(x, y) = left + (right - left) * x .

    Именно эта комбинация (``bc_walls(100, 40)`` + ``adiabatic_y=True``)
    воспроизводит верификационный расчёт Задания №1 из ``Lab1.java``
    (метод ``task1_verification``): сетка 4x4 интервалов, 50 итераций.
    """
    left = float(left)
    right = float(right)

    def bc(U: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        U[0, :] = left
        U[-1, :] = right

    return bc


_BC_STRAIGHT_LINE = bc_walls(1.0, 0.0)


def bc_straight_line(U: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Граничные условия тестовой (одномерной) задачи: u(0, y) = 1, u(1, y) = 0.

    Горизонтальные границы "не пропускают энергию" (параметр ``adiabatic_y``
    в :func:`liebmann`), поэтому решение вырождается в ``u = 1 - x``.
    """
    _BC_STRAIGHT_LINE(U, x, y)


def laplace_analytic(x, y):
    """Точное решение задачи 1.1/1.2: ``sin(pi*y)*sinh(pi*(1-x))/sinh(pi)``.

    Поддерживает как скалярные аргументы, так и сетки одинаковой формы.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    return np.sin(np.pi * y) * np.sinh(np.pi * (1.0 - x)) / np.sinh(np.pi)


def straight_line_analytic(x, left: float = 100.0, right: float = 40.0):
    """Точное решение тестовой задачи п. 1.2.3.2: ``left + (right - left) * x``."""
    return float(left) + (float(right) - float(left)) * np.asarray(x, dtype=float)


def analytic_on_grid(n: int) -> np.ndarray:
    """Точное решение, разложенное на сетку ``n x n`` (индексация ``[i, j]``)."""
    x, y, _, _ = grid(n)
    return np.sinh(np.pi * (1.0 - x))[:, None] * np.sin(np.pi * y)[None, :] / np.sinh(np.pi)


# ---------------------------------------------------------------------------
# Метод Либмана (он же метод Гаусса-Зейделя для пятиточечной аппроксимации)
# ---------------------------------------------------------------------------
def liebmann(
    n: int = 5,
    eps: float | None = 1e-3,
    omega: float = 1.0,
    fixed_iterations: int | None = None,
    max_iter: int = 20000,
    bc: BoundaryCondition | None = None,
    adiabatic_y: bool = False,
    u0: np.ndarray | None = None,
) -> dict:
    """Решение уравнения Лапласа методом Либмана на сетке ``n x n`` узлов.

    Параметры
    ---------
    n : число узлов по каждой координате (шаг ``h = 1 / (n - 1)``);
    eps : заданная точность по норме ``max|u^(k+1) - u^k|``; при ``None``
        итерации ограничиваются ``max_iter`` / ``fixed_iterations``;
    omega : коэффициент релаксации ускоренного метода Либмана
        (``omega = 1`` -- обычный метод Либмана; методичка рекомендует
        ``0 < omega <= 1.5``);
    fixed_iterations : выполнить ровно заданное число итераций
        (Задание 1 практического занятия 1.1);
    adiabatic_y : "отсутствие перетекания энергии" через горизонтальные
        границы (``j = 0`` и ``j = n - 1`` копируют соседний узел) -- адаптация
        модели под тестовую задачу п. 1.2.3.2;
    u0 : начальное приближение.

    Возвращает ``dict``: ``U``, ``x``, ``y``, ``h``, ``iterations``,
    ``history`` (норма разности между итерациями), ``delta``, ``eps``, ``omega``.
    """
    n = int(n)
    if n < 3:
        raise ValueError("нужно минимум 3 узла по каждой координате")
    if not 0.0 < omega <= 1.5:
        raise ValueError("коэффициент релаксации должен лежать в (0; 1.5]")

    x, y, hx, _ = grid(n)
    bc = bc_lab1 if bc is None else bc

    if u0 is None:
        U = np.zeros((n, n), dtype=float)
    else:
        U = np.array(u0, dtype=float, copy=True)
        if U.shape != (n, n):
            raise ValueError("форма начального приближения u0 должна быть (n, n)")
    bc(U, x, y)

    history: list[float] = []
    iterations = 0
    delta = float("inf")
    while True:
        iterations += 1
        prev = U.copy()

        # Одна итерация метода Либмана: обход узлов с использованием уже
        # обновлённых значений (Гаусс-Зейдель) + усреднение с предыдущей
        # итерацией через коэффициент релаксации.
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                value = 0.25 * (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1])
                U[i, j] = omega * value + (1.0 - omega) * U[i, j]

        if adiabatic_y:
            U[1:-1, 0] = U[1:-1, 1]
            U[1:-1, -1] = U[1:-1, -2]
            bc(U, x, y)

        delta = float(np.abs(U - prev).max())
        history.append(delta)

        if fixed_iterations is not None:
            if iterations >= int(fixed_iterations):
                break
        elif eps is not None and delta < eps:
            break
        if iterations >= int(max_iter):
            break

    return {
        "U": U,
        "x": x,
        "y": y,
        "h": hx,
        "iterations": iterations,
        "history": np.asarray(history, dtype=float),
        "delta": delta,
        "eps": eps,
        "omega": omega,
    }


def liebmann_traces(
    n: int = 5,
    iterations: Sequence[int] = (1, 2, 3, 5, 10, 20, 50),
    **kwargs,
) -> dict:
    """Снимки решения после заданного числа итераций (иллюстрация сходимости)."""
    kwargs.pop("eps", None)
    kwargs.pop("fixed_iterations", None)
    return {
        int(k): liebmann(n=n, eps=None, fixed_iterations=int(k), **kwargs)["U"].copy()
        for k in iterations
    }


def verification_test(
    intervals: int = 4,
    iterations: int = 50,
    left: float = 100.0,
    right: float = 40.0,
) -> dict:
    """Верификационный расчёт Задания №1 (порт ``task1_verification`` из ``Lab1.java``).

    Тестовая задача п. 1.2.3.2 методички -- одномерное уравнение Лапласа
    ``d2u/dx2 = 0`` с условиями ``u(0, y) = left``, ``u(1, y) = right`` и
    отсутствием перетекания энергии через горизонтальные границы. Точное
    решение: ``u(x, y) = left + (right - left) * x``.

    Расчёт выполняется обычным методом Либмана (``omega = 1``) ровно
    ``iterations`` итераций на сетке из ``intervals`` интервалов
    (``intervals + 1`` узлов по каждой оси).

    Возвращает ``dict``: ``U`` (индексация ``[i, j]``, ``i`` -- по ``x``),
    ``rows`` (матрица в ориентации ``Lab1.java``: строки -- ``y``,
    столбцы -- ``x``), ``x``, ``y``, ``h``, ``iterations``, ``history``,
    ``analytic`` (точное решение по узлам ``x``), ``max_error`` и
    ``intervals``, ``left``, ``right``.
    """
    res = liebmann(
        n=int(intervals) + 1,
        eps=None,
        fixed_iterations=int(iterations),
        bc=bc_walls(left, right),
        adiabatic_y=True,
    )
    U = res["U"]
    analytic = straight_line_analytic(res["x"], left, right)
    res.update(
        {
            "rows": U.T,
            "analytic": analytic,
            "max_error": float(np.abs(U - analytic[:, None]).max()),
            "intervals": int(intervals),
            "left": float(left),
            "right": float(right),
        }
    )
    return res


# ---------------------------------------------------------------------------
# Метод конечных разностей: формирование и решение СЛАУ (занятие 1.2)
# ---------------------------------------------------------------------------
def laplace_ssau(nx: int = 4, ny: int | None = None, bc: BoundaryCondition | None = None) -> dict:
    """Сформировать СЛАУ ``A u = b`` для пятиточечной аппроксимации Лапласа.

    Для внутреннего узла ``(i, j)`` дискретный аналог::

        (u[i-1,j] - 2u[i,j] + u[i+1,j]) / hx^2
      + (u[i,j-1] - 2u[i,j] + u[i,j+1]) / hy^2 = 0

    Возвращает ``dict``: ``A``, ``b``, ``u`` (решение, разложенное на сетку),
    ``interior``, ``index``, ``x``, ``y``, ``hx``, ``hy``, ``n``, ``n_unknown``.
    """
    ny = nx if ny is None else ny
    x, y, hx, hy = grid(nx, ny)
    bc = bc_lab1 if bc is None else bc

    U = np.zeros((nx, ny), dtype=float)
    bc(U, x, y)

    interior = [(i, j) for i in range(1, nx - 1) for j in range(1, ny - 1)]
    index = {node: k for k, node in enumerate(interior)}
    n_unknown = len(interior)
    A = np.zeros((n_unknown, n_unknown), dtype=float)
    b = np.zeros(n_unknown, dtype=float)
    cx, cy = 1.0 / hx**2, 1.0 / hy**2

    for node in interior:
        i, j = node
        row = index[node]
        A[row, row] = -2.0 * (cx + cy)
        for neighbor, coef in (
            ((i - 1, j), cx),
            ((i + 1, j), cx),
            ((i, j - 1), cy),
            ((i, j + 1), cy),
        ):
            if neighbor in index:
                A[row, index[neighbor]] += coef
            else:
                b[row] -= coef * U[neighbor]

    u = U.copy()
    if n_unknown:
        solution = np.linalg.solve(A, b)
        for node in interior:
            u[node] = solution[index[node]]

    return {
        "A": A,
        "b": b,
        "u": u,
        "interior": interior,
        "index": index,
        "x": x,
        "y": y,
        "hx": hx,
        "hy": hy,
        "n": (nx, ny),
        "n_unknown": n_unknown,
        "solution_vector": np.linalg.solve(A, b) if n_unknown else np.zeros(0),
    }


def solve_laplace_direct(n: int = 9, bc: BoundaryCondition | None = None) -> dict:
    """Решение задачи Лапласа прямым методом (LU-разложение) на сетке ``n x n``."""
    return laplace_ssau(nx=n, ny=n, bc=bc)


def jacobi_ssau(
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-6,
    max_iter: int = 20000,
    method: str = "gauss_seidel",
    omega: float = 1.0,
) -> dict:
    """Итерационное решение СЛАУ ``A u = b`` (метод Якоби или Гаусса-Зейделя).

    Возвращает ``dict``: ``u``, ``iterations``, ``residual`` (история нормы
    разности итераций), ``residual_norm`` (норма ``|A u - b|``).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
        raise ValueError("несогласованные размеры матрицы и правой части")
    if method not in {"jacobi", "gauss_seidel"}:
        raise ValueError("method должен быть 'jacobi' или 'gauss_seidel'")

    n = b.size
    if n == 0:
        return {"u": np.zeros(0), "iterations": 0, "residual": np.zeros(0), "residual_norm": 0.0}

    u = np.zeros(n, dtype=float)
    diag = np.diag(A)
    if np.any(diag == 0.0):
        raise ValueError("на диагонали матрицы есть нули: метод не применим")

    history: list[float] = []
    iterations = 0
    while iterations < int(max_iter):
        iterations += 1
        prev = u.copy()
        if method == "jacobi":
            u = (b - (A @ prev - diag * prev)) / diag
        else:
            for i in range(n):
                known = A[i] @ u - A[i, i] * u[i]
                u[i] = omega * (b[i] - known) / A[i, i] + (1.0 - omega) * u[i]
        delta = float(np.abs(u - prev).max())
        history.append(delta)
        if delta < eps:
            break

    return {
        "u": u,
        "iterations": iterations,
        "residual": np.asarray(history, dtype=float),
        "residual_norm": float(np.linalg.norm(A @ u - b)),
    }


# ---------------------------------------------------------------------------
# Исследование сходимости
# ---------------------------------------------------------------------------
def iterations_vs_eps(
    n: int = 5,
    eps_values: Iterable[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
    omega: float = 1.0,
) -> list[dict]:
    """Зависимость числа итераций от точности (Задание 2, рис. 1.2)."""
    return [
        {"eps": float(eps), "iterations": liebmann(n=n, eps=float(eps), omega=omega)["iterations"]}
        for eps in eps_values
    ]


def iterations_vs_omega(
    n: int = 5,
    eps: float = 1e-7,
    omegas: Iterable[float] | None = None,
) -> list[dict]:
    """Зависимость числа итераций от коэффициента релаксации (рис. 1.1)."""
    if omegas is None:
        omegas = np.round(np.arange(0.9, 1.51, 0.02), 2)
    return [
        {"omega": float(w), "iterations": liebmann(n=n, eps=eps, omega=float(w))["iterations"]}
        for w in omegas
    ]


def best_omega(n: int = 5, eps: float = 1e-7, omegas: Iterable[float] | None = None) -> dict:
    """Оптимальный коэффициент релаксации (минимум числа итераций)."""
    data = iterations_vs_omega(n=n, eps=eps, omegas=omegas)
    # при равенстве итераций предпочитаем omega, близкий к 1.2
    return min(data, key=lambda row: (row["iterations"], abs(row["omega"] - 1.2)))


# ---------------------------------------------------------------------------
# Оценка точности и порядка сходимости
# ---------------------------------------------------------------------------
def max_error(U: np.ndarray, n: int | None = None) -> float:
    """Максимальная погрешность численного решения относительно аналитического."""
    U = np.asarray(U, dtype=float)
    n = U.shape[0] if n is None else int(n)
    return float(np.abs(U - analytic_on_grid(n)).max())


def convergence_order(errors: Sequence[float], hs: Sequence[float]) -> list[float]:
    """Наблюдаемый порядок сходимости ``p = ln(e1/e2) / ln(h1/h2)``."""
    errors = [float(e) for e in errors]
    hs = [float(h) for h in hs]
    if len(errors) != len(hs):
        raise ValueError("длины errors и hs должны совпадать")
    orders = []
    for k in range(len(errors) - 1):
        if errors[k + 1] <= 0.0 or hs[k + 1] <= 0.0 or errors[k] <= 0.0:
            orders.append(float("nan"))
            continue
        orders.append(float(np.log(errors[k] / errors[k + 1]) / np.log(hs[k] / hs[k + 1])))
    return orders
