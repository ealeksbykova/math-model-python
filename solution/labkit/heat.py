"""Уравнение параболического типа: явная схема (практическое занятие 2).

Постановка задачи (одномерный стержень длины ``L``, ``0 <= x <= L``,
``0 < t <= t_max``)::

    dU/dt = a * d2U/dx2,      a = lambda / (rho * c)

Условия однозначности:

* геометрические: ``0 <= x <= 1``, ``0 < t <= tau``;
* граничные условия первого рода: ``U(0, t) = mu1(t)``, ``U(1, t) = mu2(t)``;
* граничные условия третьего рода (теплообмен с окружающей средой)::

      -lambda * dU/dx |_(x=0) = alfa1 * (U1 - U_(x=0))
       lambda * dU/dx |_(x=L) = alfa2 * (U2 - U_(x=L))

* начальное условие ``U(x, 0) = U0(x)``.

Явная разностная схема (2.4 методички)::

    u[i, j+1] = u[i, j] + (a * dt / dx^2) * (u[i-1, j] - 2*u[i, j] + u[i+1, j])

Условие устойчивости (2.5)::

    Fo = a * dt / dx^2 <= 0.5
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "stability_number",
    "is_stable",
    "solve_heat_explicit",
    "heat_profile",
    "analytic_steady_heat",
    "analytic_transient_dirichlet",
    "compare_time_steps",
    "steady_error",
]

InitialCondition = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Устойчивость явной схемы
# ---------------------------------------------------------------------------
def stability_number(a: float, dt: float, dx: float) -> float:
    """Показатель устойчивости ``Fo = a*dt/dx^2`` (условие (2.5): ``Fo <= 0.5``)."""
    if dx <= 0.0:
        raise ValueError("шаг dx должен быть положительным")
    return float(a * dt / dx**2)


def is_stable(a: float, dt: float, dx: float) -> bool:
    """Устойчива ли явная схема при данных шагах (``Fo <= 0.5``)."""
    return stability_number(a, dt, dx) <= 0.5 + 1e-12


# ---------------------------------------------------------------------------
# Явная схема
# ---------------------------------------------------------------------------
def _evaluate_initial(u0, x: np.ndarray) -> np.ndarray:
    """Начальное условие ``U(x, 0)`` в виде массива значений на сетке."""
    if callable(u0):
        values = np.asarray(u0(x), dtype=float)
        if values.shape != x.shape:
            raise ValueError("функция u0 должна возвращать массив размера nx")
        return values
    return np.full(x.shape, float(u0), dtype=float)


def solve_heat_explicit(
    nx: int = 11,
    tmax: float = 2.0,
    nt: int = 300,
    a: float = 0.1333,
    length: float = 1.0,
    u0: float | InitialCondition = 40.0,
    T1: float = 150.0,
    T2: float = 40.0,
    alfa1: float | None = None,
    alfa2: float | None = None,
    U1: float = 150.0,
    U2: float = 40.0,
    lam: float = 0.1333,
    snapshot_every: int = 1,
    divergence_limit: float | None = None,
) -> dict:
    """Решение уравнения теплопроводности явной схемой.

    Параметры
    ---------
    nx : число узлов по пространству (шаг ``dx = length / (nx - 1)``);
    tmax, nt : время процесса и число временных шагов (``dt = tmax / nt``);
    a : коэффициент температуропроводности ``a = lambda / (rho * c)``;
    u0 : начальное распределение (число либо функция ``x -> U0``);
    T1, T2 : значения на границах для ГУ **первого** рода;
    alfa1, alfa2 : коэффициенты теплоотдачи для ГУ **третьего** рода; если
        заданы (не ``None``), используются ГУ 3-го рода с температурами
        окружающей среды ``U1`` и ``U2`` (формулы (2.7), (2.8) методички);
    lam : коэффициент теплопроводности (используется только для ГУ 3-го рода);
    snapshot_every : сохранять профили с заданной частотой;
    divergence_limit : если ``max|u|`` превышает это значение (или появляется
        ``nan``), расчёт прерывается и выставляется ``diverged = True`` --
        используется в демонстрации неустойчивой схемы (п. 2.2.4).

    Возвращает ``dict``: ``x``, ``dt``, ``dx``, ``fourier`` (число ``Fo``),
    ``stable``, ``u`` (поле на последнем шаге), ``snapshots``
    (``{номер шага: профиль}``), ``t`` (массив времён), ``final_time``,
    ``diverged``, ``steps_done``, ``max_abs``.
    """
    nx = int(nx)
    if nx < 3:
        raise ValueError("нужно минимум 3 узла по пространству")
    if nt < 1:
        raise ValueError("число временных шагов должно быть >= 1")
    if tmax <= 0.0:
        raise ValueError("время расчёта должно быть положительным")

    x = np.linspace(0.0, float(length), nx)
    dx = float(x[1] - x[0])
    dt = float(tmax) / int(nt)
    fourier = stability_number(a, dt, dx)

    u = _evaluate_initial(u0, x)

    robin = alfa1 is not None and alfa2 is not None
    if not robin:
        u[0] = float(T1)
        u[-1] = float(T2)

    coef = a * dt / dx**2
    # снимок "t = 0" хранит именно начальное условие U(x, 0):
    # граничные условия вступают в силу начиная с первого временного шага.
    snapshots: dict[int, np.ndarray] = {0: np.asarray(_evaluate_initial(u0, x), dtype=float)}
    diverged = False
    steps_done = 0
    for step in range(1, int(nt) + 1):
        prev = u.copy()

        if robin:
            # балансовые соотношения (2.6)-(2.8): поток на границе равен
            # потоку теплопроводности внутрь стержня
            u[0] = (lam / dx * prev[1] + alfa1 * U1) / (lam / dx + alfa1)
            u[-1] = (lam / dx * prev[-2] + alfa2 * U2) / (lam / dx + alfa2)

        # явная схема (2.4) для внутренних узлов
        u[1:-1] = prev[1:-1] + coef * (prev[:-2] - 2.0 * prev[1:-1] + prev[2:])

        if step % max(1, int(snapshot_every)) == 0 or step == int(nt):
            snapshots[step] = u.copy()

        steps_done = step
        if divergence_limit is not None:
            if not np.all(np.isfinite(u)) or float(np.abs(u).max()) > divergence_limit:
                diverged = True
                snapshots[step] = u.copy()
                break

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "t": np.linspace(0.0, float(tmax), int(nt) + 1),
        "fourier": fourier,
        "stable": fourier <= 0.5 + 1e-12,
        "u": u.copy(),
        "snapshots": snapshots,
        "nt": int(nt),
        "tmax": float(tmax),
        "final_time": float(tmax),
        "a": float(a),
        "robin": robin,
        "diverged": diverged,
        "steps_done": int(steps_done),
        "max_abs": float(np.abs(u).max()) if np.all(np.isfinite(u)) else float("inf"),
    }


def heat_profile(result: dict, step: int) -> np.ndarray:
    """Профиль температуры на заданном временном шаге (или на последнем)."""
    snaps = result["snapshots"]
    if step in snaps:
        return snaps[step].copy()
    available = sorted(snaps)
    nearest = min(available, key=lambda k: abs(k - step))
    return snaps[nearest].copy()


# ---------------------------------------------------------------------------
# Аналитические решения
# ---------------------------------------------------------------------------
def analytic_steady_heat(
    T1: float = 150.0,
    T2: float = 40.0,
    alfa1: float = 1.0,
    alfa2: float = 1.0,
    lam: float = 0.1333,
    length: float = 1.0,
    U1: float | None = None,
    U2: float | None = None,
) -> dict:
    """Аналитическое стационарное решение задачи с ГУ третьего рода.

    Плотность теплового потока::

        q = (U1 - U2) / (1/alfa1 + L/lambda + 1/alfa2)

    и температуры на границах ``u(0) = U1 - q/alfa1``, ``u(L) = U2 + q/alfa2``.
    При ``alfa -> inf`` решение совпадает со случаем ГУ первого рода.
    """
    U1 = T1 if U1 is None else U1
    U2 = T2 if U2 is None else U2
    if alfa1 <= 0.0 or alfa2 <= 0.0:
        raise ValueError("коэффициенты теплоотдачи должны быть положительными")
    resistance = 1.0 / alfa1 + length / lam + 1.0 / alfa2
    q = (U1 - U2) / resistance
    u_left = U1 - q / alfa1
    u_right = U2 + q / alfa2
    return {
        "q": q,
        "u_left": float(u_left),
        "u_right": float(u_right),
        "resistance": resistance,
        "linear": lambda x: u_left + (u_right - u_left) * np.asarray(x, float) / length,
    }


def analytic_transient_dirichlet(
    x: np.ndarray,
    t: float,
    a: float = 0.1333,
    length: float = 1.0,
    T1: float = 150.0,
    T2: float = 40.0,
    u0: float = 40.0,
    terms: int = 400,
) -> np.ndarray:
    """Аналитическое нестационарное решение при постоянном начальном условии.

    Для ГУ первого рода ``u(0,t)=T1``, ``u(L,t)=T2`` и ``u(x,0)=u0``::

        u(x,t) = u_ss(x) + sum_k b_k sin(k*pi*x/L) exp(-a (k*pi/L)^2 t)

    где ``u_ss`` -- линейное стационарное распределение, а коэффициенты ``b_k``
    начального отклонения вычисляются численным интегрированием.
    """
    x = np.asarray(x, dtype=float)
    L = float(length)
    uss = T1 + (T2 - T1) * x / L
    v0 = float(u0) - uss

    # коэффициенты Фурье начального отклонения (аналитически для постоянного u0)
    k = np.arange(1, int(terms) + 1)
    # integral_0^L (u0 - uss) sin(k pi x / L) dx,  uss линейна
    # => используем точную формулу: v0(x) = u0 - T1 - (T2-T1) x / L
    c1 = float(u0) - T1
    c2 = -(T2 - T1) / L
    # integral (c1 + c2 x) sin(k pi x / L) dx on [0, L]
    kp = k * np.pi / L
    int_sin = (1.0 - np.cos(k * np.pi)) / kp
    int_xsin = (np.sin(k * np.pi) - k * np.pi * np.cos(k * np.pi)) / kp**2
    b = (2.0 / L) * (c1 * int_sin + c2 * int_xsin)

    decay = np.exp(-a * kp**2 * float(t))
    profile = uss.copy()
    if x.ndim == 1:
        basis = np.sin(np.outer(x, kp))  # (nx, nk)
        profile = uss + basis @ (b * decay)
    else:
        profile = uss + np.sin(kp * x[..., None]) @ (b * decay)
    return profile


def steady_error(result: dict, alfa1: float, alfa2: float, U1: float, U2: float,
                 lam: float = 0.1333) -> dict:
    """Сопоставление расчётных и аналитических стационарных значений на границах."""
    ref = analytic_steady_heat(alfa1=alfa1, alfa2=alfa2, lam=lam, U1=U1, U2=U2)
    u = result["u"]
    return {
        "computed_left": float(u[0]),
        "computed_right": float(u[-1]),
        "analytic_left": ref["u_left"],
        "analytic_right": ref["u_right"],
        "abs_error_left": abs(float(u[0]) - ref["u_left"]),
        "abs_error_right": abs(float(u[-1]) - ref["u_right"]),
    }


# ---------------------------------------------------------------------------
# Исследование зависимости от временного шага (п. 2.2.5)
# ---------------------------------------------------------------------------
def compare_time_steps(
    nodes: Sequence[int] = (300, 600),
    times: Iterable[float] = (0.027, 0.067, 0.133, 0.333),
    tmax: float = 2.0,
    nx: int = 11,
    a: float = 0.1333,
    **kwargs,
) -> dict:
    """Профили температуры в одинаковые моменты времени для разных ``nt``.

    Соответствует п. 2.2.5 методички: при ``m = 300`` узлам 5, 11, 21, 51
    отвечают времена 0.027, 0.067, 0.133, 0.333 с; при ``m = 600`` тем же
    временам отвечают узлы 9, 21, 41, 101.
    """
    times = [float(t) for t in times]
    runs = {}
    for m in nodes:
        m = int(m)
        res = solve_heat_explicit(nx=nx, tmax=tmax, nt=m, a=a, **kwargs)
        fields = {}
        for t in times:
            step = int(round(t / (tmax / m)))
            fields[t] = heat_profile(res, step)
        index_of = {t: int(round(t / (tmax / m))) for t in times}
        runs[m] = {
            "result": res,
            "profiles": fields,
            "steps": index_of,
            "dt": res["dt"],
        }
    return {"runs": runs, "times": times, "tmax": float(tmax), "nx": int(nx)}
