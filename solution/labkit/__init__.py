"""Численные модели для лабораторных работ по математическому моделированию.

Пакет содержит реализации, используемые в ноутбуках:

* :mod:`labkit.laplace` -- уравнение Лапласа (метод Либмана и метод конечных
  разностей, практические занятия 1.1 и 1.2);
* :mod:`labkit.heat` -- уравнение параболического типа, явная схема
  (практическое занятие 2).

Все функции возвращают ``dict`` с результатами расчёта, поэтому ноутбуки
остаются "тонкими" (теория + вызовы + таблицы + графики), а численная часть
покрыта модульными тестами (``tests/test_labkit.py``).
"""

from .laplace import (
    analytic_on_grid,
    bc_lab1,
    bc_straight_line,
    bc_walls,
    best_omega,
    convergence_order,
    grid,
    iterations_vs_eps,
    iterations_vs_omega,
    jacobi_ssau,
    laplace_analytic,
    laplace_ssau,
    liebmann,
    liebmann_traces,
    max_error,
    mesh,
    solve_laplace_direct,
    straight_line_analytic,
    verification_test,
)
from .heat import (
    analytic_steady_heat,
    analytic_transient_dirichlet,
    compare_time_steps,
    heat_profile,
    is_stable,
    solve_heat_explicit,
    stability_number,
    steady_error,
)

__all__ = [
    "analytic_on_grid",
    "bc_lab1",
    "bc_straight_line",
    "bc_walls",
    "best_omega",
    "convergence_order",
    "grid",
    "iterations_vs_eps",
    "iterations_vs_omega",
    "jacobi_ssau",
    "laplace_analytic",
    "laplace_ssau",
    "liebmann",
    "liebmann_traces",
    "max_error",
    "mesh",
    "solve_laplace_direct",
    "straight_line_analytic",
    "verification_test",
    "analytic_steady_heat",
    "analytic_transient_dirichlet",
    "compare_time_steps",
    "heat_profile",
    "is_stable",
    "solve_heat_explicit",
    "stability_number",
    "steady_error",
]

__version__ = "1.0.0"
