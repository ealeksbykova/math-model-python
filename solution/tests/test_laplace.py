"""Тесты численных моделей (пакет labkit).

Проверяются:
* аналитика уравнения Лапласа (символьно и численно);
* метод Либмана: эталонные числа методички, сходимость, порядок аппроксимации;
* ускоренный метод Либмана: влияние коэффициента релаксации;
* метод конечных разностей: корректность СЛАУ, диагональное преобладание;
* явная схема теплопроводности: устойчивость, стационарные значения,
  аналитическое нестационарное решение.
"""

from __future__ import annotations

import numpy as np
import pytest

import labkit as lk


# ---------------------------------------------------------------------------
# Аналитическое решение уравнения Лапласа
# ---------------------------------------------------------------------------
class TestLaplaceAnalytic:
    def test_satisfies_laplace_equation_symbolically(self):
        sympy = pytest.importorskip("sympy")
        x, y = sympy.symbols("x y", real=True)
        u = sympy.sin(sympy.pi * y) * sympy.sinh(sympy.pi * (1 - x)) / sympy.sinh(sympy.pi)
        assert sympy.simplify(sympy.diff(u, x, 2) + sympy.diff(u, y, 2)) == 0

    def test_boundary_conditions(self):
        y = np.linspace(0.0, 1.0, 21)
        assert np.allclose(lk.laplace_analytic(0.0, y), np.sin(np.pi * y))
        assert np.allclose(lk.laplace_analytic(1.0, y), 0.0, atol=1e-16)
        x = np.linspace(0.0, 1.0, 21)
        assert np.allclose(lk.laplace_analytic(x, 0.0), 0.0, atol=1e-16)
        assert np.allclose(lk.laplace_analytic(x, 1.0), 0.0, atol=1e-15)

    def test_value_at_center(self):
        # эталон, который выводится в ноутбуке 1.1
        expected = np.sinh(np.pi / 2) / np.sinh(np.pi)
        assert lk.laplace_analytic(0.5, 0.5) == pytest.approx(expected)
        assert lk.laplace_analytic(0.5, 0.5) == pytest.approx(0.199268, abs=1e-6)

    def test_analytic_on_grid_matches_broadcast(self):
        n = 9
        grid_vals = lk.analytic_on_grid(n)
        x, y, _, _ = lk.grid(n)
        assert grid_vals.shape == (n, n)
        assert np.allclose(grid_vals, lk.laplace_analytic(x[:, None], y[None, :]))


# ---------------------------------------------------------------------------
# Сетка и граничные условия
# ---------------------------------------------------------------------------
class TestGridAndBC:
    def test_grid_steps(self):
        x, y, hx, hy = lk.grid(5)
        assert hx == pytest.approx(0.25) and hy == pytest.approx(0.25)
        assert x[0] == 0.0 and x[-1] == 1.0

    def test_grid_rejects_too_small(self):
        with pytest.raises(ValueError):
            lk.grid(2)

    def test_non_square_grid(self):
        x, y, hx, hy = lk.grid(4, 3)
        assert hx == pytest.approx(1 / 3) and hy == pytest.approx(1 / 2)

    def test_bc_lab1(self):
        _, y, _, _ = lk.grid(5)
        U = np.zeros((5, 5))
        lk.bc_lab1(U, np.linspace(0, 1, 5), y)
        assert np.allclose(U[0, :], np.sin(np.pi * y))
        assert np.allclose(U[-1, :], 0.0)


# ---------------------------------------------------------------------------
# Метод Либмана
# ---------------------------------------------------------------------------
class TestLiebmann:
    def test_manual_reference_iteration_counts(self):
        """Эталонные значения методички: eps=1e-3 -> 9, eps=1e-4 -> 12."""
        assert lk.liebmann(n=5, eps=1e-3)["iterations"] == 9
        assert lk.liebmann(n=5, eps=1e-4)["iterations"] == 12

    def test_converges_to_analytic_solution(self):
        res = lk.liebmann(n=33, eps=1e-10, max_iter=100_000)
        assert lk.max_error(res["U"], 33) < 1e-3

    def test_boundary_conditions_preserved(self):
        res = lk.liebmann(n=9, eps=1e-8)
        U = res["U"]
        assert np.allclose(U[0, :], np.sin(np.pi * res["y"]))
        assert np.allclose(U[-1, :], 0.0)
        assert np.allclose(U[:, 0], 0.0)
        assert np.allclose(U[:, -1], 0.0)

    def test_fixed_iterations_mode(self):
        res = lk.liebmann(n=5, eps=None, fixed_iterations=7)
        assert res["iterations"] == 7
        assert len(res["history"]) == 7

    def test_history_is_decreasing(self):
        res = lk.liebmann(n=9, eps=1e-8)
        history = res["history"]
        # норма разности итераций должна в целом убывать
        assert history[-1] < history[0]
        assert np.all(np.diff(history) < 1e-12) or history[-1] < history[0] * 1e-3

    def test_second_order_convergence(self):
        ns = [9, 17, 33]
        errors, hs = [], []
        for n in ns:
            res = lk.liebmann(n=n, eps=1e-12, max_iter=200_000)
            errors.append(lk.max_error(res["U"], n))
            hs.append(res["h"])
        orders = lk.convergence_order(errors, hs)
        assert all(p > 1.8 for p in orders), orders

    def test_rejects_bad_omega(self):
        with pytest.raises(ValueError):
            lk.liebmann(n=5, omega=0.0)
        with pytest.raises(ValueError):
            lk.liebmann(n=5, omega=1.6)

    def test_rejects_bad_u0_shape(self):
        with pytest.raises(ValueError):
            lk.liebmann(n=5, u0=np.zeros((3, 3)))

    def test_verification_straight_line_case(self):
        """Тестовая задача п. 1.2.3.2: решение вырождается в u = 1 - x."""
        res = lk.liebmann(n=9, eps=1e-10, bc=lk.bc_straight_line,
                          adiabatic_y=True, max_iter=50_000)
        expected = 1.0 - res["x"]
        assert np.allclose(res["U"], expected[:, None], atol=1e-8)

    def test_bc_walls_zero_right_equals_straight_line(self):
        """``bc_walls(1, 0)`` -- та же тестовая задача, что и ``bc_straight_line``."""
        a = lk.liebmann(n=9, eps=1e-10, bc=lk.bc_walls(1.0, 0.0),
                        adiabatic_y=True, max_iter=50_000)
        b = lk.liebmann(n=9, eps=1e-10, bc=lk.bc_straight_line,
                        adiabatic_y=True, max_iter=50_000)
        assert np.allclose(a["U"], b["U"])


# ---------------------------------------------------------------------------
# Задание №1 практического занятия 1.1: верификационный расчёт
# (порт task1_verification() из Lab1.java)
# ---------------------------------------------------------------------------
class TestVerificationTask1:
    def test_matches_java_reference_matrix(self):
        """Эталон Lab1.java: 100 / 85 / 70 / 55 / 40 во всех строках."""
        res = lk.verification_test()
        expected = np.array([100.0, 85.0, 70.0, 55.0, 40.0])
        assert res["rows"].shape == (5, 5)
        assert np.allclose(res["rows"], np.broadcast_to(expected, (5, 5)), atol=1e-3)
        # ровно 50 итераций, сетка 4x4 интервалов (5x5 узлов)
        assert res["iterations"] == 50
        assert res["intervals"] == 4
        assert res["U"].shape == (5, 5)
        assert res["h"] == pytest.approx(0.25)

    def test_rows_orientation_matches_java(self):
        """В ``rows`` строки -- это y, столбцы -- x (как u[i][j] в Lab1.java)."""
        res = lk.verification_test()
        # все строки одинаковы: решение зависит только от x
        assert np.allclose(res["rows"], res["rows"][[0], :])
        # вдоль каждой строки значения совпадают с точным решением 100 - 60x
        assert np.allclose(res["rows"][0], lk.straight_line_analytic(res["x"]), atol=1e-3)

    def test_agrees_with_analytic_solution(self):
        """Верификация: совпадение с точным решением u = 100 - 60x."""
        res = lk.verification_test()
        assert res["max_error"] < 1e-3
        assert np.allclose(res["U"], res["analytic"][:, None], atol=1e-3)

    def test_no_energy_flow_through_horizontal_walls(self):
        """Отсутствие перетекания энергии: решение не должно зависеть от y."""
        res = lk.verification_test()
        assert float(np.abs(np.diff(res["U"], axis=1)).max()) < 1e-4

    def test_converges_with_iterations(self):
        """Ошибка убывает с ростом числа итераций."""
        errors = [lk.verification_test(iterations=k)["max_error"] for k in (2, 10, 50)]
        assert errors[0] > errors[1] > errors[2]

    def test_custom_wall_values(self):
        """Обобщение на произвольные значения на стенках."""
        res = lk.verification_test(left=10.0, right=0.0, iterations=60)
        assert np.allclose(res["U"], (10.0 - 10.0 * res["x"])[:, None], atol=1e-3)


class TestAcceleratedLiebmann:
    def test_optimal_omega_reduces_iterations(self):
        plain = lk.liebmann(n=5, eps=1e-7, omega=1.0)["iterations"]
        best = lk.best_omega(n=5, eps=1e-7, omegas=np.round(np.arange(1.0, 1.41, 0.02), 2))
        assert best["iterations"] < plain
        assert 1.1 <= best["omega"] <= 1.3

    def test_manual_optimum_around_1_2(self):
        """Методичка: минимум при ω ≈ 1.20 (11 итераций для eps = 1e-7)."""
        res = lk.liebmann(n=5, eps=1e-7, omega=1.2)
        assert res["iterations"] <= 12

    def test_omega_equal_one_is_plain_method(self):
        a = lk.liebmann(n=5, eps=1e-6, omega=1.0)
        b = lk.liebmann(n=5, eps=1e-6)
        assert a["iterations"] == b["iterations"]
        assert np.allclose(a["U"], b["U"])

    def test_lower_relaxation_slower(self):
        slow = lk.liebmann(n=5, eps=1e-6, omega=0.5)["iterations"]
        plain = lk.liebmann(n=5, eps=1e-6, omega=1.0)["iterations"]
        assert slow > plain

    def test_iterations_vs_eps_monotone(self):
        table = lk.iterations_vs_eps(n=5, eps_values=(1e-2, 1e-3, 1e-4, 1e-5))
        counts = [row["iterations"] for row in table]
        assert counts == sorted(counts)
        assert len(set(counts)) > 1

    def test_liebmann_traces(self):
        traces = lk.liebmann_traces(n=5, iterations=(1, 5, 20))
        assert set(traces) == {1, 5, 20}
        # решение монотонно "заполняется": норма растёт от итерации к итерации
        assert np.abs(traces[1]).max() <= np.abs(traces[20]).max()


# ---------------------------------------------------------------------------
# Метод конечных разностей: СЛАУ
# ---------------------------------------------------------------------------
class TestSSAU:
    def test_ssau_dimensions(self):
        res = lk.laplace_ssau(nx=4, ny=3)
        assert res["n_unknown"] == (4 - 2) * (3 - 2) == 2
        assert res["A"].shape == (2, 2)
        assert res["b"].shape == (2,)

    def test_ssau_matrix_is_symmetric(self):
        res = lk.laplace_ssau(nx=7, ny=7)
        assert np.allclose(res["A"], res["A"].T)

    def test_ssau_diagonal_dominance(self):
        res = lk.laplace_ssau(nx=9, ny=9)
        A = res["A"]
        for row in range(A.shape[0]):
            off = np.abs(A[row]).sum() - abs(A[row, row])
            assert abs(A[row, row]) >= off - 1e-12

    def test_ssau_solution_matches_analytic(self):
        res = lk.laplace_ssau(nx=33, ny=33)
        assert lk.max_error(res["u"], 33) < 5e-4

    def test_ssau_matches_liebmann(self):
        n = 17
        direct = lk.solve_laplace_direct(n=n)["u"]
        libman = lk.liebmann(n=n, eps=1e-12, max_iter=200_000)["U"]
        assert np.abs(direct - libman).max() < 1e-8

    def test_ssau_unknown_nodes_are_interior(self):
        res = lk.laplace_ssau(nx=4, ny=3)
        assert res["interior"] == [(1, 1), (2, 1)]


class TestJacobiSSAU:
    def test_matches_direct_solution(self):
        res = lk.laplace_ssau(nx=9, ny=9)
        it = lk.jacobi_ssau(res["A"], res["b"], eps=1e-12)
        assert np.allclose(it["u"], res["solution_vector"], atol=1e-8)
        assert it["residual_norm"] < 1e-8

    def test_gauss_seidel_faster_than_jacobi(self):
        res = lk.laplace_ssau(nx=9, ny=9)
        jac = lk.jacobi_ssau(res["A"], res["b"], eps=1e-10, method="jacobi")
        gs = lk.jacobi_ssau(res["A"], res["b"], eps=1e-10, method="gauss_seidel")
        assert gs["iterations"] < jac["iterations"]

    def test_rejects_unknown_method(self):
        A = np.array([[2.0, 0.0], [0.0, 2.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            lk.jacobi_ssau(A, b, method="sor")


# ---------------------------------------------------------------------------
# Ошибки и порядок сходимости
# ---------------------------------------------------------------------------
class TestErrorHelpers:
    def test_max_error_zero_for_analytic(self):
        assert lk.max_error(lk.analytic_on_grid(9), 9) == pytest.approx(0.0, abs=1e-15)

    def test_convergence_order_quadratic(self):
        hs = [0.2, 0.1, 0.05]
        errors = [h**2 for h in hs]
        orders = lk.convergence_order(errors, hs)
        assert all(p == pytest.approx(2.0) for p in orders)

    def test_convergence_order_length_mismatch(self):
        with pytest.raises(ValueError):
            lk.convergence_order([1.0, 2.0], [0.1])
