"""Тесты модели параболического типа (явная схема, практическое занятие 2)."""

from __future__ import annotations

import numpy as np
import pytest

import labkit as lk

A_PHYS = 0.1333
LAM_PHYS = 0.1333


# ---------------------------------------------------------------------------
# Показатель устойчивости
# ---------------------------------------------------------------------------
class TestStabilityNumber:
    def test_value_for_manual_parameters(self):
        # Δx = 0.1, Δt = 2/300, a = 0.1333
        fo = lk.stability_number(A_PHYS, 2.0 / 300, 0.1)
        assert fo == pytest.approx(0.0889, abs=1e-4)

    def test_zero_dx_rejected(self):
        with pytest.raises(ValueError):
            lk.stability_number(A_PHYS, 0.01, 0.0)

    def test_is_stable_threshold(self):
        assert lk.is_stable(A_PHYS, 2.0 / 300, 0.1)
        assert not lk.is_stable(10 * A_PHYS, 2.0 / 300, 0.1)

    def test_exact_half_borderline(self):
        # Fo = 0.5 ровно -> устойчиво (нестрогое неравенство)
        a, dt, dx = 0.5, 1.0, 1.0
        assert lk.stability_number(a, dt, dx) == pytest.approx(0.5)
        assert lk.is_stable(a, dt, dx)


# ---------------------------------------------------------------------------
# Явная схема: базовая работоспособность
# ---------------------------------------------------------------------------
class TestSolveHeatExplicit:
    def test_basic_run(self):
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=300, a=A_PHYS)
        assert res["stable"] is True
        assert res["u"].shape == (11,)
        assert not res["diverged"]
        assert res["steps_done"] == 300

    def test_dirichlet_boundaries_fixed(self):
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=300, a=A_PHYS,
                                     T1=150.0, T2=40.0)
        assert res["u"][0] == pytest.approx(150.0)
        assert res["u"][-1] == pytest.approx(40.0)

    def test_initial_condition_respected(self):
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=300, a=A_PHYS, u0=40.0)
        # снимок t = 0 -- это ровно начальное условие U(x, 0)
        assert np.allclose(res["snapshots"][0], 40.0)
        # а на первом шаге уже действуют граничные условия
        assert res["snapshots"][1][0] == pytest.approx(150.0)

    def test_initial_condition_callable(self):
        res = lk.solve_heat_explicit(nx=11, tmax=0.1, nt=5, a=A_PHYS,
                                     u0=lambda x: 40.0 + 100.0 * x)
        assert res["snapshots"][0][0] == pytest.approx(40.0)
        assert res["snapshots"][0][-1] == pytest.approx(140.0)

    def test_input_validation(self):
        with pytest.raises(ValueError):
            lk.solve_heat_explicit(nx=2)
        with pytest.raises(ValueError):
            lk.solve_heat_explicit(nt=0)
        with pytest.raises(ValueError):
            lk.solve_heat_explicit(tmax=0.0)

    def test_bad_callable_shape_rejected(self):
        with pytest.raises(ValueError):
            lk.solve_heat_explicit(nx=11, u0=lambda x: np.zeros(3))

    def test_boundary_conditions_apply_from_first_step(self):
        """U(x,0) задаётся начальным условием, ГУ -- начиная с t = dt."""
        res = lk.solve_heat_explicit(nx=11, tmax=1.0, nt=150, a=A_PHYS,
                                     T1=150.0, T2=40.0, u0=40.0)
        assert np.allclose(res["snapshots"][0], 40.0)
        assert res["snapshots"][1][0] == pytest.approx(150.0)
        assert res["snapshots"][1][-1] == pytest.approx(40.0)

    def test_steady_state_is_linear_first_kind(self):
        """При ГУ 1-го рода стационар линеен: U = 150 - 110x."""
        res = lk.solve_heat_explicit(nx=11, tmax=60.0, nt=9000, a=A_PHYS,
                                     T1=150.0, T2=40.0, u0=40.0)
        expected = 150.0 - 110.0 * res["x"]
        assert np.allclose(res["u"], expected, atol=0.2)


# ---------------------------------------------------------------------------
# Устойчивость / неустойчивость
# ---------------------------------------------------------------------------
class TestStabilityBehaviour:
    def test_stable_case_stays_bounded(self):
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=300, a=A_PHYS)
        assert np.abs(res["u"]).max() < 200.0

    def test_lambda_times_ten_diverges(self):
        """п. 2.2.4: увеличение λ в 10 раз делает явную схему неустойчивой."""
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=300, a=10 * A_PHYS,
                                     divergence_limit=1e4)
        assert res["stable"] is False
        assert res["diverged"] is True
        assert res["steps_done"] < 300

    def test_unstable_without_limit_produces_huge_values(self):
        with np.errstate(over="ignore", invalid="ignore"):
            res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=300, a=10 * A_PHYS)
        assert not res["stable"]
        assert not np.all(np.isfinite(res["u"])) or np.abs(res["u"]).max() > 1e6

    @pytest.mark.parametrize("nt,expected_stable", [
        (50, False),    # Fo = 0.533
        (100, True),    # Fo = 0.267
        (300, True),    # Fo = 0.089
        (600, True),    # Fo = 0.044
    ])
    def test_stability_threshold(self, nt, expected_stable):
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=nt, a=A_PHYS)
        assert res["stable"] is expected_stable

    def test_fo_just_above_half_is_unstable(self):
        dx = 0.1
        dt = 0.5 * dx**2 / A_PHYS * 1.05   # Fo = 0.525
        res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=int(2.0 / dt), a=A_PHYS)
        assert res["fourier"] > 0.5
        assert res["stable"] is False


# ---------------------------------------------------------------------------
# Аналитическое стационарное решение
# ---------------------------------------------------------------------------
class TestAnalyticSteady:
    @pytest.mark.parametrize("alfa1,alfa2,U1,U2,left,right", [
        (1.0, 1.0, 150.0, 40.0, 138.423, 51.577),   # вариант A методички
        (0.1, 1.0, 150.0, 40.0, 90.547, 45.945),    # вариант B
        (1.0, 0.1, 150.0, 40.0, 144.055, 99.453),   # вариант C
        (0.1, 0.1, 150.0, 40.0, 110.003, 79.997),   # вариант D
    ])
    def test_manual_values(self, alfa1, alfa2, U1, U2, left, right):
        ref = lk.analytic_steady_heat(alfa1=alfa1, alfa2=alfa2, U1=U1, U2=U2,
                                      lam=LAM_PHYS)
        assert ref["u_left"] == pytest.approx(left, abs=2e-3)
        assert ref["u_right"] == pytest.approx(right, abs=2e-3)

    def test_infinite_alpha_approaches_dirichlet(self):
        ref = lk.analytic_steady_heat(alfa1=1e6, alfa2=1e6, U1=150.0, U2=40.0,
                                      lam=LAM_PHYS)
        assert ref["u_left"] == pytest.approx(150.0, abs=1e-3)
        assert ref["u_right"] == pytest.approx(40.0, abs=1e-3)

    def test_heat_flux_balance(self):
        """Плотность потока одинакова на обеих границах в стационаре."""
        ref = lk.analytic_steady_heat(alfa1=2.0, alfa2=0.5, U1=200.0, U2=20.0,
                                      lam=LAM_PHYS)
        # закон Фурье: q = -lambda * dU/dx = lambda * (U(0) - U(L)) / L
        q_conduction = LAM_PHYS * (ref["u_left"] - ref["u_right"]) / 1.0
        assert ref["q"] == pytest.approx(q_conduction, rel=1e-9)
        # и совпадает с потоком теплоотдачи на каждой границе
        assert ref["q"] == pytest.approx(2.0 * (200.0 - ref["u_left"]), rel=1e-9)
        assert ref["q"] == pytest.approx(0.5 * (ref["u_right"] - 20.0), rel=1e-9)

    def test_zero_alpha_rejected(self):
        with pytest.raises(ValueError):
            lk.analytic_steady_heat(alfa1=0.0, alfa2=1.0)

    def test_linear_profile_endpoints(self):
        ref = lk.analytic_steady_heat(alfa1=1.0, alfa2=1.0, U1=150.0, U2=40.0)
        assert ref["linear"](0.0) == pytest.approx(ref["u_left"])
        assert ref["linear"](1.0) == pytest.approx(ref["u_right"])


# ---------------------------------------------------------------------------
# ГУ 3-го рода: численное решение против аналитики
# ---------------------------------------------------------------------------
class TestRobinBoundaryConditions:
    @pytest.mark.parametrize("alfa1,alfa2,U1,U2", [
        (1.0, 1.0, 150.0, 40.0),
        (0.1, 1.0, 150.0, 40.0),
        (1.0, 0.1, 150.0, 40.0),
        (0.1, 0.1, 150.0, 40.0),
    ])
    def test_steady_state_matches_analytic(self, alfa1, alfa2, U1, U2):
        """Длительный расчёт должен выйти на аналитические стационарные значения."""
        res = lk.solve_heat_explicit(
            nx=11, tmax=200.0, nt=int(round(200.0 / 0.0066667)), a=A_PHYS,
            u0=40.0, alfa1=alfa1, alfa2=alfa2, U1=U1, U2=U2,
        )
        err = lk.steady_error(res, alfa1, alfa2, U1, U2, lam=LAM_PHYS)
        assert err["abs_error_left"] < 1e-2
        assert err["abs_error_right"] < 1e-2

    def test_robin_flag_set(self):
        res = lk.solve_heat_explicit(nx=11, tmax=1.0, nt=150, a=A_PHYS,
                                     alfa1=1.0, alfa2=1.0, U1=150.0, U2=40.0)
        assert res["robin"] is True

    def test_boundary_not_clamped_to_ambient(self):
        """При конечном alfa граница не равна температуре среды."""
        res = lk.solve_heat_explicit(nx=11, tmax=200.0, nt=30000, a=A_PHYS,
                                     alfa1=1.0, alfa2=1.0, U1=150.0, U2=40.0, u0=40.0)
        assert res["u"][0] < 150.0 - 1.0
        assert res["u"][-1] > 40.0 + 1.0

    def test_alpha_increase_moves_boundary_to_ambient(self):
        low = lk.solve_heat_explicit(nx=11, tmax=200.0, nt=30000, a=A_PHYS, u0=40.0,
                                     alfa1=0.1, alfa2=1.0, U1=150.0, U2=40.0)
        high = lk.solve_heat_explicit(nx=11, tmax=200.0, nt=30000, a=A_PHYS, u0=40.0,
                                      alfa1=10.0, alfa2=1.0, U1=150.0, U2=40.0)
        assert high["u"][0] > low["u"][0]


# ---------------------------------------------------------------------------
# Аналитическое нестационарное решение
# ---------------------------------------------------------------------------
class TestAnalyticTransient:
    def test_matches_steady_state_at_large_time(self):
        x = np.linspace(0, 1, 11)
        t = 500.0
        an = lk.analytic_transient_dirichlet(x, t, a=A_PHYS)
        assert np.allclose(an, 150.0 - 110.0 * x, atol=1e-6)

    def test_boundary_values_for_all_time(self):
        x = np.linspace(0, 1, 11)
        for t in (0.0, 0.05, 1.0, 10.0):
            an = lk.analytic_transient_dirichlet(x, t, a=A_PHYS)
            assert an[0] == pytest.approx(150.0, abs=1e-6)
            assert an[-1] == pytest.approx(40.0, abs=1e-6)

    def test_initial_condition(self):
        x = np.linspace(0, 1, 11)
        an = lk.analytic_transient_dirichlet(x, 0.0, a=A_PHYS, terms=5000)
        # в точках, далёких от границ, значение близко к u0 = 40
        assert an[5] == pytest.approx(40.0, abs=0.5)

    def test_second_order_spatial_convergence(self):
        """Погрешность явной схемы должна убывать как O(dx²)."""
        t_end = 0.5
        errors, dxs = [], []
        for nx in (11, 21, 41, 81):
            dx = 1.0 / (nx - 1)
            dt = dx**2 / (4 * A_PHYS)          # Fo = 0.25
            nt = max(1, int(round(t_end / dt)))
            res = lk.solve_heat_explicit(nx=nx, tmax=t_end, nt=nt, a=A_PHYS,
                                         T1=150.0, T2=40.0, u0=40.0)
            an = lk.analytic_transient_dirichlet(res["x"], t_end, a=A_PHYS)
            errors.append(float(np.abs(res["u"] - an).max()))
            dxs.append(res["dx"])
        orders = lk.convergence_order(errors, dxs)
        assert all(p > 1.6 for p in orders), orders


# ---------------------------------------------------------------------------
# Сравнение временных шагов (п. 2.2.5) и вспомогательные функции
# ---------------------------------------------------------------------------
class TestCompareTimeSteps:
    def test_structure_and_dt(self):
        cmp = lk.compare_time_steps(nodes=(300, 600), tmax=2.0, nx=11, a=A_PHYS)
        assert set(cmp["runs"]) == {300, 600}
        assert cmp["runs"][300]["dt"] == pytest.approx(2.0 / 300)
        assert cmp["runs"][600]["dt"] == pytest.approx(2.0 / 600)

    def test_first_order_convergence_in_dt(self):
        """Явная схема имеет первый порядок по времени: при halving dt разница ~в 2 раза.

        Проверяется в «гладкие» моменты времени (t = 0.333 и 1.0 с). В самый
        ранний момент (t = 0.027 с) начальное условие разрывно, и там гладкой
        сходимости нет — это отдельный, физически обусловленный случай.
        """
        for t_end in (0.333, 1.0):
            profiles = []
            for nt in (300, 600, 1200):
                res = lk.solve_heat_explicit(nx=11, tmax=2.0, nt=nt, a=A_PHYS,
                                             T1=150.0, T2=40.0, u0=40.0)
                step = int(round(t_end / res["dt"]))
                profiles.append(res["snapshots"][step])
            d1 = np.abs(profiles[0] - profiles[1]).max()
            d2 = np.abs(profiles[1] - profiles[2]).max()
            # уменьшение dt вдвое уменьшает расхождение примерно в 2 раза
            assert d1 / d2 == pytest.approx(2.0, rel=0.35), (t_end, d1, d2)
            assert d2 < d1

    def test_profiles_from_compare_time_steps_close_late_in_time(self):
        """Поздние моменты времени практически не зависят от dt."""
        cmp = lk.compare_time_steps(nodes=(300, 1200), times=(0.333, 1.0),
                                    tmax=2.0, nx=11, a=A_PHYS)
        for t in (0.333, 1.0):
            diff = np.abs(cmp["runs"][300]["profiles"][t]
                          - cmp["runs"][1200]["profiles"][t]).max()
            assert diff < 0.25, f"t={t}: расхождение {diff:.3f} °C"

    def test_heat_profile_returns_copy(self):
        res = lk.solve_heat_explicit(nx=11, tmax=1.0, nt=150, a=A_PHYS)
        profile = lk.heat_profile(res, 10)
        profile[0] = -999.0
        assert res["snapshots"][10][0] != -999.0

    def test_heat_profile_nearest_step(self):
        res = lk.solve_heat_explicit(nx=11, tmax=1.0, nt=10, a=A_PHYS)
        profile = lk.heat_profile(res, 7)
        assert np.allclose(profile, res["snapshots"][7])
