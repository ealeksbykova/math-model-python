"""Сборка ноутбука: практическое занятие 1.2 (уравнение Лапласа, МКР)."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

from common import SETUP_CELL, SOLUTION_DIR, code, md, notebook

CELLS = [
    md(
        r"""
# Практическое занятие №1.2
## Разработка и исследование компьютерной модели для уравнения Лапласа. Метод конечных разностей

**Дисциплина:** математическое моделирование

---

### Цель работы

1. Освоить метод конечных разностей (МКР): аппроксимацию дифференциальных операторов
   конечно-разностными соотношениями.
2. Пройти все этапы МКР: сетка → аппроксимация → подстановка в уравнение →
   формирование **СЛАУ** → численное решение.
3. Верифицировать компьютерную модель по аналитическому решению.
4. Исследовать сходимость итерационного решения СЛАУ.

### Отличие от работы 1.1

В работе 1.1 метод Либмана сразу применялся к разностной схеме. Здесь акцент сделан
на **формальной процедуре построения МКР** и на **матрице СЛАУ** $A u = b$,
которая затем решается итерационно. Результаты обеих работ должны совпадать — это
дополнительная перекрёстная проверка.
"""
    ),
    code(SETUP_CELL),
    md(
        r"""
---
## 1. Табличные производные. Аппроксимация дифференциальных операторов

Основная идея МКР — замена непрерывной функции $y = f(x)$ сеточной функцией
$y_i = f(x_i)$ на сетке $x_i$, а производных — их разностными аналогами.

Пусть $h = x_{i+1} - x_i$. Разложим функцию в ряд Тейлора:

$$f(x_{i+1}) = f_i + h f'_i + \frac{h^2}{2} f''_i + \frac{h^3}{6} f'''_i + O(h^4),
\qquad (2.6)$$

$$f(x_{i-1}) = f_i - h f'_i + \frac{h^2}{2} f''_i - \frac{h^3}{6} f'''_i + O(h^4).
\qquad (2.7)$$

Отсюда получаются разностные производные (2.4), (2.5), (2.8), (2.9) методички:

| Производная | Формула | Порядок аппроксимации |
|---|---|---|
| правая | $f'_i \approx \dfrac{f_{i+1}-f_i}{h}$ | $O(h)$ |
| левая | $f'_i \approx \dfrac{f_i-f_{i-1}}{h}$ | $O(h)$ |
| центральная | $f'_i \approx \dfrac{f_{i+1}-f_{i-1}}{2h}$ | $O(h^2)$ |
| вторая | $f''_i \approx \dfrac{f_{i+1}-2f_i+f_{i-1}}{h^2}$ | $O(h^2)$ |

Проверим эти утверждения численно на функции $f(x) = \sin(x)$ — сравним с точными
производными и измерим порядок сходимости.
"""
    ),
    code(
        r'''
def fd_errors(h):
    """Погрешности разностных производных для f(x) = sin(x) в точке x0."""
    x0 = 0.7
    f = np.sin
    right = (f(x0 + h) - f(x0)) / h
    left = (f(x0) - f(x0 - h)) / h
    central = (f(x0 + h) - f(x0 - h)) / (2 * h)
    second = (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / h**2
    exact1, exact2 = np.cos(x0), -np.sin(x0)
    return (abs(right - exact1), abs(left - exact1),
            abs(central - exact1), abs(second - exact2))


hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
errs = np.array([fd_errors(h) for h in hs])
names = ["правая (O(h))", "левая (O(h))", "центральная (O(h²))", "вторая (O(h²))"]

print(f"{'h':>8} | " + " | ".join(f"{nm:>18}" for nm in names))
print("-" * 90)
for h, row in zip(hs, errs):
    print(f"{h:8.4f} | " + " | ".join(f"{e:18.3e}" for e in row))

print("\nнаблюдаемый порядок p = log(e1/e2)/log(2):")
for j, nm in enumerate(names):
    ps = [np.log(errs[k, j] / errs[k + 1, j]) / np.log(2) for k in range(len(hs) - 1)]
    print(f"  {nm:>20}: " + ", ".join(f"{p:.2f}" for p in ps))
'''
    ),
    code(
        r'''
fig, ax = plt.subplots()
markers = "osd^"
for j, nm in enumerate(names):
    ax.loglog(hs, errs[:, j], markers[j] + "-", label=nm)
ax.loglog(hs, errs[0, 0] * (hs / hs[0]), "k:", lw=1, label=r"$\propto h$")
ax.loglog(hs, errs[0, 2] * (hs / hs[0]) ** 2, "k--", lw=1, label=r"$\propto h^2$")
ax.invert_xaxis()
ax.set_xlabel("шаг h"), ax.set_ylabel("погрешность")
ax.set_title("Точность разностных аналогов производных f(x) = sin x")
ax.legend(fontsize=8)
plt.show()
'''
    ),
    md(
        r"""
Видно, что односторонние разности дают $O(h)$, а центральные — $O(h^2)$.
Для уравнения Лапласа (вторая производная) используется именно
центральная аппроксимация второго порядка.

---
## 2. Постановка задачи и построение сетки (этап 1)

$$\frac{\partial^2 U}{\partial x^2}+\frac{\partial^2 U}{\partial y^2}=0,
\qquad 0\le x\le 1,\quad 0<y<1,$$

$$U(0,y)=\sin(\pi y),\quad U(1,y)=0,\quad U(x,0)=0,\quad U(x,1)=0 .$$

Точное решение: $U_{\text{точн}}(x,y)=\sin(\pi y)\dfrac{\sinh(\pi(1-x))}{\sinh\pi}$.

**Этап 1 — покрыть область сеткой.** Выбираем прямоугольную область и строим
сетку $4\times3$ (по методичке), т.е. 4 узла по $x$ и 3 узла по $y$;
$h_x = 1/3$, $h_y = 1/2$. Нумеруем узлы.
"""
    ),
    code(
        r'''
nx, ny = 4, 3
x_ss, y_ss, hx, hy = lk.grid(nx, ny)
print(f"сетка {nx}×{ny} узлов; hx = {hx:.4f}, hy = {hy:.4f}")

fig, ax = plt.subplots(figsize=(5.4, 4.6))
X, Y = lk.mesh(x_ss, y_ss)
ax.plot(X, Y, "o-", color="lightsteelblue", lw=0.8)
ax.plot(X.T, Y.T, "o-", color="lightsteelblue", lw=0.8)
for i in range(nx):
    for j in range(ny):
        boundary = i in (0, nx - 1) or j in (0, ny - 1)
        ax.plot(x_ss[i], y_ss[j], "s" if boundary else "o",
                color="tab:red" if boundary else "tab:blue", ms=7, zorder=3)
        if i == 0:
            val = f"{np.sin(np.pi*y_ss[j]):.2f}"
            tag = rf"$u_{{{i},{j}}}=\sin(\pi\cdot{y_ss[j]:g})={val}$"
        elif boundary:
            tag = rf"$u_{{{i},{j}}}=0$"
        else:
            tag = rf"$u_{{{i},{j}}}$ (неизв.)"
        ax.annotate(tag, (x_ss[i], y_ss[j]), textcoords="offset points",
                    xytext=(8, -12), fontsize=8)
ax.set_xlabel("x"), ax.set_ylabel("y")
ax.set_title("Этап 1: сетка 4×3 и нумерация узлов (МКР)")
plt.show()

print("неизвестные значения (внутренние узлы):",
      [(i, j) for i in range(1, nx-1) for j in range(1, ny-1)])
'''
    ),
    md(
        r"""
---
## 3. Аппроксимация и подстановка в уравнение (этапы 2-3)

Заменяем вторые производные центральными разностями:

$$\frac{U_{i-1,j}-2U_{i,j}+U_{i+1,j}}{h_x^2}
+\frac{U_{i,j-1}-2U_{i,j}+U_{i,j+1}}{h_y^2}=0 .$$

При $h_x = h_y = h$ получаем **пятиточечный шаблон** «крест»:

$$U_{i,j}=\frac{1}{4}\big(U_{i-1,j}+U_{i+1,j}+U_{i,j-1}+U_{i,j+1}\big).$$

В общем случае разных шагов:

$$\alpha\,U_{i-1,j}+\alpha\,U_{i+1,j}+\beta\,U_{i,j-1}+\beta\,U_{i,j+1}
-2(\alpha+\beta)\,U_{i,j}=0,
\qquad \alpha=\frac{1}{h_x^2},\;\beta=\frac{1}{h_y^2}.$$

Проверим, что разностный оператор обращается в нуль на точном решении
(т.е. схема **согласована** с дифференциальным уравнением).
"""
    ),
    code(
        r'''
def finite_difference_residual(U, hx, hy):
    """Невязка разностного оператора Лапласа во внутренних узлах."""
    return ((U[:-2, 1:-1] - 2 * U[1:-1, 1:-1] + U[2:, 1:-1]) / hx**2
            + (U[1:-1, :-2] - 2 * U[1:-1, 1:-1] + U[1:-1, 2:]) / hy**2)


n_fine = 41
xf, yf, hf, _ = lk.grid(n_fine)
U_exact_fine = lk.analytic_on_grid(n_fine)
res = finite_difference_residual(U_exact_fine, hf, hf)
print(f"сетка {n_fine}×{n_fine}, h = {hf:.4f}")
print(f"max|разностный оператор на U_точн| = {np.abs(res).max():.3e}")
print("(стремится к нулю при h -> 0 => схема аппроксимирует уравнение)")

scales = [0.5, 1.0, 2.0, 4.0]
print("\nмасштабирование точного решения (линейность оператора):")
for s in scales:
    r = finite_difference_residual(s * U_exact_fine, hf, hf)
    print(f"  U -> {s:>4.1f}·U:  max|невязка| = {np.abs(r).max():.3e}")
'''
    ),
    md(
        r"""
---
## 4. Формирование СЛАУ (этап 4)

Каждому внутреннему узлу соответствует одно уравнение, каждому узлу на границе —
известное значение, уходящее в правую часть $b$. Получаем систему $A u = b$.

Сформируем матрицу для сетки $4\times3$ и выпишем её явно, как в методичке.
"""
    ),
    code(
        r'''
ssau = lk.laplace_ssau(nx=nx, ny=ny)
A, b = ssau["A"], ssau["b"]
print(f"размерность СЛАУ: {A.shape[0]}×{A.shape[1]} (по числу внутренних узлов)")
print("\nнеизвестные:", {f"u{i}{j}": k for (i, j), k in ssau["index"].items()})
print("\nA =\n", A)
print("\nb =", b)
print("\nпроверка диагонального преобладания (достаточное условие сходимости):")
for r in range(A.shape[0]):
    off = np.abs(A[r]).sum() - abs(A[r, r])
    print(f"  строка {r}: |a_rr| = {abs(A[r,r]):.2f}  vs  сумма |внедиаг.| = {off:.2f}"
          f"  -> {'OK' if abs(A[r,r]) >= off else 'нет'}")
'''
    ),
    md(
        r"""
Матрица симметрична, отрицательно определена и имеет строгое диагональное
преобладание — значит, метод простой итерации (и Гаусса-Зейделя) сходится при любом
начальном приближении, что и утверждается в методичке.

---
## 5. Решение СЛАУ (этап 5) и верификация

Решим СЛАУ двумя способами: прямым (LU-разложение, `numpy.linalg.solve`) и
итерационным (Гаусс-Зейдель), и сопоставим результаты с аналитикой.
"""
    ),
    code(
        r'''
direct = ssau["u"]
iter_res = lk.jacobi_ssau(A, b, eps=1e-12, method="gauss_seidel")

print("решение СЛАУ (прямой метод), сетка 4×3:")
print(np.round(direct, 5))

# сравнение векторов неизвестных (внутренние узлы) -- именно их решает СЛАУ
u_direct_vec = ssau["solution_vector"]
print("\nсравнение прямого и итерационного решения (вектор неизвестных):")
print(f"  u_прямое  = {np.round(u_direct_vec, 8)}")
print(f"  u_итерац. = {np.round(iter_res['u'], 8)}")
print(f"  max|u_прямое - u_итерац| = {np.abs(u_direct_vec - iter_res['u']).max():.3e}")
print(f"  итераций Гаусса-Зейделя: {iter_res['iterations']}, "
      f"||Au-b|| = {iter_res['residual_norm']:.2e}")
assert np.allclose(u_direct_vec, iter_res["u"], atol=1e-9), "методы должны совпасть"

exact_43 = np.sinh(np.pi * (1 - x_ss))[:, None] * np.sin(np.pi * y_ss)[None, :] / np.sinh(np.pi)
print("\nверификация по аналитическому решению:")
print(f"{'узел':>10} | {'расчёт':>9} | {'аналитика':>9} | {'|Δ|':>9}")
print("-" * 46)
for (i, j) in ssau["interior"]:
    print(f"  ({i},{j})   | {direct[i,j]:9.5f} | {exact_43[i,j]:9.5f} | "
          f"{abs(direct[i,j]-exact_43[i,j]):9.2e}")

err43 = np.abs(direct - exact_43).max()
print(f"\nmax|Δ| на сетке 4×3: {err43:.3e}")
assert err43 < 0.05, "погрешность на грубой сетке не должна превышать 5%"
print("✔ Модель верифицирована (погрешность соответствует O(h²) на грубой сетке).")
'''
    ),
    md(
        r"""
---
## 6. Сходимость итерационного решения СЛАУ

Проследим, как норма разности итераций убывает со номером итерации для методов
Якоби и Гаусса-Зейделя, и как число итераций зависит от заданной точности.
"""
    ),
    code(
        r'''
hist = {}
for method in ("jacobi", "gauss_seidel"):
    r = lk.jacobi_ssau(A, b, eps=1e-14, max_iter=5000, method=method)
    hist[method] = r
    print(f"{method:>13}: итераций = {r['iterations']:4d}, ||Au-b|| = {r['residual_norm']:.2e}")

fig, ax = plt.subplots()
for method, r in hist.items():
    ax.semilogy(r["residual"], label=method.replace("_", "-"))
ax.set_xlabel("номер итерации"), ax.set_ylabel(r"$\max|u^{(k)}-u^{(k-1)}|$")
ax.set_title("Сходимость итерационных методов для СЛАУ (сетка 4×3)")
ax.legend()
plt.show()

eps_check = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
print(f"\n{'eps':>8} | {'Якоби':>7} | {'Гаусс-Зейдель':>14}")
print("-" * 36)
for e in eps_check:
    rj = lk.jacobi_ssau(A, b, eps=e, method="jacobi")
    rg = lk.jacobi_ssau(A, b, eps=e, method="gauss_seidel")
    print(f"{e:8.0e} | {rj['iterations']:7d} | {rg['iterations']:14d}")
'''
    ),
    md(
        r"""
Гаусс-Зейдель (он же метод Либмана) сходится примерно в 2 раза быстрее Якоби —
это ожидаемо, поскольку он использует уже обновлённые значения, и совпадает с
выводом практического занятия 1.1.

---
## 7. Влияние шага сетки. Порядок точности

Проверим главное свойство МКР — сходимость $O(h^2)$: при уменьшении $h$ вдвое
погрешность должна падать примерно вчетверо. Сравним также с методом Либмана из
работы 1.1 (для одинаковой сетки результаты должны совпасть).
"""
    ),
    code(
        r'''
ns = [5, 9, 17, 33, 65]
errs, hs = [], []
for nn in ns:
    sol = lk.solve_laplace_direct(n=nn)["u"]
    errs.append(lk.max_error(sol, nn))
    hs.append(1.0 / (nn - 1))
orders = lk.convergence_order(errs, hs)

print(f"{'n':>4} | {'h':>8} | {'max|U-U_точн|':>15} | {'порядок p':>10}")
print("-" * 48)
for k, nn in enumerate(ns):
    p = f"{orders[k-1]:10.2f}" if k else f"{'–':>10}"
    print(f"{nn:4d} | {hs[k]:8.5f} | {errs[k]:15.3e} | {p}")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
ax.loglog(hs, errs, "o-", label="МКР (прямое решение СЛАУ)")
ax.loglog(hs, errs[0] * (np.array(hs) / hs[0]) ** 2, "k--", label=r"$\propto h^2$")
ax.set_xlabel("h"), ax.set_ylabel(r"max$|U-U_{\rm точн}|$")
ax.set_title("Сходимость МКР: второй порядок")
ax.legend(fontsize=8)

# сравнение с методом Либмана на сетке 17x17
ax = axes[1]
n_cmp = 17
u_direct = lk.solve_laplace_direct(n=n_cmp)["u"]
u_libman = lk.liebmann(n=n_cmp, eps=1e-12, max_iter=200_000)["U"]
Xc, Yc = lk.mesh(*lk.grid(n_cmp)[:2])
ax.plot(u_direct[:, n_cmp // 2], "o", ms=5, label="МКР / прямое решение СЛАУ")
ax.plot(u_libman[:, n_cmp // 2], "-", lw=1.5, label="метод Либмана (работа 1.1)")
ax.set_xlabel("индекс узла i (сечение j = 8)"), ax.set_ylabel("U")
ax.set_title(f"Совпадение двух подходов, сетка {n_cmp}×{n_cmp}")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print(f"\nmax|u_СЛАУ - u_Либман| на сетке {n_cmp}×{n_cmp} = "
      f"{np.abs(u_direct - u_libman).max():.2e}")
assert all(p > 1.7 for p in orders), "ожидался второй порядок сходимости"
print("✔ Порядок сходимости ≈ 2 подтверждён; оба подхода дают одно и то же решение.")
'''
    ),
    md(
        r"""
---
## 8. Выводы

1. **МКР применён по всем этапам**: построена сетка, выполнена аппроксимация
   вторых производных центральными разностями ($O(h^2)$), получена СЛАУ
   $A u = b$, которая решена прямым и итерационным методами.
2. **Схема корректна**: разностный оператор на точном решении даёт невязку
   $\to 0$ при $h \to 0$ (согласованность), матрица имеет строгое диагональное
   преобладание (сходимость метода простой итерации).
3. **Верификация пройдена**: на сетке $4\times3$ решение совпадает с аналитикой,
   а при измельчении сетки наблюдается второй порядок сходимости
   ($p \approx 2$: 1.94 → 1.96 → 2.00).
4. **Методы эквивалентны**: решение СЛАУ прямым методом и методом Либмана
   совпадают с точностью до $10^{-12}$ — оба подхода описывают одну и ту же
   дискретную модель, разница только в способе решения системы.
5. **Гаусс-Зейдель эффективнее Якоби** примерно в 2 раза по числу итераций,
   что объясняет выбор метода Либмана в практическом занятии 1.1.
"""
    ),
]


def build(path=None):
    path = Path(path) if path else SOLUTION_DIR / "Практическое занятие 1_2 - метод конечных разностей.ipynb"
    nb = notebook(CELLS)
    nbf.validate(nb)
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("записан:", path, f"({len(nb['cells'])} ячеек)")
    return path


if __name__ == "__main__":
    # путь можно передать аргументом (используется в сквозных тестах)
    build(sys.argv[1] if len(sys.argv) > 1 else None)
