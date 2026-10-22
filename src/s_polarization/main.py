import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import fft
from scipy import linalg
import matplotlib.colors as colors
import cmocean
import warnings


DEG = np.pi / 180.0

# =============== PARAMETERS ===============
# структура
d = 100.0                 # период решетки, нм
t = 90.0                  # высота штрихов, нм
gamma = 0.4               # Γ = d_a / d — доля периода
chi = -2 * 1.703e-3 + 2j * 3.99e-5    # χ материала штрихов
chi_b = 0.0               # χ между штрихами
chi_sub = chi             # χ подложки
# фотоны
ev = 520.0
wl = 1239.8 / ev          # нм
K = 2 * np.pi / wl        # нм^-1
# углы, градусы
alpha_ = 6.0
phi_ = 0.0
# число учитываемых порядков дифракции
n_max = 10
m = np.arange(-n_max, n_max + 1)   # порядки m, h = 2π m / d
D = len(m)                          # размерность задачи, 2 n_max + 1

PARAMS = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=chi_b,
              chi_sub=chi_sub, m=m)


# =============== GEOMETRY ===============
def incident_wavevector(K, alpha, phi):
    Kx = K * np.cos(alpha * DEG) * np.sin(phi * DEG)
    Ky = K * np.cos(alpha * DEG) * np.cos(phi * DEG)
    Kz = K * np.sin(alpha * DEG)
    return Kx, Ky, Kz


def lateral_wavevectors(Kx, Ky, m, d):
    kx = Kx + 2 * np.pi * m / d
    return kx, kx ** 2 + Ky ** 2


def choose_branch(kz2):
    kz = np.sqrt(np.asarray(kz2, dtype=complex))
    kz_real = abs(kz.real)
    kz_imag = abs(kz.imag)
    kz = kz_real + 1j * kz_imag
    return kz


# =============== FOURIER COEFFICIENTS ===============
def fourier_coefficients(chi_a, chi_b, gamma, d, orders):
    orders = np.asarray(orders)
    h = 2 * np.pi * orders / d
    chi_m = (chi_a - chi_b) * gamma * np.sinc(orders * gamma)
    chi_m = chi_m.astype(complex)
    chi_m[orders == 0] += chi_b
    return chi_m


def chi_profile(x, chi_a, chi_b, gamma, d):
    xp = (x + d / 2) % d - d / 2
    return np.where(np.abs(xp) <= gamma * d / 2, chi_a, chi_b).astype(complex)


def reconstruct_fourier(x, orders, chi_m, d):
    h = 2 * np.pi * np.asarray(orders) / d
    return np.exp(1j * np.outer(x, h)) @ chi_m


# =============== EIGENVALUE PROBLEM IN A STRUCTURED LAYER ===============
def build_A_matrix(chi_m_all, m, Kx, Ky, K, d):
    n2 = (len(chi_m_all) - 1) // 2
    diff = m[:, None] - m[None, :]              # h − g в единицах порядков
    chi_toeplitz = chi_m_all[diff + n2]         # матрица теплица χ_{h−g}
    _, kp2 = lateral_wavevectors(Kx, Ky, m, d)
    A = K ** 2 * chi_toeplitz + np.diag(K ** 2 - kp2)
    return A


def solve_structured_layer(chi_m_all, m, Kx, Ky, K, d):
    A = build_A_matrix(chi_m_all, m, Kx, Ky, K, d)
    eig, E = linalg.eig(A)
    kz = choose_branch(eig)
    return kz, E


def homogeneous_layer(chi_h, m, Kx, Ky, K, d):
    _, kp2 = lateral_wavevectors(Kx, Ky, m, d)
    return choose_branch(K ** 2 * (1 + chi_h) - kp2), np.eye(len(m), dtype=complex)


def layer_wavevectors(alpha, phi, *, K, d, t, gamma, chi, chi_b, chi_sub, m):
    n_max = int(m.max())
    Kx, Ky, Kz = incident_wavevector(K, alpha, phi)
    kx, _ = lateral_wavevectors(Kx, Ky, m, d)
    chi_m_all = fourier_coefficients(chi, chi_b, gamma, d, np.arange(-2 * n_max, 2 * n_max + 1))
    kz_vac, _ = homogeneous_layer(0.0, m, Kx, Ky, K, d)
    kz_lay, E_lay = solve_structured_layer(chi_m_all, m, Kx, Ky, K, d)
    kz_sub, _ = homogeneous_layer(chi_sub, m, Kx, Ky, K, d)
    return dict(Kx=Kx, Ky=Ky, Kz=Kz, kx=kx, kz_vac=kz_vac, kz_lay=kz_lay,
                E_lay=E_lay, kz_sub=kz_sub, chi_m_all=chi_m_all)


# =============== BOUNDARY MATRICES ===============
def P_matrix(E, kz):
    Ek = E * kz[None, :]                       # E @ diag(kz)
    return np.block([[E, E], [Ek, -Ek]])


def Q_matrix(kz, t):
    return np.diag(np.concatenate([np.exp(-1j * kz * t), np.exp(1j * kz * t)]))


def interface_matrix(P_upper, P_lower):
    return linalg.solve(P_upper, P_lower)


# =============== SOLVERS ===============
def solve_grating(alpha, phi, **p):
    """
    [R_vac, T_lay, R_lay, T_sub], dim = D.
        z = 0:  δ + R_vac            = E (T + q R)
                κ_vac (δ − R_vac)    = E k_z (T − q R)
        z = t:  E (q T + R)          = T_sub
                E k_z (q T − R)      = κ_sub T_sub
    """
    L = layer_wavevectors(alpha, phi, **p)
    m_ = p['m']
    t_ = p['t']
    D_ = len(m_)
    i0 = int(np.flatnonzero(m_ == 0)[0])

    E = L['E_lay']
    kz = L['kz_lay']
    q = np.exp(1j * kz * t_)
    Ek = E * kz[None, :]
    I = np.eye(D_, dtype=complex)
    Z = np.zeros((D_, D_), dtype=complex)

    A = np.block([
        [-I,                   E,                E * q[None, :],    Z],
        [np.diag(L['kz_vac']), Ek,               -Ek * q[None, :],  Z],
        [Z,                    E * q[None, :],   E,                 -I],
        [Z,                    Ek * q[None, :],  -Ek,               -np.diag(L['kz_sub'])],
    ])
    b = np.zeros(4 * D_, dtype=complex)
    b[i0] = 1.0                       # T_vac = (0,…,0,1,0,…,0) - падающая волна в порядке h = 0
    b[D_ + i0] = L['kz_vac'][i0]

    u = linalg.solve(A, b)
    R_vac, T_lay, R_lay, T_sub = u[:D_], u[D_:2 * D_], u[2 * D_:3 * D_], u[3 * D_:]

    Kz = L['Kz']
    refl = np.abs(R_vac) ** 2 * L['kz_vac'].real / Kz        # отражательная способность (49)
    trans = np.abs(T_sub) ** 2 * L['kz_sub'].real / Kz       # поток в подложку (для проверки баланса)

    L.update(alpha=alpha, phi=phi, m=m_, t=t_, i0=i0, R_vac=R_vac, T_lay=T_lay,
             R_lay=R_lay, T_sub=T_sub, refl=refl, trans=trans)
    return L


def solve_grating_transfer_matrix(alpha, phi, **p):
    """
    Формулировка статьи: матрица переноса (38), (47), (48).
        M = P_{01} Q^{(1)} P_{12}  (Q^{(2)} = I - фазы в подложке отсчитываются от её верхней границы),
        T_sub = M11^{−1} T_vac,  R_vac = M21 T_sub.
    Численно неустойчива при больших Im(k_zn) t (см. cond(M11)); оставлена для сравнения.
    """
    L = layer_wavevectors(alpha, phi, **p)
    m_, t_ = p['m'], p['t']
    D_ = len(m_)
    i0 = int(np.flatnonzero(m_ == 0)[0])
    I = np.eye(D_, dtype=complex)

    P_vac = P_matrix(I, L['kz_vac'])
    P_lay = P_matrix(L['E_lay'], L['kz_lay'])
    P_sub = P_matrix(I, L['kz_sub'])
    Q_lay = Q_matrix(L['kz_lay'], t_)

    M = interface_matrix(P_vac, P_lay) @ Q_lay @ interface_matrix(P_lay, P_sub)
    M11, M21 = M[:D_, :D_], M[D_:, :D_]

    T_vac = np.zeros(D_, dtype=complex)
    T_vac[i0] = 1.0
    T_sub = linalg.solve(M11, T_vac)
    R_vac = M21 @ T_sub
    refl = np.abs(R_vac) ** 2 * L['kz_vac'].real / L['Kz']
    L.update(alpha=alpha, phi=phi, m=m_, i0=i0, R_vac=R_vac, T_sub=T_sub, refl=refl,
             cond_M11=np.linalg.cond(M11))
    return L


# =============== NEAR FIELD ===============
def compute_field(sol, x, z):
    """
    Поле E(x, z) в одном периоде (без общего множителя exp(i K_y y)).
    z — глубина, положительная вниз: z < 0 вакуум, 0 <= z <= t решётка, z > t подложка.
        вакуум:   e^{i K_x x} e^{i K_z z} + Σ_h R_h e^{i k_hx x} e^{−i K_hz z}
        решётка:  Σ_h e^{i k_hx x} Σ_n [T_n e^{i k_zn z} + R_n e^{−i k_zn (z − t)}] E_hn
        подложка: Σ_h T^s_h e^{i k_hx x} e^{i k^s_hz (z − t)}
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    t_ = sol['t']
    D_ = len(sol['m'])
    C = np.zeros((len(z), D_), dtype=complex)   # амплитуды гармоник h на каждой глубине

    vac = z < 0
    lay = (z >= 0) & (z <= t_)
    sub = z > t_

    zv = z[vac][:, None]
    C[vac] = sol['R_vac'][None, :] * np.exp(-1j * sol['kz_vac'][None, :] * zv)
    C[vac, sol['i0']] += np.exp(1j * sol['Kz'] * z[vac])

    zl = z[lay][:, None]
    a = (sol['T_lay'][None, :] * np.exp(1j * sol['kz_lay'][None, :] * zl)
         + sol['R_lay'][None, :] * np.exp(-1j * sol['kz_lay'][None, :] * (zl - t_)))
    C[lay] = a @ sol['E_lay'].T                  # Σ_n E_hn a_n(z)

    zs = z[sub][:, None]
    C[sub] = sol['T_sub'][None, :] * np.exp(1j * sol['kz_sub'][None, :] * (zs - t_))

    phase_x = np.exp(1j * np.outer(sol['kx'], x))   # (D, Nx)
    return C @ phase_x                              # (Nz, Nx)


# =============== FLUORESCENCE (NIKOLAEV 2020, SEC. 2.5) ===============
def integrate_2d(f, x, z):
    """
    Двумерное численное интегрирование массива f(z, x) по координатам z (ось 0) и x (ось 1)
    методом трапеций. Совместимо со всеми версиями NumPy (np.trapezoid / np.trapz).
    """
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return trapz(trapz(f, x, axis=1), z, axis=0)


def fluorescence_mask(x, z, p=PARAMS, region='grating'):
    """
    Пространственное распределение концентрации излучающих атомов p(r)
    (Николаев 2020, разд. 2.5, ур. 27).

    Параметры:
        x, z: 1D массивы координат (нм).
        p: словарь геометрических параметров (d, t, gamma).
        region:
            'grating'   — атомы локализованы строго в штрихах решетки:
                          0 <= z <= t и |x| <= gamma * d / 2 (с учетом периодичности);
            'substrate' — атомы локализованы в подложке (z > t);
            'all'       — материал штрихов и подложки;
            callable    — пользовательская функция p(X, Z).

    Возвращает:
        2D массив формы (len(z), len(x)) со значениями концентрации p(x, z).
    """
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    X, Z = np.meshgrid(x, z)
    xp = (X + d_ / 2) % d_ - d_ / 2

    if region == 'grating':
        in_x = np.abs(xp) <= g_ * d_ / 2
        in_z = (Z >= 0.0) & (Z <= t_)
        return (in_x & in_z).astype(float)
    elif region == 'substrate':
        return (Z > t_).astype(float)
    elif region == 'all':
        in_grating = (np.abs(xp) <= g_ * d_ / 2) & (Z >= 0.0) & (Z <= t_)
        in_sub = Z > t_
        return (in_grating | in_sub).astype(float)
    elif callable(region):
        return region(X, Z).astype(float)
    else:
        raise ValueError(f"Неизвестный тип области: {region}")


def compute_fluorescence_from_field(I_field, x, z, p=PARAMS, mu_att=0.0, region='grating'):
    """
    Расчет интенсивности флуоресценции Y по уравнению Шермана (Николаев 2020, ур. 27)
    путем прямого интегрирования распределения интенсивности ближнего поля |E(x, z)|^2:

        Y = ∫∫ |E(x, z)|^2 * p(x, z) * exp(-mu_att * z) dx dz

    Параметры:
        I_field: 2D массив |E|^2 формы (len(z), len(x)).
        x, z: 1D массивы координат сетки (нм).
        p: словарь параметров (d, t, gamma).
        mu_att: эффективный коэффициент самопоглощения квантов флуоресценции (нм^-1).
                Описывает ослабление флуоресценции exp(-mu_att * z) при выходе из глубины z к детектору.
        region: область излучения ('grating', 'substrate', 'all' или 2D маска).

    Возвращает:
        Скалярную величину интегральной интенсивности флуоресценции (произв. ед.).
    """
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    hw = g_ * d_ / 2

    # Оптимизация: если сетка уже точно задана в границах штриха
    if isinstance(region, str) and region == 'grating':
        if (x.min() >= -hw - 1e-6 and x.max() <= hw + 1e-6 and
                z.min() >= -1e-6 and z.max() <= t_ + 1e-6):
            mask = 1.0
        else:
            mask = fluorescence_mask(x, z, p, region='grating')
    elif isinstance(region, np.ndarray):
        mask = region
    else:
        mask = fluorescence_mask(x, z, p, region=region)

    # Вес затухания фотонов флуоресценции при движении к верхней поверхности z=0
    z_nonneg = np.maximum(z, 0.0)
    att_weight = np.exp(-mu_att * z_nonneg)[:, None] if mu_att != 0.0 else 1.0

    integrand = I_field * mask * att_weight
    return integrate_2d(integrand, x, z)


def compute_fluorescence_analytical(sol, p=PARAMS, mu_att=0.0):
    """
    Аналитическое вычисление интеграла интенсивности флуоресценции через матричный формализм
    (Николаев 2020, разд. 2.5, ур. 28–32):

        Y = ∑_{g, h} [ U ⊙ (E W E^H) ]_{gh}

    где:
        U_{gh} — интеграл от пространственной плотности атомов в штрихе (ур. 30),
        W_{mn} — интеграл перекрытия мод a_m^*(z) * a_n(z) * exp(-mu_att * z) по глубине (ур. 31–32).
    """
    d_ = p['d']
    t_ = p['t']
    g_ = p['gamma']
    m_ = p['m']

    kz = sol['kz_lay']
    T = sol['T_lay']
    R = sol['R_lay']
    E = sol['E_lay']

    # Латеральная матрица U_{gh} = ∫_{-w/2}^{w/2} exp(i (h - g) 2π x / d) dx = w * sinc((h - g) * gamma)
    diff = m_[:, None] - m_[None, :]
    U = g_ * d_ * np.sinc(diff * g_)

    # Аналитический интеграл ∫_0^t exp(kappa * z) dz
    def int_exp(kappa, t_val):
        res = np.empty_like(kappa, dtype=complex)
        small = np.abs(kappa * t_val) < 1e-8
        res[small] = t_val * (1.0 + 0.5 * kappa[small] * t_val)
        res[~small] = (np.exp(kappa[~small] * t_val) - 1.0) / kappa[~small]
        return res

    kz_m_conj = np.conj(kz[:, None])  # (D, 1)
    kz_n = kz[None, :]               # (1, D)
    T_m_conj = np.conj(T[:, None])
    T_n = T[None, :]
    R_m_conj = np.conj(R[:, None])
    R_n = R[None, :]

    # 4 слагаемых в произведении мод a_m^*(z) a_n(z)
    k1 = 1j * (kz_n - kz_m_conj) - mu_att
    W1 = T_m_conj * T_n * int_exp(k1, t_)

    k2 = 1j * (-kz_n - kz_m_conj) - mu_att
    W2 = T_m_conj * (R_n * np.exp(1j * kz_n * t_)) * int_exp(k2, t_)

    k3 = 1j * (kz_n + kz_m_conj) - mu_att
    W3 = (R_m_conj * np.exp(-1j * kz_m_conj * t_)) * T_n * int_exp(k3, t_)

    k4 = 1j * (-kz_n + kz_m_conj) - mu_att
    W4 = (R_m_conj * np.exp(-1j * kz_m_conj * t_)) * (R_n * np.exp(1j * kz_n * t_)) * int_exp(k4, t_)

    W = W1 + W2 + W3 + W4

    # F = U ⊙ (E^* @ W @ E^T)
    M = np.conj(E) @ W @ E.T
    return np.sum(U * M).real


def fluorescence_vs_alpha(alphas, phi=0.0, p=PARAMS, nx=100, nz=100,
                          mu_att=0.0, method='grid', region='grating', return_refl=False):
    """
    Расчет угловой зависимости интенсивности флуоресценции I_fluor(alpha)
    для одномерного случая (фиксированный азимутальный угол phi).

    Параметры:
        alphas: 1D массив углов падения рентгена (градусы), например np.linspace(0.05, 20.0, 200).
        phi: азимутальный угол (градусы), по умолчанию 0.0 (коническая дифракция).
        p: словарь параметров решетки и излучения.
        nx, nz: число точек сетки ближнего поля по x и z.
        mu_att: коэффициент самопоглощения фотонов флуоресценции (нм^-1).
        method: 'grid' (интегрирование E^2 по сетке ближнего поля)
                или 'analytical' (матричный метод Николаев 2020).
        region: область излучения ('grating', 'substrate', 'all').
        return_refl: если True, возвращает также массив полного коэффициента отражения R_tot(alpha).

    Возвращает:
        alphas, I_fluor (и R_tot, если return_refl=True).
    """
    alphas = np.asarray(alphas, dtype=float)
    I_fluor = np.zeros(len(alphas), dtype=float)
    R_tot = np.zeros(len(alphas), dtype=float)

    d_ = p['d']
    t_ = p['t']
    g_ = p['gamma']
    hw = g_ * d_ / 2

    # Создаем пространственную сетку для интегрирования
    if method == 'grid':
        if region == 'grating':
            x = np.linspace(-hw, hw, nx)
            z = np.linspace(0.0, t_, nz)
        else:
            x = np.linspace(-d_ / 2, d_ / 2, nx)
            z = np.linspace(0.0, t_, nz)

    for i, a in enumerate(alphas):
        sol = solve_grating(a, phi, **p)
        R_tot[i] = np.sum(sol['refl'])

        if method == 'grid':
            E = compute_field(sol, x, z)
            I_field = np.abs(E) ** 2
            I_fluor[i] = compute_fluorescence_from_field(I_field, x, z, p=p, mu_att=mu_att, region=region)
        elif method == 'analytical':
            I_fluor[i] = compute_fluorescence_analytical(sol, p=p, mu_att=mu_att)
        else:
            raise ValueError(f"Неизвестный метод расчета: {method}")

    if return_refl:
        return alphas, I_fluor, R_tot
    return alphas, I_fluor


def plot_fluorescence_vs_alpha(alphas, I_fluor, phi=0.0, p=PARAMS, refl=None):
    """
    Построение графика одномерной кривой интенсивности флуоресценции
    в зависимости от угла падения рентгена alpha в диапазоне (0, 20] градусов.
    """
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    fig, ax = plt.subplots(figsize=(8, 5))

    line_fl, = ax.plot(alphas, I_fluor, 'b-', lw=2.0, label=r'Флуоресценция $I_{\mathrm{fluor}}(\alpha)$')
    ax.set_xlabel(r'Угол падения $\alpha$, градусы', fontsize=11)
    ax.set_ylabel(r'Интенсивность флуоресценции, произв. ед.', fontsize=11, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_xlim(alphas[0], alphas[-1])
    ax.grid(True, linestyle='--', alpha=0.5)

    title = (f"Интенсивность флуоресценции $I_{{\\mathrm{{fluor}}}}(\\alpha)$\n"
             f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$, $\\phi = {phi:.1f}^\\circ$")
    ax.set_title(title, fontsize=12)

    # # При наличии данных отражения добавляем вторую ось Y для сопоставления с дифракционными минимумами
    # if refl is not None:
    #     ax2 = ax.twinx()
    #     line_r, = ax2.plot(alphas, refl, 'r--', lw=1.2, alpha=0.75, label=r'Отражение $R_{\mathrm{tot}}(\alpha)$')
    #     ax2.set_ylabel(r'Коэффициент отражения $R_{\mathrm{tot}}$', fontsize=11, color='red')
    #     ax2.tick_params(axis='y', labelcolor='red')
    #     ax2.set_ylim(-0.02, 1.05)
    #     lines = [line_fl, line_r]
    #     labels = [l.get_label() for l in lines]
    #     ax.legend(lines, labels, loc='center right', framealpha=0.9)
    # else:
    #     ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    return fig


def resonant_lines(alphas, m_orders=(-2, -1, 1, 2), p=PARAMS):
    """
    Расчет теоретических резонансных линий дифракции (уравнение Лауэ / сфера Эвальда)
    согласно Николаев 2020, разд. 3.1, ур. (33):

        sin(phi) = (sin^2(alpha) - xi^2) / (2 * xi * cos(alpha)),  где xi = m * lambda / d

    Параметры:
        alphas: 1D массив углов падения alpha (градусы).
        m_orders: кортеж или список порядков дифракции m.
        p: словарь параметров (K, d).

    Возвращает:
        Словарь {m: (phi_deg, alpha_deg)} для каждого порядка m в пределах физических углов.
    """
    wl = 2 * np.pi / p['K']
    d_ = p['d']
    alphas = np.asarray(alphas, dtype=float)
    cos_a = np.cos(alphas * DEG)
    sin_a = np.sin(alphas * DEG)

    lines = {}
    for m_val in m_orders:
        if m_val == 0:
            continue
        xi = m_val * wl / d_
        sin_phi = (sin_a ** 2 - xi ** 2) / (2.0 * xi * cos_a)
        valid = np.abs(sin_phi) <= 1.0
        if np.any(valid):
            phi_deg = np.arcsin(sin_phi[valid]) / DEG
            lines[m_val] = (phi_deg, alphas[valid])
    return lines


def compute_fluorescence_map_2d(alphas, phis, p=PARAMS, nx=35, nz=35,
                                mu_att=0.0, method='analytical', region='grating'):
    """
    Расчет 2D карты интенсивности флуоресценции I(alpha, phi) на сетке углов падения alpha
    и азимутальных углов phi (Николаев 2020, разд. 3.1, Рис. 4).

    Параметры:
        alphas: 1D массив углов падения (градусы), например np.linspace(0.05, 3.0, 100).
        phis: 1D массив азимутальных углов (градусы), например np.linspace(-1.5, 1.5, 101).
        p: словарь параметров решетки и излучения.
        nx, nz: число точек сетки ближнего поля по x и z (для method='grid').
        mu_att: коэффициент самопоглощения фотонов флуоресценции (нм^-1).
        method: 'analytical' (полуаналитический матричный метод Николаев 2020, быстрый и точный)
                или 'grid' (интегрирование E^2 по сетке ближнего поля).
        region: область излучения ('grating', 'substrate', 'all').

    Возвращает:
        2D массив map_I формы (len(alphas), len(phis)).
    """
    alphas = np.asarray(alphas, dtype=float)
    phis = np.asarray(phis, dtype=float)
    map_I = np.zeros((len(alphas), len(phis)), dtype=float)

    d_ = p['d']
    t_ = p['t']
    g_ = p['gamma']
    hw = g_ * d_ / 2

    # Создаем пространственную сетку для интегрирования ближнего поля (если выбран метод grid)
    if method == 'grid':
        if region == 'grating':
            x = np.linspace(-hw, hw, nx)
            z = np.linspace(0.0, t_, nz)
        else:
            x = np.linspace(-d_ / 2, d_ / 2, nx)
            z = np.linspace(0.0, t_, nz)

    for i, a in enumerate(alphas):
        for j, phi_val in enumerate(phis):
            sol = solve_grating(a, phi_val, **p)
            if method == 'grid':
                E = compute_field(sol, x, z)
                I_field = np.abs(E) ** 2
                map_I[i, j] = compute_fluorescence_from_field(I_field, x, z, p=p, mu_att=mu_att, region=region)
            elif method == 'analytical':
                map_I[i, j] = compute_fluorescence_analytical(sol, p=p, mu_att=mu_att)
            else:
                raise ValueError(f"Неизвестный метод расчета: {method}")

    return map_I


def plot_fluorescence_map_2d(alphas, phis, map_I, p=PARAMS,
                             overlay_lines=True, m_orders=(-2, -1, 1, 2),
                             cmap='turbo', norm=None, two_panel=True,
                             line_color='white'):
    """
    Построение 2D тепловой карты интенсивности флуоресценции I(phi, alpha)
    в соответствии с Рис. 4 статьи Николаев 2020.

    Параметры:
        alphas: 1D массив углов падения alpha (градусы), ось Y.
        phis: 1D массив азимутальных углов phi (градусы), ось X.
        map_I: 2D массив интенсивности флуоресценции формы (len(alphas), len(phis)).
        p: словарь параметров решетки и излучения.
        overlay_lines: отображать ли теоретические резонансные линии (ур. 33 статьи).
        m_orders: порядки дифракции для отображения резонансных линий.
        cmap: цветовая карта ('turbo', 'inferno', 'plasma', 'cmo.thermal' и др.). По умолчанию 'turbo'.
        norm: нормализация шкалы цветов (None для линейной шкалы, 'sqrt' для степенной, или экземпляр Normalize).
        two_panel: если True, строит 2 панели рядом:
                   1) чистая тепловая карта (Рис. 4b),
                   2) тепловая карта с наложенными линиями Лауэ (Рис. 4c).
                   Если False, строит одну отдельную карту.
        line_color: цвет резонансных линий ('white', 'black', 'cyan' и др.).
    """
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    extent = [phis[0], phis[-1], alphas[0], alphas[-1]]

    # Обработка строковых параметров нормализации
    if isinstance(norm, str):
        if norm == 'sqrt':
            norm = colors.PowerNorm(gamma=0.5, vmin=np.nanmin(map_I), vmax=np.nanmax(map_I))
        elif norm == 'log':
            pos_min = np.nanmin(map_I[map_I > 0]) if np.any(map_I > 0) else 1.0
            norm = colors.LogNorm(vmin=pos_min, vmax=np.nanmax(map_I))
        else:
            raise ValueError(f"Неизвестный тип нормализации: {norm}")

    if two_panel:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

        # Панель 1: чистая 2D карта (Рис. 4b)
        im1 = ax1.imshow(map_I, extent=extent, origin='lower', aspect='auto',
                         cmap=cmap, norm=norm, interpolation='bilinear')
        ax1.set_xlabel(r'Азимутальный угол $\phi$, градусы', fontsize=11)
        ax1.set_ylabel(r'Угол падения $\alpha$, градусы', fontsize=11)
        ax1.set_title(r'(a) Моделирование 2D карты GIXRF', fontsize=12)
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label(r'Интенсивность флуоресценции, произв. ед.', fontsize=10)

        # Панель 2: карта с наложением резонансных линий (Рис. 4c)
        im2 = ax2.imshow(map_I, extent=extent, origin='lower', aspect='auto',
                         cmap=cmap, norm=norm, interpolation='bilinear')
        if overlay_lines:
            lines = resonant_lines(np.linspace(alphas[0], alphas[-1], 500), m_orders=m_orders, p=p)
            for m_val, (p_deg, a_deg) in lines.items():
                inside = (p_deg >= phis[0]) & (p_deg <= phis[-1])
                if np.any(inside):
                    ax2.plot(p_deg[inside], a_deg[inside], linestyle='--', color=line_color,
                             lw=1.3, alpha=0.9, label=f'$m = {m_val}$')
            ax2.legend(loc='lower right', fontsize=9, framealpha=0.7)

        ax2.set_xlim(phis[0], phis[-1])
        ax2.set_ylim(alphas[0], alphas[-1])
        ax2.set_xlabel(r'Азимутальный угол $\phi$, градусы', fontsize=11)
        ax2.set_ylabel(r'Угол падения $\alpha$, градусы', fontsize=11)
        ax2.set_title(r'(b) Резонансные линии Лауэ (ур. 33)', fontsize=12)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label(r'Интенсивность флуоресценции, произв. ед.', fontsize=10)

        fig.suptitle(f"2D карта интенсивности флуоресценции (Николаев 2020, Рис. 4)\n"
                     f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$", fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(map_I, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap, norm=norm, interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'Интенсивность флуоресценции, произв. ед.', fontsize=11)

        if overlay_lines:
            lines = resonant_lines(np.linspace(alphas[0], alphas[-1], 500), m_orders=m_orders, p=p)
            for m_val, (p_deg, a_deg) in lines.items():
                inside = (p_deg >= phis[0]) & (p_deg <= phis[-1])
                if np.any(inside):
                    ax.plot(p_deg[inside], a_deg[inside], linestyle='--', color=line_color,
                            lw=1.3, alpha=0.9, label=f'$m = {m_val}$')
            ax.legend(loc='lower right', fontsize=9, framealpha=0.7)
            title = (f"2D карта флуоресценции с линиями Лауэ (палитра {cmap})\n"
                     f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$")
        else:
            title = (f"2D карта интенсивности флуоресценции (палитра {cmap})\n"
                     f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$")

        ax.set_xlabel(r'Азимутальный угол $\phi$, градусы', fontsize=11)
        ax.set_ylabel(r'Угол падения $\alpha$, градусы', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(phis[0], phis[-1])
        ax.set_ylim(alphas[0], alphas[-1])

    fig.tight_layout()
    return fig



# =============== PLOTS ===============
def plot_fourier_check(p=PARAMS):
    orders = np.arange(-2 * int(p['m'].max()), 2 * int(p['m'].max()) + 1)
    chi_m_all = fourier_coefficients(p['chi'], p['chi_b'], p['gamma'], p['d'], orders)
    x = np.linspace(-p['d'] / 2, p['d'] / 2, 1001)
    f = chi_profile(x, p['chi'], p['chi_b'], p['gamma'], p['d'])
    f_rec = reconstruct_fourier(x, p['m'], chi_m_all[np.isin(orders, p['m'])], p['d'])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, f.real, 'b-', label='original')
    ax[0].plot(x, f_rec.real, 'r--', label=f'|m| <= {p["m"].max()}')
    ax[0].set_title('Re χ(x)')
    ax[1].plot(x, f.imag, 'g-', label='original')
    ax[1].plot(x, f_rec.imag, 'y--', label=f'|m| <= {p["m"].max()}')
    ax[1].set_title('Im χ(x)')
    for a in ax:
        a.set_xlabel('x, nm')
        a.grid(True)
        a.legend()
    fig.tight_layout()
    return fig


def plot_kz_vs_alpha(alphas, phi, p=PARAMS):
    kz_all = []
    for a in alphas:
        L = layer_wavevectors(a, phi, **p)
        kz_all.append(L['kz_lay'])       # сортировка — eig не упорядочивает моды
    kz_all = np.array(kz_all)

    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax[0].plot(alphas, kz_all.real, lw=0.8)
    ax[0].set_ylabel('Re k_zn, nm^-1')
    ax[0].set_title(f'k_zn(α) in the structured layer, φ = {phi}°')
    ax[1].plot(alphas, kz_all.imag, lw=0.8)
    ax[1].set_ylabel('Im k_zn, nm^-1')
    ax[1].set_xlabel('α, deg')
    for a in ax:
        a.grid(True)
    fig.tight_layout()
    return fig


def plot_eigenvectors(sol):
    E_mag = np.abs(sol['E_lay'])
    mm = sol['m']
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(E_mag, aspect='auto', cmap='viridis', interpolation='none',
                   extent=[-0.5, len(mm) - 0.5, mm[-1] + 0.5, mm[0] - 0.5], vmin=0, vmax=1)
    ax.set_xlabel('Номер моды n', fontsize=11)
    ax.set_ylabel('Дифракционный порядок m', fontsize=11)
    ax.set_title(f"|E_hn|, α = {sol['alpha']}°, φ = {sol['phi']}°")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='normalized amplitude')
    fig.tight_layout()
    return fig


def plot_near_field(sol, p=PARAMS, nx=400, nz=300, pad=20.0, cmap='cmo.thermal'):
    """Карта |E|^2 в одном периоде. scale='log' — log10|E|^2 (11 декад), scale='linear' — |E|^2 с обрезкой по 99.5%."""
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    x = np.linspace(-d_ / 2, d_ / 2, nx)
    z = np.linspace(-pad, t_ + pad, nz)                  # глубина вниз
    E = compute_field(sol, x, z)
    I_field = np.abs(E) ** 2
    height = t_ - z                                       # для рисунка: высота над подложкой

    fig, ax = plt.subplots(figsize=(8, 8))
    data, vmin, vmax, label = I_field, 0.0, np.percentile(I_field, 99.5), '|E|²'
    cmap = cmap or 'gnuplot2'
    im = ax.imshow(data, extent=[x[0], x[-1], height[-1], height[0]], vmin=vmin, vmax=vmax,
                   aspect='auto', cmap=cmap, origin='upper', interpolation='bilinear')
    hw = g_ * d_ / 2
    ax.plot([-hw, hw], [t_, t_], 'w-', lw=2)
    ax.plot([-hw, -hw], [0, t_], 'w-', lw=2)
    ax.plot([hw, hw], [0, t_], 'w-', lw=2)
    ax.plot([-d_ / 2, -hw], [0, 0], 'w-', lw=2)
    ax.plot([hw, d_ / 2], [0, 0], 'w-', lw=2)
    ax.set_xlabel('x, nm')
    ax.set_ylabel('height above substrate, nm')
    ax.set_title(f"{label},  d = {d_} nm, t = {t_} nm, Γ = {g_}, α = {sol['alpha']}°, φ = {sol['phi']}°")
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlim(-d_ / 2, d_ / 2)
    ax.set_ylim(height[-1], height[0])
    fig.tight_layout()
    return fig


# =============== MAIN ===============
def main(save_dir=None):

    sol = solve_grating(alpha_, phi_, **PARAMS)

    figs = {}
    figs['fourier'] = plot_fourier_check()
    figs['kz_alpha'] = plot_kz_vs_alpha(np.linspace(0.05, 20, 200), phi_)
    figs['eigenvectors'] = plot_eigenvectors(sol)
    figs['near_field'] = plot_near_field(sol)                                      # линейная шкала |E|^2
    # figs['near_field_a3_phi1'] = plot_near_field(solve_grating(3.0, 1.0, **PARAMS))  # второй случай: α = 3°, φ = 1°

    # Расчет одномерной кривой флуоресценции от угла падения α ∈ (0, 20] градусов
    alphas_fl = np.linspace(0.05, 20.0, 200)
    alphas_fl, I_fl, R_fl = fluorescence_vs_alpha(alphas_fl, phi_, p=PARAMS, nx=80, nz=80, return_refl=True)
    figs['fluorescence'] = plot_fluorescence_vs_alpha(alphas_fl, I_fl, phi_, p=PARAMS, refl=R_fl)

    # Расчет 2D тепловой карты флуоресценции I(phi, alpha) (Николаев 2020, Рис. 4)
    alphas_map = np.linspace(0.05, 3.0, 100)
    phis_map = np.linspace(-1.5, 1.5, 101)
    map_I = compute_fluorescence_map_2d(alphas_map, phis_map, p=PARAMS, method='analytical')

    # 1) Чистая 2D тепловая карта без линий резонанса (палитра turbo)
    figs['fluorescence_map_2d_clean'] = plot_fluorescence_map_2d(
        alphas_map, phis_map, map_I, p=PARAMS,
        overlay_lines=False, two_panel=False, cmap='turbo'
    )

    # 2) 2D тепловая карта с наложением теоретических резонансных линий Лауэ (палитра turbo)
    figs['fluorescence_map_2d_lines'] = plot_fluorescence_map_2d(
        alphas_map, phis_map, map_I, p=PARAMS,
        overlay_lines=True, two_panel=False, cmap='turbo'
    )

    # 3) Двухпанельная 2D карта (чистая карта и с линиями рядом, Николаев 2020, Рис. 4b,c)
    figs['fluorescence_map_2d'] = plot_fluorescence_map_2d(
        alphas_map, phis_map, map_I, p=PARAMS,
        overlay_lines=True, two_panel=True, cmap='turbo'
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, name + '.png'), dpi=120)
        print('figures saved to', save_dir)
    else:
        plt.show()


if __name__ == '__main__':
    save_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'figures')
    main(save_path)
