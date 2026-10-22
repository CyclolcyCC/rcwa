import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
from scipy import fft
from scipy import linalg
import cmocean
import warnings


DEG = np.pi / 180.0

# =============== PARAMETERS ===============
# структура
d = 100.0                 # период решетки, нм
t = 90.0                  # высота штрихов, нм
gamma = 0.4               # Γ = d_a / d — доля периода
chi = -2 * 1.703e-3 + 2j * 3.99e-5    # χ материала штрихов (Si3N4 при 520 эВ)
chi_b = 0.0               # χ между штрихами (вакуум)
chi_sub = chi             # χ подложки
eps_1 = 1.0 + chi         # комплексная диэлектрическая проницаемость штрихов
eps_b = 1.0 + chi_b       # проницаемость межштриховой области
eps_sub = 1.0 + chi_sub   # проницаемость подложки

# фотоны
ev = 520.0                # энергия фотонов, эВ
wl = 1239.8 / ev          # длина волны, нм (~2.384 нм)
K = 2 * np.pi / wl        # волновое число в вакууме, нм^-1

# углы, градусы
alpha_ = 6.0
phi_ = 0.0

# число учитываемых порядков дифракции
n_max = 15
m = np.arange(-n_max, n_max + 1)   # порядки m, h = 2π m / d
D = len(m)                          # размерность задачи, 2 n_max + 1

PARAMS = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=chi_b,
              chi_sub=chi_sub, eps_1=eps_1, eps_b=eps_b, eps_sub=eps_sub, m=m)


# =============== GEOMETRY ===============
def incident_wavevector(K, alpha, phi):
    """Компоненты падающего волнового вектора (Kx, Ky, Kz)."""
    Kx = K * np.cos(alpha * DEG) * np.sin(phi * DEG)
    Ky = K * np.cos(alpha * DEG) * np.cos(phi * DEG)
    Kz = K * np.sin(alpha * DEG)
    return Kx, Ky, Kz


def lateral_wavevectors(Kx, Ky, m, d):
    """Латеральные волновые числа kx и модуль kp^2 = kx^2 + Ky^2."""
    kx = Kx + 2 * np.pi * m / d
    return kx, kx ** 2 + Ky ** 2


def choose_branch(kz2):
    """
    Выбор физической ветви квадратного корня для kz:
    Re(kz) >= 0 и Im(kz) >= 0 (затухание вглубь поглощающей среды).
    """
    kz = np.sqrt(np.asarray(kz2, dtype=complex))
    kz_real = np.abs(kz.real)
    kz_imag = np.abs(kz.imag)
    return kz_real + 1j * kz_imag


# =============== FOURIER COEFFICIENTS (LI 1996 RULES) ===============
def fourier_coefficients_eps(eps_a, eps_b, gamma, d, orders):
    """
    Аналитические коэффициенты Фурье диэлектрической проницаемости ε(x)
    для ламеллярной прямоугольной решетки (прямое правило Лорана).
    """
    orders = np.asarray(orders, dtype=float)
    eps_m = (eps_a - eps_b) * gamma * np.sinc(orders * gamma)
    eps_m = eps_m.astype(complex)
    eps_m[orders == 0] += eps_b
    return eps_m


def fourier_coefficients_eta(eps_a, eps_b, gamma, d, orders):
    """
    Аналитические коэффициенты Фурье обратной проницаемости η(x) = 1 / ε(x)
    для ламеллярной прямоугольной решетки (обратное правило Ли, 1996).
    """
    orders = np.asarray(orders, dtype=float)
    eta_a = 1.0 / eps_a
    eta_b = 1.0 / eps_b
    eta_m = (eta_a - eta_b) * gamma * np.sinc(orders * gamma)
    eta_m = eta_m.astype(complex)
    eta_m[orders == 0] += eta_b
    return eta_m


def eps_profile(x, eps_a, eps_b, gamma, d):
    """Профиль диэлектрической проницаемости ε(x) в действительном пространстве."""
    xp = (x + d / 2) % d - d / 2
    return np.where(np.abs(xp) <= gamma * d / 2, eps_a, eps_b).astype(complex)


def eta_profile(x, eps_a, eps_b, gamma, d):
    """Профиль обратной диэлектрической проницаемости η(x) = 1/ε(x)."""
    xp = (x + d / 2) % d - d / 2
    return np.where(np.abs(xp) <= gamma * d / 2, 1.0 / eps_a, 1.0 / eps_b).astype(complex)


def reconstruct_fourier(x, orders, coeffs, d):
    """Синтез периодической функции из усеченного ряда Фурье."""
    h = 2 * np.pi * np.asarray(orders) / d
    return np.exp(1j * np.outer(x, h)) @ coeffs


def build_fourier_matrices(eps_a, eps_b, gamma, d, m):
    """
    Построение тёплицевых матриц [ε] и Z = [1/ε] размера D x D
    согласно правилам факторизации Ли (Li, 1996) для p-поляризации.
    """
    n_max = int(np.max(m))
    orders_extended = np.arange(-2 * n_max, 2 * n_max + 1)
    eps_all = fourier_coefficients_eps(eps_a, eps_b, gamma, d, orders_extended)
    eta_all = fourier_coefficients_eta(eps_a, eps_b, gamma, d, orders_extended)

    diff = m[:, None] - m[None, :]  # h - g в единицах порядков
    n2 = 2 * n_max
    eps_mat = eps_all[diff + n2]
    Z_mat = eta_all[diff + n2]
    return eps_mat, Z_mat


# =============== EIGENVALUE PROBLEM FOR P-POLARIZATION ===============
def build_A_matrix_p(eps_mat, Z_mat, m, Kx, Ky, K, d):
    """
    Построение матрицы взаимодействия A^(p) для p-поляризации согласно правилам факторизации Ли (Li, 1996):
        d^2 u / dz^2 + A^(p) u = 0,
        A^(p) = Z^{-1} [ K^2 I - Kx [ε]^{-1} Kx ] - Ky^2 I,
    где Z = [1/ε] — тёплицева матрица обратной проницаемости,
    а слагаемое -Ky^2 I возникает от второй производной ∂^2/∂y^2 при конической дифракции.
    """
    D = len(m)
    kx = Kx + 2 * np.pi * m / d
    Kx_diag = np.diag(kx)

    Z_inv = linalg.inv(Z_mat)
    eps_inv = linalg.inv(eps_mat)

    bracket = K ** 2 * np.eye(D, dtype=complex) - Kx_diag @ eps_inv @ Kx_diag
    A = Z_inv @ bracket - (Ky ** 2) * np.eye(D, dtype=complex)
    return A


def solve_structured_layer_p(eps_mat, Z_mat, m, Kx, Ky, K, d):
    """
    Решение задачи на собственные значения для слоя периодической решетки в p-поляризации:
        A^(p) E_n = k_zn^2 E_n.
    Возвращает вертикальные волновые числа kz_lay и матрицу собственных векторов E_lay.
    """
    A = build_A_matrix_p(eps_mat, Z_mat, m, Kx, Ky, K, d)
    eig, E = linalg.eig(A)
    kz = choose_branch(eig)
    return kz, E


def homogeneous_layer_p(eps_h, m, Kx, Ky, K, d):
    """Вертикальные волновые числа для однородной среды в p-поляризации."""
    kx, kp2 = lateral_wavevectors(Kx, Ky, m, d)
    kz2 = eps_h * K ** 2 - kp2
    return choose_branch(kz2), np.eye(len(m), dtype=complex)


def layer_wavevectors(alpha, phi, *, K, d, t, gamma, chi, chi_b, chi_sub,
                      eps_1, eps_b, eps_sub, m):
    """
    Расчет волновых векторов и собственных мод во всех трех областях
    (вакуум, слой решетки, подложка) для p-поляризации.
    """
    Kx, Ky, Kz = incident_wavevector(K, alpha, phi)
    kx, _ = lateral_wavevectors(Kx, Ky, m, d)

    eps_mat, Z_mat = build_fourier_matrices(eps_1, eps_b, gamma, d, m)

    kz_vac, _ = homogeneous_layer_p(1.0, m, Kx, Ky, K, d)
    kz_lay, E_lay = solve_structured_layer_p(eps_mat, Z_mat, m, Kx, Ky, K, d)
    kz_sub, _ = homogeneous_layer_p(eps_sub, m, Kx, Ky, K, d)

    return dict(Kx=Kx, Ky=Ky, Kz=Kz, kx=kx, kz_vac=kz_vac, kz_lay=kz_lay,
                E_lay=E_lay, kz_sub=kz_sub, eps_mat=eps_mat, Z_mat=Z_mat)


# =============== BOUNDARY CONDITIONS SOLVER (P-POL) ===============
def solve_grating(alpha, phi, **p):
    """
    Численно устойчивое решение СЛАУ граничных условий для p-поляризации (ур. 3.85 статьи Hench & Strakos):
        Непрерывность поля u на z=0 и z=t;
        Непрерывность (1/ε) ∂u/∂z на z=0 и z=t (с учетом матрицы Z = [1/ε] по правилу Ли).

    Глобальная СЛАУ 4D x 4D:
        [-I,           E,            E * q,         0      ] [R_vac]   [  δ_0   ]
        [K_vac,        P,           -P * q,         0      ] [T_lay] = [kz0_vac ]
        [0,            E * q,        E,            -I      ] [R_lay]   [   0    ]
        [0,            P * q,       -P,            -K_sub~ ] [T_sub]   [   0    ]
    где P = Z @ E @ diag(kz_lay), K_sub~ = diag(kz_sub / eps_sub).
    """
    L = layer_wavevectors(alpha, phi, **p)
    m_ = p['m']
    t_ = p['t']
    eps_sub = p['eps_sub']
    D_ = len(m_)
    i0 = int(np.flatnonzero(m_ == 0)[0])

    E = L['E_lay']
    kz = L['kz_lay']
    Z_mat = L['Z_mat']

    q = np.exp(1j * kz * t_)
    P = Z_mat @ (E * kz[None, :])
    I = np.eye(D_, dtype=complex)
    Z_zero = np.zeros((D_, D_), dtype=complex)

    A_bc = np.block([
        [-I,                   E,               E * q[None, :],   Z_zero],
        [np.diag(L['kz_vac']), P,               -P * q[None, :],  Z_zero],
        [Z_zero,               E * q[None, :],  E,                -I],
        [Z_zero,               P * q[None, :],  -P,               -np.diag(L['kz_sub'] / eps_sub)],
    ])

    b = np.zeros(4 * D_, dtype=complex)
    b[i0] = 1.0                       # падающая волна амплитудой 1 в порядке h=0
    b[D_ + i0] = L['kz_vac'][i0]

    u = linalg.solve(A_bc, b)
    R_vac, T_lay, R_lay, T_sub = u[:D_], u[D_:2 * D_], u[2 * D_:3 * D_], u[3 * D_:]

    Kz = L['Kz']
    # Отражательная способность каждого порядка дифракции (поток вектора Пойнтинга)
    refl = np.abs(R_vac) ** 2 * L['kz_vac'].real / Kz
    trans = np.abs(T_sub) ** 2 * (L['kz_sub'] / eps_sub).real / Kz

    L.update(alpha=alpha, phi=phi, m=m_, t=t_, i0=i0, R_vac=R_vac, T_lay=T_lay,
             R_lay=R_lay, T_sub=T_sub, refl=refl, trans=trans, p=p)
    return L


# =============== NEAR FIELD RECONSTRUCTION (P-POL) ===============
def compute_field(sol, x, z):
    """
    Распределение комплексного волнового поля E(x, z) в одном периоде для p-поляризации
    (без общего множителя exp(i K_y y)).
    z — глубина (нм): z < 0 вакуум, 0 <= z <= t решетка, z > t подложка.
        вакуум:   e^{i K_x x} e^{i K_z z} + Σ_h R_h e^{i k_hx x} e^{−i κ_hz z}
        решетка:  Σ_h e^{i k_hx x} Σ_n [T_n e^{i k_zn z} + R_n e^{−i k_zn (z − t)}] E_hn
        подложка: Σ_h T^s_h e^{i k_hx x} e^{i κ^s_hz (z − t)}
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    t_ = sol['t']
    D_ = len(sol['m'])
    C = np.zeros((len(z), D_), dtype=complex)

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


# =============== FLUORESCENCE (SEMI-ANALYTICAL P-POL) ===============
def integrate_2d(f, x, z):
    """Двумерное численное интегрирование методом трапеций."""
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return trapz(trapz(f, x, axis=1), z, axis=0)


def fluorescence_mask(x, z, p=PARAMS, region='grating'):
    """Геометрическая маска для численного интегрирования флуоресценции."""
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    hw = g_ * d_ / 2
    x_mesh, z_mesh = np.meshgrid(x, z)
    if region == 'grating':
        return (np.abs(x_mesh) <= hw) & (z_mesh >= 0.0) & (z_mesh <= t_)
    elif region == 'substrate':
        return z_mesh > t_
    elif region == 'all':
        return ((np.abs(x_mesh) <= hw) & (z_mesh >= 0.0) & (z_mesh <= t_)) | (z_mesh > t_)
    else:
        raise ValueError(f"Неизвестная область: {region}")


def compute_fluorescence_from_field(I_field, x, z, p=PARAMS, mu_att=0.0, region='grating'):
    """Численный расчет выхода флуоресценции интегрированием |E|^2 по сетке."""
    mask = fluorescence_mask(x, z, p=p, region=region)
    att = np.exp(-mu_att * z)[:, None]
    integrand = I_field * att * mask
    return integrate_2d(integrand, x, z)


def compute_fluorescence_analytical(sol, p=PARAMS, mu_att=0.0):
    """
    Аналитическое вычисление интеграла интенсивности флуоресценции через матричный формализм
    (Николаев 2020, разд. 2.5, ур. 28–32) для p-поляризации:

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
    return float(np.sum(U * M).real)


def fluorescence_vs_alpha(alphas, phi=0.0, p=PARAMS, nx=100, nz=100,
                          mu_att=0.0, method='analytical', region='grating', return_refl=False):
    """
    Расчет угловой зависимости выхода флуоресценции I_fluor(alpha) для p-поляризации
    в диапазоне углов падения alpha ∈ (0, 20] градусов.
    """
    alphas = np.asarray(alphas, dtype=float)
    I_fluor = np.zeros(len(alphas), dtype=float)
    R_tot = np.zeros(len(alphas), dtype=float)

    d_ = p['d']
    t_ = p['t']
    g_ = p['gamma']
    hw = g_ * d_ / 2

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

        if method == 'analytical':
            I_fluor[i] = compute_fluorescence_analytical(sol, p=p, mu_att=mu_att)
        elif method == 'grid':
            field = compute_field(sol, x, z)
            I_field = np.abs(field) ** 2
            I_fluor[i] = compute_fluorescence_from_field(I_field, x, z, p=p, mu_att=mu_att, region=region)
        else:
            raise ValueError(f"Неизвестный метод расчета: {method}")

    if return_refl:
        return alphas, I_fluor, R_tot
    return alphas, I_fluor

    if return_refl:
        return alphas, I_fluor, R_tot
    return alphas, I_fluor


# =============== 2D FLUORESCENCE MAP & RESONANT LINES ===============
def resonant_lines(alphas, m_orders=(-2, -1, 1, 2), p=PARAMS):
    """
    Расчет кинематических резонансных линий Рэлея-Вуда (ур. 766 в теории p-поляризации):
        sin(phi) = (sin^2(alpha) - beta^2) / (2 * beta * cos(alpha)),  где beta = m * lambda / d.
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
        beta = m_val * wl / d_
        sin_phi = (sin_a ** 2 - beta ** 2) / (2.0 * beta * cos_a)
        valid = np.abs(sin_phi) <= 1.0
        if np.any(valid):
            phi_deg = np.arcsin(sin_phi[valid]) / DEG
            lines[m_val] = (phi_deg, alphas[valid])
    return lines


def compute_fluorescence_map_2d(alphas, phis, p=PARAMS, nx=35, nz=35,
                                mu_att=0.0, method='analytical', region='grating'):
    """
    Расчет 2D карты интенсивности флуоресценции I(phi, alpha) в p-поляризации.
    """
    alphas = np.asarray(alphas, dtype=float)
    phis = np.asarray(phis, dtype=float)
    map_I = np.zeros((len(alphas), len(phis)), dtype=float)

    d_ = p['d']
    t_ = p['t']
    g_ = p['gamma']
    hw = g_ * d_ / 2

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
            if method == 'analytical':
                map_I[i, j] = compute_fluorescence_analytical(sol, p=p, mu_att=mu_att)
            elif method == 'grid':
                field = compute_field(sol, x, z)
                I_field = np.abs(field) ** 2
                map_I[i, j] = compute_fluorescence_from_field(I_field, x, z, p=p, mu_att=mu_att, region=region)
            else:
                raise ValueError(f"Неизвестный метод расчета: {method}")

    return map_I


def plot_fluorescence_map_2d(alphas, phis, map_I, p=PARAMS,
                             overlay_lines=True, m_orders=(-2, -1, 1, 2),
                             cmap='turbo', norm=None, two_panel=True,
                             line_color='white'):
    """
    Построение 2D тепловой карты интенсивности флуоресценции I(phi, alpha)
    для p-поляризации в палитре turbo с наложением резонансных линий Рэлея.
    """
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    extent = [phis[0], phis[-1], alphas[0], alphas[-1]]

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

        # Панель 1: чистая 2D карта (без линий)
        im1 = ax1.imshow(map_I, extent=extent, origin='lower', aspect='auto',
                         cmap=cmap, norm=norm, interpolation='bilinear')
        ax1.set_xlabel(r'Азимутальный угол $\phi$, градусы', fontsize=11)
        ax1.set_ylabel(r'Угол падения $\alpha$, градусы', fontsize=11)
        ax1.set_title(r'(a) Моделирование 2D карты GIXRF (p-поляризация)', fontsize=12)
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label(r'Интенсивность флуоресценции, произв. ед.', fontsize=10)

        # Панель 2: карта с наложением резонансных линий Рэлея
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
        ax2.set_title(r'(b) Резонансные линии Рэлея-Вуда (p-поляризация)', fontsize=12)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label(r'Интенсивность флуоресценции, произв. ед.', fontsize=10)

        fig.suptitle(f"2D карта интенсивности флуоресценции (p-поляризация, палитра {cmap})\n"
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
            title = (f"2D карта флуоресценции с линиями Рэлея (p-поляризация, {cmap})\n"
                     f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$")
        else:
            title = (f"2D карта интенсивности флуоресценции (p-поляризация, {cmap})\n"
                     f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$")

        ax.set_xlabel(r'Азимутальный угол $\phi$, градусы', fontsize=11)
        ax.set_ylabel(r'Угол падения $\alpha$, градусы', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(phis[0], phis[-1])
        ax.set_ylim(alphas[0], alphas[-1])

    fig.tight_layout()
    return fig


# =============== PLOTS FOR PIPELINE OUTPUTS ===============
def plot_fourier_check(p=PARAMS):
    """Проверка сходимости рядов Фурье для ε(x) и η(x) = 1/ε(x) (правила Ли, 1996)."""
    orders = p['m']
    orders_ext = np.arange(-2 * int(orders.max()), 2 * int(orders.max()) + 1)
    d_ = p['d']

    eps_all = fourier_coefficients_eps(p['eps_1'], p['eps_b'], p['gamma'], d_, orders_ext)
    eta_all = fourier_coefficients_eta(p['eps_1'], p['eps_b'], p['gamma'], d_, orders_ext)

    x = np.linspace(-d_ / 2, d_ / 2, 1001)
    eps_real_step = eps_profile(x, p['eps_1'], p['eps_b'], p['gamma'], d_)
    eta_real_step = eta_profile(x, p['eps_1'], p['eps_b'], p['gamma'], d_)

    mask_orders = np.isin(orders_ext, orders)
    eps_rec = reconstruct_fourier(x, orders, eps_all[mask_orders], d_)
    eta_rec = reconstruct_fourier(x, orders, eta_all[mask_orders], d_)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # Re ε(x)
    ax[0, 0].plot(x, eps_real_step.real, 'k-', lw=1.5, label='Аналитический ступеньчатый')
    ax[0, 0].plot(x, eps_rec.real, 'r--', lw=1.2, label=f'Фурье (D={len(orders)})')
    ax[0, 0].set_ylabel(r'Re $\varepsilon(x)$')
    ax[0, 0].set_title(r'Диэлектрическая проницаемость Re $\varepsilon(x)$')
    ax[0, 0].grid(True, alpha=0.5)
    ax[0, 0].legend(loc='lower center', fontsize=9)

    # Im ε(x)
    ax[0, 1].plot(x, eps_real_step.imag, 'k-', lw=1.5, label='Аналитический')
    ax[0, 1].plot(x, eps_rec.imag, 'r--', lw=1.2, label='Фурье')
    ax[0, 1].set_ylabel(r'Im $\varepsilon(x)$')
    ax[0, 1].set_title(r'Поглощение Im $\varepsilon(x)$')
    ax[0, 1].grid(True, alpha=0.5)

    # Re η(x) = 1/ε(x)
    ax[1, 0].plot(x, eta_real_step.real, 'k-', lw=1.5, label='Аналитический')
    ax[1, 0].plot(x, eta_rec.real, 'b--', lw=1.2, label=f'Фурье (обратное правило Ли)')
    ax[1, 0].set_ylabel(r'Re $\eta(x) = \operatorname{Re}(1/\varepsilon(x))$')
    ax[1, 0].set_xlabel('x, нм')
    ax[1, 0].set_title(r'Обратная проницаемость Re $1/\varepsilon(x)$ (правило Ли)')
    ax[1, 0].grid(True, alpha=0.5)
    ax[1, 0].legend(loc='upper center', fontsize=9)

    # Im η(x)
    ax[1, 1].plot(x, eta_real_step.imag, 'k-', lw=1.5, label='Аналитический')
    ax[1, 1].plot(x, eta_rec.imag, 'b--', lw=1.2, label='Фурье')
    ax[1, 1].set_ylabel(r'Im $\eta(x) = \operatorname{Im}(1/\varepsilon(x))$')
    ax[1, 1].set_xlabel('x, нм')
    ax[1, 1].set_title(r'Im $1/\varepsilon(x)$')
    ax[1, 1].grid(True, alpha=0.5)

    fig.suptitle(f"Ряды Фурье для p-поляризации: правила Ли (Li 1996)\n"
                 f"$d = {d_:.0f}$ нм, $\\Gamma = {p['gamma']}$, число гармоник $D = {len(orders)}$", fontsize=12)
    fig.tight_layout()
    return fig


def plot_kz_vs_alpha(alphas, phi, p=PARAMS):
    """Дисперсионные кривые k_zn(alpha) для мод в структурированном слое (p-поляризация)."""
    kz_all = []
    for a in alphas:
        L = layer_wavevectors(a, phi, **p)
        kz_all.append(L['kz_lay'])
    kz_all = np.array(kz_all)

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(alphas, kz_all.real, lw=0.9)
    ax[0].set_ylabel(r'Re $k_{zn}$, нм$^{-1}$', fontsize=11)
    ax[0].set_title(f'Дисперсионные кривые $k_{{zn}}(\\alpha)$ в решетке (p-поляризация), $\\phi = {phi}^\\circ$', fontsize=12)
    ax[0].grid(True, linestyle='--', alpha=0.5)

    ax[1].plot(alphas, kz_all.imag, lw=0.9)
    ax[1].set_ylabel(r'Im $k_{zn}$, нм$^{-1}$', fontsize=11)
    ax[1].set_xlabel(r'Угол падения $\alpha$, градусы', fontsize=11)
    ax[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    return fig


def plot_eigenvectors(sol):
    """2D карта матрицы амплитуд собственных векторов |E_hn| (p-поляризация)."""
    E_mag = np.abs(sol['E_lay'])
    mm = sol['m']
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(E_mag, aspect='auto', cmap='viridis', interpolation='none',
                   extent=[-0.5, len(mm) - 0.5, mm[-1] + 0.5, mm[0] - 0.5], vmin=0, vmax=1)
    ax.set_xlabel('Номер моды n', fontsize=11)
    ax.set_ylabel('Дифракционный порядок h', fontsize=11)
    ax.set_title(f"Собственные векторы $|E_{{hn}}|$ (p-поляризация)\n"
                 f"$\\alpha = {sol['alpha']}^\\circ$, $\\phi = {sol['phi']}^\\circ$", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Нормированная амплитуда')
    fig.tight_layout()
    return fig


def plot_near_field(sol, p=PARAMS, nx=400, nz=300, pad=20.0, cmap='gnuplot2'):
    """Распределение энергии ближнего поля |E|^2 = |Ex|^2 + |Ez|^2 в одном периоде структуры."""
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    x = np.linspace(-d_ / 2, d_ / 2, nx)
    z = np.linspace(-pad, t_ + pad, nz)
    I_field = np.abs(compute_field(sol, x, z)) ** 2
    height = t_ - z  # высота над границей подложки

    fig, ax = plt.subplots(figsize=(8, 7.5))
    data, vmin, vmax = I_field, 0.0, np.percentile(I_field, 99.5)
    im = ax.imshow(data, extent=[x[0], x[-1], height[-1], height[0]], vmin=vmin, vmax=vmax,
                   aspect='auto', cmap=cmap, origin='upper', interpolation='bilinear')

    # Контур штриха нанорешетки
    hw = g_ * d_ / 2
    ax.plot([-hw, hw], [t_, t_], 'w-', lw=2)
    ax.plot([-hw, -hw], [0, t_], 'w-', lw=2)
    ax.plot([hw, hw], [0, t_], 'w-', lw=2)
    ax.plot([-d_ / 2, -hw], [0, 0], 'w-', lw=2)
    ax.plot([hw, d_ / 2], [0, 0], 'w-', lw=2)

    ax.set_xlabel('x, нм', fontsize=11)
    ax.set_ylabel('Высота над подложкой, нм', fontsize=11)
    ax.set_title(f"Ближнее поле $|E|^2$ (p-поляризация, правила Ли)\n"
                 f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$, $\\alpha = {sol['alpha']}^\\circ$, $\\phi = {sol['phi']}^\\circ$", fontsize=12)
    fig.colorbar(im, ax=ax, label=r'Плотность энергии поля $|E|^2 / |E_{\mathrm{inc}}|^2$')
    ax.set_xlim(-d_ / 2, d_ / 2)
    ax.set_ylim(height[-1], height[0])
    fig.tight_layout()
    return fig


def plot_fluorescence_vs_alpha(alphas, I_fluor, phi=0.0, p=PARAMS):
    """1D кривая флуоресценции I_fluor(alpha) для p-поляризации."""
    d_, t_, g_ = p['d'], p['t'], p['gamma']
    fig, ax = plt.subplots(figsize=(8, 5))

    line_fl, = ax.plot(alphas, I_fluor, 'r-', lw=2.0, label=r'Флуоресценция $I_{\mathrm{fluor}}(\alpha)$ (p-pol)')
    ax.set_xlabel(r'Угол падения $\alpha$, градусы', fontsize=11)
    ax.set_ylabel(r'Интенсивность флуоресценции, произв. ед.', fontsize=11)
    ax.set_xlim(alphas[0], alphas[-1])
    ax.grid(True, linestyle='--', alpha=0.5)

    title = (f"Интенсивность флуоресценции $I_{{\\mathrm{{fluor}}}}(\\alpha)$ (p-поляризация)\n"
             f"$d = {d_:.0f}$ нм, $t = {t_:.0f}$ нм, $\\Gamma = {g_}$, $\\phi = {phi:.1f}^\\circ$")
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', framealpha=0.85)

    fig.tight_layout()
    return fig


# =============== MAIN PIPELINE ===============
def main(save_dir=None):
    print("=== Запуск расчета RCWA для p-поляризации ===")
    sol = solve_grating(alpha_, phi_, **PARAMS)

    figs = {}
    print("1. Построение проверки рядов Фурье...")
    figs['fourier'] = plot_fourier_check()

    print("2. Построение дисперсионных кривых k_zn(alpha)...")
    figs['kz_alpha'] = plot_kz_vs_alpha(np.linspace(0.05, 20, 200), phi_)

    print("3. Построение карты собственных векторов |E_hn|...")
    figs['eigenvectors'] = plot_eigenvectors(sol)

    print("4. Расчет и построение ближнего поля |E|^2...")
    figs['near_field'] = plot_near_field(sol)

    print("5. Расчет 1D кривой флуоресценции от alpha in (0, 20]...")
    alphas_fl = np.linspace(0.05, 20.0, 200)
    alphas_fl, I_fl = fluorescence_vs_alpha(alphas_fl, phi_, p=PARAMS, method='analytical', return_refl=False)
    figs['fluorescence'] = plot_fluorescence_vs_alpha(alphas_fl, I_fl, phi_, p=PARAMS)

    print("6. Расчет 2D карты флуоресценции I(phi, alpha)...")
    alphas_map = np.linspace(0.05, 3.0, 200)
    phis_map = np.linspace(-1.5, 1.5, 51)
    map_I = compute_fluorescence_map_2d(alphas_map, phis_map, p=PARAMS, method='analytical')

    # Чистая 2D карта (без линий)
    figs['fluorescence_map_2d_clean'] = plot_fluorescence_map_2d(
        alphas_map, phis_map, map_I, p=PARAMS,
        overlay_lines=False, two_panel=False, cmap='turbo'
    )

    # 2D карта с наложением резонансных линий
    figs['fluorescence_map_2d_lines'] = plot_fluorescence_map_2d(
        alphas_map, phis_map, map_I, p=PARAMS,
        overlay_lines=True, two_panel=False, cmap='turbo'
    )

    # Двухпанельная 2D карта
    figs['fluorescence_map_2d'] = plot_fluorescence_map_2d(
        alphas_map, phis_map, map_I, p=PARAMS,
        overlay_lines=True, two_panel=True, cmap='turbo'
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            out_file = os.path.join(save_dir, name + '.png')
            fig.savefig(out_file, dpi=120)
        print(f"Все графики для p-поляризации успешно сохранены в: {save_dir}")
    else:
        plt.show()


if __name__ == '__main__':
    save_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'figures')
    main(save_path)
