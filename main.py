import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import fft
from scipy import linalg
import cmocean
import warnings


DEG = np.pi / 180.0

# =============== PARAMETERS ===============
# структура
d = 100.0                 # период решетки, нм
t = 50.0                  # высота штрихов, нм
gamma = 0.4               # Γ = d_a / d — доля периода
chi = -0.036 + 0.0009j    # χ материала штрихов
chi_b = 0.0               # χ между штрихами
chi_sub = chi             # χ подложки
# фотоны
ev = 500.0
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
    ax.set_xlabel('mode n (sorted by Re k_zn)')
    ax.set_ylabel('diffraction order m')
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
    figs['near_field_a3_phi1'] = plot_near_field(solve_grating(3.0, 1.0, **PARAMS))  # второй случай: α = 3°, φ = 1°

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, name + '.png'), dpi=120)
        print('figures saved to', save_dir)
    else:
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
