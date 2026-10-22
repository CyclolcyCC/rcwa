"""
Решатель RCWA для многослойных периодических структур (s-поляризация / TE).
Поддерживает произвольное число слоев N >= 1, наслоенных вдоль оси z.
"""

import numpy as np
from scipy import linalg


def choose_branch(kz2):
    """Выбор физической ветви волнового числа: Re(kz) > 0 или (Re(kz) == 0 и Im(kz) >= 0)."""
    kz2 = np.asarray(kz2, dtype=complex)
    root = np.sqrt(kz2)
    re = root.real
    im = root.imag
    mask = (re < -1e-12) | ((np.abs(re) <= 1e-12) & (im < 0.0))
    root[mask] = -root[mask]
    return root


def incident_wavevector(K, alpha, phi):
    """
    Вектор падающей волны K_inc = (Kx, Ky, Kz) в лабораторной СК:
    Kx = K * cos(alpha) * sin(phi)
    Ky = K * cos(alpha) * cos(phi)
    Kz = K * sin(alpha)
    """
    a = np.radians(alpha)
    p = np.radians(phi)
    return (K * np.cos(a) * np.sin(p),
            K * np.cos(a) * np.cos(p),
            K * np.sin(a))


def lateral_wavevectors(Kx, Ky, m, d):
    """Латеральные компоненты k_{x,m} и k_{rho,m}^2 = k_{x,m}^2 + Ky^2."""
    kx = Kx + 2.0 * np.pi * m / d
    kp2 = kx ** 2 + Ky ** 2
    return kx, kp2


def solve_multislice_s(structure, alpha, phi, K, m):
    """
    Решение СЛАУ граничных условий для многослойной структуры (s-поляризация).
    Размер СЛАУ: 2(N+1)D x 2(N+1)D.
    """
    N = structure.n_layers
    d = structure.period
    D = len(m)
    i0 = int(np.flatnonzero(m == 0)[0])

    Kx, Ky, Kz = incident_wavevector(K, alpha, phi)
    kx, kp2 = lateral_wavevectors(Kx, Ky, m, d)
    kz_vac = choose_branch(K ** 2 - kp2)
    kz_sub = choose_branch(K ** 2 * (1.0 + structure.chi_sub) - kp2)

    # 1. Решение задачи на собственные значения для каждого слоя
    layer_data = []
    z_bounds = structure.layer_z_boundaries()

    for j in range(N):
        h_j = structure.layers[j].thickness
        chi_m_all = structure.fourier_coefficients_layer(j, np.arange(-2 * m.max(), 2 * m.max() + 1))
        diff = m[:, None] - m[None, :]
        C_j = chi_m_all[diff + 2 * m.max()]

        A_j = K ** 2 * C_j + np.diag(K ** 2 - kp2)
        eig_j, E_j = linalg.eig(A_j)
        kz_j = choose_branch(eig_j)
        q_j = np.exp(1j * kz_j * h_j)
        P_j = E_j * kz_j[None, :]

        layer_data.append({
            'E': E_j, 'kz': kz_j, 'q': q_j, 'P': P_j,
            'h': h_j, 'z_top': z_bounds[j], 'z_bot': z_bounds[j + 1],
            'layer_def': structure.layers[j]
        })

    # 2. Сборка глобальной блочной СЛАУ размера 2(N+1)D
    M_size = 2 * (N + 1) * D
    M_mat = np.zeros((M_size, M_size), dtype=complex)
    b_vec = np.zeros(M_size, dtype=complex)

    I = np.eye(D, dtype=complex)
    K_vac = np.diag(kz_vac)
    K_sub = np.diag(kz_sub)

    # Граница 0 (вакуум / слой 0)
    M_mat[0:D, 0:D] = -I
    M_mat[0:D, D:2 * D] = layer_data[0]['E']
    M_mat[0:D, 2 * D:3 * D] = layer_data[0]['E'] * layer_data[0]['q'][None, :]
    b_vec[i0] = 1.0

    M_mat[D:2 * D, 0:D] = K_vac
    M_mat[D:2 * D, D:2 * D] = layer_data[0]['P']
    M_mat[D:2 * D, 2 * D:3 * D] = -layer_data[0]['P'] * layer_data[0]['q'][None, :]
    b_vec[D + i0] = kz_vac[i0]

    # Промежуточные границы слой j -> j+1
    for j in range(N - 1):
        row_f = 2 * (j + 1) * D
        row_d = (2 * (j + 1) + 1) * D
        col_Tj = (2 * j + 1) * D
        col_Rj = (2 * j + 2) * D
        col_Tj1 = (2 * (j + 1) + 1) * D
        col_Rj1 = (2 * (j + 1) + 2) * D

        # Непрерывность поля
        M_mat[row_f:row_f + D, col_Tj:col_Tj + D] = layer_data[j]['E'] * layer_data[j]['q'][None, :]
        M_mat[row_f:row_f + D, col_Rj:col_Rj + D] = layer_data[j]['E']
        M_mat[row_f:row_f + D, col_Tj1:col_Tj1 + D] = -layer_data[j + 1]['E']
        M_mat[row_f:row_f + D, col_Rj1:col_Rj1 + D] = -layer_data[j + 1]['E'] * layer_data[j + 1]['q'][None, :]

        # Непрерывность производной
        M_mat[row_d:row_d + D, col_Tj:col_Tj + D] = layer_data[j]['P'] * layer_data[j]['q'][None, :]
        M_mat[row_d:row_d + D, col_Rj:col_Rj + D] = -layer_data[j]['P']
        M_mat[row_d:row_d + D, col_Tj1:col_Tj1 + D] = -layer_data[j + 1]['P']
        M_mat[row_d:row_d + D, col_Rj1:col_Rj1 + D] = layer_data[j + 1]['P'] * layer_data[j + 1]['q'][None, :]

    # Граница N-1 -> подложка
    row_sub_f = 2 * N * D
    row_sub_d = (2 * N + 1) * D
    col_TN = (2 * (N - 1) + 1) * D
    col_RN = (2 * (N - 1) + 2) * D
    col_Tsub = (2 * N + 1) * D

    M_mat[row_sub_f:row_sub_f + D, col_TN:col_TN + D] = layer_data[-1]['E'] * layer_data[-1]['q'][None, :]
    M_mat[row_sub_f:row_sub_f + D, col_RN:col_RN + D] = layer_data[-1]['E']
    M_mat[row_sub_f:row_sub_f + D, col_Tsub:col_Tsub + D] = -I

    M_mat[row_sub_d:row_sub_d + D, col_TN:col_TN + D] = layer_data[-1]['P'] * layer_data[-1]['q'][None, :]
    M_mat[row_sub_d:row_sub_d + D, col_RN:col_RN + D] = -layer_data[-1]['P']
    M_mat[row_sub_d:row_sub_d + D, col_Tsub:col_Tsub + D] = -K_sub

    # 3. Решение СЛАУ
    u = linalg.solve(M_mat, b_vec)
    R_vac = u[:D]
    T_sub = u[col_Tsub:col_Tsub + D]

    for j in range(N):
        col_T = (2 * j + 1) * D
        col_R = (2 * j + 2) * D
        layer_data[j]['T'] = u[col_T:col_T + D]
        layer_data[j]['R'] = u[col_R:col_R + D]

    refl = np.abs(R_vac) ** 2 * kz_vac.real / Kz
    trans = np.abs(T_sub) ** 2 * kz_sub.real / Kz

    return {
        'structure': structure,
        'alpha': alpha, 'phi': phi, 'K': K, 'm': m, 'i0': i0,
        'Kx': Kx, 'Ky': Ky, 'Kz': Kz, 'kx': kx,
        'kz_vac': kz_vac, 'kz_sub': kz_sub,
        'R_vac': R_vac, 'T_sub': T_sub,
        'refl': refl, 'trans': trans,
        'layers': layer_data
    }


def compute_field_s(sol, x, z):
    """
    Расчет комплексного поля E_y(x, z) на 2D-сетке (x, z).
    z < 0: вакуум, 0 <= z <= H: слои структуры, z > H: подложка.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    Nz, Nx = len(z), len(x)
    C = np.zeros((Nz, len(sol['m'])), dtype=complex)

    H = sol['structure'].total_thickness

    # Вакуум (z < 0)
    vac_mask = z < 0.0
    if np.any(vac_mask):
        zv = z[vac_mask][:, None]
        inc = np.zeros(len(sol['m']), dtype=complex)
        inc[sol['i0']] = 1.0
        # Падающая волна exp(i Kz z) + отраженные волны R_h exp(-i kz_vac z)
        C[vac_mask] = (inc[None, :] * np.exp(1j * sol['Kz'] * zv) +
                       sol['R_vac'][None, :] * np.exp(-1j * sol['kz_vac'][None, :] * zv))

    # Подложка (z > H)
    sub_mask = z > H
    if np.any(sub_mask):
        zs = z[sub_mask][:, None]
        C[sub_mask] = sol['T_sub'][None, :] * np.exp(1j * sol['kz_sub'][None, :] * (zs - H))

    # Внутри каждого слоя
    for ldata in sol['layers']:
        z_top = ldata['z_top']
        z_bot = ldata['z_bot']
        lay_mask = (z >= z_top) & (z <= z_bot)
        if np.any(lay_mask):
            zl = z[lay_mask][:, None]
            h = ldata['h']
            kz = ldata['kz'][None, :]
            T = ldata['T'][None, :]
            R = ldata['R'][None, :]
            E = ldata['E']

            # Моды a_n(z) = T_n exp(i kz (z - z_top)) + R_n exp(-i kz (z - z_bot))
            a_z = T * np.exp(1j * kz * (zl - z_top)) + R * np.exp(-1j * kz * (zl - z_bot))
            C[lay_mask] = a_z @ E.T

    phase_x = np.exp(1j * np.outer(sol['kx'], x))
    field = C @ phase_x
    return field


def _int_exp(kappa, t_val):
    """Аналитический интеграл ∫_0^t exp(kappa * z) dz с регуляризацией при kappa -> 0."""
    res = np.empty_like(kappa, dtype=complex)
    small = np.abs(kappa * t_val) < 1e-8
    res[small] = t_val * (1.0 + 0.5 * kappa[small] * t_val)
    res[~small] = (np.exp(kappa[~small] * t_val) - 1.0) / kappa[~small]
    return res


def compute_fluorescence_analytical_s(sol, mu_att=0.0):
    """
    Полуаналитический расчет интенсивности флуоресценции для s-поляризации
    по формуле Шермана-Николаева (суммирование по всем флуоресцентным блокам во всех слоях).
    """
    total_fluor = 0.0
    d = sol['structure'].period
    m = sol['m']
    diff = m[:, None] - m[None, :]

    for ldata in sol['layers']:
        layer_def = ldata['layer_def']
        h = ldata['h']
        z_top = ldata['z_top']
        T = ldata['T']
        R = ldata['R']
        kz = ldata['kz']
        E = ldata['E']

        kz_m_conj = np.conj(kz[:, None])
        kz_n = kz[None, :]
        T_m_conj = np.conj(T[:, None])
        T_n = T[None, :]
        R_m_conj = np.conj(R[:, None])
        R_n = R[None, :]

        # 4 слагаемых произведения мод
        k1 = 1j * (kz_n - kz_m_conj) - mu_att
        W1 = T_m_conj * T_n * _int_exp(k1, h)

        k2 = 1j * (-kz_n - kz_m_conj) - mu_att
        W2 = T_m_conj * (R_n * np.exp(1j * kz_n * h)) * _int_exp(k2, h)

        k3 = 1j * (kz_n + kz_m_conj) - mu_att
        W3 = (R_m_conj * np.exp(-1j * kz_m_conj * h)) * T_n * _int_exp(k3, h)

        k4 = 1j * (-kz_n + kz_m_conj) - mu_att
        W4 = (R_m_conj * np.exp(-1j * kz_m_conj * h)) * (R_n * np.exp(1j * kz_n * h)) * _int_exp(k4, h)

        W = (W1 + W2 + W3 + W4) * np.exp(-mu_att * z_top)
        M = np.conj(E) @ W @ E.T

        # Суммирование по флуоресцентным боксам слоя
        for b in layer_def.boxes:
            if not b.is_fluorescent:
                continue
            delta_x = b.width
            mid_x = b.center
            phase = np.exp(1j * diff * 2.0 * np.pi * mid_x / d)
            U = delta_x * np.sinc(diff * delta_x / d) * phase
            total_fluor += np.sum(U * M).real

    return float(total_fluor)


def compute_fluorescence_numerical_s(sol, nx=80, nz=40, mu_att=0.0):
    """
    Численное 2D-интегрирование интенсивности поля |E(x, z)|^2 по флуоресцентным блокам
    структуры для s-поляризации методом трапеций.
    """
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    total_fluor = 0.0

    for ldata in sol['layers']:
        layer_def = ldata['layer_def']
        if not any(b.is_fluorescent for b in layer_def.boxes):
            continue
        z_top = ldata['z_top']
        z_bot = ldata['z_bot']
        z_grid = np.linspace(z_top, z_bot, nz)

        for b in layer_def.boxes:
            if not b.is_fluorescent:
                continue
            x_grid = np.linspace(b.left, b.right, nx)
            E = compute_field_s(sol, x_grid, z_grid)
            I_field = np.abs(E) ** 2
            if mu_att != 0.0:
                I_field = I_field * np.exp(-mu_att * z_grid[:, None])
            total_fluor += trapz(trapz(I_field, x_grid, axis=1), z_grid, axis=0)

    return float(total_fluor)

