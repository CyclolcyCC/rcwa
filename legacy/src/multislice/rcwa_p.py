"""
Решатель RCWA для многослойных периодических структур (p-поляризация / TM).
Применяет правила факторизации Ли (Li, 1996) для каждого горизонтального среза.
Поддерживает произвольное число слоев N >= 1, наслоенных вдоль оси z.
"""

import numpy as np
from scipy import linalg
from .rcwa_s import choose_branch, incident_wavevector, lateral_wavevectors, _int_exp


def solve_multislice_p(structure, alpha, phi, K, m):
    """
    Решение СЛАУ граничных условий для многослойной структуры (p-поляризация).
    Учитывает непрерывность u (H_y) и (1/eps) du/dz через матрицу Z = [1/eps].
    Размер СЛАУ: 2(N+1)D x 2(N+1)D.
    """
    N = structure.n_layers
    d = structure.period
    D = len(m)
    eps_sub = 1.0 + structure.chi_sub
    i0 = int(np.flatnonzero(m == 0)[0])

    Kx, Ky, Kz = incident_wavevector(K, alpha, phi)
    kx, kp2 = lateral_wavevectors(Kx, Ky, m, d)
    kz_vac = choose_branch(K ** 2 - kp2)
    kz_sub = choose_branch(eps_sub * K ** 2 - kp2)
    Kx_diag = np.diag(kx)

    layer_data = []
    z_bounds = structure.layer_z_boundaries()

    # 1. Решение задачи на собственные значения для каждого слоя по правилам Ли
    for j in range(N):
        h_j = structure.layers[j].thickness
        eps_mat, Z_mat = structure.fourier_matrices_layer_p(j, m)

        bracket = K ** 2 * np.eye(D, dtype=complex) - Kx_diag @ linalg.inv(eps_mat) @ Kx_diag
        A_j = linalg.inv(Z_mat) @ bracket - (Ky ** 2) * np.eye(D, dtype=complex)
        eig_j, E_j = linalg.eig(A_j)
        kz_j = choose_branch(eig_j)
        q_j = np.exp(1j * kz_j * h_j)
        P_j = Z_mat @ (E_j * kz_j[None, :])

        layer_data.append({
            'E': E_j, 'kz': kz_j, 'q': q_j, 'P': P_j,
            'eps_mat': eps_mat, 'Z_mat': Z_mat,
            'h': h_j, 'z_top': z_bounds[j], 'z_bot': z_bounds[j + 1],
            'layer_def': structure.layers[j]
        })

    # 2. Сборка глобальной блочной СЛАУ размера 2(N+1)D
    M_size = 2 * (N + 1) * D
    M_mat = np.zeros((M_size, M_size), dtype=complex)
    b_vec = np.zeros(M_size, dtype=complex)

    I = np.eye(D, dtype=complex)
    K_vac = np.diag(kz_vac)
    K_sub = np.diag(kz_sub / eps_sub)

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

        # Непрерывность поля H_y
        M_mat[row_f:row_f + D, col_Tj:col_Tj + D] = layer_data[j]['E'] * layer_data[j]['q'][None, :]
        M_mat[row_f:row_f + D, col_Rj:col_Rj + D] = layer_data[j]['E']
        M_mat[row_f:row_f + D, col_Tj1:col_Tj1 + D] = -layer_data[j + 1]['E']
        M_mat[row_f:row_f + D, col_Rj1:col_Rj1 + D] = -layer_data[j + 1]['E'] * layer_data[j + 1]['q'][None, :]

        # Непрерывность (1/eps) dH_y/dz
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
    trans = np.abs(T_sub) ** 2 * (kz_sub / eps_sub).real / Kz

    return {
        'structure': structure,
        'alpha': alpha, 'phi': phi, 'K': K, 'm': m, 'i0': i0,
        'Kx': Kx, 'Ky': Ky, 'Kz': Kz, 'kx': kx,
        'kz_vac': kz_vac, 'kz_sub': kz_sub, 'eps_sub': eps_sub,
        'R_vac': R_vac, 'T_sub': T_sub,
        'refl': refl, 'trans': trans,
        'layers': layer_data
    }


def compute_field_p(sol, x, z):
    """
    Расчет комплексного поля E(x, z) для p-поляризации на 2D-сетке (x, z).
    Возвращает матрицу интенсивности |E(x, z)|^2 = |E_x|^2 + |E_z|^2.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    Nz, Nx = len(z), len(x)
    m = sol['m']
    D = len(m)
    K = sol['K']
    kx = sol['kx']

    Ex_grid = np.zeros((Nz, Nx), dtype=complex)
    Ez_grid = np.zeros((Nz, Nx), dtype=complex)

    phase_x = np.exp(1j * np.outer(kx, x))
    H = sol['structure'].total_thickness

    # Вакуум (z < 0)
    vac_mask = z < 0.0
    if np.any(vac_mask):
        zv = z[vac_mask][:, None]
        inc = np.zeros(D, dtype=complex)
        inc[sol['i0']] = 1.0
        # H_y
        Hy_vac = inc[None, :] * np.exp(1j * sol['Kz'] * zv) + sol['R_vac'][None, :] * np.exp(-1j * sol['kz_vac'][None, :] * zv)
        # dHy/dz
        dHy_dz = 1j * sol['Kz'] * inc[None, :] * np.exp(1j * sol['Kz'] * zv) - 1j * sol['kz_vac'][None, :] * sol['R_vac'][None, :] * np.exp(-1j * sol['kz_vac'][None, :] * zv)
        # Ex = i / K * dHy/dz, Ez = -1 / K * kx * Hy
        Ex_fourier = (1j / K) * dHy_dz
        Ez_fourier = (-1.0 / K) * (kx[None, :] * Hy_vac)
        Ex_grid[vac_mask] = Ex_fourier @ phase_x
        Ez_grid[vac_mask] = Ez_fourier @ phase_x

    # Подложка (z > H)
    sub_mask = z > H
    if np.any(sub_mask):
        zs = z[sub_mask][:, None]
        eps_s = sol['eps_sub']
        Hy_sub = sol['T_sub'][None, :] * np.exp(1j * sol['kz_sub'][None, :] * (zs - H))
        dHy_dz = 1j * sol['kz_sub'][None, :] * Hy_sub
        Ex_fourier = (1j / (K * eps_s)) * dHy_dz
        Ez_fourier = (-1.0 / (K * eps_s)) * (kx[None, :] * Hy_sub)
        Ex_grid[sub_mask] = Ex_fourier @ phase_x
        Ez_grid[sub_mask] = Ez_fourier @ phase_x

    # Внутри каждого слоя
    for ldata in sol['layers']:
        z_top = ldata['z_top']
        z_bot = ldata['z_bot']
        lay_mask = (z >= z_top) & (z <= z_bot)
        if np.any(lay_mask):
            zl = z[lay_mask][:, None]
            kz = ldata['kz'][None, :]
            T = ldata['T'][None, :]
            R = ldata['R'][None, :]
            E = ldata['E']
            Z_mat = ldata['Z_mat']

            a_z = T * np.exp(1j * kz * (zl - z_top)) + R * np.exp(-1j * kz * (zl - z_bot))
            da_dz = 1j * kz * (T * np.exp(1j * kz * (zl - z_top)) - R * np.exp(-1j * kz * (zl - z_bot)))

            Hy_fourier = a_z @ E.T
            dHy_fourier = da_dz @ E.T

            # Правила Ли: Ex = (i / K) * Z @ dHy, Ez = (-1 / K) * Z @ (kx * Hy)
            Ex_fourier = (1j / K) * (dHy_fourier @ Z_mat.T)
            Ez_fourier = (-1.0 / K) * ((Hy_fourier * kx[None, :]) @ Z_mat.T)

            Ex_grid[lay_mask] = Ex_fourier @ phase_x
            Ez_grid[lay_mask] = Ez_fourier @ phase_x

    field_intensity = np.abs(Ex_grid) ** 2 + np.abs(Ez_grid) ** 2
    return field_intensity


def compute_fluorescence_analytical_p(sol, mu_att=0.0):
    """
    Полуаналитический расчет интенсивности флуоресценции для p-поляризации
    (суммирование по всем флуоресцентным блокам во всех слоях структуры).
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

        for b in layer_def.boxes:
            if not b.is_fluorescent:
                continue
            delta_x = b.width
            mid_x = b.center
            phase = np.exp(1j * diff * 2.0 * np.pi * mid_x / d)
            U = delta_x * np.sinc(diff * delta_x / d) * phase
            total_fluor += np.sum(U * M).real

    return float(total_fluor)


def compute_fluorescence_numerical_p(sol, nx=80, nz=40, mu_att=0.0):
    """
    Численное 2D-интегрирование интенсивности волновой функции TM моды |Hy(x, z)|^2
    по флуоресцентным блокам структуры для p-поляризации методом трапеций.
    """
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    total_fluor = 0.0
    kx = sol['kx']

    for ldata in sol['layers']:
        layer_def = ldata['layer_def']
        if not any(b.is_fluorescent for b in layer_def.boxes):
            continue
        z_top = ldata['z_top']
        z_bot = ldata['z_bot']
        z_grid = np.linspace(z_top, z_bot, nz)
        zl = z_grid[:, None]
        kz = ldata['kz'][None, :]
        a_z = ldata['T'][None, :] * np.exp(1j * kz * (zl - z_top)) + ldata['R'][None, :] * np.exp(-1j * kz * (zl - z_bot))
        Hy_fourier = a_z @ ldata['E'].T

        for b in layer_def.boxes:
            if not b.is_fluorescent:
                continue
            x_grid = np.linspace(b.left, b.right, nx)
            Hy_grid = Hy_fourier @ np.exp(1j * np.outer(kx, x_grid))
            I_field = np.abs(Hy_grid) ** 2
            if mu_att != 0.0:
                I_field = I_field * np.exp(-mu_att * z_grid[:, None])
            total_fluor += trapz(trapz(I_field, x_grid, axis=1), z_grid, axis=0)

    return float(total_fluor)

