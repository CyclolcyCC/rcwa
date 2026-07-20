import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import fft
from scipy import linalg
import cmocean
import warnings

warnings.filterwarnings('ignore')

'''
hw = 500
wl = 1239.8/hw

#Structure
K = 2*np.pi/wl
pitch = 100
number_of_modes = 11
height = 50
gamma = 0.4
sigma = 0
vacuum_offset = 20
substrate_offset = 20
chi = -0.036 + 0.0009j
'''

# ===============PARAMETERS===============
# structure
length = 100
height = 50
gamma = 0.4
# photons
ev = 500
wl = 1239.8 / ev
K = 2 * np.pi / wl
# angles
alpha_ = 6
phi_ = 0
# dielectric susceptibility
chi = -0.036 + 0.0009j
n_max = 2
nmodes = n_max * 2 + 1


# ===============PULSE===============
# fft reconstruction function
def reconstruct_fourier(length, nmodes, frq, f_h, x):
    f_rec = np.zeros(nmodes, dtype=complex)

    for (freq, coeff) in zip(frq, f_h):
        h = 2 * np.pi * freq / length
        f_rec += coeff * np.exp(1j * h * x)

    return f_rec


# linspace
N = 1001  # a lot of dots for fft
x = np.linspace(0, length, N, endpoint=False)
f = np.zeros(N, dtype=complex)
shift = -length // 2
mask_fft = (x >= -shift-(length*(gamma/2))) & (x < -shift+(length*(gamma/2)))
f[mask_fft] = chi

# FFT
f_h_all = fft.fftshift(fft.fft(f) / N)
frq_all = fft.fftshift(fft.fftfreq(N, d=1/N))

# only required harmonics
indices = np.where(np.abs(frq_all) <= n_max)[0]
f_h = f_h_all[indices]
frq = frq_all[indices]

# reconstruction
f_rec = reconstruct_fourier(length, N, frq_all, f_h_all, x)

# visualization and comparison
plt.figure(figsize=(15, 5))
# orig
plt.subplot(1, 3, 1)
plt.plot(x, f.real, 'b-', label='real', alpha=0.7)
plt.plot(x, f.imag, 'r--', label='imag', alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('orig function')
plt.legend()
plt.subplot(1, 3, 2)
plt.legend()
plt.grid(True)
# comparison
# real
plt.subplot(1, 3, 2)
plt.plot(x, f.real, 'b-', label='original', alpha=0.7)
plt.plot(x, f_rec.real, 'r--', label='reconstructed', alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f're() comparison')
plt.legend()
plt.grid(True)
# imaginary
plt.subplot(1, 3, 3)
plt.plot(x, f.imag, 'g-', label='original', alpha=0.7)
plt.plot(x, f_rec.imag, 'y--', label='reconstructed', alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'im() comparison')
plt.legend()
plt.grid(True)
#misc
plt.tight_layout()
plt.show()


# ===============EIGENVALUES===============
# A matrix build function
def build_A_matrix(f_h, h_vector, alpha_i, phi_i):
    # toeplitz with 0 diagonal and 2 n_max + 1 dim
    padding = len(f_h)//2  # сколько нулей добавить с каждой стороны
    f_h_padded = np.pad(f_h, (padding, padding), mode='constant', constant_values=0)

    chi_m = linalg.toeplitz(f_h_padded[n_max+padding::1], f_h_padded[n_max+padding::-1])
    # 1 + chi_0 (only this works)
    for i in range(len(h_vector)):
        chi_m[i][i] += 1
    # k and kappa diagonal matrix
    deg = np.pi / 180
    k_0x = K * np.cos(alpha_i * deg) * np.sin(phi_i * deg)
    k_0y = K * np.cos(alpha_i * deg) * np.cos(phi_i * deg)

    # k_hx and k_hy arrays
    k_hx = k_0x + h_vector
    k_hy = k_0y * np.ones_like(k_hx)
    # kappa array
    kappa = k_hx ** 2 + k_hy ** 2

    # final matrix A
    A = K ** 2 * chi_m - np.diag(kappa)

    return A, kappa


# harmonics space dimension
h_dim = 2 * n_max + 1 # = nmodes
# h vector over all frequencies
h_vector = 2 * np.pi * frq / length

# eigenvalues computation preparation
n = 30
# alpha array
alpha = np.linspace(0.01, n, n * 10) # 0 is not included for numerical stability
# phi = const
phi_i = 0  # coplanar

# kzn all values array
k_z_all_layers = []

# computation cycle
for alpha_i in alpha:
    # eigenvalues problem solution
    A, k_hp = build_A_matrix(f_h, h_vector, alpha_i, phi_i)
    eigenvalues, E_arr = linalg.eig(A)
    # kzn yield
    k_z_arr = np.sqrt(eigenvalues)

    # we set signs of real and imag parts as they have to be
    k_z_arr_real = np.abs(k_z_arr.real)
    k_z_arr_imag = -np.abs(k_z_arr.imag)
    k_z_s = k_z_arr_real + 1j * k_z_arr_imag

    k_z_all_layers.append(k_z_s)

# list -> numpy array
k_z_all_layers = np.array(k_z_all_layers)

# kzn values demonstration
plt.figure(figsize=(10, 5))
# real
plt.subplot(1, 2, 1)
plt.plot(alpha, k_z_all_layers.real, alpha=0.7)
plt.xlabel('alpha')
plt.ylabel('re. kzn')
plt.title(f're() part')
plt.legend()
plt.grid(True)
# imaginary
plt.subplot(1, 2, 2)
plt.plot(alpha, k_z_all_layers.imag, alpha=0.7)
plt.xlabel('alpha')
plt.ylabel('im. kzn')
plt.title(f'im() part')
plt.legend()
plt.grid(True)
#misc
plt.tight_layout()
plt.show()


# ===============BOUNDARY===============
# P matrix build function
def build_P_matrix(E_, kz, homogenous=False):
    # 2D x 2D
    D = len(kz)
    P = np.zeros((2 * D, 2 * D), dtype=complex)

    # homogenous condition
    if homogenous:
        E = np.eye(D)
    else:
        E = E_

    # matrix construction
    P[:D, :D] = E # up-left
    P[:D, D:] = E # up-right
    P[D:, :D] = E @ np.diag(kz) # down-right
    P[D:, D:] = -E @ np.diag(kz) # down-left

    return P

# Q matrix build function
def build_Q_matrix(kz, heigth):
    # 2D x 2D
    D = len(kz)
    Q = np.zeros((2 * D, 2 * D), dtype=complex)

    # matrix construction
    Q[:D, :D] = np.diag(np.exp(1j * kz * height)) # up-left
    Q[D:, D:] = np.diag(np.exp(-1j * kz * height)) # down-right

    return Q

# interface matrix build function
def build_interface_matrix(P_upper, P_lower):
    return linalg.inv(P_upper) @ P_lower

# M matrix build function
def build_M_matrix(kz, P_vac_struct, Q_struct, P_struct_sub):
    D = len(kz)
    # matrix construction
    M = P_vac_struct @ Q_struct @ P_struct_sub

    # blocking
    M11 = M[:D, :D]
    M12 = M[:D, D:]
    M21 = M[D:, :D]
    M22 = M[D:, D:]

    return M11, M12, M21, M22

# intermediate calculations
Kz = K * np.sin(alpha_)
Kp = K * np.cos(alpha_)
# diffraction orders
m_max = n_max
m_values = np.arange(-m_max, m_max + 1)  # [-2, -1, 0, 1, 2]
# eigenvalues and eigenvectors for alpha_ and phi_
A_, k_hp = build_A_matrix(f_h, h_vector, alpha_, phi_)
eigenvalues_, E_ = linalg.eig(A_)


# ===============DISCRETE EIGENVECTOR VISUALIZATION (CHESSBOARD STYLE)===============
def plot_eigenvectors_discrete(E_, m_values, alpha_, phi_, title="Собственные векторы (дискретная карта)"):
    """
    Визуализация собственных векторов в виде дискретной шахматной доски
    Каждая клетка - это амплитуда для конкретного дифракционного порядка и моды
    """
    D = E_.shape[1]  # количество мод

    # Выбираем сколько мод показывать (все или ограничиваем)
    n_modes_to_show = min(D, 15)

    # Берем модуль амплитуд
    E_mag = np.abs(E_[:, :n_modes_to_show]).T  # транспонируем: строки - моды, столбцы - порядки

    # Нормируем каждую строку (моду) для лучшей визуализации
    # Или можно нормировать всю матрицу
    E_mag_norm = E_mag / (np.max(E_mag) + 1e-12)

    # Создаем фигуру
    fig, axes = plt.subplots(1, 1, figsize=(6.5, 6))

    # ===== 1. Тепловая карта с четкими границами (без интерполяции) =====
    ax1 = axes

    # Используем imshow с interpolation='none' для четких пикселей
    im1 = ax1.imshow(E_mag_norm,
                     aspect='auto',
                     cmap='viridis',
                     interpolation='none',  # КЛЮЧЕВОЙ ПАРАМЕТР - отключает интерполяцию
                     extent=[m_values[0] - 0.5, m_values[-1] + 0.5,
                             n_modes_to_show - 0.5, -0.5],
                     vmin=0, vmax=1)

    # Добавляем сетку для четкого разделения клеток
    ax1.set_xticks(range(m_values[0], m_values[-1] + 1))
    ax1.set_yticks(range(n_modes_to_show))
    ax1.set_xticklabels(m_values)
    ax1.set_yticklabels([f'Mode {n}' for n in range(n_modes_to_show)])

    # Рисуем линии сетки поверх карты
    # ax1.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    ax1.set_xlabel('Diffraction order m', fontsize=12)
    ax1.set_ylabel('Mode number n', fontsize=12)
    ax1.set_title(f'|Eigenvector components|\nα={alpha_}°, φ={phi_}°', fontsize=12)

    # Цветовая шкала
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Normalized amplitude')

    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

    return fig


# Вызов функции
plot_eigenvectors_discrete(E_, m_values, alpha_, phi_)

# calculating sqrt of eigenvalues
kz = np.sqrt(eigenvalues_)
# we set signs of real and imag parts as they have to be
kz_real = np.abs(kz.real)
kz_imag = -np.abs(kz.imag)
kzs = kz + 1j * kz

kz = kzs

D = len(kz)
# every layer calculation
# 0 vacuum
kz_vac = np.sqrt(K**2 - k_hp**2 + 0j)
P_vac = build_P_matrix(np.eye(D), kz_vac, True)

# 1 structure
P_struct = build_P_matrix(E_, kz, False)
Q_struct = build_Q_matrix(kz, height)

# 2 substrate
kz_sub = np.sqrt(K**2 *(1 + chi) - k_hp**2 + 0j)
P_sub = build_P_matrix(np.eye(D), kz_sub, True)

# interface matrices
P_vac_struct = build_interface_matrix(P_vac, P_struct) # vac -> struct
P_struct_sub = build_interface_matrix(P_struct, P_sub) # struct -> sub

# blocks of transfer matrix
M11, M12, M21, M22 = build_M_matrix(kz, P_vac_struct, Q_struct, P_struct_sub)

# boundary solution
# vac incident
T_vac = np.zeros(D, dtype=complex)
T_vac[np.argmin(np.abs(m_values))] = 1.0

# sub reflected
R_sub = np.zeros(D, dtype=complex)

# [T_vac; R_vac] = M * [T_sub; R_sub]
# -> T_vac = M11 * T_sub, R_vac = M21 * T_sub

# -> T_sub = inv(M11) * T_vac
T_sub = linalg.solve(M11, T_vac)
# ->  R_vac = M21 * T_sub
R_vac = M21 @ T_sub

# ===============AMPLITUDES===============
# amplitudes inside layer
def get_amplitudes_inside_layer(P_vac_struct, Q_struct, P_struct_sub, M11, T_vac, kz, D):
    # Находим T_sub
    T_sub = linalg.solve(M11, T_vac)

    # На границе с подложкой (z = 0)
    vec_at_bottom = P_struct_sub @ np.concatenate([T_sub, np.zeros(D, dtype=complex)])
    T_bottom = vec_at_bottom[:D]
    R_bottom = vec_at_bottom[D:]

    # Функция для получения амплитуд на глубине z
    def get_at_z(z):
        # Матрица распространения от дна до глубины z
        Q_z = build_Q_matrix(kz, z)
        # Амплитуды на глубине z
        vec_at_z = Q_z @ np.concatenate([T_bottom, R_bottom])
        return vec_at_z[:D], vec_at_z[D:]

    return get_at_z, T_bottom, R_bottom, T_sub

# ===============FIELD COMPUTATION===============
def compute_field_correct(X, Z, height, K, alpha_, R_vac, E_, kz, T_bottom, R_bottom, T_sub, kz_sub, D, m_values, length):
    """
    Правильное вычисление поля с учетом фаз из статьи
    """
    E_field = np.zeros_like(X, dtype=complex)
    deg = np.pi / 180

    # Горизонтальные компоненты для каждого дифракционного порядка
    Kx_h = K * np.cos(alpha_ * deg) + 2 * np.pi / length * m_values
    Kz = K * np.sin(alpha_ * deg)
    kz_vac = np.sqrt(K ** 2 - Kx_h ** 2 + 0j)

    # Индекс нулевого порядка
    zero_idx = np.argmin(np.abs(m_values))

    print(f"Computing field on grid {X.shape[0]} x {X.shape[1]}")
    print(f"D = {D}, modes = {len(m_values)}")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            z = Z[i, j]

            if z > height:
                # ===== ВАКУУМ (z > H) =====
                # Падающая волна
                E_inc = np.exp(1j * (Kx_h[zero_idx] * x + Kz * (z - height)))

                # Отраженные волны
                E_ref = 0
                for idx in range(D):
                    if idx < len(R_vac):
                        E_ref += R_vac[idx] * np.exp(1j * (Kx_h[idx] * x - kz_vac[idx] * (z - height)))

                E_field[i, j] = E_inc + E_ref

            elif z >= 0:
                # ===== СТРУКТУРИРОВАННЫЙ СЛОЙ (0 <= z <= H) =====
                E_layer = 0

                for n in range(D):  # по модам
                    # Амплитуда моды n на глубине z
                    # Используем T_bottom и R_bottom (на z=0)
                    amp_n = T_bottom[n] * np.exp(1j * kz[n] * z) + R_bottom[n] * np.exp(-1j * kz[n] * z)

                    # Суммируем по дифракционным порядкам
                    for m_idx in range(D):
                        if m_idx < E_.shape[0] and n < E_.shape[1]:
                            E_layer += amp_n * E_[m_idx, n] * np.exp(1j * Kx_h[m_idx] * x)

                E_field[i, j] = E_layer

            else:
                # ===== ПОДЛОЖКА (z < 0) =====
                E_sub = 0
                for idx in range(D):
                    if idx < len(T_sub):
                        E_sub += T_sub[idx] * np.exp(1j * (Kx_h[idx] * x + kz_sub[idx] * z))

                E_field[i, j] = E_sub

    return E_field

# ===============GET AMPLITUDES FUNCTION===============
get_T_R, T_bottom, R_bottom, T_sub = get_amplitudes_inside_layer(P_vac_struct, Q_struct, P_struct_sub, M11, T_vac, kz, D)

# ===============CREATE GRID (ОДИН ПЕРИОД)===============
# Показываем только один период: от -length/2 до length/2
x_range = np.linspace(-length / 2, length / 2, 400)  # от -50 до 50 нм
z_range = np.linspace(-20, height + 20, 300)  # от -10 до 65 нм

X, Z = np.meshgrid(x_range, z_range)

print("Вычисление поля...")
print(f"  Сетка: {X.shape[0]} x {X.shape[1]} точек")
print(f"  x: от {x_range[0]:.1f} до {x_range[-1]:.1f} нм")
print(f"  z: от {z_range[0]:.1f} до {z_range[-1]:.1f} нм")

# Преобразуем углы в радианы для расчетов
deg = np.pi / 180
alpha_rad = alpha_ * deg

# Горизонтальные компоненты для каждого порядка
h_values = 2 * np.pi / length * m_values
Kx_h = K * np.cos(alpha_rad) + h_values

# kz_vac с правильной размерностью
kz_vac_full = np.sqrt(K ** 2 - Kx_h ** 2 + 0j)

# Вычисляем поле
E_field = compute_field_correct(
    X, Z, height, K, alpha_, R_vac, E_, kz,
    T_bottom, R_bottom, T_sub, kz_sub,
    D, m_values, length
)

print("Построение тепловой карты...")


# ===============PLOT HEATMAP (ОДИН ПЕРИОД)===============
def plot_heatmap(X, Z, E_field, height, length, gamma, alpha_):
    # Интенсивность
    I_field = np.abs(E_field) ** 2

    fig, ax = plt.subplots(figsize=(8, 8))

    # Тепловая карта с логарифмической шкалой и палитрой nipy_spectral
    im = ax.imshow(
        np.log10(I_field + 1e-12),
        extent=[X[0, 0], X[0, -1], Z[0, 0], Z[-1, 0]],
        aspect='auto',
        cmap='cmo.thermal',  # черный -> синий -> фиолетовый -> оранжевый
        origin='lower',
        interpolation='bilinear'
    )

    # ===== РИСУЕМ ГРАНИЦЫ СТОЛБИКА ДЛЯ ОРИЕНТАЦИИ =====
    pillar_width = gamma * length
    pillar_half = pillar_width / 2

    # Рисуем контур столбика (белые линии)
    # Верхняя грань
    ax.plot([-pillar_half, pillar_half], [height, height], 'w-', linewidth=2, alpha=0.9)
    # Нижняя грань
    ax.plot([-pillar_half, -length//2], [0, 0], 'w-', linewidth=2, alpha=0.9)
    ax.plot([pillar_half, length // 2], [0, 0], 'w-', linewidth=2, alpha=0.9)
    # Левая стенка
    ax.plot([-pillar_half, -pillar_half], [0, height], 'w-', linewidth=2, alpha=0.9)
    # Правая стенка
    ax.plot([pillar_half, pillar_half], [0, height], 'w-', linewidth=2, alpha=0.9)

    # Подписи
    ax.set_xlabel('x (нм)', fontsize=12)
    ax.set_ylabel('z (нм)', fontsize=12)
    ax.set_title(
        f'Распределение интенсивности |E|² (один период)\n'
        f'd = {length} нм, H = {height} нм, Γ = {gamma}, α = {alpha_:.1f}°',
        fontsize=14
    )

    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(|E|²)')

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', linewidth=2, label='Контур столбика'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Сетка
    # ax.grid(True, alpha=0.2, linestyle=':')

    # Отметки границ
    # ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    # ax.axhline(y=height, color='gray', linestyle=':', alpha=0.3, linewidth=1)

    # Ограничиваем оси
    ax.set_xlim(-length / 2, length / 2)
    ax.set_ylim(z_range[0], z_range[-1])

    plt.tight_layout()
    plt.savefig('field_heatmap_one_period.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, ax


# Рисуем тепловую карту для одного периода
plot_heatmap(X, Z, E_field, height, length, gamma, alpha_)
