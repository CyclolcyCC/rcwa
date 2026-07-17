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
m_max = 2
m_values = np.arange(-m_max, m_max + 1)  # [-2, -1, 0, 1, 2]
# eigenvalues and eigenvectors for alpha_ and phi_
A_, k_hp = build_A_matrix(f_h, h_vector, alpha_, phi_)
eigenvalues_, E_ = linalg.eig(A_)

# sorting by arguments instead of setting signs
idx_sorted = np.argsort(-np.imag(np.sqrt(eigenvalues_)))
eigenvalues_ = eigenvalues_[idx_sorted]
E_ = E_[:, idx_sorted]

# calculating sqrt of eigenvalues
kz = np.sqrt(eigenvalues_)
D = len(kz)
# every layer calculation
# 0 vacuum
kz_vac = np.sqrt(K**2 - k_hp**2 + 0j)
P_vac = build_P_matrix(np.eye(D), kz_vac, True)

# 1 structure
P_struct = build_P_matrix(E_, kz, False)
Q_struct = build_Q_matrix(kz, height)

# 2 substrate
kz_sub = np.sqrt(K**2 * chi - k_hp**2 + 0j)
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
    """
    Возвращает функцию для получения амплитуд на любой глубине внутри слоя
    """
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
def compute_field(X, Z, height, K, Kp, Kz, kz_vac_full, R_vac, E_, kz, T_sub, kz_sub, D, m_values, length, Kx_h):
    """
    Вычисляет поле E(x,z) для сетки координат
    """
    E_field = np.zeros_like(X, dtype=complex)

    # Для каждого порядка получаем амплитуды
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            z = Z[i, j]

            if z > height:
                # ===== ВАКУУМ (z > H) =====
                # Падающая волна (только нулевой порядок)
                E_inc = np.exp(1j * (Kx_h[np.argmin(np.abs(m_values))] * x + Kz * (z - height)))

                # Отраженные волны (идут вверх)
                E_ref = 0
                for idx in range(D):
                    if idx < len(R_vac):
                        E_ref += R_vac[idx] * np.exp(1j * (Kx_h[idx] * x - kz_vac_full[idx] * (z - height)))

                E_field[i, j] = E_inc + E_ref

            elif z >= 0:
                # ===== СТРУКТУРИРОВАННЫЙ СЛОЙ (0 <= z <= H) =====
                # Получаем амплитуды на этой глубине
                T_z, R_z = get_T_R(z)

                # Суммируем по всем модам
                E_layer = 0
                for n in range(D):
                    # Амплитуда моды n на глубине z
                    amplitude = T_z[n] * np.exp(1j * kz[n] * z) + R_z[n] * np.exp(-1j * kz[n] * z)

                    # Суммируем по дифракционным порядкам
                    for idx in range(D):
                        if idx < E_.shape[0] and n < E_.shape[1]:
                            E_layer += amplitude * E_[idx, n] * np.exp(1j * Kx_h[idx] * x)

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
get_T_R, T_bottom, R_bottom, T_sub = get_amplitudes_inside_layer(
    P_vac_struct, Q_struct, P_struct_sub, M11, T_vac, kz, D
)

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
E_field = compute_field(
    X, Z, height, K, Kp, Kz,
    kz_vac_full, R_vac, E_, kz,
    T_sub, kz_sub, D, m_values, length, Kx_h
)

print("Построение тепловой карты...")


# ===============PLOT HEATMAP (ОДИН ПЕРИОД)===============
def plot_heatmap(X, Z, E_field, height, length, gamma, alpha_):
    """
    Строит тепловую карту для одного периода
    """
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
    ax.grid(True, alpha=0.2, linestyle=':')

    # Отметки границ
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=height, color='gray', linestyle=':', alpha=0.3, linewidth=1)

    # Ограничиваем оси
    ax.set_xlim(-length / 2, length / 2)
    ax.set_ylim(z_range[0], z_range[-1])

    plt.tight_layout()
    plt.savefig('field_heatmap_one_period.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, ax


# Рисуем тепловую карту для одного периода
plot_heatmap(X, Z, E_field, height, length, gamma, alpha_)
