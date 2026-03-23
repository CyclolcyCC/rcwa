import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import linalg

# Параметры задачи
D = 10.0  # период
N = 7  # количество слагаемых
n_max = N // 2
pulse = -1 + 1j
lambda_ = 1.0
K = 2*np.pi/lambda_

n_periods = 3
# значения функции на периоде
x = np.linspace(0, n_periods*D, n_periods*N, endpoint=False)
xn = np.linspace(0, D, N, endpoint=False)

# создание функции для одного периода(фурье)
T = len(x)//n_periods # длина периода в количестве расчетных точек
fn = np.zeros(T, dtype=complex)
mask = (xn >= 0) & (xn < D/2)
fn[mask] = pulse

# вычисление коэффициентов фурье с помощью fft
chi_h = fft.fft(fn) / T # нормировка для комплексного ряда фурье

# изменение порядка коэффициентов
chi_h_shifted = fft.fftshift(chi_h)
# соответствующие частоты
n = fft.fftshift(fft.fftfreq(T, d=1 / T))

def h_arr_gen(n):
    return np.array([2*np.pi*m/D for m in range(-n, n + 1)])

# восстановление функции по первым 2000 гармоникам
# построение идеальной функции для нескольких периодов
def reconstruct_fourier(chi_h, n_original, x_points, n_max, n_periods, fn):
    N = len(x_points)
    f_reconstructed = np.zeros(N, dtype=complex)
    f = np.zeros(N, dtype=complex)

    for i in range(n_periods):
        ps = i * D
        pn = ps + D/2
        mask = (x_points >= ps) & (x_points < pn)
        f[mask] = pulse

    for m in range(-n_max, n_max + 1):
        idx = np.where(n_original == m)[0]
        if len(idx) > 0:
            h = 2*np.pi*m/D
            f_reconstructed += chi_h[idx[0]] * np.exp(1j * h * x_points)

    return f, f_reconstructed


# восстанавливаем функцию
f, f_reconstructed = reconstruct_fourier(chi_h_shifted, n, x, n_max, n_periods, fn)

# визуализация действительной и мнимой частей и сравнение с построенным рядом

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, f.real, label='Re(f)')
plt.plot(x, f.imag, label='Im(f)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Исходная функция')
plt.legend()
plt.grid(True)

# Сравнение
plt.subplot(1, 3, 2)
plt.plot(x, f.real, 'b-', label='Original Re(f)', alpha=0.7)
plt.plot(x, f_reconstructed.real, 'r--', label='Reconstructed Re(f)', alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Сравнение действительной части')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, f.imag, 'g-', label='Original Im(f)', alpha=0.7)
plt.plot(x, f_reconstructed.imag, 'y--', label='Reconstructed Im(f)', alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Сравнение комплексной части')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# решение задачи на собственные значение и векторы

# создание матрицы A
def matrix(chi_h, alphai, phii):
    chi_matrix = linalg.toeplitz(chi_h[n_max:], (chi_h[::-1])[n_max:])

    deg = np.pi/180
    k_p_x = np.zeros(n_max+1, dtype=complex)
    k_p_y = np.zeros(n_max+1, dtype=complex)
    k_p_x += K * np.cos(alphai*deg) * np.sin(phii*deg)
    k_p_y += K * np.cos(alphai*deg) * np.cos(phii*deg)

    h = h_arr_gen(n_max//2)
    k_p_x = k_p_x + h
    k_h_p2 = (k_p_x + h)**2 + k_p_y**2
    A = K**2 * chi_matrix - np.diag(k_h_p2)
    print("shape: ", A.shape)
    print("rank: ", np.linalg.matrix_rank(A))
    return A


# решение волнового уравнения
nn = 1000
alpha = np.linspace(0, 30, nn, endpoint=False)
phi = np.linspace(1.0e-5, 30, nn, endpoint=False)
k_a_ = []
for alphai in alpha:
    eigenvalues, eigenvectors = linalg.eig(matrix(chi_h_shifted, alphai, 0))
    k_a_.append(np.sqrt(eigenvalues))
k_a_ = np.array(k_a_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(alpha, k_a_.real)
plt.xlabel('alpha')
plt.ylabel('Re(k_zn)')
plt.title('1')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(alpha, k_a_.imag)
plt.xlabel('alpha')
plt.ylabel('Im(k_zn)')
plt.title('2')
plt.grid(True)

plt.tight_layout()
plt.show()