import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import linalg

# Параметры задачи
D = 10.0e-6  # период
N = 1001  # количество слагаемых (-500 до 500)
n_max = N // 2  # 500
sigma = 0.2
n_periods = 3
pulse = -0.6e-5 + 1j * 0.6e-7
lambd = 1.0e-10
K = 2*np.pi/lambd

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
def matrix(K, chi_h, n, alpha):
    A = np.zeros((n_max, n_max), dtype=complex)
    Kp = K * np.cos(alpha/180*np.pi)
    for i in range(0, n_max):
        for j in range(0, n_max):
            sum = n_max-i-j
            inv = n_max - 1 - i
            idx = np.where(n == sum)[0]
            if i != j:
                A[inv, j]  = K**2*chi_h[idx[0]]
            else:
                h = 2 * np.pi * (i - n_max / 2) / D
                kh2 = Kp ** 2 + h ** 2
                A[inv, j] = K**2*chi_h[idx[0]]-kh2
    return A

nn = 1000
angle = np.linspace(1.0e-5, 1, nn, endpoint=False)
k_a_ = np.zeros(nn, dtype=complex)
for i in range(nn):
    A = matrix(K, chi_h_shifted, n, angle[i])
    kzn2, En = linalg.eig(A)
    k_a_[i] = np.sqrt(kzn2[0])


print("Собственные значения (k^2):")
print(len(kzn2))
print("\nСобственные векторы (E):")
print(len(En[0]))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(angle, k_a_.real, label='kzn(a)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График зависимости действительных значений kzn от угла alpha')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(angle, k_a_.imag, label='kzn(a)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График зависимости мнимых значений kzn от угла alpha')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()