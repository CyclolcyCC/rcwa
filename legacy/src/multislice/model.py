"""
Модель многослойной периодической структуры (multi-slice / staircase approximation).

Позволяет задавать профиль штриха решетки в виде стопки горизонтальных прямоугольных слоев (boxes),
как в папке GIXRF_Ruthenium, с поддержкой:
- Одиночного прямоугольника (текущая базовая модель: 1 слой толщиной t, шириной w);
- Многослойных приближений произвольных форм (трапеция, скругленные углы, сэндвич, TEM-профиль).
"""

import numpy as np


class BoxDef:
    """Прямоугольный блок в слое: границы по горизонтали [left, right] в нм и восприимчивость chi."""
    def __init__(self, left, right, chi, name="grating", is_fluorescent=True):
        self.left = float(min(left, right))
        self.right = float(max(left, right))
        self.chi = complex(chi)
        self.name = name
        self.is_fluorescent = is_fluorescent

    @property
    def width(self):
        return self.right - self.left

    @property
    def center(self):
        return (self.left + self.right) / 2.0

    def __repr__(self):
        return f"Box({self.name}: x=[{self.left:.1f}, {self.right:.1f}] nm, chi={self.chi.real:+.4e}{self.chi.imag:+.4e}j)"


class LayerDef:
    """Один горизонтальный срез (слой) структуры постоянной толщины thickness (нм)."""
    def __init__(self, thickness, boxes=None, name=""):
        self.thickness = float(thickness)
        self.boxes = list(boxes) if boxes is not None else []
        self.name = name

    def add_box(self, left, right, chi, name="grating", is_fluorescent=True):
        box = BoxDef(left, right, chi, name=name, is_fluorescent=is_fluorescent)
        self.boxes.append(box)
        return self

    @property
    def total_box_width(self):
        return sum(b.width for b in self.boxes)

    def __repr__(self):
        return f"Layer(thickness={self.thickness:.1f} nm, boxes={len(self.boxes)})"


class MultiSliceStructure:
    """
    Полная модель периодической структуры с периодом d (нм).
    Содержит упорядоченный сверху вниз список слоев (LayerDef),
    вакуумную полубесконечную область сверху (z < 0) и подложку снизу (z > H).
    """
    def __init__(self, period=100.0, chi_sub=complex(-0.003406, 0.0000798)):
        self.period = float(period)
        self.chi_sub = complex(chi_sub)
        self.layers = []

    def add_layer(self, thickness, boxes=None, name=""):
        layer = LayerDef(thickness, boxes, name=name)
        self.layers.append(layer)
        return layer

    @property
    def total_thickness(self):
        return sum(layer.thickness for layer in self.layers)

    @property
    def n_layers(self):
        return len(self.layers)

    def layer_z_boundaries(self):
        """Возвращает массив z-границ слоев [0, z_1, z_2, ..., H]."""
        zs = [0.0]
        cur = 0.0
        for lay in self.layers:
            cur += lay.thickness
            zs.append(cur)
        return np.array(zs, dtype=float)

    def layer_bounds(self, idx):
        """Границы z (z_top, z_bot) для слоя с индексом idx."""
        zs = self.layer_z_boundaries()
        return zs[idx], zs[idx + 1]

    # --- Пресеты структур ---
    @classmethod
    def single_rectangle(cls, period=100.0, height=90.0, cd=40.0,
                         chi=complex(-0.003406, 0.0000798),
                         chi_sub=complex(-0.003406, 0.0000798)):
        """Базовый одиночный прямоугольник (текущий классический случай)."""
        model = cls(period=period, chi_sub=chi_sub)
        lay = model.add_layer(height, name="Slab")
        lay.add_box(-cd / 2.0, cd / 2.0, chi, name="Grating", is_fluorescent=True)
        return model

    @classmethod
    def trapezoid(cls, period=100.0, height=90.0, cd_mid=40.0, swa_deg=85.0,
                  n_slices=5, chi=complex(-0.003406, 0.0000798),
                  chi_sub=complex(-0.003406, 0.0000798)):
        """Трапеция, разбитая на n_slices срезов равной толщины."""
        model = cls(period=period, chi_sub=chi_sub)
        h_slice = height / n_slices
        swa_rad = np.radians(swa_deg)
        # Полуширина стенки: dx = h / tan(swa)
        slope_dx = (height / 2.0) / np.tan(swa_rad) if swa_deg < 89.9 else 0.0

        for i in range(n_slices):
            # z в середине среза относительно центра трапеции
            z_mid = (i + 0.5) * h_slice
            # Сверху (z=0) ширина меньше, снизу (z=height) больше
            w_i = cd_mid + 2.0 * (z_mid - height / 2.0) / np.tan(swa_rad)
            w_i = max(w_i, 1.0)
            lay = model.add_layer(h_slice, name=f"Slice {i+1}")
            lay.add_box(-w_i / 2.0, w_i / 2.0, chi, name="Grating", is_fluorescent=True)
        return model

    @classmethod
    def rounded_trapezoid(cls, period=100.0, height=90.0, cd_mid=40.0, swa_deg=85.0,
                          r_top=5.0, n_slices=10, chi=complex(-0.003406, 0.0000798),
                          chi_sub=complex(-0.003406, 0.0000798)):
        """Трапеция со скруглением верхних углов радиусом r_top."""
        model = cls(period=period, chi_sub=chi_sub)
        h_slice = height / n_slices
        swa_rad = np.radians(swa_deg)
        for i in range(n_slices):
            z_mid = (i + 0.5) * h_slice
            w_i = cd_mid + 2.0 * (z_mid - height / 2.0) / np.tan(swa_rad)
            if z_mid < r_top:
                # Круговой скос верхнего угла
                dz = r_top - z_mid
                w_i -= 2.0 * (r_top - np.sqrt(max(0.0, r_top**2 - dz**2)))
            w_i = max(w_i, 1.0)
            lay = model.add_layer(h_slice, name=f"Slice {i+1}")
            lay.add_box(-w_i / 2.0, w_i / 2.0, chi, name="Grating", is_fluorescent=True)
        return model

    # --- Аналитический расчет компонент Фурье для слоя ---
    def fourier_coefficients_layer(self, layer_idx, orders):
        """
        Расчет гармоник Фурье chi_m для заданного слоя layer_idx.
        chi(x) = sum_k chi_k * rect_{[a_k, b_k]}(x).
        """
        layer = self.layers[layer_idx]
        d = self.period
        orders = np.asarray(orders, dtype=int)
        chi_m = np.zeros(len(orders), dtype=complex)

        for b in layer.boxes:
            w = b.width
            x0 = b.center
            # \int_{x0 - w/2}^{x0 + w/2} exp(-i m 2\pi x / d) dx = w * sinc(m * w / d) * exp(-i m 2\pi x0 / d)
            phase = np.exp(-1j * orders * 2.0 * np.pi * x0 / d)
            sinc_val = np.sinc(orders * w / d)
            chi_m += b.chi * (w / d) * sinc_val * phase

        return chi_m

    def fourier_matrices_layer_p(self, layer_idx, m):
        """
        Построение теплицевых матриц [eps] и Z = [1/eps] по правилу факторизации Ли (Li, 1996)
        для слоя layer_idx.
        """
        layer = self.layers[layer_idx]
        d = self.period
        n_max = int(np.max(m))
        orders_extended = np.arange(-2 * n_max, 2 * n_max + 1)

        # eps(x) = 1 + chi(x) -> eps_0 = 1 + chi_0, eps_m = chi_m (m != 0)
        eps_all = np.zeros(len(orders_extended), dtype=complex)
        eps_all[orders_extended == 0] = 1.0

        # eta(x) = 1/eps(x) = 1 + sum_k [ 1/(1 + chi_k) - 1 ] * rect_k(x)
        eta_all = np.zeros(len(orders_extended), dtype=complex)
        eta_all[orders_extended == 0] = 1.0

        for b in layer.boxes:
            w = b.width
            x0 = b.center
            phase = np.exp(-1j * orders_extended * 2.0 * np.pi * x0 / d)
            sinc_val = np.sinc(orders_extended * w / d)
            eps_box = 1.0 + b.chi
            eta_box = 1.0 / eps_box

            eps_all += b.chi * (w / d) * sinc_val * phase
            eta_all += (eta_box - 1.0) * (w / d) * sinc_val * phase

        diff = m[:, None] - m[None, :]
        n2 = 2 * n_max
        eps_mat = eps_all[diff + n2]
        Z_mat = eta_all[diff + n2]
        return eps_mat, Z_mat
