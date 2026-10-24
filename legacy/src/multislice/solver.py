"""
Модуль для проведения расчетов и построения графиков многослойного RCWA
для s- и p-поляризаций.

ТРЕБОВАНИЯ:
1. Дисперсионные кривые содержат обе части: Re(k_zn) и Im(k_zn).
2. Ближнее поле строится точно по образцу (ось Y: height above substrate, белые контуры штриха и подложки, палитра thermal/inferno, поддержка alpha и phi).
3. Возможность задания детализации сетки (Nx, Nz, N_alpha, N_phi).
4. На графиках флуоресценции НЕТ отражательной способности!
5. Карты и графики вычитания (разности) поляризаций сохраняются в папку без вывода на экран.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import cmocean
    CMAP_THERMAL = cmocean.cm.thermal
except Exception:
    CMAP_THERMAL = 'inferno'

from .rcwa_s import (
    solve_multislice_s,
    compute_field_s,
    compute_fluorescence_analytical_s,
    compute_fluorescence_numerical_s
)
from .rcwa_p import (
    solve_multislice_p,
    compute_field_p,
    compute_fluorescence_analytical_p,
    compute_fluorescence_numerical_p
)


# =========================================================================
# 1. Эскиз профиля геометрии структуры
# =========================================================================
def plot_structure_geometry(structure, ax=None, title="Геометрический профиль многослойной наноструктуры"):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True
    else:
        fig = ax.figure

    d = structure.period
    H = structure.total_thickness
    z_bounds = structure.layer_z_boundaries()

    # Вакуум и подложка
    ax.axhspan(-15.0, 0.0, color='#e0f2fe', alpha=0.5, label='Вакуум')
    ax.axhspan(H, H + 25.0, color='#cbd5e1', alpha=0.7, label='Подложка')

    box_colors = ['#f97316', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899']

    for j, layer in enumerate(structure.layers):
        z_top = z_bounds[j]
        h = layer.thickness
        ax.axhline(z_top, color='#94a3b8', ls='--', lw=0.8, alpha=0.8)

        for bi, b in enumerate(layer.boxes):
            c = box_colors[bi % len(box_colors)]
            rect = Rectangle((b.left, z_top), b.width, h,
                             facecolor=c, edgecolor='#1e293b', lw=1.2, alpha=0.85,
                             label=b.name if (j == 0 and bi == 0) else None)
            ax.add_patch(rect)
            # Периодические копии
            rect_left = Rectangle((b.left - d, z_top), b.width, h,
                                  facecolor=c, edgecolor='#64748b', lw=0.8, alpha=0.35)
            rect_right = Rectangle((b.left + d, z_top), b.width, h,
                                   facecolor=c, edgecolor='#64748b', lw=0.8, alpha=0.35)
            ax.add_patch(rect_left)
            ax.add_patch(rect_right)

    ax.axhline(H, color='#475569', ls='-', lw=1.5)
    ax.axvline(-d / 2.0, color='#64748b', ls=':', lw=1.2)
    ax.axvline(d / 2.0, color='#64748b', ls=':', lw=1.2, label='Границы периода d')

    ax.set_xlim(-0.75 * d, 0.75 * d)
    ax.set_ylim(H + 25.0, -15.0)  # глубина z растет вниз
    ax.set_xlabel('x, нм', fontsize=11)
    ax.set_ylabel('Глубина z, нм', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    if created_fig:
        fig.tight_layout()
    return fig


# =========================================================================
# 2. Дисперсионные кривые k_zn(alpha) — ОБЕ ЧАСТИ: Re и Im
# =========================================================================
def compute_and_plot_dispersion(structure, alphas, phi, K, m, pol='s'):
    """
    Построение дисперсионных кривых k_zn(alpha):
    Верхняя панель: Re(k_zn), нижняя панель: Im(k_zn).
    """
    kz_re_list = []
    kz_im_list = []

    solve_fn = solve_multislice_s if pol == 's' else solve_multislice_p
    for a in alphas:
        sol = solve_fn(structure, a, phi, K, m)
        kz0 = sol['layers'][0]['kz']
        kz_re_list.append(np.sort(kz0.real)[::-1])
        kz_im_list.append(np.sort(kz0.imag))

    kz_re_arr = np.array(kz_re_list)
    kz_im_arr = np.array(kz_im_list)

    pol_name = r'$s$-поляризация' if pol == 's' else r'$p$-поляризация'

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)

    # Верхняя панель: Re
    ax[0].plot(alphas, kz_re_arr, lw=1.0)
    ax[0].set_ylabel(r'$\mathrm{Re}(k_{zn})$, нм$^{-1}$', fontsize=11)
    ax[0].set_title(f'Дисперсионные кривые $k_{{zn}}(\\alpha)$ ({pol_name}, $\\phi = {phi:.1f}^\\circ$)',
                    fontsize=12, fontweight='bold')
    ax[0].grid(True, linestyle=':', alpha=0.6)

    # Нижняя панель: Im
    ax[1].plot(alphas, kz_im_arr, lw=1.0)
    ax[1].set_ylabel(r'$\mathrm{Im}(k_{zn})$, нм$^{-1}$', fontsize=11)
    ax[1].set_xlabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax[1].grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    return fig


# =========================================================================
# 3. Карты собственных векторов
# =========================================================================
def plot_eigenvectors(structure, alpha, phi, K, m, pol='s'):
    solve_fn = solve_multislice_s if pol == 's' else solve_multislice_p
    sol = solve_fn(structure, alpha, phi, K, m)
    E0 = sol['layers'][0]['E']
    intensity_E = np.abs(E0)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(intensity_E, aspect='auto', cmap='viridis', origin='lower',
                   extent=[-0.5, len(m) - 0.5, m[0] - 0.5, m[-1] + 0.5])
    fig.colorbar(im, ax=ax, label=r'Амплитуда $|E_{hn}|$')

    pol_name = r'$s$-поляризация' if pol == 's' else r'$p$-поляризация'
    ax.set_xlabel('Номер моды $n$', fontsize=11)
    ax.set_ylabel('Дифракционный порядок $h$', fontsize=11)
    ax.set_title(f'2D карта собственных векторов ({pol_name}, $\\alpha={alpha:.1f}^\\circ$, $\\phi={phi:.1f}^\\circ$)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# =========================================================================
# 4. Ближнее электрическое поле |E|² (точно по скриншоту)
# =========================================================================
def draw_near_field_contour(ax, structure, d, H):
    """
    Отрисовка белого контура структуры в координатах 'height above substrate, nm'.
    """
    z_bounds = structure.layer_z_boundaries()

    if structure.n_layers == 1 and structure.layers[0].boxes:
        b = structure.layers[0].boxes[0]
        hw = b.width / 2.0
        # Верхняя грань
        ax.plot([-hw, hw], [H, H], 'w-', lw=2.0)
        # Боковые стенки
        ax.plot([-hw, -hw], [0.0, H], 'w-', lw=2.0)
        ax.plot([hw, hw], [0.0, H], 'w-', lw=2.0)
        # Поверхность подложки
        ax.plot([-d / 2.0, -hw], [0.0, 0.0], 'w-', lw=2.0)
        ax.plot([hw, d / 2.0], [0.0, 0.0], 'w-', lw=2.0)
    else:
        # Для многослойных срезов
        for j, lay in enumerate(structure.layers):
            y_top = H - z_bounds[j]
            y_bot = H - z_bounds[j + 1]
            for b in lay.boxes:
                # Прямоугольник каждого среза
                ax.plot([b.left, b.right], [y_top, y_top], 'w-', lw=1.6)
                ax.plot([b.left, b.right], [y_bot, y_bot], 'w-', lw=1.6)
                ax.plot([b.left, b.left], [y_bot, y_top], 'w-', lw=1.6)
                ax.plot([b.right, b.right], [y_bot, y_top], 'w-', lw=1.6)

        # Нижняя граница на подложке y = 0
        bottom_boxes = structure.layers[-1].boxes
        if bottom_boxes:
            b_bot = bottom_boxes[0]
            ax.plot([-d / 2.0, b_bot.left], [0.0, 0.0], 'w-', lw=2.0)
            ax.plot([b_bot.right, d / 2.0], [0.0, 0.0], 'w-', lw=2.0)
        else:
            ax.plot([-d / 2.0, d / 2.0], [0.0, 0.0], 'w-', lw=2.0)


def compute_and_plot_near_field(sol, pol='s', nx=300, nz=300, pad=20.0):
    """
    Построение карты ближнего поля |E|² в точности как на скриншоте:
    - Ось X: x, nm ([-d/2, d/2])
    - Ось Y: height above substrate, nm ([-pad, H + pad])
    - Контур: сплошная белая линия 'w-', lw=2
    - Цветовая шкала: cmocean.cm.thermal / inferno
    - Заголовок: |E|², d = ... nm, t = ... nm, Γ = ..., α = ...°, φ = ...°
    """
    structure = sol['structure']
    d = structure.period
    H = structure.total_thickness

    x = np.linspace(-d / 2.0, d / 2.0, nx)
    z = np.linspace(-pad, H + pad, nz)

    if pol == 's':
        E = compute_field_s(sol, x, z)
        I_field = np.abs(E) ** 2
    else:
        I_field = compute_field_p(sol, x, z)

    height = H - z

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    vmin = 0.0
    vmax = float(np.percentile(I_field, 99.5))

    im = ax.imshow(I_field, extent=[x[0], x[-1], height[-1], height[0]],
                   vmin=vmin, vmax=vmax, aspect='auto', cmap=CMAP_THERMAL,
                   origin='upper', interpolation='bilinear')

    draw_near_field_contour(ax, structure, d, H)

    ax.set_xlabel('x, nm', fontsize=11)
    ax.set_ylabel('height above substrate, nm', fontsize=11)

    gamma = structure.layers[0].total_box_width / d if structure.layers else 0.4
    pol_tag = r" ($s$-пол.)" if pol == 's' else r" ($p$-пол.)"
    ax.set_title(f"|E|²{pol_tag},  d = {d:.1f} nm, t = {H:.1f} nm, Γ = {gamma:.2f}, α = {sol['alpha']:.1f}°, φ = {sol['phi']:.1f}°",
                 fontsize=11.5)

    fig.colorbar(im, ax=ax, label='|E|²')
    ax.set_xlim(-d / 2.0, d / 2.0)
    ax.set_ylim(height[-1], height[0])
    fig.tight_layout()
    return fig


def compute_and_save_near_field_diff(sol_s, sol_p, nx=300, nz=300, pad=20.0, save_path=None):
    """
    Разность ближних полей |Es|^2 - |Ep|^2.
    Строится и сохраняется в файл БЕЗ вывода отдельного окна на экран.
    """
    structure = sol_s['structure']
    d = structure.period
    H = structure.total_thickness

    x = np.linspace(-d / 2.0, d / 2.0, nx)
    z = np.linspace(-pad, H + pad, nz)

    I_s = np.abs(compute_field_s(sol_s, x, z)) ** 2
    I_p = compute_field_p(sol_p, x, z)
    diff = I_s - I_p
    height = H - z

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    vmax = float(np.max(np.abs(diff))) or 1.0

    im = ax.imshow(diff, extent=[x[0], x[-1], height[-1], height[0]],
                   vmin=-vmax, vmax=vmax, aspect='auto', cmap='seismic',
                   origin='upper', interpolation='bilinear')

    draw_near_field_contour(ax, structure, d, H)

    ax.set_xlabel('x, nm', fontsize=11)
    ax.set_ylabel('height above substrate, nm', fontsize=11)
    ax.set_title(f"Разность |E_s|² - |E_p|², d = {d:.1f} nm, t = {H:.1f} nm, α = {sol_s['alpha']:.1f}°, φ = {sol_s['phi']:.1f}°",
                 fontsize=11.5)

    fig.colorbar(im, ax=ax, label=r'$|E_s|^2 - |E_p|^2$')
    ax.set_xlim(-d / 2.0, d / 2.0)
    ax.set_ylim(height[-1], height[0])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# 5. 1D Кривые флуоресценции (БЕЗ ОТРАЖЕНИЯ!)
# =========================================================================
def scan_fluorescence_1d(structure, alphas, phi, K, m_s, m_p, mu_att=0.0, method='analytical'):
    alphas = np.asarray(alphas, dtype=float)
    Is_arr = np.zeros(len(alphas), dtype=float)
    Ip_arr = np.zeros(len(alphas), dtype=float)

    for i, a in enumerate(alphas):
        sol_s = solve_multislice_s(structure, a, phi, K, m_s)
        sol_p = solve_multislice_p(structure, a, phi, K, m_p)
        if method == 'numerical':
            Is_arr[i] = compute_fluorescence_numerical_s(sol_s, mu_att=mu_att)
            Ip_arr[i] = compute_fluorescence_numerical_p(sol_p, mu_att=mu_att)
        else:
            Is_arr[i] = compute_fluorescence_analytical_s(sol_s, mu_att=mu_att)
            Ip_arr[i] = compute_fluorescence_analytical_p(sol_p, mu_att=mu_att)

    return Is_arr, Ip_arr


def plot_fluorescence_single(alphas, I_fluor, pol='s'):
    """Кривая флуоресценции одной поляризации — СТРОГО БЕЗ ОТРАЖЕНИЯ."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    pol_name = r'$s$-поляризация' if pol == 's' else r'$p$-поляризация'
    color = '#1f77b4' if pol == 's' else '#ff7f0e'

    ax.plot(alphas, I_fluor, color=color, lw=2.2, label=rf'Флуоресценция $I_{{\mathrm{{fluor}}}}$ ({pol_name})')
    ax.set_xlabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=11)
    ax.set_title(f'Кривая флуоресценции $I_{{fluor}}(\\alpha)$ ({pol_name})', fontsize=12, fontweight='bold')
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(0, np.max(I_fluor) * 1.08)
    ax.grid(True, linestyle='-', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_fluorescence_comparison_onscreen(alphas, Is, Ip):
    """
    Вывод на экран совмещенного графика s и p поляризаций (БЕЗ ВЫЧИТАНИЯ И БЕЗ ОТРАЖЕНИЯ).
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(alphas, Is, color='#1f77b4', lw=2.2, label=r'$s$-поляризация ($I_s$)')
    ax.plot(alphas, Ip, color='#ff7f0e', lw=1.8, ls='--', label=r'$p$-поляризация ($I_p$)')
    ax.set_xlabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=11)
    ax.set_title(r'Сравнение кривых флуоресценции: $s$- и $p$-поляризации',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(0, max(np.max(Is), np.max(Ip)) * 1.08)
    ax.grid(True, linestyle='-', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    return fig


def save_fluorescence_comparison_with_diff(alphas, Is, Ip, save_path):
    """
    Сохранение двухпанельного файла (сравнение + относительная разность) в папку figures.
    """
    fig, (ax_main, ax_diff) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax_main.plot(alphas, Is, color='#1f77b4', lw=2.2, label=r'$s$-поляризация ($I_s$)')
    ax_main.plot(alphas, Ip, color='#ff7f0e', lw=1.8, ls='--', label=r'$p$-поляризация ($I_p$)')
    ax_main.set_xlabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax_main.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=11)
    ax_main.set_title(r'Сравнение $I_s(\alpha)$ и $I_p(\alpha)$', fontsize=12, fontweight='bold')
    ax_main.set_xlim(alphas[0], alphas[-1])
    ax_main.set_ylim(0, max(np.max(Is), np.max(Ip)) * 1.08)
    ax_main.grid(True, linestyle='-', alpha=0.6)
    ax_main.legend(loc='lower right', framealpha=0.9)

    diff_rel = (Is - Ip) / np.maximum(Is, 1e-12) * 100.0
    ax_diff.plot(alphas, diff_rel, color='#dc2626', lw=2.0, label=r'Относ. разность $\frac{I_s - I_p}{I_s} \times 100\%$')
    ax_diff.axhline(0, color='black', lw=0.8, ls=':')
    ax_diff.set_xlabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax_diff.set_ylabel(r'Расхождение $(I_s - I_p)/I_s$, %', fontsize=11)
    ax_diff.set_title(r'Поляризационное расхождение $(I_s - I_p) / I_s$', fontsize=12, fontweight='bold')
    ax_diff.set_xlim(alphas[0], alphas[-1])
    ax_diff.grid(True, linestyle='-', alpha=0.6)
    ax_diff.legend(loc='upper right', framealpha=0.9)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =========================================================================
# 6. 2D Карты флуоресценции
# =========================================================================
def scan_fluorescence_2d(structure, alphas, phis, K, m_s, m_p, mu_att=0.0, method='analytical'):
    Na, Np = len(alphas), len(phis)
    map_s = np.zeros((Na, Np), dtype=float)
    map_p = np.zeros((Na, Np), dtype=float)

    for i, a in enumerate(alphas):
        for j, p in enumerate(phis):
            sol_s = solve_multislice_s(structure, a, p, K, m_s)
            sol_p = solve_multislice_p(structure, a, p, K, m_p)
            if method == 'numerical':
                map_s[i, j] = compute_fluorescence_numerical_s(sol_s, mu_att=mu_att)
                map_p[i, j] = compute_fluorescence_numerical_p(sol_p, mu_att=mu_att)
            else:
                map_s[i, j] = compute_fluorescence_analytical_s(sol_s, mu_att=mu_att)
                map_p[i, j] = compute_fluorescence_analytical_p(sol_p, mu_att=mu_att)

    return map_s, map_p


def plot_fluorescence_2d_map(alphas, phis, map_data, which='s', title=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    extent = [phis[0], phis[-1], alphas[0], alphas[-1]]

    im = ax.imshow(map_data, origin='lower', aspect='auto',
                   extent=extent, cmap='turbo', interpolation='bilinear')
    pol_label = r'$s$-поляризация' if which == 's' else r'$p$-поляризация'
    default_title = f'2D карта выхода флуоресценции ({pol_label})'
    fig.colorbar(im, ax=ax, label=r'Интенсивность $I_{\mathrm{fluor}}$')

    ax.set_xlabel(r'Азимутальный угол $\phi$, град', fontsize=11)
    ax.set_ylabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax.set_title(title or default_title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def save_fluorescence_2d_diff_map(alphas, phis, map_s, map_p, save_path):
    """Сохранение 2D карты разности флуоресценции Is - Ip без показа на экран."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    extent = [phis[0], phis[-1], alphas[0], alphas[-1]]
    diff = map_s - map_p
    vmax = float(np.max(np.abs(diff))) or 1.0

    im = ax.imshow(diff, origin='lower', aspect='auto',
                   extent=extent, cmap='seismic', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear')
    fig.colorbar(im, ax=ax, label=r'Разность $I_s - I_p$')

    ax.set_xlabel(r'Азимутальный угол $\phi$, град', fontsize=11)
    ax.set_ylabel(r'Угол падения $\alpha$, град', fontsize=11)
    ax.set_title(r'2D карта разности флуоресценции $I_s(\alpha, \phi) - I_p(\alpha, \phi)$',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
