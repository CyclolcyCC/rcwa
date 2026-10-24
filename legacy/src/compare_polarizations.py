"""
Сравнение угловых зависимостей интенсивности флуоресценции
для s- и p-поляризаций рентгеновского излучения в диапазоне [0, 40] градусов.

Включает:
1. Расчет для s-поляризации (n_s = 15) и p-поляризации с увеличенным базисом мод (n_p = 25).
2. Построение 3-панельного графика на диапазоне [0, 40] град.
3. Построение графика сходимости по числу мод n_p in [5, 10, 15, 20, 25, 30] для p-поляризации.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.s_polarization.main import PARAMS as Ps, fluorescence_vs_alpha as f_s
from src.p_polarization.main import PARAMS as Pp, fluorescence_vs_alpha as f_p


def main():
    phi = 0.0
    alphas = np.linspace(0.05, 40.0, 500)
    
    # Задаем число мод: для s-поляризации n_s = 15, для p-поляризации повышенное n_p = 25
    n_s = 15
    n_p = 25
    
    p_s = dict(Ps)
    p_s['m'] = np.arange(-n_s, n_s + 1)
    
    p_p = dict(Pp)
    p_p['m'] = np.arange(-n_p, n_p + 1)
    
    print(f"1. Расчет флуоресценции s-pol (n_max={n_s}, D={len(p_s['m'])}) на [0.05, 40] град...")
    _, Is, Rs = f_s(alphas, phi, p=p_s, method='analytical', return_refl=True)
    
    print(f"2. Расчет флуоресценции p-pol (n_max={n_p}, D={len(p_p['m'])}) на [0.05, 40] град...")
    _, Ip, Rp = f_p(alphas, phi, p=p_p, method='analytical', return_refl=True)
    
    diff_abs = Is - Ip
    diff_rel = (Is - Ip) / np.maximum(Is, 1e-12) * 100.0
    
    # ------------------ ГРАФИК 1: СРАВНЕНИЕ НА [0, 40] ГРАДУСОВ ------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # Панель (а): Полный диапазон [0, 40] град
    ax1 = axes[0]
    line_s, = ax1.plot(alphas, Is, color='#1f77b4', lw=2.2, label=f'$s$-поляризация ($n_{{\\max}}={n_s}$)')
    line_p, = ax1.plot(alphas, Ip, color='#ff7f0e', lw=1.8, ls='--', label=f'$p$-поляризация ($n_{{\\max}}={n_p}$)')
    ax1.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax1.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=12)
    ax1.set_title(r'(а) Сравнение $I_{\mathrm{fluor}}(\alpha)$ на $\alpha \in [0^\circ, 40^\circ]$', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, max(np.max(Is), np.max(Ip)) * 1.08)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(True, which='major', linestyle='-', alpha=0.6)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    ax1_r = ax1.twinx()
    line_rs, = ax1_r.plot(alphas, Rs, color='#2ca02c', lw=1.2, ls=':', alpha=0.7, label=r'Отражение $R_s(\alpha)$')
    ax1_r.set_ylabel(r'Коэффициент отражения $R_{\mathrm{tot}}$', fontsize=11, color='#2ca02c')
    ax1_r.tick_params(axis='y', labelcolor='#2ca02c')
    ax1_r.set_ylim(0, 1.05)
    ax1_r.grid(False)
    
    lines = [line_s, line_p, line_rs]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Панель (б): Резонансная область [2.5, 7.5] град
    ax2 = axes[1]
    mask_zoom = (alphas >= 2.5) & (alphas <= 7.5)
    ax2.plot(alphas[mask_zoom], Is[mask_zoom], color='#1f77b4', lw=2.2, label=r'$s$-поляризация')
    ax2.plot(alphas[mask_zoom], Ip[mask_zoom], color='#ff7f0e', lw=1.8, ls='--', label=r'$p$-поляризация')
    ax2.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax2.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=12)
    ax2.set_title(r'(б) Область резонансных пиков ($\alpha \in [2.5^\circ, 7.5^\circ]$)', fontsize=13, fontweight='bold')
    ax2.set_xlim(2.5, 7.5)
    
    idx_p1 = np.argmax(Is[alphas < 5.0])
    ax2.annotate(f'Главный пик: $\\alpha \\approx {alphas[idx_p1]:.2f}^\\circ$\n$I_s={Is[idx_p1]:.1f}, I_p={Ip[idx_p1]:.1f}$',
                 xy=(alphas[idx_p1], Is[idx_p1]), xytext=(alphas[idx_p1] + 0.4, Is[idx_p1] - 500),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=10, backgroundcolor='#ffffffcc')
    
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(True, which='major', linestyle='-', alpha=0.6)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Панель (в): Количественная разница
    ax3 = axes[2]
    color_diff = '#d62728'
    color_rel = '#9467bd'
    
    line_diff, = ax3.plot(alphas, diff_abs, color=color_diff, lw=2.0, label=r'Абс. разность $\Delta I = I_s - I_p$')
    ax3.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax3.set_ylabel(r'Абсолютная разность $\Delta I$, отн. ед.', fontsize=12, color=color_diff)
    ax3.tick_params(axis='y', labelcolor=color_diff)
    ax3.set_xlim(0, 40)
    ax3.set_ylim(0, np.max(diff_abs) * 1.15)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.grid(True, which='major', linestyle='-', alpha=0.6)
    ax3.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    ax3_rel = ax3.twinx()
    line_rel, = ax3_rel.plot(alphas, diff_rel, color=color_rel, lw=1.8, ls='-.', label=r'Отн. разность $(I_s - I_p)/I_s$ (%)')
    ax3_rel.set_ylabel(r'Относительная разность $\delta$, %', fontsize=12, color=color_rel)
    ax3_rel.tick_params(axis='y', labelcolor=color_rel)
    ax3_rel.set_ylim(0, max(np.max(diff_rel) * 1.25, 1.2))
    ax3_rel.grid(False)
    
    lines3 = [line_diff, line_rel]
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax3.set_title(r'(в) Различие между поляризациями на $[0^\circ, 40^\circ]$', fontsize=13, fontweight='bold')
    
    out_dir = os.path.join(ROOT_DIR, 'src', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fluorescence_sp_comparison.png')
    fig.savefig(out_path, dpi=300)
    print(f"Сохранен основной график (до 40 град): {out_path}")
    
    # ------------------ ГРАФИК 2: СХОДИМОСТЬ ПО ЧИСЛУ МОД P-ПОЛЯРИЗАЦИИ ------------------
    print("3. Расчет модовой сходимости для p-поляризации (n_max = 5, 10, 15, 20, 25, 30)...")
    n_modes_list = [5, 10, 15, 20, 25, 30]
    p_results = {}
    
    fig_conv, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    
    colors_list = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_modes_list)))
    
    for idx_n, n_val in enumerate(n_modes_list):
        p_temp = dict(Pp)
        p_temp['m'] = np.arange(-n_val, n_val + 1)
        _, Ip_temp = f_p(alphas, phi, p=p_temp, method='analytical')
        p_results[n_val] = Ip_temp
        
        ax_c1.plot(alphas, Ip_temp, lw=1.5, color=colors_list[idx_n],
                   label=f'$n_{{\\max}} = {n_val}$ ($D={2*n_val+1}$)')
        
        if n_val > 5:
            rel_change = np.abs(Ip_temp - p_results[n_modes_list[idx_n - 1]]) / np.maximum(Ip_temp, 1e-12) * 100.0
            ax_c2.plot(alphas, rel_change, lw=1.5, color=colors_list[idx_n],
                       label=f'$n={n_val}$ vs ${n_modes_list[idx_n - 1]}$')
    
    # Также наносим s-поляризацию пунктиром для ориентира
    ax_c1.plot(alphas, Is, 'k--', lw=2.0, alpha=0.6, label=r'$s$-поляризация (эталон)')
    
    ax_c1.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax_c1.set_ylabel(r'Интенсивность $I_{\mathrm{fluor}}^{(p)}$, отн. ед.', fontsize=12)
    ax_c1.set_title(r'(а) Кривые $I_{\mathrm{fluor}}^{(p)}(\alpha)$ при различном числе мод $n_{\max}$', fontsize=13, fontweight='bold')
    ax_c1.set_xlim(0, 40)
    ax_c1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax_c1.grid(True, alpha=0.5)
    
    ax_c2.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax_c2.set_ylabel(r'Относительное изменение при добавлении мод, %', fontsize=12)
    ax_c2.set_title(r'(б) Сходимость: шаг $\Delta n_{\max} = 5$', fontsize=13, fontweight='bold')
    ax_c2.set_xlim(0, 40)
    ax_c2.set_ylim(0, 2.5)
    ax_c2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax_c2.grid(True, alpha=0.5)
    
    out_path_conv = os.path.join(out_dir, 'modal_convergence_p_pol.png')
    fig_conv.savefig(out_path_conv, dpi=300)
    print(f"Сохранен график сходимости: {out_path_conv}")
    
    # Печать контрольных точек
    print("\n" + "="*80)
    print(f"{'alpha, deg':>10} | {'I_s (n=15)':>12} | {'I_p (n=25)':>12} | {'Delta I':>10} | {'Delta I / I_s (%)':>18}")
    print("="*80)
    for a in [0.5, 1.0, 2.0, 2.75, 3.75, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
        idx = np.argmin(np.abs(alphas - a))
        print(f"{alphas[idx]:10.2f} | {Is[idx]:12.2f} | {Ip[idx]:12.2f} | {diff_abs[idx]:10.2f} | {diff_rel[idx]:17.3f}%")
    print("="*80)


if __name__ == '__main__':
    main()
