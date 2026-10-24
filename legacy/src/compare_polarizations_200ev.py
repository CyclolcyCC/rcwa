"""
Расчет выхода флуоресценции для s- и p-поляризаций при энергии E = 200 эВ
(длина волны lambda = 6.199 нм, мягкий рентген / край EUV).

Оптические константы Si3N4 взяты из базы данных Henke / CXRO:
    E = 200 эВ: delta = 0.0109, beta = 0.0067, chi = -0.0218 + 0.0134j
Критический угол полного внешнего отражения:
    alpha_c = sqrt(2 * delta) approx 8.46 градусов (вместо ~3.3 град при 520 эВ).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.s_polarization.main import fluorescence_vs_alpha as f_s
from src.p_polarization.main import fluorescence_vs_alpha as f_p


def main():
    ev = 200.0
    wl = 1239.84 / ev        # 6.1992 нм
    K = 2 * np.pi / wl       # 1.0135 нм^-1
    
    # Оптические константы Si3N4 при 200 эВ (CXRO Henke)
    delta = 0.0109
    beta = 0.0067
    chi = -2.0 * delta + 2.0j * beta
    eps_1 = 1.0 + chi
    
    # Геометрия решетки
    d = 100.0
    t = 90.0
    gamma = 0.4
    
    n_s = 15
    n_p = 25
    m_s = np.arange(-n_s, n_s + 1)
    m_p = np.arange(-n_p, n_p + 1)
    
    p_s = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=0.0, chi_sub=chi, m=m_s)
    p_p = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=0.0, chi_sub=chi,
               eps_1=eps_1, eps_b=1.0, eps_sub=eps_1, m=m_p)
    
    alphas = np.linspace(0.05, 40.0, 500)
    phi = 0.0
    
    print(f"Расчет s-поляризации при E={ev} эВ (lambda={wl:.3f} нм)...")
    _, Is, Rs = f_s(alphas, phi, p=p_s, method='analytical', return_refl=True)
    
    print(f"Расчет p-поляризации при E={ev} эВ (lambda={wl:.3f} нм)...")
    _, Ip, Rp = f_p(alphas, phi, p=p_p, method='analytical', return_refl=True)
    
    diff_abs = Is - Ip
    diff_rel = (Is - Ip) / np.maximum(Is, 1e-12) * 100.0
    
    alpha_c = np.degrees(np.sqrt(2.0 * delta))
    
    # ------------------ ГРАФИК 1: 3-ПАНЕЛЬНЫЙ ДЛЯ 200 эВ ------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # Панель (а): Полный обзор [0, 40] град
    ax1 = axes[0]
    line_s, = ax1.plot(alphas, Is, color='#1f77b4', lw=2.2, label=r'$s$-поляризация ($I_s$)')
    line_p, = ax1.plot(alphas, Ip, color='#ff7f0e', lw=1.8, ls='--', label=r'$p$-поляризация ($I_p$)')
    ax1.axvline(alpha_c, color='gray', ls=':', lw=1.5, label=f'$\\alpha_c \\approx {alpha_c:.1f}^\\circ$')
    ax1.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax1.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=12)
    ax1.set_title(rf'(а) $E = {ev:.0f}$ эВ ($\lambda = {wl:.2f}$ нм): полный диапазон $[0^\circ, 40^\circ]$',
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, max(np.max(Is), np.max(Ip)) * 1.08)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(True, which='major', linestyle='-', alpha=0.6)
    
    ax1_r = ax1.twinx()
    line_rs, = ax1_r.plot(alphas, Rs, color='#2ca02c', lw=1.2, ls='-.', alpha=0.7, label=r'Отражение $R_s(\alpha)$')
    ax1_r.set_ylabel(r'Коэффициент отражения $R_{\mathrm{tot}}$', fontsize=11, color='#2ca02c')
    ax1_r.tick_params(axis='y', labelcolor='#2ca02c')
    ax1_r.set_ylim(0, 1.05)
    ax1_r.grid(False)
    
    lines = [line_s, line_p, line_rs]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Панель (б): Область критического угла и резонансов [1, 15] град
    ax2 = axes[1]
    mask_zoom = (alphas >= 1.0) & (alphas <= 15.0)
    ax2.plot(alphas[mask_zoom], Is[mask_zoom], color='#1f77b4', lw=2.2, label=r'$s$-поляризация ($I_s$)')
    ax2.plot(alphas[mask_zoom], Ip[mask_zoom], color='#ff7f0e', lw=1.8, ls='--', label=r'$p$-поляризация ($I_p$)')
    ax2.axvline(alpha_c, color='gray', ls=':', lw=1.5, label=f'$\\alpha_c \\approx {alpha_c:.1f}^\\circ$')
    ax2.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax2.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=12)
    ax2.set_title(r'(б) Увеличенный масштаб вблизи $\alpha_c$ ($\alpha \in [1^\circ, 15^\circ]$)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(1.0, 15.0)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(True, which='major', linestyle='-', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Панель (в): Разность между поляризациями
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
    
    ax3_rel = ax3.twinx()
    line_rel, = ax3_rel.plot(alphas, diff_rel, color=color_rel, lw=1.8, ls='-.', label=r'Отн. разность $(I_s - I_p)/I_s$ (%)')
    ax3_rel.set_ylabel(r'Относительная разность $\delta$, %', fontsize=12, color=color_rel)
    ax3_rel.tick_params(axis='y', labelcolor=color_rel)
    ax3_rel.set_ylim(0, max(np.max(diff_rel) * 1.25, 7.0))
    ax3_rel.grid(False)
    
    lines3 = [line_diff, line_rel]
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc='center right', frameon=True, facecolor='white', framealpha=0.9)
    ax3.set_title(r'(в) Различие между поляризациями при $E = 200$ эВ', fontsize=13, fontweight='bold')
    
    out_dir = os.path.join(ROOT_DIR, 'src', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fluorescence_sp_200ev.png')
    fig.savefig(out_path, dpi=300)
    print(f"Сохранен график для 200 эВ: {out_path}")
    
    # ------------------ ГРАФИК 2: СОПОСТАВЛЕНИЕ 520 эВ И 200 эВ ------------------
    # Рассчитаем для 520 эВ относительную разность на той же сетке
    from src.s_polarization.main import PARAMS as P520_s
    from src.p_polarization.main import PARAMS as P520_p
    p520_s = dict(P520_s); p520_s['m'] = m_s
    p520_p = dict(P520_p); p520_p['m'] = m_p
    _, Is_520 = f_s(alphas, phi, p=p520_s, method='analytical')
    _, Ip_520 = f_p(alphas, phi, p=p520_p, method='analytical')
    rel_520 = (Is_520 - Ip_520) / np.maximum(Is_520, 1e-12) * 100.0
    
    fig_comp, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    
    # Левая панель: Сравнение относительного расхождения delta (%)
    ax_e1.plot(alphas, diff_rel, color='#9467bd', lw=2.2, label=r'$E = 200$ эВ ($\lambda = 6.20$ нм, $\chi \approx -0.022$)')
    ax_e1.plot(alphas, rel_520, color='#1f77b4', lw=2.0, ls='--', label=r'$E = 520$ эВ ($\lambda = 2.38$ нм, $\chi \approx -0.0034$)')
    ax_e1.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax_e1.set_ylabel(r'Относительное расхождение $\delta = (I_s - I_p)/I_s$, %', fontsize=12)
    ax_e1.set_title(r'(а) Сравнение поляризационного расхождения: 200 эВ vs 520 эВ', fontsize=13, fontweight='bold')
    ax_e1.set_xlim(0, 40)
    ax_e1.set_ylim(0, max(np.max(diff_rel) * 1.15, 7.0))
    ax_e1.grid(True, alpha=0.5)
    ax_e1.legend(fontsize=11, loc='upper right')
    
    # Правая панель: Сравнение нормированных кривых выхода флуоресценции
    norm_s_200 = Is / np.max(Is)
    norm_p_200 = Ip / np.max(Is)
    norm_s_520 = Is_520 / np.max(Is_520)
    norm_p_520 = Ip_520 / np.max(Is_520)
    
    ax_e2.plot(alphas, norm_s_200, color='#9467bd', lw=2.0, label=r'$I_s$ ($200$ эВ)')
    ax_e2.plot(alphas, norm_p_200, color='#ff7f0e', lw=1.8, ls='--', label=r'$I_p$ ($200$ эВ)')
    ax_e2.plot(alphas, norm_s_520, color='#1f77b4', lw=1.5, ls=':', alpha=0.8, label=r'$I_s$ ($520$ эВ)')
    ax_e2.plot(alphas, norm_p_520, color='#2ca02c', lw=1.5, ls=':', alpha=0.8, label=r'$I_p$ ($520$ эВ)')
    ax_e2.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax_e2.set_ylabel(r'Нормированный выход флуоресценции $I / I_{\max}$', fontsize=12)
    ax_e2.set_title(r'(б) Форма кривых флуоресценции при 200 эВ и 520 эВ', fontsize=13, fontweight='bold')
    ax_e2.set_xlim(0, 40)
    ax_e2.grid(True, alpha=0.5)
    ax_e2.legend(fontsize=10, loc='lower right')
    
    out_path_comp = os.path.join(out_dir, 'energy_comparison_520_vs_200ev.png')
    fig_comp.savefig(out_path_comp, dpi=300)
    print(f"Сохранен график сравнения энергий: {out_path_comp}")
    
    # Печать контрольной таблицы
    print("\n" + "="*80)
    print(f"{'alpha, deg':>10} | {'I_s (200 eV)':>14} | {'I_p (200 eV)':>14} | {'Delta I':>10} | {'Delta I / I_s (%)':>18}")
    print("="*80)
    for a in [0.5, 1.0, 3.0, 5.0, 8.46, 12.0, 16.0, 20.0, 30.0, 40.0]:
        idx = np.argmin(np.abs(alphas - a))
        print(f"{alphas[idx]:10.2f} | {Is[idx]:14.2f} | {Ip[idx]:14.2f} | {diff_abs[idx]:10.2f} | {diff_rel[idx]:17.3f}%")
    print("="*80)


if __name__ == '__main__':
    main()
