"""
Расчет выхода флуоресценции для s- и p-поляризаций при энергии E = 395 эВ
(длина волны lambda = 3.1388 нм, мягкий рентген вблизи K-края азота).

Оптические константы Si3N4 взяты из базы данных Henke / CXRO (LBNL):
    Для плотности образца PTB rho = 2.836 г/см^3:
        E = 395.0 эВ: delta = 0.002721, beta = 0.0007892
        chi = -2 * delta + 2j * beta = -0.005442 + 0.001578j
    Критический угол полного внешнего отражения:
        alpha_c = sqrt(2 * delta) approx 4.23 градусов.
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
    ev = 395.0
    wl = 1239.841984 / ev     # 3.13884 нм
    K = 2 * np.pi / wl        # 2.00175 нм^-1
    
    # Оптические константы Si3N4 при 395 эВ (CXRO Henke, rho = 2.83595 г/см^3)
    delta = 2.720537e-3
    beta = 7.892305e-4
    chi = -2.0 * delta + 2.0j * beta
    eps_1 = 1.0 + chi
    
    # Геометрия решетки (период 100 нм, высота 90 нм, скважность 0.4)
    d = 100.0
    t = 90.0
    gamma = 0.4
    
    # Базис мод: для s-поляризации 15, для p-поляризации 25
    n_s = 15
    n_p = 25
    m_s = np.arange(-n_s, n_s + 1)
    m_p = np.arange(-n_p, n_p + 1)
    
    p_s = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=0.0, chi_sub=chi, m=m_s)
    p_p = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=0.0, chi_sub=chi,
               eps_1=eps_1, eps_b=1.0, eps_sub=eps_1, m=m_p)
    
    alphas = np.linspace(0.05, 40.0, 500)
    phi = 0.0
    
    print(f"1. Расчет s-поляризации при E={ev} эВ (lambda={wl:.3f} нм)...")
    _, Is, Rs = f_s(alphas, phi, p=p_s, method='analytical', return_refl=True)
    
    print(f"2. Расчет p-поляризации при E={ev} эВ (lambda={wl:.3f} нм)...")
    _, Ip, Rp = f_p(alphas, phi, p=p_p, method='analytical', return_refl=True)
    
    diff_abs = Is - Ip
    diff_rel = (Is - Ip) / np.maximum(Is, 1e-12) * 100.0
    
    alpha_c = np.degrees(np.sqrt(2.0 * delta))
    
    # ------------------ ГРАФИК 1: 3-ПАНЕЛЬНЫЙ ДЛЯ 395 эВ ------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # Панель (а): Полный обзор [0, 40] град
    ax1 = axes[0]
    line_s, = ax1.plot(alphas, Is, color='#1f77b4', lw=2.2, label=r'$s$-поляризация ($I_s$)')
    line_p, = ax1.plot(alphas, Ip, color='#ff7f0e', lw=1.8, ls='--', label=r'$p$-поляризация ($I_p$)')
    ax1.axvline(alpha_c, color='gray', ls=':', lw=1.5, label=rf'$\alpha_c \approx {alpha_c:.2f}^\circ$')
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
    ax2.axvline(alpha_c, color='gray', ls=':', lw=1.5, label=rf'$\alpha_c \approx {alpha_c:.2f}^\circ$')
    ax2.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax2.set_ylabel(r'Интенсивность флуоресценции $I_{\mathrm{fluor}}$, отн. ед.', fontsize=12)
    ax2.set_title(r'(б) Увеличенный масштаб вблизи $\alpha_c$ ($\alpha \in [1^\circ, 15^\circ]$)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(1.0, 15.0)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(True, which='major', linestyle='-', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Панель (в): Относительное и абсолютное расхождение
    ax3 = axes[2]
    color_diff = '#d62728'
    ax3.plot(alphas, diff_rel, color=color_diff, lw=2.0, label=r'Относ. разность $\frac{I_s - I_p}{I_s} \times 100\%$')
    ax3.axhline(0, color='black', lw=0.8, ls=':')
    ax3.axvline(alpha_c, color='gray', ls=':', lw=1.5, label=rf'$\alpha_c \approx {alpha_c:.2f}^\circ$')
    ax3.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax3.set_ylabel(r'Относительная разность $(I_s - I_p)/I_s$, %', fontsize=12, color=color_diff)
    ax3.tick_params(axis='y', labelcolor=color_diff)
    ax3.set_title(r'(в) Разность между $s$- и $p$-поляризациями', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, 40)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.grid(True, which='major', linestyle='-', alpha=0.6)
    
    ax3_abs = ax3.twinx()
    color_abs = '#9467bd'
    ax3_abs.plot(alphas, diff_abs, color=color_abs, lw=1.3, ls='--', alpha=0.8,
                 label=r'Абсолютная разность $I_s - I_p$')
    ax3_abs.set_ylabel(r'Абсолютная разность $I_s - I_p$, отн. ед.', fontsize=11, color=color_abs)
    ax3_abs.tick_params(axis='y', labelcolor=color_abs)
    ax3_abs.grid(False)
    
    lines_diff = [ax3.get_lines()[0], ax3_abs.get_lines()[0]]
    labels_diff = [l.get_label() for l in lines_diff]
    ax3.legend(lines_diff, labels_diff, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    out_dir = os.path.join(ROOT_DIR, 'src', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fluorescence_sp_395ev.png')
    plt.savefig(out_path, dpi=300)
    print(f"График успешно сохранен в: {out_path}")
    plt.close()
    
    # ------------------ СТАТИСТИКА ------------------
    print("\n--- Сводка расхождения для 395 эВ ---")
    print(f"Критический угол alpha_c = {alpha_c:.2f} град")
    test_angles = [0.5, 1.0, 2.0, 3.0, 4.23, 6.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    for a in test_angles:
        idx = np.argmin(np.abs(alphas - a))
        s_val = Is[idx]
        p_val = Ip[idx]
        rel = diff_rel[idx]
        print(f"alpha = {alphas[idx]:5.2f} deg: Is = {s_val:8.2f}, Ip = {p_val:8.2f}, (Is-Ip)/Is = {rel:+6.3f}%")


if __name__ == '__main__':
    main()
