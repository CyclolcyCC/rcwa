"""
Построение сводного графика сравнения поляризационного расхождения
для трех энергий: 520 эВ, 395 эВ и 200 эВ.
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


def calc_energy(ev, delta, beta, alphas, n_s=15, n_p=25):
    wl = 1239.841984 / ev
    K = 2 * np.pi / wl
    chi = -2.0 * delta + 2.0j * beta
    eps_1 = 1.0 + chi
    
    d, t, gamma = 100.0, 90.0, 0.4
    m_s = np.arange(-n_s, n_s + 1)
    m_p = np.arange(-n_p, n_p + 1)
    
    p_s = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=0.0, chi_sub=chi, m=m_s)
    p_p = dict(K=K, d=d, t=t, gamma=gamma, chi=chi, chi_b=0.0, chi_sub=chi,
               eps_1=eps_1, eps_b=1.0, eps_sub=eps_1, m=m_p)
    
    _, Is, Rs = f_s(alphas, 0.0, p=p_s, method='analytical', return_refl=True)
    _, Ip, Rp = f_p(alphas, 0.0, p=p_p, method='analytical', return_refl=True)
    
    diff_rel = (Is - Ip) / np.maximum(Is, 1e-12) * 100.0
    alpha_c = np.degrees(np.sqrt(2.0 * delta))
    return Is, Ip, diff_rel, alpha_c, wl


def main():
    alphas = np.linspace(0.05, 40.0, 300)
    
    cases = [
        {"ev": 520.0, "delta": 1.703e-3, "beta": 3.99e-5, "color": "#1f77b4", "name": "520 эВ (λ = 2.38 нм)"},
        {"ev": 395.0, "delta": 2.721e-3, "beta": 7.89e-4, "color": "#2ca02c", "name": "395 эВ (λ = 3.14 нм)"},
        {"ev": 200.0, "delta": 0.0109,   "beta": 0.0067, "color": "#d62728", "name": "200 эВ (λ = 6.20 нм)"},
    ]
    
    results = []
    for c in cases:
        print(f"Расчет для {c['ev']} эВ...")
        Is, Ip, diff_rel, alpha_c, wl = calc_energy(c['ev'], c['delta'], c['beta'], alphas)
        c['Is'] = Is
        c['Ip'] = Ip
        c['diff_rel'] = diff_rel
        c['alpha_c'] = alpha_c
        c['wl'] = wl
        results.append(c)
        
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    
    # Левая панель: Относительная разность (Is - Ip)/Is
    ax1 = axes[0]
    for c in results:
        ax1.plot(alphas, c['diff_rel'], color=c['color'], lw=2.2,
                 label=rf"{c['name']}, $\alpha_c \approx {c['alpha_c']:.1f}^\circ$")
        ax1.axvline(c['alpha_c'], color=c['color'], ls=':', lw=1.2, alpha=0.7)
    
    ax1.axhline(0, color='black', lw=0.8, ls='-')
    ax1.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax1.set_ylabel(r'Относительное расхождение $(I_s - I_p)/I_s$, %', fontsize=12)
    ax1.set_title(r'(а) Эволюция поляризационного расхождения с энергией', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 40)
    ax1.set_ylim(-1.5, 12.0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(True, which='major', linestyle='-', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    # Правая панель: Увеличение в диапазоне [0, 15] град
    ax2 = axes[1]
    for c in results:
        mask = (alphas <= 15.0)
        ax2.plot(alphas[mask], c['diff_rel'][mask], color=c['color'], lw=2.2,
                 label=rf"{c['name']}")
        ax2.axvline(c['alpha_c'], color=c['color'], ls=':', lw=1.2, alpha=0.7)
        
    ax2.axhline(0, color='black', lw=0.8, ls='-')
    ax2.set_xlabel(r'Угол падения $\alpha$, град', fontsize=12)
    ax2.set_ylabel(r'Относительное расхождение $(I_s - I_p)/I_s$, %', fontsize=12)
    ax2.set_title(r'(б) Масштаб $\alpha \in [0^\circ, 15^\circ]$ (область полного внешнего отражения)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-1.0, 12.0)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(True, which='major', linestyle='-', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    out_dir = os.path.join(ROOT_DIR, 'src', 'figures')
    out_path = os.path.join(out_dir, 'energy_comparison_520_395_200ev.png')
    plt.savefig(out_path, dpi=300)
    print(f"Сводный график сохранен в: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
