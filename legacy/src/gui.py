"""
Интерактивный графический интерфейс для RCWA-моделирования
многослойных периодических наноструктур и анализа различий s- и p-поляризаций.

Возможности интерфейса:
1. Задание восприимчивости chi материала через чистые поля Re/Im или delta/beta.
2. База данных ранее сохраненных значений восприимчивости (энергия, дельта, бета)
   с возможностью выбора из списка и сохранения новых значений (с проверкой дубликатов).
3. Послойное задание формы структуры в виде наслоенных прямоугольников.
4. Выбор метода расчета флуоресценции: полуаналитический (по умолчанию)
   или численный (галочка "Численное интегрирование флуоресценции").
5. Выбор графиков для расчета с выводом каждого выбранного графика в отдельное окно.
6. Настройка детализации сеток по обеим осям для всех карт.
7. Настройки ближнего поля: угол alpha, галочка расчета для phi != 0 с вводом phi.
8. Дисперсионные кривые отображают обе части: действительную Re и мнимую Im.
9. Графики и карты сравнений с вычитанием сохраняются в папку figures без вывода на экран.
10. СТРОГОЕ ПРАВИЛО: ни на одном графике флуоресценции не рисуется кривая отражательной способности!
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# Добавление корневой директории в sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.multislice.model import MultiSliceStructure, LayerDef, BoxDef
from src.multislice.rcwa_s import solve_multislice_s
from src.multislice.rcwa_p import solve_multislice_p
from src.multislice.solver import (
    plot_structure_geometry,
    compute_and_plot_dispersion,
    plot_eigenvectors,
    compute_and_plot_near_field,
    compute_and_save_near_field_diff,
    scan_fluorescence_1d,
    plot_fluorescence_single,
    plot_fluorescence_comparison_onscreen,
    save_fluorescence_comparison_with_diff,
    scan_fluorescence_2d,
    plot_fluorescence_2d_map,
    save_fluorescence_2d_diff_map
)


class RcwaGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RCWA GIXRF: Многослойные наноструктуры и поляризации")
        self.root.geometry("1420x880")
        self.root.minsize(1150, 720)

        self.figures_dir = os.path.join(ROOT_DIR, "src", "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.db_path = os.path.join(ROOT_DIR, "src", "materials_db.json")

        # Текущая структура (по умолчанию - одиночный прямоугольник 90x40 нм)
        self.structure = MultiSliceStructure.single_rectangle(
            period=100.0, height=90.0, cd=40.0,
            chi=complex(-0.003406, 0.0000798),
            chi_sub=complex(-0.003406, 0.0000798)
        )

        self.is_calculating = False
        self._build_layout()
        self._load_materials_into_combo()
        self._load_structure_into_table()
        self.update_geometry_preview()

    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ================= ЛЕВАЯ ПАНЕЛЬ: ЭСКИЗ СТРУКТУРЫ И СВОДКА =================
        left_frame = ttk.Frame(paned, padding=6)
        paned.add(left_frame, weight=3)

        lbl_geom_title = ttk.Label(
            left_frame,
            text="Эскиз геометрического профиля структуры",
            font=("TkDefaultFont", 11, "bold"),
            foreground="#1e293b"
        )
        lbl_geom_title.pack(anchor="w", pady=(0, 4))

        self.fig_geom = Figure(figsize=(7, 5.8), dpi=100)
        self.canvas_geom = FigureCanvasTkAgg(self.fig_geom, master=left_frame)
        self.canvas_geom.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_geom = NavigationToolbar2Tk(self.canvas_geom, left_frame)
        self.toolbar_geom.update()

        # Информационная плашка свойств структуры
        self.info_frame = ttk.LabelFrame(left_frame, text="Сводная информация о текущей модели", padding=6)
        self.info_frame.pack(fill="x", pady=(6, 2))

        self.lbl_info_geom = ttk.Label(
            self.info_frame,
            text="Период d: 100.0 нм | Высота H: 90.0 нм | Число срезов: 1 | Доля заполнения Γ: 0.40",
            font=("TkDefaultFont", 9)
        )
        self.lbl_info_geom.pack(anchor="w")

        self.lbl_info_mat = ttk.Label(
            self.info_frame,
            text="Материал: χ = -3.4060e-03 + 7.9800e-05j (δ = 1.7030e-03, β = 3.9900e-05)",
            font=("TkDefaultFont", 9),
            foreground="#0369a1"
        )
        self.lbl_info_mat.pack(anchor="w")

        # ================= ПРАВАЯ ПАНЕЛЬ: УПРАВЛЕНИЕ И ПАРАМЕТРЫ =================
        right_container = ttk.Frame(paned, padding=4)
        paned.add(right_container, weight=2)

        ctrl_canvas = tk.Canvas(right_container, borderwidth=0, highlightthickness=0, width=500)
        scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=ctrl_canvas.yview)
        self.ctrl_frame = ttk.Frame(ctrl_canvas, padding=6)

        self.ctrl_frame.bind(
            "<Configure>", lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
        )
        ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)

        ctrl_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_beam_section()
        self._build_material_section()
        self._build_layers_section()
        self._build_grid_and_maps_section()
        self._build_plots_selection_section()
        self._build_actions_section()

    # ------------------ СЕКЦИЯ 1: ИЗЛУЧЕНИЕ И УГЛЫ ------------------
    def _build_beam_section(self):
        sec = ttk.LabelFrame(self.ctrl_frame, text="1. Параметры излучения и 1D сканирования", padding=6)
        sec.pack(fill="x", pady=4)

        f1 = ttk.Frame(sec)
        f1.pack(fill="x", pady=2)
        ttk.Label(f1, text="Энергия фотонов E, эВ:").pack(side="left")
        self.var_energy = tk.StringVar(value="520.0")
        self.var_energy.trace_add("write", self._on_energy_change)
        e_entry = ttk.Entry(f1, textvariable=self.var_energy, width=8)
        e_entry.pack(side="left", padx=4)

        self.lbl_wl = ttk.Label(f1, text="λ = 2.384 нм", foreground="#0369a1", font=("TkDefaultFont", 9, "bold"))
        self.lbl_wl.pack(side="left", padx=6)

        f2 = ttk.Frame(sec)
        f2.pack(fill="x", pady=2)
        ttk.Label(f2, text="Угол α от:").pack(side="left")
        self.var_alpha_min = tk.StringVar(value="0.05")
        ttk.Entry(f2, textvariable=self.var_alpha_min, width=5).pack(side="left", padx=2)
        ttk.Label(f2, text="до:").pack(side="left")
        self.var_alpha_max = tk.StringVar(value="20.0")
        ttk.Entry(f2, textvariable=self.var_alpha_max, width=5).pack(side="left", padx=2)
        ttk.Label(f2, text="град, точек:").pack(side="left")
        self.var_n_alpha = tk.StringVar(value="150")
        ttk.Entry(f2, textvariable=self.var_n_alpha, width=5).pack(side="left", padx=2)

        f3 = ttk.Frame(sec)
        f3.pack(fill="x", pady=2)
        ttk.Label(f3, text="Мод s:").pack(side="left")
        self.var_ns = tk.StringVar(value="15")
        ttk.Entry(f3, textvariable=self.var_ns, width=4).pack(side="left", padx=2)
        ttk.Label(f3, text="Мод p:").pack(side="left", padx=(8, 0))
        self.var_np = tk.StringVar(value="25")
        ttk.Entry(f3, textvariable=self.var_np, width=4).pack(side="left", padx=2)

    def _on_energy_change(self, *args):
        try:
            ev = float(self.var_energy.get())
            if ev > 0:
                wl = 1239.841984 / ev
                self.lbl_wl.config(text=f"λ = {wl:.3f} нм")
                self._update_info_panel()
        except ValueError:
            pass

    # ------------------ СЕКЦИЯ 2: ВОСПРИИМЧИВОСТЬ МАТЕРИАЛА ------------------
    def _build_material_section(self):
        sec = ttk.LabelFrame(self.ctrl_frame, text="2. Восприимчивость материала", padding=6)
        sec.pack(fill="x", pady=4)

        # Выбор из базы сохраненных материалов
        f_db = ttk.Frame(sec)
        f_db.pack(fill="x", pady=2)
        ttk.Label(f_db, text="Сохраненные:").pack(side="left")
        self.combo_saved_materials = ttk.Combobox(f_db, state="readonly", width=34)
        self.combo_saved_materials.pack(side="left", padx=4)
        self.combo_saved_materials.bind("<<ComboboxSelected>>", self._on_select_saved_material)

        # Переключатель формата ввода chi vs delta/beta
        f_mode = ttk.Frame(sec)
        f_mode.pack(fill="x", pady=2)
        ttk.Label(f_mode, text="Формат:").pack(side="left")
        self.var_mat_mode = tk.StringVar(value="chi")
        r1 = ttk.Radiobutton(f_mode, text="χ' и χ''", value="chi", variable=self.var_mat_mode, command=self._on_mat_mode_toggle)
        r2 = ttk.Radiobutton(f_mode, text="δ и β", value="delta_beta", variable=self.var_mat_mode, command=self._on_mat_mode_toggle)
        r1.pack(side="left", padx=4)
        r2.pack(side="left", padx=4)

        # Поля ввода для решетки
        self.f_chi = ttk.Frame(sec)
        self.f_chi.pack(fill="x", pady=2)
        self.lbl_chi_re = ttk.Label(self.f_chi, text="Решетка: Re:")
        self.lbl_chi_re.pack(side="left")
        self.var_chi_re = tk.StringVar(value="-0.003406")
        ttk.Entry(self.f_chi, textvariable=self.var_chi_re, width=11).pack(side="left", padx=2)
        self.lbl_chi_im = ttk.Label(self.f_chi, text="Im:")
        self.lbl_chi_im.pack(side="left")
        self.var_chi_im = tk.StringVar(value="0.0000798")
        ttk.Entry(self.f_chi, textvariable=self.var_chi_im, width=11).pack(side="left", padx=2)

        # Поля ввода для подложки
        f_sub = ttk.Frame(sec)
        f_sub.pack(fill="x", pady=2)
        self.lbl_sub_re = ttk.Label(f_sub, text="Подложка: Re:")
        self.lbl_sub_re.pack(side="left")
        self.var_chi_sub_re = tk.StringVar(value="-0.003406")
        ttk.Entry(f_sub, textvariable=self.var_chi_sub_re, width=11).pack(side="left", padx=2)
        self.lbl_sub_im = ttk.Label(f_sub, text="Im:")
        self.lbl_sub_im.pack(side="left")
        self.var_chi_sub_im = tk.StringVar(value="0.0000798")
        ttk.Entry(f_sub, textvariable=self.var_chi_sub_im, width=11).pack(side="left", padx=2)

        # Кнопки применения и сохранения
        f_btns = ttk.Frame(sec)
        f_btns.pack(fill="x", pady=3)
        ttk.Button(f_btns, text="Применить χ к структуре", command=self._apply_chi_to_structure).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(f_btns, text="Сохранить текущие значения в базу", command=self._on_save_current_material).pack(side="left", expand=True, fill="x", padx=1)

    def _read_materials_db(self):
        if not os.path.exists(self.db_path):
            defaults = [
                {"energy_eV": 520.0, "delta": 0.001703, "beta": 3.99e-05, "label": "520 эВ"},
                {"energy_eV": 395.0, "delta": 0.002955, "beta": 0.0006455, "label": "395 эВ"},
                {"energy_eV": 200.0, "delta": 0.01157, "beta": 0.002456, "label": "200 эВ"}
            ]
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=2, ensure_ascii=False)
            return defaults
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_materials_db(self, data):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_materials_into_combo(self):
        items = self._read_materials_db()
        values = []
        for it in items:
            ev = it.get("energy_eV", 0.0)
            d = it.get("delta", 0.0)
            b = it.get("beta", 0.0)
            values.append(f"{ev:.1f} эВ | δ = {d:.4e}, β = {b:.4e}")
        self.combo_saved_materials["values"] = values
        if values:
            self.combo_saved_materials.current(0)

    def _on_select_saved_material(self, event=None):
        idx = self.combo_saved_materials.current()
        if idx < 0:
            return
        items = self._read_materials_db()
        if idx >= len(items):
            return
        it = items[idx]
        ev = float(it.get("energy_eV", 520.0))
        delta = float(it.get("delta", 0.001703))
        beta = float(it.get("beta", 3.99e-5))

        self.var_energy.set(f"{ev:.1f}")

        mode = self.var_mat_mode.get()
        if mode == "delta_beta":
            self.var_chi_re.set(f"{delta:.6e}")
            self.var_chi_im.set(f"{beta:.6e}")
            self.var_chi_sub_re.set(f"{delta:.6e}")
            self.var_chi_sub_im.set(f"{beta:.6e}")
        else:
            chi_re = -2.0 * delta
            chi_im = 2.0 * beta
            self.var_chi_re.set(f"{chi_re:.6e}")
            self.var_chi_im.set(f"{chi_im:.6e}")
            self.var_chi_sub_re.set(f"{chi_re:.6e}")
            self.var_chi_sub_im.set(f"{chi_im:.6e}")

        self._apply_chi_to_structure(silent=True)

    def _on_save_current_material(self):
        try:
            ev = float(self.var_energy.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное значение энергии излучения!")
            return

        mode = self.var_mat_mode.get()
        try:
            v1 = float(self.var_chi_re.get())
            v2 = float(self.var_chi_im.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные значения восприимчивости!")
            return

        if mode == "delta_beta":
            delta, beta = v1, v2
        else:
            delta = -v1 / 2.0
            beta = v2 / 2.0

        items = self._read_materials_db()
        # Проверка на наличие уже сохраненной записи
        for it in items:
            e_diff = abs(it.get("energy_eV", 0.0) - ev)
            d_diff = abs(it.get("delta", 0.0) - delta)
            b_diff = abs(it.get("beta", 0.0) - beta)
            if e_diff < 0.05 and d_diff < 1e-8 and b_diff < 1e-8:
                messagebox.showinfo("Сохранение", "Уже сохранено!")
                return

        new_entry = {
            "energy_eV": round(ev, 2),
            "delta": delta,
            "beta": beta,
            "label": f"{ev:.1f} эВ"
        }
        items.append(new_entry)
        self._write_materials_db(items)
        self._load_materials_into_combo()
        self.combo_saved_materials.current(len(items) - 1)
        messagebox.showinfo("Сохранение", f"Успешно сохранено: {ev:.1f} эВ")

    def _on_mat_mode_toggle(self):
        mode = self.var_mat_mode.get()
        if mode == "delta_beta":
            self.lbl_chi_re.config(text="Решетка: δ:")
            self.lbl_chi_im.config(text="β:")
            self.lbl_sub_re.config(text="Подложка: δ:")
            self.lbl_sub_im.config(text="β:")
            try:
                c_re = float(self.var_chi_re.get())
                c_im = float(self.var_chi_im.get())
                self.var_chi_re.set(f"{-c_re / 2.0:.6e}")
                self.var_chi_im.set(f"{c_im / 2.0:.6e}")
                s_re = float(self.var_chi_sub_re.get())
                s_im = float(self.var_chi_sub_im.get())
                self.var_chi_sub_re.set(f"{-s_re / 2.0:.6e}")
                self.var_chi_sub_im.set(f"{s_im / 2.0:.6e}")
            except ValueError:
                pass
        else:
            self.lbl_chi_re.config(text="Решетка: Re:")
            self.lbl_chi_im.config(text="Im:")
            self.lbl_sub_re.config(text="Подложка: Re:")
            self.lbl_sub_im.config(text="Im:")
            try:
                d_val = float(self.var_chi_re.get())
                b_val = float(self.var_chi_im.get())
                self.var_chi_re.set(f"{-2.0 * d_val:.6e}")
                self.var_chi_im.set(f"{2.0 * b_val:.6e}")
                sd_val = float(self.var_chi_sub_re.get())
                sb_val = float(self.var_chi_sub_im.get())
                self.var_chi_sub_re.set(f"{-2.0 * sd_val:.6e}")
                self.var_chi_sub_im.set(f"{2.0 * sb_val:.6e}")
            except ValueError:
                pass
        self._update_info_panel()

    def _get_current_chi(self):
        mode = self.var_mat_mode.get()
        v1 = float(self.var_chi_re.get())
        v2 = float(self.var_chi_im.get())
        if mode == "delta_beta":
            return complex(-2.0 * v1, 2.0 * v2)
        return complex(v1, v2)

    def _get_current_chi_sub(self):
        mode = self.var_mat_mode.get()
        v1 = float(self.var_chi_sub_re.get())
        v2 = float(self.var_chi_sub_im.get())
        if mode == "delta_beta":
            return complex(-2.0 * v1, 2.0 * v2)
        return complex(v1, v2)

    def _apply_chi_to_structure(self, silent=False):
        chi = self._get_current_chi()
        chi_sub = self._get_current_chi_sub()
        self.structure.chi_sub = chi_sub
        for lay in self.structure.layers:
            for b in lay.boxes:
                b.chi = chi
        self.update_geometry_preview()
        if not silent:
            messagebox.showinfo("Успешно", "Восприимчивость обновлена для всех слоев и подложки!")

    # ------------------ СЕКЦИЯ 3: ПОСЛОЙНАЯ СТРУКТУРА ------------------
    def _build_layers_section(self):
        sec = ttk.LabelFrame(self.ctrl_frame, text="3. Послойное задание формы структуры", padding=6)
        sec.pack(fill="x", pady=4)

        fp = ttk.Frame(sec)
        fp.pack(fill="x", pady=2)
        ttk.Label(fp, text="Период d, нм:").pack(side="left")
        self.var_period = tk.StringVar(value="100.0")
        ttk.Entry(fp, textvariable=self.var_period, width=8).pack(side="left", padx=4)

        fpr = ttk.Frame(sec)
        fpr.pack(fill="x", pady=3)
        ttk.Label(fpr, text="Готовые пресеты:").pack(side="left")
        self.combo_preset = ttk.Combobox(fpr, state="readonly", values=[
            "Одиночный прямоугольник 90x40 нм",
            "Трапеция 5 слоев, SWA = 85°",
            "Трапеция со скруглением 10 слоев, r = 5 нм",
            "Ступенчатый профиль 3 слоя"
        ], width=34)
        self.combo_preset.current(0)
        self.combo_preset.pack(side="left", padx=4)
        ttk.Button(sec, text="Загрузить выбранный пресет", command=self._on_load_preset).pack(fill="x", pady=2)

        ttk.Label(sec, text="Слои структуры сверху вниз:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(6, 2))
        table_frame = ttk.Frame(sec)
        table_frame.pack(fill="x")

        cols = ("idx", "thick", "left", "right", "width")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=5)
        self.tree.heading("idx", text="№")
        self.tree.heading("thick", text="h, нм")
        self.tree.heading("left", text="x_лев")
        self.tree.heading("right", text="x_прав")
        self.tree.heading("width", text="w, нм")

        self.tree.column("idx", width=35, anchor="center")
        self.tree.column("thick", width=65, anchor="center")
        self.tree.column("left", width=65, anchor="center")
        self.tree.column("right", width=65, anchor="center")
        self.tree.column("width", width=65, anchor="center")
        self.tree.pack(side="left", fill="x", expand=True)
        self.tree.bind("<Double-1>", lambda event: self._on_edit_layer_dialog())

        sc_tree = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sc_tree.set)
        sc_tree.pack(side="right", fill="y")

        f_edit = ttk.Frame(sec)
        f_edit.pack(fill="x", pady=3)
        ttk.Button(f_edit, text="➕ Добавить слой", command=self._on_add_layer_dialog).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(f_edit, text="✏ Редактировать", command=self._on_edit_layer_dialog).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(f_edit, text="❌ Удалить", command=self._on_delete_layer).pack(side="left", expand=True, fill="x", padx=1)

        ttk.Button(sec, text="🔄 Обновить эскиз структуры", command=self.update_geometry_preview).pack(fill="x", pady=3)

    def _load_structure_into_table(self):
        self.tree.delete(*self.tree.get_children())
        for j, lay in enumerate(self.structure.layers):
            if lay.boxes:
                b = lay.boxes[0]
                left = f"{b.left:.1f}"
                right = f"{b.right:.1f}"
                width = f"{b.width:.1f}"
            else:
                left, right, width = "-", "-", "-"
            self.tree.insert("", "end", values=(j + 1, f"{lay.thickness:.1f}", left, right, width))
        self._update_info_panel()

    def _on_load_preset(self):
        choice = self.combo_preset.get()
        period = float(self.var_period.get() or 100.0)
        chi = self._get_current_chi()
        chi_sub = self._get_current_chi_sub()

        if "Одиночный прямоугольник" in choice:
            self.structure = MultiSliceStructure.single_rectangle(period=period, height=90.0, cd=40.0, chi=chi, chi_sub=chi_sub)
        elif "Трапеция 5 слоев" in choice:
            self.structure = MultiSliceStructure.trapezoid(period=period, height=90.0, cd_mid=40.0, swa_deg=85.0, n_slices=5, chi=chi, chi_sub=chi_sub)
        elif "скруглением" in choice:
            self.structure = MultiSliceStructure.rounded_trapezoid(period=period, height=90.0, cd_mid=40.0, swa_deg=85.0, r_top=5.0, n_slices=10, chi=chi, chi_sub=chi_sub)
        elif "Ступенчатый" in choice:
            self.structure = MultiSliceStructure(period=period, chi_sub=chi_sub)
            self.structure.add_layer(30.0).add_box(-15.0, 15.0, chi)
            self.structure.add_layer(30.0).add_box(-20.0, 20.0, chi)
            self.structure.add_layer(30.0).add_box(-25.0, 25.0, chi)

        self._load_structure_into_table()
        self.update_geometry_preview()

    def _on_add_layer_dialog(self):
        thick = simpledialog.askfloat("Новый слой", "Толщина слоя h, нм:", initialvalue=18.0, minvalue=0.1)
        if thick is None:
            return
        left = simpledialog.askfloat("Новый слой", "Левая координата x_лев, нм:", initialvalue=-20.0)
        if left is None:
            return
        right = simpledialog.askfloat("Новый слой", "Правая координата x_прав, нм:", initialvalue=20.0)
        if right is None:
            return

        chi = self._get_current_chi()
        lay = self.structure.add_layer(thick, name=f"Slice {self.structure.n_layers + 1}")
        lay.add_box(left, right, chi)
        self._load_structure_into_table()
        self.update_geometry_preview()

    def _on_edit_layer_dialog(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Выбор", "Выберите слой в таблице для редактирования.")
            return
        idx = int(self.tree.item(sel[0])['values'][0]) - 1
        lay = self.structure.layers[idx]
        b = lay.boxes[0] if lay.boxes else None

        thick = simpledialog.askfloat("Редактирование", "Толщина слоя h, нм:", initialvalue=lay.thickness, minvalue=0.1)
        if thick is not None:
            lay.thickness = thick
        if b:
            l_val = simpledialog.askfloat("Редактирование", "Левая координата x_лев, нм:", initialvalue=b.left)
            r_val = simpledialog.askfloat("Редактирование", "Правая координата x_прав, нм:", initialvalue=b.right)
            if l_val is not None and r_val is not None:
                b.left = min(l_val, r_val)
                b.right = max(l_val, r_val)

        self._load_structure_into_table()
        self.update_geometry_preview()

    def _on_delete_layer(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(self.tree.item(sel[0])['values'][0]) - 1
        if len(self.structure.layers) <= 1:
            messagebox.showwarning("Ошибка", "В структуре должен оставаться хотя бы один слой!")
            return
        self.structure.layers.pop(idx)
        self._load_structure_into_table()
        self.update_geometry_preview()

    # ------------------ СЕКЦИЯ 4: НАСТРОЙКИ СЕТОК И МЕТОД РАСЧЕТА ------------------
    def _build_grid_and_maps_section(self):
        sec = ttk.LabelFrame(self.ctrl_frame, text="4. Параметры сеток и метод расчета", padding=6)
        sec.pack(fill="x", pady=4)

        # Выбор метода расчета флуоресценции
        f_method = ttk.Frame(sec)
        f_method.pack(fill="x", pady=2)
        self.var_numerical_fluor = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f_method,
            text="Численное интегрирование флуоресценции",
            variable=self.var_numerical_fluor
        ).pack(side="left")

        ttk.Separator(sec, orient="horizontal").pack(fill="x", pady=4)

        # Ближнее поле
        ttk.Label(sec, text="Параметры ближнего поля |E|²:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

        f_nf_angles = ttk.Frame(sec)
        f_nf_angles.pack(fill="x", pady=2)
        ttk.Label(f_nf_angles, text="Угол α_NF, град:").pack(side="left")
        self.var_nf_alpha = tk.StringVar(value="6.0")
        ttk.Entry(f_nf_angles, textvariable=self.var_nf_alpha, width=6).pack(side="left", padx=3)

        self.var_nf_enable_phi = tk.BooleanVar(value=False)
        self.chk_nf_phi = ttk.Checkbutton(
            f_nf_angles,
            text="посчитать ближнее поле для φ ≠ 0",
            variable=self.var_nf_enable_phi,
            command=self._on_toggle_nf_phi
        )
        self.chk_nf_phi.pack(side="left", padx=(8, 3))

        ttk.Label(f_nf_angles, text="φ:").pack(side="left")
        self.var_nf_phi = tk.StringVar(value="0.0")
        self.entry_nf_phi = ttk.Entry(f_nf_angles, textvariable=self.var_nf_phi, width=5, state="disabled")
        self.entry_nf_phi.pack(side="left", padx=2)

        f_nf_grid = ttk.Frame(sec)
        f_nf_grid.pack(fill="x", pady=2)
        ttk.Label(f_nf_grid, text="Сетка ближнего поля: Nx =").pack(side="left")
        self.var_nf_nx = tk.StringVar(value="200")
        ttk.Entry(f_nf_grid, textvariable=self.var_nf_nx, width=5).pack(side="left", padx=2)
        ttk.Label(f_nf_grid, text="Nz =").pack(side="left")
        self.var_nf_nz = tk.StringVar(value="200")
        ttk.Entry(f_nf_grid, textvariable=self.var_nf_nz, width=5).pack(side="left", padx=2)

        ttk.Separator(sec, orient="horizontal").pack(fill="x", pady=4)

        # 2D Карты флуоресценции
        ttk.Label(sec, text="Параметры 2D карт флуоресценции:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

        f_2d_grid = ttk.Frame(sec)
        f_2d_grid.pack(fill="x", pady=2)
        ttk.Label(f_2d_grid, text="Сетка 2D: N_α =").pack(side="left")
        self.var_2d_na = tk.StringVar(value="40")
        ttk.Entry(f_2d_grid, textvariable=self.var_2d_na, width=5).pack(side="left", padx=2)
        ttk.Label(f_2d_grid, text="N_φ =").pack(side="left")
        self.var_2d_np = tk.StringVar(value="40")
        ttk.Entry(f_2d_grid, textvariable=self.var_2d_np, width=5).pack(side="left", padx=2)

        f_2d_range = ttk.Frame(sec)
        f_2d_range.pack(fill="x", pady=2)
        ttk.Label(f_2d_range, text="α 2D: [").pack(side="left")
        self.var_2d_amin = tk.StringVar(value="0.05")
        ttk.Entry(f_2d_range, textvariable=self.var_2d_amin, width=4).pack(side="left")
        ttk.Label(f_2d_range, text=",").pack(side="left")
        self.var_2d_amax = tk.StringVar(value="3.0")
        ttk.Entry(f_2d_range, textvariable=self.var_2d_amax, width=4).pack(side="left")
        ttk.Label(f_2d_range, text="]°  φ 2D: [").pack(side="left")
        self.var_2d_pmin = tk.StringVar(value="-1.5")
        ttk.Entry(f_2d_range, textvariable=self.var_2d_pmin, width=4).pack(side="left")
        ttk.Label(f_2d_range, text=",").pack(side="left")
        self.var_2d_pmax = tk.StringVar(value="1.5")
        ttk.Entry(f_2d_range, textvariable=self.var_2d_pmax, width=4).pack(side="left")
        ttk.Label(f_2d_range, text="]°").pack(side="left")

    def _on_toggle_nf_phi(self):
        if self.var_nf_enable_phi.get():
            self.entry_nf_phi.config(state="normal")
            if self.var_nf_phi.get() == "0.0":
                self.var_nf_phi.set("1.0")
        else:
            self.entry_nf_phi.config(state="disabled")
            self.var_nf_phi.set("0.0")

    # ------------------ СЕКЦИЯ 5: ВЫБОР ГРАФИКОВ ------------------
    def _build_plots_selection_section(self):
        sec = ttk.LabelFrame(self.ctrl_frame, text="5. Выбор графиков для расчета", padding=6)
        sec.pack(fill="x", pady=4)

        self.chk_vars = {
            "geom_save": tk.BooleanVar(value=True),
            "fluor_1d_sp": tk.BooleanVar(value=True),
            "fluor_1d_s": tk.BooleanVar(value=False),
            "fluor_1d_p": tk.BooleanVar(value=False),
            "near_field_s": tk.BooleanVar(value=True),
            "near_field_p": tk.BooleanVar(value=True),
            "dispersion_s": tk.BooleanVar(value=False),
            "dispersion_p": tk.BooleanVar(value=False),
            "eigenvectors": tk.BooleanVar(value=False),
            "fluor_2d_s": tk.BooleanVar(value=False),
            "fluor_2d_p": tk.BooleanVar(value=False),
        }

        ttk.Checkbutton(sec, text="1. Эскиз геометрии структуры", variable=self.chk_vars["geom_save"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="2. Сравнение флуоресценции s и p", variable=self.chk_vars["fluor_1d_sp"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="3. Кривая флуоресценции s-поляризации", variable=self.chk_vars["fluor_1d_s"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="4. Кривая флуоресценции p-поляризации", variable=self.chk_vars["fluor_1d_p"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="5. Ближнее поле s-поляризации", variable=self.chk_vars["near_field_s"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="6. Ближнее поле p-поляризации", variable=self.chk_vars["near_field_p"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="7. Дисперсионные кривые s-поляризации", variable=self.chk_vars["dispersion_s"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="8. Дисперсионные кривые p-поляризации", variable=self.chk_vars["dispersion_p"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="9. Карты собственных векторов", variable=self.chk_vars["eigenvectors"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="10. 2D карта выхода флуоресценции s-поляризации", variable=self.chk_vars["fluor_2d_s"]).pack(anchor="w")
        ttk.Checkbutton(sec, text="11. 2D карта выхода флуоресценции p-поляризации", variable=self.chk_vars["fluor_2d_p"]).pack(anchor="w")

        lbl_diff_hint = ttk.Label(
            sec,
            text="Карты и графики вычитания сохраняются в папку figures без вывода на экран.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="#64748b",
            wraplength=460
        )
        lbl_diff_hint.pack(anchor="w", pady=(4, 2))

        fb = ttk.Frame(sec)
        fb.pack(fill="x", pady=3)
        ttk.Button(fb, text="Выбрать все", command=lambda: self._set_all_checks(True)).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(fb, text="Базовый набор", command=self._set_basic_checks).pack(side="left", expand=True, fill="x", padx=1)

    def _set_all_checks(self, val):
        for v in self.chk_vars.values():
            v.set(val)

    def _set_basic_checks(self):
        for k, v in self.chk_vars.items():
            v.set(k in ("geom_save", "fluor_1d_sp", "near_field_s", "near_field_p"))

    # ------------------ СЕКЦИЯ 6: ДЕЙСТВИЯ И ЗАПУСК ------------------
    def _build_actions_section(self):
        sec = ttk.LabelFrame(self.ctrl_frame, text="6. Запуск вычислений и экспорт", padding=6)
        sec.pack(fill="x", pady=4)

        self.btn_run = ttk.Button(sec, text="🚀 Запустить расчет", command=self.on_run_click)
        self.btn_run.pack(fill="x", ipady=5, pady=3)

        self.progress = ttk.Progressbar(sec, mode="indeterminate")
        self.progress.pack(fill="x", pady=2)

        self.lbl_status = ttk.Label(sec, text="Готов к расчету.", foreground="#475569", wraplength=460)
        self.lbl_status.pack(anchor="w", pady=2)

        f_dir = ttk.Frame(sec)
        f_dir.pack(fill="x", pady=3)
        ttk.Button(f_dir, text="📁 Открыть папку с графиками", command=self._open_figures_folder).pack(fill="x")

    def _open_figures_folder(self):
        try:
            os.startfile(self.figures_dir)
        except Exception:
            messagebox.showinfo("Папка с графиками", f"Путь к директории:\n{self.figures_dir}")

    # ================= СВОДНАЯ ПАНЕЛЬ И ЭСКИЗ ГЕОМЕТРИИ =================
    def _update_info_panel(self):
        d = float(self.var_period.get() or 100.0)
        H = self.structure.total_thickness
        n_lay = self.structure.n_layers
        avg_w = self.structure.layers[0].total_box_width if self.structure.layers else 40.0
        gamma = avg_w / d if d > 0 else 0.4

        self.lbl_info_geom.config(
            text=f"Период d: {d:.1f} нм | Высота H: {H:.1f} нм | Число срезов: {n_lay} | Доля заполнения Γ: {gamma:.2f}"
        )

        chi = self._get_current_chi()
        delta = -chi.real / 2.0
        beta = chi.imag / 2.0
        self.lbl_info_mat.config(
            text=f"Материал: χ = {chi.real:+.4e}{chi.imag:+.4e}j (δ = {delta:.4e}, β = {beta:.4e})"
        )

    def update_geometry_preview(self):
        try:
            self.structure.period = float(self.var_period.get() or 100.0)
        except ValueError:
            pass
        self.fig_geom.clear()
        ax = self.fig_geom.add_subplot(111)
        plot_structure_geometry(self.structure, ax=ax, title="Геометрический профиль структуры")
        self.canvas_geom.draw()
        self._update_info_panel()

    # ================= РАСЧЕТ В ФОНОВОМ ПОТОКЕ =================
    def on_run_click(self):
        if self.is_calculating:
            return
        self.is_calculating = True
        self.btn_run.config(state="disabled")
        self.progress.start(10)
        method_name = "численный" if self.var_numerical_fluor.get() else "полуаналитический"
        self.lbl_status.config(text=f"Идет расчет RCWA (метод: {method_name})...", foreground="#0284c7")

        thread = threading.Thread(target=self._run_simulation_thread, daemon=True)
        thread.start()

    def _run_simulation_thread(self):
        try:
            ev = float(self.var_energy.get())
            wl = 1239.841984 / ev
            K = 2.0 * np.pi / wl

            a_min = float(self.var_alpha_min.get())
            a_max = float(self.var_alpha_max.get())
            n_alpha = int(self.var_n_alpha.get())
            alphas = np.linspace(a_min, a_max, n_alpha)
            phi_1d = 0.0

            ns = int(self.var_ns.get())
            np_modes = int(self.var_np.get())
            m_s = np.arange(-ns, ns + 1)
            m_p = np.arange(-np_modes, np_modes + 1)

            a_nf = float(self.var_nf_alpha.get())
            phi_nf = float(self.var_nf_phi.get()) if self.var_nf_enable_phi.get() else 0.0
            nx_nf = int(self.var_nf_nx.get())
            nz_nf = int(self.var_nf_nz.get())

            na_2d = int(self.var_2d_na.get())
            np_2d = int(self.var_2d_np.get())
            a2d_min = float(self.var_2d_amin.get())
            a2d_max = float(self.var_2d_amax.get())
            p2d_min = float(self.var_2d_pmin.get())
            p2d_max = float(self.var_2d_pmax.get())
            alphas_2d = np.linspace(a2d_min, a2d_max, na_2d)
            phis_2d = np.linspace(p2d_min, p2d_max, np_2d)

            # Выбор метода расчета флуоресценции
            fluor_method = 'numerical' if self.var_numerical_fluor.get() else 'analytical'

            saved_files = []
            results = {}

            # 1. Сохранение эскиза геометрии структуры
            if self.chk_vars["geom_save"].get():
                path = os.path.join(self.figures_dir, "structure_geometry.png")
                self.fig_geom.savefig(path, dpi=300, bbox_inches='tight')
                saved_files.append("structure_geometry.png")

            # 2. 1D флуоресценция
            need_1d = any([
                self.chk_vars["fluor_1d_sp"].get(),
                self.chk_vars["fluor_1d_s"].get(),
                self.chk_vars["fluor_1d_p"].get()
            ])
            if need_1d:
                self._update_status_msg(f"Расчет 1D кривых флуоресценции ({fluor_method})...")
                Is, Ip = scan_fluorescence_1d(self.structure, alphas, phi_1d, K, m_s, m_p, method=fluor_method)
                results["1d"] = (alphas, Is, Ip)

                path_comp = os.path.join(self.figures_dir, "fluorescence_sp_comparison.png")
                save_fluorescence_comparison_with_diff(alphas, Is, Ip, path_comp)
                saved_files.append("fluorescence_sp_comparison.png")

                if self.chk_vars["fluor_1d_s"].get():
                    f_s = plot_fluorescence_single(alphas, Is, pol='s')
                    p_s = os.path.join(self.figures_dir, "fluorescence_s.png")
                    f_s.savefig(p_s, dpi=300)
                    plt.close(f_s)
                    saved_files.append("fluorescence_s.png")

                if self.chk_vars["fluor_1d_p"].get():
                    f_p = plot_fluorescence_single(alphas, Ip, pol='p')
                    p_p = os.path.join(self.figures_dir, "fluorescence_p.png")
                    f_p.savefig(p_p, dpi=300)
                    plt.close(f_p)
                    saved_files.append("fluorescence_p.png")

            # 3. Ближнее поле |E|²
            need_nf = any([
                self.chk_vars["near_field_s"].get(),
                self.chk_vars["near_field_p"].get()
            ])
            if need_nf:
                self._update_status_msg(f"Расчет ближнего поля (α={a_nf}°, φ={phi_nf}°, сетка {nx_nf}x{nz_nf})...")
                sol_s_nf = solve_multislice_s(self.structure, a_nf, phi_nf, K, m_s)
                sol_p_nf = solve_multislice_p(self.structure, a_nf, phi_nf, K, m_p)
                results["nf"] = (sol_s_nf, sol_p_nf, nx_nf, nz_nf)

                f_nfs = compute_and_plot_near_field(sol_s_nf, pol='s', nx=nx_nf, nz=nz_nf)
                f_nfs.savefig(os.path.join(self.figures_dir, "near_field_s.png"), dpi=300)
                plt.close(f_nfs)
                saved_files.append("near_field_s.png")

                f_nfp = compute_and_plot_near_field(sol_p_nf, pol='p', nx=nx_nf, nz=nz_nf)
                f_nfp.savefig(os.path.join(self.figures_dir, "near_field_p.png"), dpi=300)
                plt.close(f_nfp)
                saved_files.append("near_field_p.png")

                p_diff = os.path.join(self.figures_dir, "near_field_diff.png")
                compute_and_save_near_field_diff(sol_s_nf, sol_p_nf, nx=nx_nf, nz=nz_nf, save_path=p_diff)
                saved_files.append("near_field_diff.png")

            # 4. Дисперсионные кривые k_zn(α)
            if self.chk_vars["dispersion_s"].get() or self.chk_vars["dispersion_p"].get():
                self._update_status_msg("Расчет дисперсионных кривых k_zn(α)...")
                results["disp"] = (self.structure, alphas, phi_1d, K, m_s, m_p)

                if self.chk_vars["dispersion_s"].get():
                    f_ds = compute_and_plot_dispersion(self.structure, alphas, phi_1d, K, m_s, pol='s')
                    f_ds.savefig(os.path.join(self.figures_dir, "dispersion_s.png"), dpi=300)
                    plt.close(f_ds)
                    saved_files.append("dispersion_s.png")

                if self.chk_vars["dispersion_p"].get():
                    f_dp = compute_and_plot_dispersion(self.structure, alphas, phi_1d, K, m_p, pol='p')
                    f_dp.savefig(os.path.join(self.figures_dir, "dispersion_p.png"), dpi=300)
                    plt.close(f_dp)
                    saved_files.append("dispersion_p.png")

            # 5. Собственные векторы
            if self.chk_vars["eigenvectors"].get():
                self._update_status_msg("Расчет карт собственных векторов E_n(x)...")
                results["eig"] = (self.structure, a_nf, phi_nf, K, m_s, m_p)

                for pol in ('s', 'p'):
                    m_curr = m_s if pol == 's' else m_p
                    f_e = plot_eigenvectors(self.structure, a_nf, phi_nf, K, m_curr, pol=pol)
                    fname = f"eigenvectors_{pol}.png"
                    f_e.savefig(os.path.join(self.figures_dir, fname), dpi=300)
                    plt.close(f_e)
                    saved_files.append(fname)

            # 6. 2D карты флуоресценции
            need_2d = any([
                self.chk_vars["fluor_2d_s"].get(),
                self.chk_vars["fluor_2d_p"].get()
            ])
            if need_2d:
                self._update_status_msg(f"Расчет 2D карт флуоресценции ({fluor_method}, сетка {na_2d}x{np_2d})...")
                map_s, map_p = scan_fluorescence_2d(self.structure, alphas_2d, phis_2d, K, m_s, m_p, method=fluor_method)
                results["2d"] = (alphas_2d, phis_2d, map_s, map_p)

                f_2ds = plot_fluorescence_2d_map(alphas_2d, phis_2d, map_s, which='s')
                f_2ds.savefig(os.path.join(self.figures_dir, "fluorescence_map_2d_s.png"), dpi=300)
                plt.close(f_2ds)
                saved_files.append("fluorescence_map_2d_s.png")

                f_2dp = plot_fluorescence_2d_map(alphas_2d, phis_2d, map_p, which='p')
                f_2dp.savefig(os.path.join(self.figures_dir, "fluorescence_map_2d_p.png"), dpi=300)
                plt.close(f_2dp)
                saved_files.append("fluorescence_map_2d_p.png")

                p_2d_diff = os.path.join(self.figures_dir, "fluorescence_map_2d_diff.png")
                save_fluorescence_2d_diff_map(alphas_2d, phis_2d, map_s, map_p, p_2d_diff)
                saved_files.append("fluorescence_map_2d_diff.png")

            self.root.after(0, self._on_calculation_finish, results, saved_files)

        except Exception as e:
            self.root.after(0, self._on_calculation_error, str(e))

    def _update_status_msg(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))

    # ================= ВЫВОД ВЫБРАННЫХ ГРАФИКОВ В ОТДЕЛЬНЫХ ОКНАХ =================
    def _display_separate_windows(self, results):
        opened_figures = []

        if self.chk_vars["fluor_1d_sp"].get() and "1d" in results:
            alphas, Is, Ip = results["1d"]
            fig = plot_fluorescence_comparison_onscreen(alphas, Is, Ip)
            try:
                fig.canvas.manager.set_window_title("Сравнение флуоресценции s и p")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["fluor_1d_s"].get() and "1d" in results:
            alphas, Is, _ = results["1d"]
            fig = plot_fluorescence_single(alphas, Is, pol='s')
            try:
                fig.canvas.manager.set_window_title("Флуоресценция s-поляризации")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["fluor_1d_p"].get() and "1d" in results:
            alphas, _, Ip = results["1d"]
            fig = plot_fluorescence_single(alphas, Ip, pol='p')
            try:
                fig.canvas.manager.set_window_title("Флуоресценция p-поляризации")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["near_field_s"].get() and "nf" in results:
            sol_s_nf, _, nx_nf, nz_nf = results["nf"]
            fig = compute_and_plot_near_field(sol_s_nf, pol='s', nx=nx_nf, nz=nz_nf)
            try:
                fig.canvas.manager.set_window_title(f"Ближнее поле s-поляризации α={sol_s_nf['alpha']:.1f}°")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["near_field_p"].get() and "nf" in results:
            _, sol_p_nf, nx_nf, nz_nf = results["nf"]
            fig = compute_and_plot_near_field(sol_p_nf, pol='p', nx=nx_nf, nz=nz_nf)
            try:
                fig.canvas.manager.set_window_title(f"Ближнее поле p-поляризации α={sol_p_nf['alpha']:.1f}°")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["dispersion_s"].get() and "disp" in results:
            st, alphas, phi_1d, K, m_s, _ = results["disp"]
            fig = compute_and_plot_dispersion(st, alphas, phi_1d, K, m_s, pol='s')
            try:
                fig.canvas.manager.set_window_title("Дисперсионные кривые s-поляризации")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["dispersion_p"].get() and "disp" in results:
            st, alphas, phi_1d, K, _, m_p = results["disp"]
            fig = compute_and_plot_dispersion(st, alphas, phi_1d, K, m_p, pol='p')
            try:
                fig.canvas.manager.set_window_title("Дисперсионные кривые p-поляризации")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["eigenvectors"].get() and "eig" in results:
            st, a_nf, phi_nf, K, m_s, m_p = results["eig"]
            fig_s = plot_eigenvectors(st, a_nf, phi_nf, K, m_s, pol='s')
            fig_p = plot_eigenvectors(st, a_nf, phi_nf, K, m_p, pol='p')
            try:
                fig_s.canvas.manager.set_window_title("Собственные векторы s-поляризации")
                fig_p.canvas.manager.set_window_title("Собственные векторы p-поляризации")
            except Exception:
                pass
            opened_figures.extend([fig_s, fig_p])

        if self.chk_vars["fluor_2d_s"].get() and "2d" in results:
            alphas_2d, phis_2d, map_s, _ = results["2d"]
            fig = plot_fluorescence_2d_map(alphas_2d, phis_2d, map_s, which='s')
            try:
                fig.canvas.manager.set_window_title("2D карта флуоресценции s-поляризации")
            except Exception:
                pass
            opened_figures.append(fig)

        if self.chk_vars["fluor_2d_p"].get() and "2d" in results:
            alphas_2d, phis_2d, _, map_p = results["2d"]
            fig = plot_fluorescence_2d_map(alphas_2d, phis_2d, map_p, which='p')
            try:
                fig.canvas.manager.set_window_title("2D карта флуоресценции p-поляризации")
            except Exception:
                pass
            opened_figures.append(fig)

        if opened_figures:
            plt.show(block=False)

    def _on_calculation_finish(self, results, saved_files):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self.is_calculating = False

        self._display_separate_windows(results)

        msg = f"Расчет успешно завершен!\nСохранено файлов в src/figures: {len(saved_files)}\nОткрыто окон с графиками на экране."
        self.lbl_status.config(text=f"Готово. Сохранено {len(saved_files)} файлов в src/figures/", foreground="#16a34a")
        messagebox.showinfo("Расчет завершен", msg)

    def _on_calculation_error(self, err_msg):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self.is_calculating = False
        self.lbl_status.config(text=f"Ошибка: {err_msg}", foreground="#dc2626")
        messagebox.showerror("Ошибка вычислений", err_msg)


def main():
    root = tk.Tk()
    app = RcwaGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
