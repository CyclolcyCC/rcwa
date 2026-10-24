#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive TEM annotation GUI for MBDDT layer models.

The image is assumed to be aligned to the horizon (no rotation): depth Z runs
vertically (image rows), the lateral coordinate X runs horizontally (columns).
Layer boundaries are horizontal lines of constant depth Z; a layer is the band
between two adjacent boundaries. A box is a rectangle inside one layer, bounded
laterally (left/right in X) and assigned a material.

Layout: a Matplotlib canvas (the micrograph) on the left, a Tk control panel on
the right with image / calibration / material / box tools.

Workflow:
    Load image     open a TEM micrograph.
    Calibrate      click two points, type the physical distance (nm); scale only.
    Set origin     click the point that is physical (0, 0); axes read out nm.
    Add layers     click to drop horizontal Z boundaries (drag to move,
                   right-click to delete).
    Materials      add named materials with complex chi; mark exactly one as
                   fluorescent.
    Add box        pick a material, then click two X positions inside a layer.

Geometry and the annotation data model are pure Python (TemCalibration,
Annotation), decoupled from Tk so they can be unit-tested headless.
@author: Konstantin V. Nikolaev
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pure geometry: pixel <-> physical (nm) mapping. No GUI dependency.
# ---------------------------------------------------------------------------

class TemCalibration:
    """Axis-aligned pixel<->nm map: isotropic scale + origin (no rotation).

    Physical coordinates are (z, x) in nm, measured from the origin:
        z = depth, vertical, grows downward with image rows (into the sample);
        x = lateral, horizontal, grows with image columns (to the right).
    """

    def __init__(self):
        self.scale = None          # nm per pixel
        self.origin_px = None      # (col, row) pixel of physical (0, 0)
        self._p1 = self._p2 = None

    def set_points(self, p1, p2):
        self._p1 = np.asarray(p1, float)
        self._p2 = np.asarray(p2, float)

    def set_distance(self, dist_nm):
        if self._p1 is None or self._p2 is None:
            raise ValueError("set_points() must precede set_distance()")
        dist_px = float(np.hypot(*(self._p2 - self._p1)))
        if dist_px == 0:
            raise ValueError("calibration points coincide")
        self.scale = dist_nm / dist_px
        if self.origin_px is None:
            self.origin_px = self._p1.copy()

    def set_origin(self, p):
        self.origin_px = np.asarray(p, float)

    @property
    def ready(self):
        return self.scale is not None and self.origin_px is not None

    # axis-aligned transforms.
    # imshow uses origin="lower": row 0 is at the bottom, Y axis grows upward.
    # Z (depth) grows with row index: z = (row - origin_row) * scale.
    # Substrate (z < 0) is below the origin; sample layers (z > 0) are above.
    def row_to_z(self, row):
        return (np.asarray(row, float) - self.origin_px[1]) * self.scale

    def col_to_x(self, col):
        return (np.asarray(col, float) - self.origin_px[0]) * self.scale

    def z_to_row(self, z_nm):
        return np.asarray(z_nm, float) / self.scale + self.origin_px[1]

    def x_to_col(self, x_nm):
        return np.asarray(x_nm, float) / self.scale + self.origin_px[0]

    def px_to_nm(self, p):
        col, row = p
        return np.array([self.row_to_z(row), self.col_to_x(col)])


# ---------------------------------------------------------------------------
# Annotation data model: materials, layer boundaries, boxes. GUI-independent.
# ---------------------------------------------------------------------------

class MaterialDef:
    """A named material with complex susceptibility chi and a fluorescent flag."""

    def __init__(self, name, chi, fluorescent=False, color=None, substrate=False):
        self.name = name
        self.chi = complex(chi)
        self.fluorescent = fluorescent
        self.substrate = substrate
        self.color = color  # matplotlib color string, e.g. "C0", "C1", …

    def __repr__(self):
        flags = ""
        if self.fluorescent:
            flags += " *fluor*"
        if self.substrate:
            flags += " *substr*"
        return "%s: chi=%g%+gj%s" % (self.name, self.chi.real, self.chi.imag, flags)


class BoxDef:
    """A box inside a layer: lateral [left, right] in nm + a material name."""

    def __init__(self, layer_index, left_nm, right_nm, material_name):
        self.layer_index = layer_index
        self.left = min(left_nm, right_nm)
        self.right = max(left_nm, right_nm)
        self.material = material_name

    def __repr__(self):
        return "layer %d  x=[%.1f, %.1f] nm  %s" % (
            self.layer_index, self.left, self.right, self.material)


class Annotation:
    """All annotation state independent of any GUI toolkit.

    layers_nm : sorted Z boundaries (nm). N boundaries -> N-1 bounded layers
                (band i spans boundaries[i]..boundaries[i+1]).
    materials : list of MaterialDef (exactly one may be fluorescent).
    boxes     : list of BoxDef.
    """

    def __init__(self):
        self.layers_nm = []
        self.materials = []
        self.boxes = []

    # -- layers -----------------------------------------------------------
    def add_layer(self, z_nm):
        self.layers_nm.append(float(z_nm))
        self.layers_nm.sort()

    def move_layer(self, index, z_nm):
        self.layers_nm[index] = float(z_nm)
        self.layers_nm.sort()

    def remove_layer(self, index):
        self.layers_nm.pop(index)

    def layer_bounds(self, i):
        """(z_top, z_bottom) in nm of bounded layer i (between boundary i, i+1)."""
        zs = sorted(self.layers_nm)
        return zs[i], zs[i + 1]

    def n_bounded_layers(self):
        return max(0, len(self.layers_nm) - 1)

    def layer_at_z(self, z_nm):
        """Index of the bounded layer containing depth z_nm, or None."""
        zs = sorted(self.layers_nm)
        for i in range(len(zs) - 1):
            if zs[i] <= z_nm <= zs[i + 1]:
                return i
        return None

    # -- materials --------------------------------------------------------
    def add_material(self, name, chi, fluorescent=False, color=None):
        if any(m.name == name for m in self.materials):
            raise ValueError("material name %r already exists" % name)
        if color is None:
            color = "C%d" % len(self.materials)
        m = MaterialDef(name, chi, fluorescent, color=color)
        if fluorescent:
            for other in self.materials:
                other.fluorescent = False
        self.materials.append(m)
        return m

    def set_fluorescent(self, name):
        """Mark exactly one material (by name) as fluorescent, clear the rest."""
        for m in self.materials:
            m.fluorescent = (m.name == name)

    def set_substrate(self, name):
        """Mark exactly one material (by name) as substrate, clear the rest."""
        for m in self.materials:
            m.substrate = (m.name == name)

    def substrate_material(self):
        for m in self.materials:
            if m.substrate:
                return m
        return None

    def remove_material(self, name):
        self.materials = [m for m in self.materials if m.name != name]
        self.boxes = [b for b in self.boxes if b.material != name]

    def material_names(self):
        return [m.name for m in self.materials]

    def material_color(self, name):
        """Return the color assigned to a material by name, or 'gray' if unknown."""
        for m in self.materials:
            if m.name == name:
                return m.color
        return "gray"

    def fluorescent_material(self):
        for m in self.materials:
            if m.fluorescent:
                return m
        return None

    # -- boxes ------------------------------------------------------------
    def add_box(self, layer_index, left_nm, right_nm, material_name):
        self.boxes.append(BoxDef(layer_index, left_nm, right_nm, material_name))

    def clear(self):
        self.__init__()


# ---------------------------------------------------------------------------
# Tk + Matplotlib front-end. Imported lazily so the module stays headless-safe.
# ---------------------------------------------------------------------------

def launch(image_path=None):
    """Open the annotation window. Optional image_path preloads a micrograph."""
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    import matplotlib.image as mpimg
    import tkinter as tk
    from tkinter import ttk, filedialog, simpledialog, messagebox

    # Tk variables must be created *after* the root window exists, otherwise
    # tkinter raises "Too early to create variable: no default root window".
    root = tk.Tk()
    root.title("TEM layer annotation")
    root.geometry("1200x760")

    S = {
        "cal": TemCalibration(),
        "ann": Annotation(),
        "image": None,
        "mode": tk.StringVar(master=root, value="none"),
        "cal_clicks": [],
        "box_clicks": [],
        "box_layer": None,
        "drag": None,
    }

    # === left: matplotlib canvas ==========================================
    left = ttk.Frame(root)
    left.pack(side="left", fill="both", expand=True)
    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    canvas = FigureCanvasTkAgg(fig, master=left)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, left)

    art = {"img": None, "cal_pts": None, "cal_line": None,
           "origin": None, "axes": [], "layers": [], "boxes": [],
           "substrate": None}
    title = ax.set_title("Load a TEM image to begin", fontsize=10)

    # === right: control panel =============================================
    right = ttk.Frame(root, padding=8)
    right.pack(side="right", fill="y")

    def section(label):
        ttk.Label(right, text=label, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(10, 2))

    # --- image / reset
    section("Image")
    ttk.Button(right, text="Load image", command=lambda: load_image()).pack(fill="x")
    ttk.Button(right, text="Clear all", command=lambda: clear_all()).pack(fill="x", pady=2)

    # --- modes
    section("Annotate (click on image)")
    for val, lbl in [("cal", "Calibrate (2 pts)"),
                     ("origin", "Set origin"),
                     ("layers", "Add layer lines"),
                     ("box", "Add box")]:
        ttk.Radiobutton(right, text=lbl, value=val, variable=S["mode"],
                        command=lambda: on_mode()).pack(anchor="w")

    # --- materials
    section("Materials")
    mat_frame = ttk.Frame(right)
    mat_frame.pack(fill="x")
    ttk.Label(mat_frame, text="Name").grid(row=0, column=0, sticky="w")
    ttk.Label(mat_frame, text="chi (e.g. -4.4e-5+2.3e-6j)").grid(row=1, column=0, sticky="w")
    name_var = tk.StringVar(master=root)
    chi_var = tk.StringVar(master=root)
    ttk.Entry(mat_frame, textvariable=name_var, width=22).grid(row=0, column=1, padx=2)
    ttk.Entry(mat_frame, textvariable=chi_var, width=22).grid(row=1, column=1, padx=2)
    ttk.Button(right, text="Add material", command=lambda: add_material()).pack(fill="x")
    mat_list = tk.Listbox(right, height=6, exportselection=False)
    mat_list.pack(fill="x", pady=2)
    mrow = ttk.Frame(right)
    mrow.pack(fill="x")
    ttk.Button(mrow, text="Set fluorescent", command=lambda: set_fluor()).pack(side="left", expand=True, fill="x")
    ttk.Button(mrow, text="Set substrate", command=lambda: set_substrate_mat()).pack(side="left", expand=True, fill="x")
    ttk.Button(mrow, text="Remove", command=lambda: remove_material()).pack(side="left", expand=True, fill="x")

    # --- boxes
    section("Boxes")
    ttk.Label(right, text="Material for new box:").pack(anchor="w")
    box_mat = ttk.Combobox(right, state="readonly", values=[])
    box_mat.pack(fill="x")
    ttk.Label(right, text="(then choose 'Add box' mode and click two X points "
                          "inside a layer)", wraplength=240,
              foreground="gray").pack(anchor="w")
    box_list = tk.Listbox(right, height=6, exportselection=False)
    box_list.pack(fill="x", pady=2)
    ttk.Button(right, text="Remove selected box",
               command=lambda: remove_box()).pack(fill="x")

    # --- code generation / tools
    section("Tools")
    ttk.Button(right, text="Center origin on boxes",
               command=lambda: center_origin_on_boxes()).pack(fill="x", pady=(0, 2))
    section("Export")
    ttk.Button(right, text="Generate code",
               command=lambda: generate_code()).pack(fill="x")

    status = ttk.Label(right, text="", wraplength=240, foreground="#0a0")
    status.pack(anchor="w", pady=(10, 0))

    def set_status(msg):
        status.config(text=msg)

    # === axes in nm ========================================================
    def apply_nm_axes():
        cal = S["cal"]
        if not cal.ready:
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("pixels"); ax.set_ylabel("pixels")
            return
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.0f" % cal.col_to_x(v)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.0f" % cal.row_to_z(v)))
        ax.set_xlabel("lateral X, nm"); ax.set_ylabel("depth Z, nm")

    # === drawing ===========================================================
    def redraw():
        cal, ann = S["cal"], S["ann"]
        for key in ("cal_pts", "cal_line", "origin"):
            if art[key] is not None:
                art[key].remove(); art[key] = None
        for a in art["axes"] + art["layers"] + art["boxes"]:
            a.remove()
        art["axes"], art["layers"], art["boxes"] = [], [], []
        if "substrate" in art and art["substrate"] is not None:
            art["substrate"].remove(); art["substrate"] = None

        if S["cal_clicks"]:
            pts = np.array(S["cal_clicks"])
            art["cal_pts"] = ax.scatter(pts[:, 0], pts[:, 1], c="yellow", s=40, zorder=5)
            if len(S["cal_clicks"]) == 2:
                (art["cal_line"],) = ax.plot(pts[:, 0], pts[:, 1], "-", c="yellow", lw=1.2, zorder=4)

        if cal.origin_px is not None:
            ox, oy = cal.origin_px
            art["origin"] = ax.scatter([ox], [oy], marker="+", c="red", s=160, lw=2, zorder=6)
            (lx,) = ax.plot(ax.get_xlim(), [oy, oy], "--", c="red", lw=0.8, alpha=0.7, zorder=3)
            (ly,) = ax.plot([ox, ox], ax.get_ylim(), "--", c="red", lw=0.8, alpha=0.7, zorder=3)
            art["axes"] += [lx, ly]

        if cal.ready:
            x0, x1 = ax.get_xlim()
            for z in ann.layers_nm:
                row = cal.z_to_row(z)
                (ln,) = ax.plot([x0, x1], [row, row], "-", c="cyan", lw=1.5, zorder=4)
                art["layers"].append(ln)
            # substrate: fill everything below z=0.
            # With origin="lower": Y grows upward, z grows with row.
            # z < 0 means row < row_zero, i.e. below row_zero on screen.
            # So we fill from ylim[0] (bottom edge) up to row_zero.
            sm = ann.substrate_material()
            if sm is not None and cal.origin_px is not None:
                x0, x1 = ax.get_xlim()
                row_zero = cal.z_to_row(0.0)
                row_bottom = ax.get_ylim()[0]  # lower edge (y_min)
                scolor = sm.color
                (srect,) = ax.fill(
                    [x0, x1, x1, x0],
                    [row_zero, row_zero, row_bottom, row_bottom],
                    alpha=0.20, color=scolor, zorder=1)
                art["substrate"] = srect
            # boxes: rectangle = layer band (Z) x [left,right] (X)
            for b in ann.boxes:
                if b.layer_index >= ann.n_bounded_layers():
                    continue
                z_top, z_bot = ann.layer_bounds(b.layer_index)
                r_top, r_bot = cal.z_to_row(z_top), cal.z_to_row(z_bot)
                c_l, c_r = cal.x_to_col(b.left), cal.x_to_col(b.right)
                bcolor = ann.material_color(b.material)
                (rect,) = ax.fill(
                    [c_l, c_r, c_r, c_l], [r_top, r_top, r_bot, r_bot],
                    alpha=0.35, color=bcolor, zorder=2)
                art["boxes"].append(rect)
        canvas.draw_idle()

    def refresh_lists():
        import matplotlib.colors as mcolors

        def _hex(color):
            """Convert any matplotlib color spec to a #rrggbb hex string."""
            try:
                r, g, b, *_ = mcolors.to_rgba(color)
                return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))
            except Exception:
                return "#888888"

        mat_list.delete(0, tk.END)
        for i, m in enumerate(S["ann"].materials):
            mat_list.insert(tk.END, repr(m))
            hex_col = _hex(m.color)
            mat_list.itemconfig(i, background=hex_col, selectbackground=hex_col)
        box_mat["values"] = S["ann"].material_names()
        if S["ann"].material_names() and not box_mat.get():
            box_mat.current(0)
        box_list.delete(0, tk.END)
        for i, b in enumerate(S["ann"].boxes):
            box_list.insert(tk.END, repr(b))
            hex_col = _hex(S["ann"].material_color(b.material))
            box_list.itemconfig(i, background=hex_col, selectbackground=hex_col)

    def report():
        cal, ann = S["cal"], S["ann"]
        if not cal.ready:
            return
        zs = sorted(ann.layers_nm)
        thick = [round(b - a, 1) for a, b in zip(zs, zs[1:])]
        set_status("scale=%.4f nm/px | boundaries=%d (layers=%d) | "
                   "thickness(nm)=%s | materials=%d | boxes=%d"
                   % (cal.scale, len(zs), ann.n_bounded_layers(),
                      thick, len(ann.materials), len(ann.boxes)))

    # === image / reset =====================================================
    def load_image(path=None):
        if path is None:
            path = filedialog.askopenfilename(
                title="Open TEM image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                           ("All files", "*.*")])
        if not path:
            return
        img = mpimg.imread(path)
        S["image"] = np.flipud(img)
        if art["img"] is not None:
            art["img"].remove()
        art["img"] = ax.imshow(S["image"], cmap="gray", origin="lower")
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(0, img.shape[0])
        clear_all(keep_image=True)
        set_status("Image loaded. Calibrate, then set origin / layers / boxes.")

    def clear_all(keep_image=False):
        S["cal"] = TemCalibration()
        S["ann"] = Annotation()
        S["cal_clicks"], S["box_clicks"], S["box_layer"], S["drag"] = [], [], None, None
        apply_nm_axes()
        refresh_lists()
        redraw()
        if not keep_image:
            set_status("Cleared calibration, origin, layers, boxes.")

    # === modes / materials / boxes =========================================
    def on_mode():
        m = S["mode"].get()
        S["cal_clicks"], S["box_clicks"] = [], []
        msgs = {"cal": "Calibrate: click point 1, then point 2.",
                "origin": "Set origin: click physical (0, 0).",
                "layers": "Layers: click to add a horizontal Z boundary; "
                          "drag to move; right-click to delete.",
                "box": "Box: click two X positions inside one layer."}
        set_status(msgs.get(m, ""))
        redraw()

    def add_material():
        name = name_var.get().strip()
        if not name:
            messagebox.showwarning("Material", "Enter a material name.")
            return
        try:
            chi = complex(chi_var.get().replace(" ", "") or "0")
        except ValueError:
            messagebox.showwarning("Material", "chi must be like -4.4e-5+2.3e-6j")
            return
        try:
            S["ann"].add_material(name, chi)
        except ValueError as e:
            messagebox.showwarning("Material", str(e))
            return
        name_var.set(""); chi_var.set("")
        refresh_lists(); report()

    def _selected_material_name():
        sel = mat_list.curselection()
        if not sel:
            return None
        return S["ann"].materials[sel[0]].name

    def set_fluor():
        nm = _selected_material_name()
        if nm:
            S["ann"].set_fluorescent(nm); refresh_lists(); report()

    def set_substrate_mat():
        nm = _selected_material_name()
        if nm:
            S["ann"].set_substrate(nm); refresh_lists(); redraw(); report()

    def remove_material():
        nm = _selected_material_name()
        if nm:
            S["ann"].remove_material(nm); refresh_lists(); redraw(); report()

    def remove_box():
        sel = box_list.curselection()
        if sel:
            S["ann"].boxes.pop(sel[0]); refresh_lists(); redraw(); report()

    def center_origin_on_boxes():
        """Shift origin_px so that X=0 coincides with the centre of mass
        of all non-substrate boxes (z > 0), then recompute box coordinates."""
        cal, ann = S["cal"], S["ann"]
        if not cal.ready:
            set_status("Calibrate first."); return

        sm = ann.substrate_material()
        substrate_name = sm.name if sm is not None else None

        # collect boxes with z > 0 (above-substrate layers)
        active = [b for b in ann.boxes if b.material != substrate_name]
        if not active:
            set_status("No non-substrate boxes to centre on."); return

        # centre of mass in nm: each box contributes its own centre, weight=1
        cx_nm = sum((b.left + b.right) / 2.0 for b in active) / len(active)

        # shift origin column by cx_nm / scale pixels
        shift_px = cx_nm / cal.scale
        cal.origin_px[0] += shift_px

        # recompute all box left/right in the new coordinate system
        for b in ann.boxes:
            b.left  -= cx_nm
            b.right -= cx_nm

        apply_nm_axes(); refresh_lists(); redraw(); report()
        set_status("Origin X shifted by %.2f nm to centre of mass." % cx_nm)

    def generate_code():
        """Build a Python snippet that reconstructs the annotated structure
        using the mbddt API (mb.Material / mb.Layer / mb.Model)."""
        ann = S["ann"]
        cal = S["cal"]

        lines = []

        # --- materials
        if ann.materials:
            lines.append("# --- materials")
        for m in ann.materials:
            kwargs = []
            if m.fluorescent:
                kwargs.append("isfluorescent=True")
            chi = m.chi
            chi_str = "%g%+gj" % (chi.real, chi.imag)
            args = ", ".join(kwargs)
            lines.append("%s = mb.Material(%s).set_chi(%s)" % (m.name, args, chi_str))

        if not ann.materials:
            lines.append("# No materials defined yet.")
            _show_code("\n".join(lines))
            return

        # map material name -> variable name (same here, but kept explicit)
        mat_var = {m.name: m.name for m in ann.materials}

        # --- structure parameters (only if calibration is available)
        if cal.ready:
            lines.append("")
            lines.append("# --- structure")
            lines.append("vacuum = mb.Material().set_chi(0)")
            lines.append("pitch = ...  # nm, set from your experiment")
            lines.append("n_modes = ...  # int, set for your calculation")
            lines.append("sym = mb.Symmetry2D(pitch, n_modes)")
            lines.append("sample = mb.Model(sym)")
            lines.append("sample.add_layer(mb.Layer(sym, 0).add_box(-pitch / 2, pitch / 2, vacuum))")
            lines.append("")

            # Layers, bottom to top in z (descending z = descending layer index)
            zs = sorted(ann.layers_nm)
            n_layers = ann.n_bounded_layers()
            for li in range(n_layers - 1, -1, -1):
                z_top, z_bot = ann.layer_bounds(li)
                thickness = round(z_bot - z_top, 3)
                boxes_in = [b for b in ann.boxes if b.layer_index == li]
                if boxes_in:
                    box_calls = ""
                    for b in boxes_in:
                        vname = mat_var.get(b.material, b.material)
                        left  = round(b.left,  3)
                        right = round(b.right, 3)
                        box_calls += ".add_box(%g, %g, %s)" % (left, right, vname)
                    lines.append(
                        "sample.add_layer(mb.Layer(sym, %g)%s)" %
                        (thickness, box_calls))
                else:
                    lines.append(
                        "sample.add_layer(mb.Layer(sym, %g))  # no boxes" %
                        thickness)

            # Substrate: thickness = distance from z=0 to bottom of image
            sm = ann.substrate_material()
            if sm is not None:
                vname = mat_var.get(sm.name, sm.name)
                # origin_px[1] is the row of z=0; row 0 is the bottom edge
                substr_thickness = round(cal.origin_px[1] * cal.scale, 3)
                lines.append(
                    "sample.add_layer(mb.Layer(sym, %g).add_box("
                    "-pitch/2, pitch/2, %s))  # substrate" %
                    (substr_thickness, vname))
        else:
            lines.append("")
            lines.append("# Calibrate the image to get layer thicknesses in nm.")

        _show_code("\n".join(lines))

    def _show_code(text):
        """Pop up a small window with the generated code, ready to copy."""
        win = tk.Toplevel(root)
        win.title("Generated code")
        win.geometry("620x380")
        frame = ttk.Frame(win, padding=6)
        frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, wrap="none", font=("Courier", 10))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", text)
        txt.config(state="disabled")
        # copy-to-clipboard button
        def _copy():
            win.clipboard_clear()
            win.clipboard_append(text)
            btn.config(text="Copied!")
            win.after(1500, lambda: btn.config(text="Copy to clipboard"))
        btn = ttk.Button(frame, text="Copy to clipboard", command=_copy)
        btn.pack(pady=(4, 0))

    # === mouse handling ====================================================
    def nearest_layer(event):
        cal, ann = S["cal"], S["ann"]
        if not cal.ready or not ann.layers_nm:
            return None, np.inf
        best, bestd = None, np.inf
        for i, z in enumerate(ann.layers_nm):
            d = abs(cal.z_to_row(z) - event.ydata)
            if d < bestd:
                best, bestd = i, d
        return best, bestd

    def on_press(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cal, ann = S["cal"], S["ann"]
        mode = S["mode"].get()
        p = (event.xdata, event.ydata)

        if mode == "cal":
            S["cal_clicks"].append(p); redraw()
            if len(S["cal_clicks"]) == 2:
                cal.set_points(*S["cal_clicks"])
                dist = simpledialog.askfloat("Calibration",
                                             "Distance between the two points (nm):",
                                             minvalue=0.0)
                if dist:
                    cal.set_distance(dist); apply_nm_axes(); redraw(); report()
                    set_status("Calibrated. Set origin / layers / boxes.")
                else:
                    S["cal_clicks"] = []; redraw()

        elif mode == "origin":
            cal.set_origin(p)
            # The origin defines z = 0; drop a layer boundary there so that
            # everything below z = 0 is treated as the substrate. Avoid a
            # duplicate if a z = 0 boundary already exists.
            if not any(abs(z) < 1e-9 for z in ann.layers_nm):
                ann.add_layer(0.0)
            apply_nm_axes(); redraw(); report()
            set_status("Origin set. z=0 boundary added; everything below is "
                       "the substrate.")

        elif mode == "layers" and cal.ready:
            if event.button == 3:
                i, d = nearest_layer(event)
                if i is not None and d < 15:
                    ann.remove_layer(i)
            elif event.button == 1:
                i, d = nearest_layer(event)
                if i is not None and d < 8:
                    S["drag"] = i
                else:
                    ann.add_layer(cal.row_to_z(event.ydata))
            redraw(); report()

        elif mode == "box" and cal.ready:
            if ann.n_bounded_layers() == 0:
                set_status("Need at least two layer lines to define a layer.")
                return
            if not box_mat.get():
                set_status("Add a material and select it for the box first.")
                return
            z = cal.row_to_z(event.ydata)
            li = ann.layer_at_z(z)
            if li is None:
                set_status("Click inside a layer (between two boundaries).")
                return
            if not S["box_clicks"]:
                S["box_clicks"] = [event.xdata]; S["box_layer"] = li
                set_status("Box: now click the second X point in the same layer.")
            else:
                if li != S["box_layer"]:
                    set_status("Second click must be in the same layer. Restart.")
                    S["box_clicks"] = []; return
                x1 = cal.col_to_x(S["box_clicks"][0])
                x2 = cal.col_to_x(event.xdata)
                new_left  = min(x1, x2)
                new_right = max(x1, x2)
                layer_idx = S["box_layer"]

                # check intersections with existing boxes in the same layer
                same = [b for b in ann.boxes if b.layer_index == layer_idx]
                overlapping = [b for b in same
                               if b.left < new_right and b.right > new_left]

                if len(overlapping) == 0:
                    # no conflict — add as-is
                    ann.add_box(layer_idx, new_left, new_right, box_mat.get())

                elif len(overlapping) == 1:
                    nb = overlapping[0]
                    # fully contained? forbid
                    if new_left >= nb.left and new_right <= nb.right:
                        set_status("Box is fully inside an existing box. Forbidden.")
                        S["box_clicks"] = []; return
                    if nb.left >= new_left and nb.right <= new_right:
                        set_status("New box fully contains an existing box. Forbidden.")
                        S["box_clicks"] = []; return
                    # partial overlap on one side — snap shared boundary to midpoint
                    if new_left < nb.left:          # new box is to the left
                        mid = (new_right + nb.left) / 2.0
                        new_right = mid
                        nb.left   = mid
                    else:                            # new box is to the right
                        mid = (nb.right + new_left) / 2.0
                        nb.right  = mid
                        new_left  = mid
                    ann.add_box(layer_idx, new_left, new_right, box_mat.get())

                else:
                    set_status("New box overlaps %d existing boxes. Forbidden." %
                               len(overlapping))
                    S["box_clicks"] = []; return

                S["box_clicks"] = []; S["box_layer"] = None
                refresh_lists(); redraw(); report()

    def on_motion(event):
        if S["drag"] is None or event.inaxes != ax or event.xdata is None:
            return
        S["ann"].move_layer(S["drag"], S["cal"].row_to_z(event.ydata))
        # re-sort may change index; re-find the dragged boundary by proximity
        i, _ = nearest_layer(event)
        S["drag"] = i
        redraw()

    def on_release(event):
        if S["drag"] is not None:
            S["drag"] = None; report()

    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)

    if image_path:
        load_image(image_path)
    refresh_lists()

    root.mainloop()


if __name__ == "__main__":
    import sys
    launch(sys.argv[1] if len(sys.argv) > 1 else None)