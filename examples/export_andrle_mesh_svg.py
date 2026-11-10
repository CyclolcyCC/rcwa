"""Export a pure-vector SVG replica of Anna Andrle's FEM structure (Figure 4.36).

Generates a vector-only Matplotlib figure matching the original thesis cross-section
geometry, FEM triangulation look, domain colors, and material labels (kept unchanged
as requested) with zero embedded raster images.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import Delaunay


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_trapezoid_boundary(
    x_bot: float,
    x_top: float,
    y_bot: float,
    y_top: float,
    *,
    r_top: float = 4.2,
    n_arc: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build closed Si3N4 trapezoid contour and smooth interior corner seed points."""
    dx = x_bot - x_top
    dy = y_top - y_bot
    slope_len = np.hypot(dx, dy)
    nx, ny = dy / slope_len, dx / slope_len

    cy_t = y_top - r_top
    cx_t = x_bot - (cy_t - y_bot) * (dx / dy) - r_top * slope_len / dy

    t_arc = np.linspace(np.pi / 2.0, np.arctan2(ny, nx), n_arc)
    top_right = np.column_stack([cx_t + r_top * np.cos(t_arc), cy_t + r_top * np.sin(t_arc)])
    right_pts = np.vstack([[0.0, y_top], top_right, [x_bot, y_bot]])

    left_pts = right_pts[::-1].copy()
    left_pts[:, 0] *= -1.0

    # Extra ring points near rounded corners prevent fan pinwheels in Delaunay mesh
    corner_seeds = []
    for scale in (0.45, 0.78):
        for ang in np.linspace(np.pi / 2.0, np.arctan2(ny, nx), 4):
            px = cx_t + (r_top * scale) * np.cos(ang)
            py = cy_t + (r_top * scale) * np.sin(ang)
            corner_seeds.append((px, py))
            corner_seeds.append((-px, py))

    return np.vstack([left_pts, right_pts]), np.array(corner_seeds)


def _make_conformal_profile(
    x_bot: float,
    x_top: float,
    y_top: float,
    *,
    y_groove: float,
    d_wall: float,
    r_top: float = 4.2,
    r_bot: float = 6.2,
    n_arc: int = 12,
) -> np.ndarray:
    """Build upper conformal profile contour (cap + sidewall + groove fillet)."""
    dx = x_bot - x_top
    dy = y_top
    slope_len = np.hypot(dx, dy)
    nx, ny = dy / slope_len, dx / slope_len

    xb = x_bot + d_wall * slope_len / dy
    yt = y_top + d_wall
    rt = r_top + d_wall
    cy_t = yt - rt
    cx_t = xb - cy_t * (dx / dy) - rt * slope_len / dy
    t_arc = np.linspace(np.pi / 2.0, np.arctan2(ny, nx), n_arc)
    top_right = np.column_stack([cx_t + rt * np.cos(t_arc), cy_t + rt * np.sin(t_arc)])

    rb = max(0.3, r_bot - d_wall)
    cy_b = y_groove + rb
    cx_b = xb - cy_b * (dx / dy) + rb * slope_len / dy
    b_arc = np.linspace(np.arctan2(-ny, -nx), -np.pi / 2.0, n_arc)
    bot_right = np.column_stack([cx_b + rb * np.cos(b_arc), cy_b + rb * np.sin(b_arc)])

    right_pts = np.vstack([[0.0, yt], top_right, bot_right, [50.0, y_groove]])
    left_pts = right_pts[::-1].copy()
    left_pts[:, 0] *= -1.0
    return np.vstack([left_pts, right_pts])


def _fem_mesh(
    poly_pts: np.ndarray,
    h_grid: float,
    *,
    seed: int = 0,
    extra_pts: np.ndarray | None = None,
) -> np.ndarray:
    """Generate an organic triangular mesh clipped cleanly inside poly_pts."""
    rng = np.random.default_rng(seed)
    path = MplPath(poly_pts)
    x0, y0 = poly_pts.min(axis=0)
    x1, y1 = poly_pts.max(axis=0)

    xs = np.arange(x0 - h_grid, x1 + 1.5 * h_grid, h_grid)
    ys = np.arange(y0 - h_grid, y1 + 1.5 * h_grid, h_grid * 0.866)
    pts = [
        (
            x + (0.5 * h_grid if j % 2 else 0.0) + rng.uniform(-0.22, 0.22) * h_grid,
            y + rng.uniform(-0.22, 0.22) * h_grid,
        )
        for j, y in enumerate(ys)
        for x in xs
    ]
    pts_arr = np.asarray(pts)
    pts_arr = pts_arr[path.contains_points(pts_arr, radius=-0.15 * h_grid)]

    bnd = []
    for i in range(len(poly_pts)):
        a, b = poly_pts[i], poly_pts[(i + 1) % len(poly_pts)]
        n_seg = max(1, int(np.ceil(np.linalg.norm(b - a) / h_grid)))
        for k in range(n_seg):
            bnd.append(a + (b - a) * (k / n_seg))

    parts = [np.asarray(bnd)]
    if len(pts_arr) > 0:
        parts.append(pts_arr)
    if extra_pts is not None and len(extra_pts) > 0:
        inside_extra = extra_pts[path.contains_points(extra_pts, radius=-0.05)]
        if len(inside_extra) > 0:
            parts.append(inside_extra)

    all_pts = np.vstack(parts)
    tri = Delaunay(all_pts)
    tris = all_pts[tri.simplices]
    centroids = tris.mean(axis=1)
    return tris[path.contains_points(centroids, radius=1e-4)]


def build_andrle_mesh_figure() -> plt.Figure:
    """Construct pure-vector matplotlib Figure matching Figure 4.36 from Anna Andrle's thesis."""
    core_pts, corner_seeds = _make_trapezoid_boundary(28.0, 17.0, 0.0, 98.2, r_top=4.2, n_arc=8)
    sio2_outer = _make_conformal_profile(28.0, 17.0, 98.2, y_groove=5.9, d_wall=2.6, r_top=4.2, r_bot=6.2)
    c2o_outer = _make_conformal_profile(28.0, 17.0, 98.2, y_groove=8.4, d_wall=5.1, r_top=4.2, r_bot=6.2)

    fig, ax = plt.subplots(figsize=(5.4, 5.97), dpi=450)

    # 1. Si substrate
    si_poly = np.array([[-50.0, -10.0], [50.0, -10.0], [50.0, -1.3], [-50.0, -1.3]])
    ax.add_patch(Polygon(si_poly, closed=True, fc="#363636", ec="none", zorder=1))
    ax.add_collection(
        PolyCollection(_fem_mesh(si_poly, 4.8, seed=1), facecolors="none", edgecolors="#606060", linewidths=0.45, zorder=2)
    )

    # 2. Thin interface native-oxide stripe
    box_poly = np.array([[-50.0, -1.3], [50.0, -1.3], [50.0, 0.0], [-50.0, 0.0]])
    ax.add_patch(Polygon(box_poly, closed=True, fc="#2c35f5", ec="none", zorder=2))
    ax.add_collection(
        PolyCollection(_fem_mesh(box_poly, 1.8, seed=2), facecolors="none", edgecolors="#7b86ff", linewidths=0.35, zorder=3)
    )

    # 3. C2O contamination layer
    c2o_ring = np.vstack([c2o_outer, sio2_outer[::-1]])
    ax.add_patch(Polygon(c2o_ring, closed=True, fc="#8a8a8a", ec="none", zorder=3))
    ax.add_collection(
        PolyCollection(_fem_mesh(c2o_ring, 2.2, seed=3), facecolors="none", edgecolors="#737373", linewidths=0.40, zorder=4)
    )

    # 4. SiO2 oxide shell
    sio2_ring = np.vstack([sio2_outer, [[50.0, 0.0]], core_pts[::-1], [[-50.0, 0.0]]])
    ax.add_patch(Polygon(sio2_ring, closed=True, fc="#bddfeb", ec="none", zorder=4))
    ax.add_collection(
        PolyCollection(_fem_mesh(sio2_ring, 2.2, seed=4), facecolors="none", edgecolors="#6882c8", linewidths=0.45, zorder=5)
    )

    # 5. Si3N4 grating core
    core_tris = _fem_mesh(core_pts, 5.6, seed=5, extra_pts=corner_seeds)
    ax.add_patch(Polygon(core_pts, closed=True, fc="#a6f1a6", ec="#56b856", lw=0.6, zorder=5))
    ax.add_collection(
        PolyCollection(core_tris, facecolors="none", edgecolors="#5fb85f", linewidths=0.45, zorder=6)
    )

    # Material badges preserved exactly as in Anna Andrle's thesis
    badge_kw = dict(boxstyle="round,pad=0.32,rounding_size=0.35", fc="white", ec="black", lw=1.3)
    ax.text(0.0, 104.5, r"$\mathrm{C_2O}$", color="#7a7a7a", fontsize=13, ha="center", va="center", bbox=badge_kw, zorder=10)
    ax.text(0.0, 14.5, r"$\mathrm{Si_3N_4}$", color="#007800", fontsize=13, ha="center", va="center", bbox=badge_kw, zorder=10)
    ax.text(-26.0, 8.8, r"$\mathrm{SiO_2}$", color="#0000e0", fontsize=13, ha="center", va="center", bbox=badge_kw, zorder=10)
    ax.text(-16.0, -7.2, "Si", color="black", fontsize=13, ha="center", va="center", bbox=badge_kw, zorder=10)

    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-10.0, 110.0)
    ax.set_aspect("equal")
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("X / nm", fontsize=13)
    ax.set_ylabel("Y / nm", fontsize=13)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    return fig


def main():
    output_dir = REPO_ROOT / "examples" / "output"
    svg_dir = output_dir / "svg_output"
    svg_dir.mkdir(parents=True, exist_ok=True)
    fig = build_andrle_mesh_figure()

    out_svg = svg_dir / "andrle_structure_mesh.svg"
    out_png = output_dir / "andrle_structure_mesh.png"
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=450)
    print(f"Saved pure-vector Andrle Figure 4.36 replica to: {out_svg} and {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
