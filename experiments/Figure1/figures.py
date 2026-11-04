"""Figures for the synthetic multi-fidelity DAG-DGP example."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Arc,
    ArrowStyle,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
)
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from . import model


def configure_style():
    """Use the paper's LaTeX text rendering and Times font."""
    plt.switch_backend("pgf")
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{amsmath}\renewcommand{\rmdefault}{ptm}",
            "font.family": "serif",
        }
    )


FIELD_CMAP = LinearSegmentedColormap.from_list(
    "field", ["#2F86C0", "#94C9E8", "#F7F5F2", "#EFA77A", "#D55E00"]
)
TUBE_CMAP = LinearSegmentedColormap.from_list(  # the same, more saturated
    "tube", ["#0B5E99", "#3C9AD3", "#F4F1EC", "#EC8A4C", "#B84A00"]
)
INK, BOX, FLOOR, FLOOR_FILL, GREEN = (
    "#1F2430",
    "#A9AFB8",
    "#9EA5AF",
    "#EEF0F3",
    "#009E73",
)
ZH = 1.05  # height of the X axis relative to the L1 extent
DENSITY, POWER, NSAMP = (
    5.6,
    1.6,
    240,
)  # opacity per unit length ~ |z|^POWER; ray samples
FOG = 0.55  # how much the field veils the geometry behind it
TUBE_R, RIM, HALO, CUBE = 0.0135, 1.30, 1.75, 0.046
VIEW, VIEW_ANGLE, SIZE = (18, -66), 9.0, (1500, 1500)  # elevation, azimuth; perspective


def field(d, nx=72):
    """E[f_H(x, l1, l2) | D] on a grid (x, l2, l1)."""
    xs = np.linspace(0, 1, nx)
    F = d["surf_lat"][None] + np.interp(xs, d["x"], d["gx_mean"])[:, None, None]
    return d["surf_l1"], d["surf_l2"], xs, F


def render(d):
    """The geometry (box, floor, trajectory, observations) is drawn by PyVista; the field
    is ray-marched with the same perspective camera and composited over the geometry
    using its depth buffer.  Returns the image and the projected box corners."""
    import pyvista as pv  # imported here: only the rendering needs PyVista/VTK
    import vtk

    l1g, l2g, xs, F = field(d)
    s = 1.0 / (l1g[-1] - l1g[0])  # one scale for the L1 and L2 axes

    def X(v):
        return (v - l1g[0]) * s

    def Y(v):
        return (v - l2g[0]) * s

    def Z(v):
        return v * ZH

    bx = (0, X(l1g[-1]), 0, Y(l2g[-1]), 0, Z(1.0))
    interp = RegularGridInterpolator(
        (Z(xs), Y(l2g), X(l1g)), F, bounds_error=False, fill_value=np.nan
    )
    med = float(np.median(F))  # centre of the diverging colour map
    span = float(max(F.max() - med, med - F.min()))
    vmin, vmax = med - span, med + span

    p = pv.Plotter(off_screen=True, window_size=SIZE)
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.add_mesh(pv.Box(bounds=bx).extract_all_edges(), color=BOX, line_width=2.4)
    p.add_mesh(
        pv.Plane(
            center=(bx[1] / 2, bx[3] / 2, -0.001),
            direction=(0, 0, 1),
            i_size=bx[1],
            j_size=bx[3],
        ),
        color=FLOOR_FILL,
        ambient=1.0,
        diffuse=0.0,
        specular=0.0,
    )
    m1, m2, x = d["L1_mean"], d["L2_mean"], d["x"]
    floor = np.c_[X(m1), Y(m2), np.full(len(m1), 0.003)]  # floor projection
    p.add_mesh(
        pv.Spline(floor, len(floor) * 2).tube(radius=0.0055, n_sides=12),
        color=FLOOR,
        ambient=0.85,
        diffuse=0.25,
        specular=0.0,
    )
    spl = pv.Spline(np.c_[X(m1), Y(m2), Z(x)], len(x) * 2)  # trajectory
    spl["H"] = np.nan_to_num(interp(spl.points[:, [2, 1, 0]]), nan=med)
    p.add_mesh(
        spl.tube(radius=TUBE_R, n_sides=28),
        scalars="H",
        cmap=TUBE_CMAP,
        clim=(vmin, vmax),
        smooth_shading=True,
        show_scalar_bar=False,
        ambient=0.42,
        diffuse=0.68,
        specular=0.45,
        specular_power=40,
    )
    for r, c in ((RIM, INK), (HALO, "white")):  # dark rim, white halo
        p.add_mesh(
            spl.tube(radius=TUBE_R * r, n_sides=28),
            color=c,
            culling="front",
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
        )
    # observations of H: a cube on the trajectory, a dashed drop line, a dot on the floor
    for xv in d["xH"]:
        c = np.array([X(np.interp(xv, x, m1)), Y(np.interp(xv, x, m2)), Z(xv)])
        zz = np.linspace(0.004, c[2] - CUBE * 0.6, 40)
        for k in range(0, 39, 2):
            p.add_mesh(
                pv.lines_from_points(
                    np.c_[np.full(2, c[0]), np.full(2, c[1]), zz[k : k + 2]]
                ),
                color=FLOOR,
                line_width=2.6,
            )
        p.add_mesh(pv.Sphere(radius=0.0105, center=(c[0], c[1], 0.004)), color=FLOOR)
        p.add_mesh(
            pv.Cube(center=tuple(c), x_length=CUBE, y_length=CUBE, z_length=CUBE),
            color=GREEN,
            ambient=0.5,
            diffuse=0.6,
            specular=0.25,
            show_edges=True,
            edge_color=INK,
            line_width=3.2,
        )

    el, az = np.radians(VIEW[0]), np.radians(VIEW[1])
    focus = np.array([bx[1] / 2, bx[3] / 2, bx[5] / 2])
    p.camera.position = focus + 6 * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    p.camera.focal_point = focus
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = VIEW_ANGLE
    p.reset_camera(bounds=bx)
    p.remove_all_lights()
    for off, inten in (([-2.5, -1.5, 4.0], 0.85), ([3.0, -3.0, 2.0], 0.40)):
        p.add_light(
            pv.Light(
                position=tuple(focus + np.array(off)),
                focal_point=focus,
                intensity=inten,
                light_type="scene light",
            )
        )
    p.render()
    geo = p.screenshot(None, return_img=True)[..., :3].astype(float) / 255.0
    depth = p.get_image_depth(fill_value=np.nan)  # minus the camera-z of the geometry
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    anchors = {}
    for i, xa in enumerate((0, bx[1])):
        for j, ya in enumerate((0, bx[3])):
            for k, za in enumerate((0, bx[5])):
                coord.SetValue(xa, ya, za)
                u, v = coord.GetComputedDoubleDisplayValue(p.renderer)
                anchors[f"c{i}{j}{k}"] = [u, SIZE[1] - v]
    pos, fp = np.array(p.camera.position), np.array(p.camera.focal_point)
    up, va = np.array(p.camera.up), np.radians(p.camera.view_angle)
    p.close()

    # rays through the pixel centres, clipped to the box and to the geometry
    W, Hh = SIZE
    dvec = (fp - pos) / np.linalg.norm(fp - pos)
    up = up - up.dot(dvec) * dvec
    up /= np.linalg.norm(up)
    right = np.cross(dvec, up)
    pix = 2 * np.linalg.norm(fp - pos) * np.tan(va / 2) / Hh
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(Hh) + 0.5)
    Dr = (
        fp[None, None]
        + ((uu - W / 2) * pix)[..., None] * right
        + ((Hh / 2 - vv) * pix)[..., None] * up
    ) - pos
    Dr /= np.linalg.norm(Dr, axis=-1, keepdims=True)
    origins = np.broadcast_to(pos, Dr.shape)
    lo, hi = np.array(bx[0::2]), np.array(bx[1::2])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - origins) / Dr, (hi - origins) / Dr
    tn = np.maximum(np.nanmax(np.minimum(t1, t2), axis=-1), 0)
    tf = np.nanmin(np.maximum(t1, t2), axis=-1)
    tgeo = -depth / (Dr * dvec).sum(-1)
    has_geo = np.isfinite(tgeo)
    tf = np.where(has_geo, np.minimum(tf, tgeo), tf)
    idx = np.nonzero(tf > tn)

    # front-to-back compositing; colour from the diverging map, opacity ~ |H - median|^POWER
    o_, d_, tn_, tf_ = origins[idx], Dr[idx], tn[idx], tf[idx]
    C, acc = np.zeros((len(tn_), 3)), np.zeros(len(tn_))
    dt = (tf_ - tn_) / NSAMP
    for k in range(NSAMP):
        t = tn_ + (tf_ - tn_) * (k + 0.5) / NSAMP
        z = np.nan_to_num((interp((o_ + t[:, None] * d_)[:, [2, 1, 0]]) - med) / span)
        a = 1 - np.exp(-DENSITY * np.abs(z) ** POWER * dt)
        C += ((1 - acc) * a)[:, None] * FIELD_CMAP(np.clip(0.5 + 0.5 * z, 0, 1))[:, :3]
        acc += (1 - acc) * a
    col, A = np.zeros((Hh, W, 3)), np.zeros((Hh, W))
    col[idx], A[idx] = C, acc
    f = np.where(has_geo, FOG, 1.0)
    img = f[..., None] * col + (1 - f * A)[..., None] * geo
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), anchors


# Figure composition: the DAG and its three node panels.
EDGE, MUTED = "#3D434E", "#8A919C"
ORANGE, VERMILLION, INDIGO = "#E69F00", "#D55E00", "#3B4A8C"
L_LINE, L_BAND, L_DRAW = "#C98300", "#FBE3AE", "#EEC56E"  # the L1, L2 posteriors
TINT = {"X": "#DCDEEA", "L1": "#FCEBC4", "L2": "#FCEBC4", "H": "#F7D4C0"}
RIM_C = {"X": INDIGO, "L1": ORANGE, "L2": ORANGE, "H": VERMILLION}
FIG_INK = "#1E222A"

W_FIG, H_FIG = 2.39, 1.385  # inches; type is scaled by FS so that it
FS = W_FIG / (0.37 * 430.0 / 72.27)  # prints at its true size in the paper
DY = 0.025
NODE = {
    "X": (0.110, 0.675 + DY),
    "L1": (0.470, 0.822 + DY),
    "L2": (0.470, 0.528 + DY),
    "H": (0.975, 0.675 + DY),
}
R_NODE, A_DIAM = 0.068, 0.082
CARD = {
    "L1": (0.025, 0.910 + DY, 0.865, 0.400),
    "L2": (0.025, 0.045 + DY, 0.865, 0.400),
    "H": (1.105, 0.045 + DY, 1.265, 1.265),
}  # x0, y0, width, height
TITLE = {"L1": r"$L_1(X)$", "L2": r"$L_2(X)$", "H": r"$H(L_1, L_2, X)$"}
LW_EDGE, LW_RIM, LW_CARD = 0.6, 0.85, 0.6
IN_ANG = 38  # edges into H meet the fusion arc at 180 +- IN_ANG
XAX_GAP = 0.045  # the X axis of the H card sits left of the box
H_PAD = (0.10, 0.15, 0.06, 0.11)  # render's margins in the H card: l, r, b, t
CONE_ALPHA = (0.42, 0.06)
CBAR = (0.540, 0.05, 0.028, 0.36)  # colour bar: offset from card centre, w, h
ARROW = ArrowStyle("-|>", head_length=3.0, head_width=1.45)
SMALL = ArrowStyle("-|>", head_length=2.4, head_width=1.1)


def mix(c, d, t):
    return tuple((1 - t) * np.array(to_rgb(c)) + t * np.array(to_rgb(d)))


def card(ax, n):
    x0, y0, w, h = CARD[n]
    f = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.04",
        fc="white",
        ec=mix(RIM_C[n], "white", 0.15),
        lw=LW_CARD,
        zorder=3,
    )
    ax.add_patch(f)
    return f


def card_title(ax, n):
    """The function's name as a tab on the card's outer edge."""
    x0, y0, w, h = CARD[n]
    ax.text(
        x0 + 0.075,
        y0 if n == "L2" else y0 + h,
        TITLE[n],
        ha="left",
        va="center",
        fontsize=7 * FS,
        color=FIG_INK,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.12", fc="white", ec="none"),
    )


def cone(ax, n, side):
    """Draw a shaded connector from a DAG node to its plot panel."""
    p = np.array(NODE[n])
    x0, y0, w, h = CARD[n]
    if side == "up":
        e = [np.array([x0 + 0.10, y0]), np.array([x0 + w - 0.10, y0])]
    elif side == "down":
        e = [np.array([x0 + 0.10, y0 + h]), np.array([x0 + w - 0.10, y0 + h])]
    else:
        e = [np.array([x0, y0 + 0.10]), np.array([x0, y0 + h - 0.10])]
    u = (e[0] + e[1]) / 2 - p
    L = np.linalg.norm(u)
    u /= L
    perp = np.array([-u[1], u[0]])
    t1, t2 = p + R_NODE * 0.9 * perp, p - R_NODE * 0.9 * perp
    e.sort(key=lambda v: -np.dot(v - p, perp))
    poly = Polygon([t1, e[0], e[1], t2], closed=True, fc="none", ec="none", zorder=1)
    ax.add_patch(poly)
    pts = np.array([t1, t2, *e])
    xa, xb, ya, yb = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    GX, GY = np.meshgrid(np.linspace(xa, xb, 240), np.linspace(ya, yb, 240))
    t = np.clip(((GX - p[0]) * u[0] + (GY - p[1]) * u[1]) / L, 0, 1)
    a = CONE_ALPHA[0] * (1 - t) ** 0.9 + CONE_ALPHA[1]
    img = 1 - a[..., None] * (1 - np.array(to_rgb(TINT[n])))
    im = ax.imshow(
        img,
        extent=(xa, xb, ya, yb),
        origin="lower",
        interpolation="bilinear",
        zorder=1,
        aspect="auto",
    )
    im.set_clip_path(poly)
    for tt, v in ((t1, e[0]), (t2, e[1])):
        ax.plot(
            *zip(tt, v, strict=True),
            color=mix(RIM_C[n], "white", 0.45),
            lw=0.35,
            zorder=1.2,
        )


def node(ax, n):
    c = NODE[n]
    if n == "X":  # the input: a diamond
        a = A_DIAM
        ax.add_patch(
            Polygon(
                [
                    (c[0] - a, c[1]),
                    (c[0], c[1] + a),
                    (c[0] + a, c[1]),
                    (c[0], c[1] - a),
                ],
                closed=True,
                fc=TINT[n],
                ec=RIM_C[n],
                lw=LW_RIM,
                joinstyle="miter",
                zorder=6,
            )
        )
    else:
        ax.add_patch(Circle(c, R_NODE, fc=TINT[n], ec=RIM_C[n], lw=LW_RIM, zorder=6))
    lab = {"X": r"$X$", "L1": r"$L_1$", "L2": r"$L_2$", "H": r"$H$"}[n]
    ax.text(
        *c, lab, ha="center", va="center", fontsize=7.5 * FS, color=FIG_INK, zorder=7
    )


def reach(n, u):
    return A_DIAM / (abs(u[0]) + abs(u[1])) if n == "X" else R_NODE


def edges(ax):
    """Straight edges; X -> H, L1 -> H, L2 -> H end on the fusion arc around H."""
    H0 = np.array(NODE["H"])
    fuse_r = R_NODE + 0.034
    kw = dict(
        arrowstyle=ARROW,
        mutation_scale=1,
        color=EDGE,
        lw=LW_EDGE,
        shrinkA=0,
        shrinkB=0,
        zorder=5,
        joinstyle="miter",
        capstyle="butt",
    )
    for a, b in (("X", "L1"), ("X", "L2")):
        pa, pb = np.array(NODE[a]), np.array(NODE[b])
        u = (pb - pa) / np.linalg.norm(pb - pa)
        ax.add_patch(
            FancyArrowPatch(
                pa + u * (reach(a, u) + 0.018), pb - u * (reach(b, -u) + 0.022), **kw
            )
        )
    for a, ang in (("L1", 180 - IN_ANG), ("X", 180), ("L2", 180 + IN_ANG)):
        th = np.radians(ang)
        end = H0 + np.array([np.cos(th), np.sin(th)]) * (fuse_r + 0.024)
        pa = np.array(NODE[a])
        u = (end - pa) / np.linalg.norm(end - pa)
        ax.add_patch(FancyArrowPatch(pa + u * (reach(a, u) + 0.018), end, **kw))
    ax.add_patch(
        Arc(
            tuple(H0),
            2 * fuse_r,
            2 * fuse_r,
            theta1=180 - IN_ANG - 14,
            theta2=180 + IN_ANG + 14,
            color=VERMILLION,
            lw=1.25,
            capstyle="round",
            zorder=6,
        )
    )


def axis_arrow(ax, s, e, lw=0.6):
    ax.add_patch(
        FancyArrowPatch(
            s,
            e,
            arrowstyle=SMALL,
            mutation_scale=1,
            color=EDGE,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


def low_fidelity(ax, n, d, frame):
    """Posterior of L1 or L2 against X: 95% band, joint draws, mean and observations."""
    x0, y0, w, h = CARD[n]
    px, pw = x0 + 0.055, w - 0.14
    py, ph = (y0 + 0.105, h - 0.16) if n == "L1" else (y0 + 0.15, h - 0.185)
    m, s = d[f"{n}_mean"], d[f"{n}_sd"]
    lo, hi = m - 1.96 * s, m + 1.96 * s
    xo, yo = (d["x1"], d["y1"]) if n == "L1" else (d["x2"], d["y2"])
    a, b = min(lo.min(), yo.min()), max(hi.max(), yo.max())

    def X(t):
        return px + t * pw

    def Y(v):
        return py + (v - a) / (b - a) * ph

    xs = X(d["x"])
    arts = [ax.fill_between(xs, Y(lo), Y(hi), color=L_BAND, lw=0, zorder=4)]
    for dr in d[f"{n}_draws"]:
        arts += ax.plot(xs, Y(dr), color=L_DRAW, lw=0.3, zorder=4.5)
    arts += ax.plot(xs, Y(m), color=L_LINE, lw=0.95, zorder=5, solid_capstyle="round")
    arts.append(
        ax.scatter(
            X(xo),
            Y(yo),
            s=2.0,
            marker="s",
            c=GREEN,
            edgecolors="white",
            linewidths=0.15,
            zorder=6,
        )
    )
    for art in arts:
        art.set_clip_path(frame)
    ya = y0 + (0.05 if n == "L1" else 0.105)
    axis_arrow(ax, (px, ya), (px + pw + 0.015, ya))
    ax.text(
        px + pw + 0.029,
        ya,
        r"$X$",
        fontsize=7 * FS,
        color=FIG_INK,
        ha="left",
        va="center",
        zorder=7,
    )


def high_fidelity(ax, frame, im, anchors):
    """The volume render of H, with the axes, their labels and the colour bar."""
    x0, y0, w, h = CARD["H"]
    cx, cy = x0 + w / 2, y0 + h / 2
    ys, xs = np.nonzero((np.asarray(im) < 250).any(-1))
    bb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)  # crop the white margin
    rgb = np.asarray(im.crop(bb))
    ih, iw = rgb.shape[:2]
    fx0, fx1, fy0, fy1 = (
        x0 + H_PAD[0],
        x0 + w - H_PAD[1],
        y0 + H_PAD[2],
        y0 + h - H_PAD[3],
    )
    sc = min((fx1 - fx0) / iw, (fy1 - fy0) / ih)
    ew, eh = iw * sc, ih * sc
    ex0, ey0 = (fx0 + fx1) / 2 - ew / 2, (fy0 + fy1) / 2 - eh / 2
    ax.imshow(
        rgb, extent=(ex0, ex0 + ew, ey0, ey0 + eh), interpolation="lanczos", zorder=4
    ).set_clip_path(frame)

    def to_in(q):
        return np.array([ex0 + (q[0] - bb[0]) * sc, ey0 + eh - (q[1] - bb[1]) * sc])

    A = {k: to_in(v) for k, v in anchors.items()}
    fl = [k for k in A if k.endswith("0")]  # floor corners
    e1 = min(
        (("c000", "c100"), ("c010", "c110")), key=lambda e: A[e[0]][1] + A[e[1]][1]
    )
    e2 = min(
        (("c000", "c010"), ("c100", "c110")), key=lambda e: A[e[0]][1] + A[e[1]][1]
    )
    centre = np.mean([A[k] for k in fl], axis=0)
    for (p_, q_), lab in ((e1, r"$L_1$"), (e2, r"$L_2$")):  # front floor edges
        s_, e_ = A[p_], A[q_]
        mid = (s_ + e_) / 2
        out = (mid - centre) / np.linalg.norm(mid - centre)
        s_, e_ = s_ + 0.02 * out, e_ + 0.02 * out
        axis_arrow(ax, s_ + 0.04 * (e_ - s_), e_ - 0.02 * (e_ - s_), lw=0.5)
        ax.text(
            *(mid + 0.06 * out),
            lab,
            fontsize=7 * FS,
            color=FIG_INK,
            ha="center",
            va="center",
            zorder=7,
        )
    corner = min(fl, key=lambda k: A[k][0])  # vertical axis: X
    s_ = A[corner] + np.array([-XAX_GAP, 0.0])
    e_ = A[corner[:3] + "1"] + np.array([-XAX_GAP, 0.0])
    axis_arrow(ax, s_, e_ + np.array([0, 0.02]))
    ax.text(
        e_[0],
        e_[1] + 0.07,
        r"$X$",
        fontsize=7 * FS,
        color=FIG_INK,
        ha="center",
        va="center",
        zorder=7,
    )
    bx0, by0, bw, bh = cx + CBAR[0], cy + CBAR[1], CBAR[2], CBAR[3]
    grad = TUBE_CMAP(np.linspace(0, 1, 256))[::-1, None, :3]
    ax.imshow(
        np.repeat(grad, 2, axis=1),
        extent=(bx0, bx0 + bw, by0, by0 + bh),
        interpolation="bilinear",
        zorder=6,
        aspect="auto",
    )
    ax.add_patch(
        Rectangle((bx0, by0), bw, bh, fill=False, ec=MUTED, lw=0.3, zorder=6.1)
    )
    ax.text(
        bx0 + bw / 2,
        by0 + bh + 0.065,
        r"$H$",
        fontsize=7 * FS,
        color=FIG_INK,
        ha="center",
        va="center",
        zorder=7,
    )


def compose(d, im, anchors, out):
    mpl.rcParams["font.size"] = 7
    fig = plt.figure(figsize=(W_FIG, H_FIG))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, H_FIG)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    cone(ax, "L1", "up")
    cone(ax, "L2", "down")
    cone(ax, "H", "right")
    for n in ("L1", "L2"):
        low_fidelity(ax, n, d, card(ax, n))
    high_fidelity(ax, card(ax, "H"), im, anchors)
    for n in ("L1", "L2", "H"):
        card_title(ax, n)
    edges(ax)
    for n in ("X", "L1", "L2", "H"):
        node(ax, n)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


APP_INK, APP_GREEN, APP_INDIGO, GREY = "#1E222A", "#009E73", "#3B4A8C", "#8A8F99"
TRUTH = {
    "L1": "#E69F00",
    "L2": "#56B4E9",
    "H": "#D55E00",
}  # Figure of the three functions
POST = {
    "L1": (L_LINE, L_BAND),
    "L2": (L_LINE, L_BAND),  # as the L cards of Figure 1
    "H": ("#D55E00", "#F4C7A6"),
}
LABEL = {"L1": "$L_1$", "L2": "$L_2$", "H": "$H$"}
OBS = {"L1": ("x1", "y1"), "L2": ("x2", "y2"), "H": ("xH", "yH")}
GP_BAND_ALPHA = 0.12


def truth(x):
    return {"L1": model.f_L1(x), "L2": model.f_L2(x), "H": model.f_H(x)}


def gp_on_x(d):
    """Baseline: a GP on X alone fitted to the high-fidelity data (SE kernel, type-II ML)."""
    hp = model.fit_se(d["xH"], d["yH"], np.random.default_rng(0), ell_min=0.05)
    m, v = model.gp_post(d["xH"], d["yH"], d["x"], hp)
    return m, np.sqrt(v)


def plot_fidelities(d, out):
    """The three functions of the data-generating process, with direct labels."""
    x, t = d["x"], truth(d["x"])
    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    ends = []
    for k in ("L1", "L2", "H"):
        ax.plot(x, t[k], color=TRUTH[k], lw=1.3)
        ends.append([t[k][-1], LABEL[k], TRUTH[k]])
    ends.sort()  # labels at the right, kept 0.2 apart
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.2)
    for y, lab, colour in ends:
        ax.text(1.015, y, lab, color=colour, ha="left", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$X$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(d, gp, out):
    """Posterior mean, 95% band, ground truth and data of each node; GP on X in the H panel."""
    x, t = d["x"], truth(d["x"])
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.65))
    for ax, k in zip(axes, ("L1", "L2", "H"), strict=True):
        line, band = POST[k]
        m, s = d[f"{k}_mean"], d[f"{k}_sd"]
        if k == "H":
            mA, sA = gp[0], gp[1]
            ax.fill_between(
                x,
                mA - 1.96 * sA,
                mA + 1.96 * sA,
                color=APP_INDIGO,
                alpha=GP_BAND_ALPHA,
                lw=0,
            )
            ax.plot(x, mA, color=APP_INDIGO, lw=0.9, zorder=3)
        ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=band, lw=0)
        ax.plot(x, m, color=line, lw=1.2)
        ax.plot(x, t[k], color=APP_INK, lw=0.7, ls=(0, (3, 2)))
        ax.scatter(
            d[OBS[k][0]],
            d[OBS[k][1]],
            marker="s",
            s=9,
            c=APP_GREEN,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        ax.set_title(LABEL[k], fontsize=9, pad=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("$X$", labelpad=1)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, pad=2)
    axes[2].set_yticks([-0.5, 0, 0.5, 1])
    handles = [
        Patch(color=GREY, alpha=0.35, lw=0),
        Line2D([], [], color=GREY, lw=1.2),
        Line2D([], [], color=APP_INK, lw=0.7, ls=(0, (3, 2))),
        Line2D([], [], marker="s", ls="", ms=3.2, mfc=APP_GREEN, mec="white", mew=0.4),
        (
            Patch(color=APP_INDIGO, alpha=GP_BAND_ALPHA, lw=0),
            Line2D([], [], color=APP_INDIGO, lw=0.9),
        ),
    ]
    fig.legend(
        handles,
        ["95\\% band", "posterior mean", "truth", "data", "GP on $X$ only"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
