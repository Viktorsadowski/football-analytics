"""
Names next to points in a scatter, without sitting on each other.

Used by the charts that label a handful of points in a crowd (report/make_figures.py, report/make_case_figures.py,
defense.py). Every name tries the spots around its point, closest first, and takes the first one that hits
nothing: no other name, no other marked point, no other text on the chart, not outside the axes. It also has to
be closer to its own point than to any other marked one, otherwise you can't tell whose name it is. If the name
ends up away from its point it gets a thin line back to it.

Call it last: after the limits are set and after tight_layout, the check is done in pixels.
"""

import numpy as np
from matplotlib import patheffects
from matplotlib.transforms import Bbox

# where a name can go, relative to its point: (x direction, y direction). right of the point first
SPOTS = [(1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)]
DIST = [5, 9, 14, 20, 28, 38, 50]        # how far out, in points


def _overlap(a: Bbox, b: Bbox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def _gap(box: Bbox, x: float, y: float) -> float:
    """distance from a point to the nearest edge of a box, 0 inside"""
    return float(np.hypot(max(box.x0 - x, 0, x - box.x1), max(box.y0 - y, 0, y - box.y1)))


def place_labels(ax, xs, ys, names, avoid_xy=None, avoid_line=None, avoid_artists=(), fontsize=6.8,
                 color="#0b0b0b", halo="#fcfcfb", line_color="#52514e", marker_pt=3.5):
    """xs, ys, names: the points that get a name. avoid_xy: other marked points, without a name. avoid_line:
    points along a line the names should stay off. avoid_artists: texts / legends already on the chart.
    marker_pt: radius of a marked point, in points."""
    xs, ys, names = np.asarray(xs, float), np.asarray(ys, float), list(names)
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = fig.dpi / 72                                   # points -> pixels
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = marker_pt * k
    # boxes a name may not touch: every marked point, then whatever else is on the chart
    dots = [Bbox.from_extents(px - r, py - r, px + r, py + r) for px, py in pts]
    marked = [tuple(p) for p in pts]
    if avoid_xy is not None and len(avoid_xy):
        for px, py in ax.transData.transform(np.asarray(avoid_xy, float)):
            dots.append(Bbox.from_extents(px - r, py - r, px + r, py + r))
            marked.append((px, py))
    if avoid_line is not None and len(avoid_line):
        for px, py in ax.transData.transform(np.asarray(avoid_line, float)):
            dots.append(Bbox.from_extents(px - 2, py - 2, px + 2, py + 2))
    fixed = [a.get_window_extent(rend) for a in avoid_artists]
    frame = ax.get_window_extent(rend)
    # the crowded points first, they have the fewest free spots
    crowd = [sum(np.hypot(*(p - q)) < 60 * k for q in pts) for p in pts]
    placed, out = [], []
    for i in np.argsort(crowd)[::-1]:
        px, py = pts[i]
        t = ax.text(0, 0, names[i], fontsize=fontsize)
        ext = t.get_window_extent(rend)
        t.remove()
        w, h = ext.width, ext.height
        best = None
        for d in DIST:
            for sx, sy in SPOTS:
                # box of the text if it sits d away in this direction (diagonals a bit closer in)
                f = 0.75 if sx and sy else 1.0
                cx = px + sx * (d * k * f + r) + (sx * w / 2 if sx else 0)
                cy = py + sy * (d * k * f + r) + (sy * h / 2 if sy else 0)
                box = Bbox.from_extents(cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2)
                area = box.width * box.height
                others = placed + fixed + [b for j, b in enumerate(dots) if j != i]
                hit = sum(_overlap(box, b) for b in others)
                # a name that leaves the axes counts as a hit too
                hit += area - _overlap(box, frame)
                # far enough out (or on a diagonal) the name gets a line back to its point
                line = d >= 14 or (bool(sx and sy) and d >= 9)
                if line:
                    # the line itself shouldn't run through a name or another point
                    ex, ey = min(max(px, box.x0), box.x1), min(max(py, box.y0), box.y1)
                    seg = [(px + (ex - px) * u, py + (ey - py) * u) for u in np.linspace(0.25, 1, 8)]
                    hit += area * sum(any(b.x0 <= qx <= b.x1 and b.y0 <= qy <= b.y1 for qx, qy in seg) for b in others)
                    unclear = 0
                else:
                    # no line: then no other marked point may be anywhere near as close to the name as its own
                    own = _gap(box, px, py)
                    unclear = sum(_gap(box, qx, qy) < 2.0 * own for j, (qx, qy) in enumerate(marked) if j != i)
                # overlap is worse than an unclear name, and a spot without a line wins a tie
                cost = 10 * hit + area * unclear + (0.01 * area if line else 0)
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, cx, cy, line, box)
                if hit == 0 and unclear == 0:
                    break
            else:
                continue
            break
        _, cx, cy, line, box = best
        placed.append(box)
        t = ax.annotate(names[i], xy=(xs[i], ys[i]), xytext=((cx - px) / k, (cy - py) / k),
                        textcoords="offset points", ha="center", va="center", fontsize=fontsize, color=color,
                        zorder=6)
        # a thin edge in the background colour, so a name stays readable on top of the grey crowd
        t.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground=halo)])
        if line:
            # line from the point to the nearest edge of the name, under the marker so it starts at its rim.
            # later names stay off it
            ex, ey = min(max(px, box.x0), box.x1), min(max(py, box.y0), box.y1)
            (x0, y0), (x1, y1) = ax.transData.inverted().transform([(px, py), (ex, ey)])
            ax.plot([x0, x1], [y0, y1], color=line_color, lw=0.5, zorder=2.5)
            for u in np.linspace(0.2, 1, 10):
                qx, qy = px + (ex - px) * u, py + (ey - py) * u
                dots.append(Bbox.from_extents(qx - 2, qy - 2, qx + 2, qy + 2))
        out.append(t)
    return out
