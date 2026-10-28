#!/usr/bin/env python3
"""
Figures for the united case PDF (season_sim.png comes from src/season_sim.py into figures/).

  report/figures/case_points.png    planned vs real points above replacement per window
  report/figures/case_players.png   the signings: projected rating in summer 2023 vs real 2023-24 rating

  python report/make_case_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROCESSED_DIR  # noqa: E402
from labels import place_labels  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLS = {"standing still": "#8a8984", "actual": "#c0392b", "A": "#2f6db3", "B": "#e08a1e"}
ORDER = ("standing still", "actual", "A", "B")


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def case_points() -> None:
    j = json.loads((PROCESSED_DIR / "united_case.json").read_text())["hindsight"]
    parts = [("planned_par", "projected in summer 2023"), ("real_par_planned_min", "real 2023-24 value, planned minutes"),
             ("real_par_real_min", "real value, minutes capped at what he played")]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    w = 0.26
    x = np.arange(len(ORDER))
    for k, (key, lab) in enumerate(parts):
        vals = [j[o][key] for o in ORDER]
        alpha = [1.0, 0.6, 0.3][k]
        bars = ax.bar(x + (k - 1) * w, vals, width=w * 0.92, color=[COLS[o] for o in ORDER], alpha=alpha,
                      label=lab, edgecolor=[COLS[o] for o in ORDER], lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                    color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["standing still", "real window", "scenario A:\nbest value", "scenario B:\nmost points"], fontsize=8, color=INK)
    ax.set_ylabel("league points above replacement level,\n2023-24", color=MUTED, fontsize=8)
    ax.set_ylim(0, max(j[o]["planned_par"] for o in ORDER) * 1.22)
    # legend in grey, the shades are what it explains
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=MUTED, alpha=a, edgecolor=MUTED, label=p[1]) for a, p in zip((1.0, 0.6, 0.3), parts)]
    ax.legend(handles=hs, frameon=False, fontsize=7.2, loc="upper left", labelcolor=INK)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "case_points.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def case_players() -> None:
    g = pd.read_parquet(PROCESSED_DIR / "united_case_moves.parquet")
    g = g[(g["dir"] == "in") & g["val_real"].notna()].drop_duplicates("name")
    # a player bought in two windows gets the colour of the first
    first = pd.read_parquet(PROCESSED_DIR / "united_case_moves.parquet")
    first = first[first["dir"] == "in"].groupby("name")["window"].agg(lambda s: "+".join(sorted(set(s))))
    g["win"] = g["name"].map(first)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "both")
    lim = (-0.04, 0.32)
    ax.plot(lim, lim, color=INK, lw=0.9)
    for w, lab in (("A", "scenario A"), ("B", "scenario B"), ("A+B", "both scenarios"), ("actual", "real window")):
        d = g[g["win"] == w]
        col = {"A": COLS["A"], "B": COLS["B"], "A+B": "#6b4fa0", "actual": COLS["actual"]}[w]
        ax.scatter(d["val_pre"], d["val_real"], s=30, color=col, label=lab, zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    note = ax.text(-0.03, 0.29, "above the line: better in 2023-24\nthan projected in summer 2023", fontsize=7.2,
                   color=INK, va="top")
    ax.set_xlabel("projected value for 2023-24, npxG difference per 90 (summer 2023)", color=MUTED, fontsize=8)
    ax.set_ylabel("real 2023-24 value, npxG difference per 90", color=MUTED, fontsize=8)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    lx = np.linspace(*lim, 100)
    place_labels(ax, g["val_pre"], g["val_real"], g["name"], avoid_line=np.column_stack([lx, lx]),
                 avoid_artists=[note, leg], marker_pt=3)
    fig.savefig(OUT / "case_players.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    case_points()
    case_players()
    print(f"figures -> {OUT}")
