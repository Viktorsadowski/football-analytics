#!/usr/bin/env python3
"""
Figures that only the report needs (the analysis figures come from src/ into figures/).

  report/figures/pipeline.png        how the pieces fit together
  report/figures/market.png          model value vs transfermarkt value, premier league, summer 2024
  report/figures/market_test.png     does the market move towards the model: next year's value change by gap
  report/figures/forecast.png        august forecasts vs the benchmarks, points off per team
  report/figures/wages.png           reported wages vs the wage model (leave one out)
  report/figures/transfer_test.png   predicted vs realised surplus of EPL signings, by quintile

  python report/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROCESSED_DIR, ROOT, SEASONS  # noqa: E402
from labels import place_labels  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, MUTED, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
GREY, GREEN, PURPLE = "#9a9893", "#2e9e6b", "#8a5cc2"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREY = "#e3eefb", "#fdeadf", "#f0efeb"


def style(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plain_log(ax):
    # 1, 10, 100 on log axes instead of 10^0, 10^1
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.5, 99)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fill, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge, lw=1))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=8.6, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=6.5, color=MUTED, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=0.9,
                                     shrinkA=0, shrinkB=0))

    w, h = 16.5, 8.5
    xs = [0, 20.5, 41, 61.5, 82]
    top, mid, bot = 25, 13, 1
    # talent
    main = [("Understat", "shots + rosters, big\nfive, 2014-15 to 2023-24"),
            ("Segments", "120k stretches with the\nsame 22 on the pitch"),
            ("RAPM", "npxG ridge with a box-\nscore prior, O + D"),
            ("Value", "aging, keepers, league\nstrength, shrink"),
            ("Points (PAR)", "calibrated on team\nresults, vs replacement")]
    for x, (t, s) in zip(xs, main):
        box(x, top, w, h, t, s, LIGHT_BLUE, BLUE)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w + 0.4, top + h / 2, b - 0.4, top + h / 2)
    # availability
    box(xs[1], mid, w, h, "Transfermarkt", "lineups, squads, values,\nfees, contracts", LIGHT_ORANGE, ORANGE)
    box(xs[2], mid, w, h, "Injuries", "inferred from matchday\nsquads, checked by hand", LIGHT_ORANGE, ORANGE)
    box(xs[3], mid, w, h, "Minutes", "injury risk + share of\nteam minutes next season", LIGHT_ORANGE, ORANGE)
    arrow(xs[1] + w + 0.4, mid + h / 2, xs[2] - 0.4, mid + h / 2)
    arrow(xs[2] + w + 0.4, mid + h / 2, xs[3] - 0.4, mid + h / 2)
    # elbow arrows so minutes -> points and points -> transfer value don't cross or run over a box
    def elbow(pts):
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=0.9, solid_capstyle="butt")
        arrow(*pts[-2], *pts[-1])
    elbow([(xs[3] + w + 0.4, mid + h / 2), (xs[4] + 4, mid + h / 2), (xs[4] + 4, top - 0.4)])
    # money
    box(xs[1], bot, w, h, "Prize money", "PL payments by place,\nCL, relegation", LIGHT_GREY, MUTED)
    box(xs[2], bot, w, h, "Wages", "59 press-reported wages,\n126 club accounts", LIGHT_GREY, MUTED)
    box(xs[3], bot, w, h, "Transfer value", "points x price - wages\n+ resale - fee", LIGHT_GREY, MUTED)
    box(xs[4], bot, w, h, "United case", "summer 2023, scenarios A\nand B, graded on 2023-24", LIGHT_GREY, MUTED)
    arrow(xs[1] + w + 0.4, bot + h / 2, xs[2] - 0.4, bot + h / 2)
    arrow(xs[2] + w + 0.4, bot + h / 2, xs[3] - 0.4, bot + h / 2)
    arrow(xs[3] + w + 0.4, bot + h / 2, xs[4] - 0.4, bot + h / 2)
    elbow([(xs[4] + w - 3, top - 0.4), (xs[4] + w - 3, bot + h + 1.7), (xs[3] + w / 2, bot + h + 1.7),
           (xs[3] + w / 2, bot + h + 0.4)])
    ax.text(0.5, mid + h / 2 - 6.5, "Side questions on\nthe same values:\npositions, age,\nbig fees, overreaction",
            fontsize=6.8, color=MUTED, va="center", linespacing=1.3)
    save(fig, "pipeline.png")


def market() -> None:
    m = pd.read_parquet(PROCESSED_DIR / "market_value.parquet")
    s = m[(m["season"] == max(SEASONS)) & (m["league"] == "EPL") & (m["proj_min"] >= 1500) & (m["mv"] > 0)].copy()
    x, y = s["mv"] / 1e6, s["model_eur"] / 1e6
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.scatter(x, y.clip(lower=0.3), s=12, color=MUTED, alpha=0.35, lw=0)
    under = s.nlargest(7, "gap")
    over = s.nsmallest(7, "gap")
    big = s[~s.index.isin(under.index) & ~s.index.isin(over.index)].nlargest(5, "model_eur")
    for d, col, lab in ((under, BLUE, "model likes him more than the market"),
                        (over, ORANGE, "market likes him more than the model"), (big, INK, "highest model value")):
        ax.scatter(d["mv"] / 1e6, (d["model_eur"] / 1e6).clip(lower=0.3), s=26, color=col, zorder=3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = (0.3, 250)
    ax.plot(lim, lim, color=INK, lw=0.9)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    plain_log(ax)
    note = ax.text(0.4, 120, "on the line: model and\nmarket agree", fontsize=7.5, color=INK, va="center")
    ax.set_xlabel("Transfermarkt market value, 1 Sept 2024, EUR m (log)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("model value: next 4 seasons of points x EUR 4.06m, EUR m (log)", color=MUTED, fontsize=8.5)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    named = pd.concat([under, over, big])
    lx = np.geomspace(*lim, 200)
    place_labels(ax, named["mv"] / 1e6, (named["model_eur"] / 1e6).clip(lower=0.3), named["name"],
                 avoid_line=np.column_stack([lx, lx]), avoid_artists=[note, leg], marker_pt=2.8)
    save(fig, "market.png")


def market_test() -> None:
    # same regression as market_value.py: next year's change in log value, age and level taken out, by gap quintile
    m = pd.read_parquet(PROCESSED_DIR / "market_value.parquet")
    t = m[(m["mv"] > 0) & (m["mv_next"] > 0) & m["age"].notna() & (m["proj_min"] >= 900)].copy()
    t["dlog"] = np.log(t["mv_next"] / t["mv"])
    X0 = np.column_stack([np.ones(len(t)), t["age"], t["age"] ** 2, np.log(t["mv"])])
    b0, *_ = np.linalg.lstsq(X0, t["dlog"], rcond=None)
    resid = t["dlog"] - X0 @ b0
    q = pd.qcut(t["gap"], 5, labels=["most\noverpriced", "2", "3", "4", "most\nunderpriced"])
    v = 100 * (np.exp(resid.groupby(q, observed=True).mean()) - 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "y")
    cols = [ORANGE, ORANGE, GREY, BLUE, BLUE]
    ax.bar(range(5), v.to_numpy(), color=cols, width=0.62)
    for i, val in enumerate(v):
        ax.text(i, val + (0.4 if val >= 0 else -0.4), f"{val:+.1f}%".replace("-", "−"), ha="center",
                va="bottom" if val >= 0 else "top", fontsize=7.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(v.index, fontsize=7.5, color=INK)
    ax.set_ylabel("market value change next year, %\n(age and value level taken out)", color=MUTED, fontsize=8)
    ax.set_xlabel(f"model vs market gap, quintiles of {len(t):,} player-seasons", color=MUTED, fontsize=8)
    ax.set_ylim(v.min() - 2.5, v.max() + 2.5)
    save(fig, "market_test.png")


def forecast() -> None:
    from forecast_backtest import score
    d = pd.read_parquet(PROCESSED_DIR / "forecast_backtest.parquet")
    rows = [("ours, no fit", "ours"), ("ours, fitted line", "p_ours_fit"), ("last season", "p_last"),
            ("Elo 1 August", "p_elo"), ("odds, first 3 rounds", "p_mkt"), ("50/50 ours + odds", "p_blend"),
            ("closing odds, whole season\n(uses in-season info)", "p_close")]
    vals = [score(d, c)["mae"] for _, c in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "x")
    cols = [BLUE, BLUE, GREY, GREY, GREY, BLUE, LIGHT_GREY]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=cols, height=0.6, edgecolor=[c if c != LIGHT_GREY else GREY for c in cols], lw=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 0.1, yy, f"{v:.1f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5, color=INK)
    ax.set_xlabel("points off per team-season, mean absolute error, 2016-17 to 2023-24", color=MUTED, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.15)
    save(fig, "forecast.png")


def wages() -> None:
    import wages as W
    r = pd.read_parquet(PROCESSED_DIR / "reported_wages.parquet")
    ok = r[r["quality"] == "ok"].reset_index(drop=True)
    terms = ("league", "age", "elo")
    A, y = W.design(ok, terms), np.log(ok["wage"]).to_numpy()
    pred = np.zeros(len(ok))
    for i in range(len(ok)):
        msk = np.arange(len(ok)) != i
        b = np.linalg.lstsq(A[msk], y[msk], rcond=None)[0]
        pred[i] = A[i] @ b
    ok["loo"] = np.exp(pred) / 1e6
    ok["w"] = ok["wage"] / 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    lc = {"EPL": BLUE, "Serie A": GREEN, "Bundesliga": INK, "Ligue 1": ORANGE}
    for lg, g in ok.groupby("league"):
        ax.scatter(g["loo"], g["w"], s=22, color=lc.get(lg, GREY), label=lg, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = (0.2, 40)
    ax.plot(lim, lim, color=INK, lw=0.9)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    plain_log(ax)
    ax.set_xlabel("wage model, left out of its own fit, EUR m a year gross (log)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("reported wage, EUR m a year gross (log)", color=MUTED, fontsize=8.5)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    ok["miss"] = np.log(ok["w"] / ok["loo"]).abs()
    named = ok.nlargest(8, "miss")
    named = pd.concat([named, ok[ok["player"].isin(["Kobbie Mainoo", "Harry Kane", "Casemiro", "Erling Haaland"])]])
    named = named.drop_duplicates("player")
    lx = np.geomspace(*lim, 200)
    place_labels(ax, named["loo"], named["w"], named["player"], avoid_xy=ok[["loo", "w"]].to_numpy(),
                 avoid_line=np.column_stack([lx, lx]), avoid_artists=[leg], marker_pt=2.8)
    save(fig, "wages.png")


def transfer_test() -> None:
    from scipy.stats import spearmanr
    import transfer_value as T
    mv, curve, params, wf, con, elo = T.load()
    ctx = T.context(mv, curve, params, wf, con)
    bt = T.backtest(mv, ctx, elo)
    q = pd.qcut(bt["pred_surplus"], 5, labels=False)
    v = bt.groupby(q)["real_surplus"].median() / 1e6
    rc = {c: spearmanr(bt[c], bt["real_surplus"]).statistic for c in ("pred_surplus", "old_surplus", "naive")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.0), dpi=200, gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor(SURFACE)
    style(ax, "y")
    ax.bar(range(5), v.to_numpy(), color=[ORANGE if x < 0 else BLUE for x in v], width=0.62)
    for i, val in enumerate(v):
        ax.text(i, val + (0.5 if val >= 0 else -0.5), f"{val:+.1f}".replace("-", "−"), ha="center",
                va="bottom" if val >= 0 else "top", fontsize=7.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["worst\npredicted", "2", "3", "4", "best\npredicted"], fontsize=7.5, color=INK)
    ax.set_ylabel("realised surplus over 3 seasons,\nmedian, EUR m", color=MUTED, fontsize=8)
    ax.set_ylim(v.min() - 4, v.max() + 4)
    ax.set_title(f"{len(bt)} PL summer signings 2015-20", loc="left", fontsize=8.5, color=INK)
    style(bx, "x")
    names = ["market value - fee", "old: no wages", "new: wages, contract,\nresale"]
    vals = [rc["naive"], rc["old_surplus"], rc["pred_surplus"]]
    bx.barh(range(3), vals, color=[GREY, GREY, BLUE], height=0.55)
    for i, val in enumerate(vals):
        bx.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=7.5, color=INK)
    bx.set_yticks(range(3))
    bx.set_yticklabels(names, fontsize=7.5, color=INK)
    bx.set_xlim(0, max(vals) * 1.25)
    bx.set_xlabel("rank correlation with realised surplus", color=MUTED, fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig, "transfer_test.png")


if __name__ == "__main__":
    pipeline()
    market()
    market_test()
    forecast()
    wages()
    transfer_test()
    print(f"figures -> {OUT}")
