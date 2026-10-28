#!/usr/bin/env python3
"""
Step 12: play the season after the window out (2023-24 for the summer 2023 case). how would united have done with
window A, B, standing still, or the real one: chance of the title, top 4 (= the CL places that season), relegation.

  ratings      every PL club on 1 Sept, like forecast_backtest.py: squad (latest lineup or transfer before then),
               val_cal as of the season before x availability.py's projected minutes share, scaled to 11 players,
               the rest at the newcomer value. = npxG difference per match, league mean taken out
  united       the real squad's rating + (window's strength - real window's strength), both from
               united_case.py's depth chart (val_cal x minutes / 90 / 38), so the two methods only meet in a
               difference
  matches      the real fixture list (transfermarkt). goals Poisson: home = h x exp(k (r_i - r_j) / 2),
               away = a x exp(-k (r_i - r_j) / 2), h and a = mean PL home / away goals in the seasons before.
               k set so a rating gap gives the points it gave historically (forecast_backtest.parquet: points on
               strength, all leagues, seasons before only)
  uncertainty  each simulated season every club's rating gets a draw from N(0, sd), sd set so the spread of
               points around the expectation matches the real forecast error (RMSE of the fitted line in the
               backtest) once match luck is counted. without it the sim is far too sure
  graded       predicted points vs the real table, and united_case.py's hindsight points (real ratings) on the chart

Output: data/processed/season_sim.parquet (club x window: expected points, title, top 4, top 5, relegation),
        figures/season_sim.png

  python src/season_sim.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT
from tm_match import us_to_tm

TM = RAW_DIR / "transfermarkt"
from united_case import DECISION
SEASON = DECISION                    # the season after the window
UNITED = 985
N_SIM = 20000
FULL = 38 * 90
rng = np.random.default_rng(7)


def clubs_and_fixtures():
    g = pd.read_csv(TM / "games.csv", usecols=["competition_id", "season", "date", "home_club_id", "away_club_id",
                                               "home_club_goals", "away_club_goals", "home_club_name", "away_club_name"])
    g = g[(g["competition_id"] == "GB1") & (g["season"] == SEASON)]
    names = pd.concat([g.set_index("home_club_id")["home_club_name"], g.set_index("away_club_id")["away_club_name"]])
    names = names[~names.index.duplicated()]
    return g, names


def squads_sept(clubs) -> pd.DataFrame:
    """tm player -> tm club on 1 sept, same rule as forecast_backtest.squads()"""
    cut = pd.Timestamp(f"{SEASON}-09-01")
    gl = pd.read_csv(TM / "game_lineups.csv", usecols=["game_id", "player_id", "club_id"])
    gd = pd.read_csv(TM / "games.csv", usecols=["game_id", "date", "competition_type"])
    gd = gd[gd["competition_type"] != "national_team_competition"]
    gl = gl.merge(gd[["game_id", "date"]], on="game_id")
    gl["date"] = pd.to_datetime(gl["date"])
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    sq = gl[gl["date"] <= cut].sort_values("date").drop_duplicates("player_id", keep="last")[["player_id", "club_id", "date"]]
    t = tr[tr["transfer_date"] <= cut].sort_values("transfer_date").drop_duplicates("player_id", keep="last")
    j = sq.merge(t[["player_id", "to_club_id", "transfer_date"]], on="player_id", how="outer")
    later = j["transfer_date"].notna() & (j["date"].isna() | (j["transfer_date"] > j["date"]))
    j["club"] = np.where(later, j["to_club_id"], j["club_id"])
    recent = (j["date"] >= cut - pd.Timedelta(days=400)) | (j["transfer_date"] >= cut - pd.Timedelta(days=400))
    j = j[recent & j["club"].isin(clubs)]
    return j[["player_id", "club"]]


def ratings(clubs, params):
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    avail = pd.read_parquet(PROCESSED_DIR / "availability.parquet")
    v = war[war["season"] == SEASON - 1].drop_duplicates("player_id").set_index("player_id")["val_cal"]
    a = avail[avail["season"] == SEASON - 1].drop_duplicates("player_id").set_index("player_id")["proj_share"]
    sq = squads_sept(clubs)
    us = {t: u for u, t in us_to_tm().items()}
    sq["us"] = sq["player_id"].map(us)
    sq["val"], sq["share"] = sq["us"].map(v), sq["us"].map(a)
    sq = sq.dropna(subset=["val", "share"])
    out = {}
    for c in clubs:
        p = sq[sq["club"] == c]
        tot = p["share"].sum()
        scale = min(1.0, 11 / tot) if tot > 0 else 1.0
        out[c] = (p["val"] * p["share"] * scale).sum() + max(0.0, 11 - tot * scale) * params["newcomer_cal"]
    r = pd.Series(out)
    return r - r.mean()


def points_slope():
    """points per unit of strength x matches, fitted on the backtest (league means out), + the fitted line's RMSE"""
    d = pd.read_parquet(PROCESSED_DIR / "forecast_backtest.parquet")
    d = d[d["season"] < SEASON]          # only what was known before the season
    x = d["strength"] * d["matches"]
    xc = x - x.groupby([d["season"], d["league"]]).transform("mean")
    yc = d["pts"] - d.groupby(["season", "league"])["pts"].transform("mean")
    b = np.sum(xc * yc) / np.sum(xc ** 2)
    rmse = float(np.sqrt(np.mean((yc - b * xc) ** 2)))
    return b, rmse


def simulate(r: pd.Series, fx: pd.DataFrame, k: float, h: float, a: float, sd: float, n: int = N_SIM):
    """r = rating per club (npxG diff per match). returns points array (n, clubs) and the club order"""
    clubs = list(r.index)
    ix = {c: i for i, c in enumerate(clubs)}
    hi = fx["home_club_id"].map(ix).to_numpy()
    ai = fx["away_club_id"].map(ix).to_numpy()
    base = np.tile(r.to_numpy(), (n, 1)) + rng.normal(0, sd, (n, len(clubs))) if sd > 0 else np.tile(r.to_numpy(), (n, 1))
    diff = base[:, hi] - base[:, ai]
    gh = rng.poisson(h * np.exp(k * diff / 2))
    ga = rng.poisson(a * np.exp(-k * diff / 2))
    ph = np.where(gh > ga, 3, np.where(gh == ga, 1, 0))
    pa = np.where(ga > gh, 3, np.where(gh == ga, 1, 0))
    pts = np.zeros((n, len(clubs)))
    np.add.at(pts.T, hi, ph.T)
    np.add.at(pts.T, ai, pa.T)
    # goal difference for tie-breaks
    gd = np.zeros((n, len(clubs)))
    np.add.at(gd.T, hi, (gh - ga).T)
    np.add.at(gd.T, ai, (ga - gh).T)
    return pts, gd, clubs


def ranks(pts, gd):
    key = pts + gd / 1000 + rng.uniform(0, 1e-6, pts.shape)
    return (-key).argsort(axis=1).argsort(axis=1) + 1


def calibrate(r, fx, h, a, b):
    """k such that expected points rise b per unit of rating x 38 (no noise, so it's the pure mapping)"""
    best = None
    for k in np.linspace(0.6, 2.5, 20):
        pts, _, _ = simulate(r, fx, k, h, a, 0.0, n=2000)
        e = pts.mean(axis=0)
        x = r.to_numpy() * 38
        slope = np.sum((x - x.mean()) * (e - e.mean())) / np.sum((x - x.mean()) ** 2)
        if best is None or abs(slope - b) < abs(best[1] - b):
            best = (k, slope)
    return best


if __name__ == "__main__":
    params = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    fx, names = clubs_and_fixtures()
    clubs = sorted(set(fx["home_club_id"]))
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    m = m[(m["league"] == "EPL") & (m["season"] < SEASON)]
    h, a = m["goals_h"].mean(), m["goals_a"].mean()
    b, rmse = points_slope()
    r = ratings(clubs, params)
    k, got = calibrate(r, fx, h, a, b)
    # luck alone: spread of a club's points around its expectation with no rating noise
    p0, _, _ = simulate(r, fx, k, h, a, 0.0, n=4000)
    luck = float(p0.std(axis=0).mean())
    extra = np.sqrt(max(rmse ** 2 - luck ** 2, 0))
    sd = extra / (b * 38)
    print(f"goals: home {h:.2f}, away {a:.2f} (PL, seasons before). points per unit of rating x match: {b:.2f} "
          f"(backtest), k = {k:.2f} gives {got:.2f}")
    print(f"forecast RMSE {rmse:.1f} points = match luck {luck:.1f} + rating error {extra:.1f} -> rating sd {sd:.3f} "
          f"npxG per match")

    # united's windows: strength from the case's depth charts, as a difference to the real window
    dc = pd.read_parquet(PROCESSED_DIR / "united_case_depth.parquet")
    s_w = dc.assign(x=dc["val_cal"] * dc["minutes"] / 90 / 38).groupby("window")["x"].sum()
    windows = {"actual": 0.0}
    for w in ("standing still", "A", "B"):
        windows[w] = float(s_w[w] - s_w["actual"])
    print("united strength vs the real window (npxG difference per match): "
          + ", ".join(f"{w} {d:+.3f}" for w, d in windows.items()))

    real = pd.concat([fx.groupby("home_club_id").apply(lambda x: (3 * (x["home_club_goals"] > x["away_club_goals"])
                                                                   + (x["home_club_goals"] == x["away_club_goals"])).sum(),
                                                       include_groups=False),
                      fx.groupby("away_club_id").apply(lambda x: (3 * (x["away_club_goals"] > x["home_club_goals"])
                                                                   + (x["home_club_goals"] == x["away_club_goals"])).sum(),
                                                       include_groups=False)], axis=1).sum(axis=1)
    rows = []
    for w, dlt in windows.items():
        rr = r.copy()
        rr[UNITED] += dlt
        rr = rr - rr.mean()
        pts, gd, order = simulate(rr, fx, k, h, a, sd)
        rk = ranks(pts, gd)
        for j, c in enumerate(order):
            rows.append({"window": w, "club": names.get(c, c), "club_id": c, "rating": rr[c],
                         "exp_pts": pts[:, j].mean(), "p10": np.percentile(pts[:, j], 10),
                         "p90": np.percentile(pts[:, j], 90), "title": (rk[:, j] == 1).mean(),
                         "top4": (rk[:, j] <= 4).mean(), "top5": (rk[:, j] <= 5).mean(),
                         "releg": (rk[:, j] >= 18).mean(), "exp_rank": rk[:, j].mean(), "real_pts": real.get(c)})
    out = pd.DataFrame(rows)
    out.to_parquet(PROCESSED_DIR / "season_sim.parquet", index=False)

    act = out[out["window"] == "actual"].sort_values("exp_pts", ascending=False)
    print(f"\n{SEASON}-{str(SEASON + 1)[-2:]} as forecast on 1 Sept {SEASON} (real squads), {N_SIM:,} seasons:")
    show = act[["club", "rating", "exp_pts", "p10", "p90", "title", "top4", "top5", "releg", "real_pts"]].copy()
    for c in ("title", "top4", "top5", "releg"):
        show[c] = (100 * show[c]).round(1)
    print(show.round(2).to_string(index=False))
    e = act["exp_pts"] - act["real_pts"]
    print(f"  vs the real table: MAE {e.abs().mean():.1f} points, r {np.corrcoef(act['exp_pts'], act['real_pts'])[0, 1]:.2f}")

    u = out[out["club_id"] == UNITED].set_index("window").loc[["standing still", "actual", "A", "B"]]
    print("\nmanchester united by window (%):")
    print(u[["exp_pts", "p10", "p90", "exp_rank", "title", "top4", "top5", "releg"]].assign(
        title=lambda x: 100 * x["title"], top4=lambda x: 100 * x["top4"], top5=lambda x: 100 * x["top5"],
        releg=lambda x: 100 * x["releg"]).round(1).to_string())
    print(f"  real {SEASON}-{str(SEASON + 1)[-2:]}: {int(u['real_pts'].iloc[0])} points")

    # chart: top = pre-season odds per window (simulated points), bottom = hindsight (real ratings that season)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    cols = {"standing still": "#8a8984", "actual": "#c0392b", "A": "#2f6db3", "B": "#e08a1e"}
    # names on the chart, the report calls the model's two windows scenario A and B
    nm = {"standing still": "standing still", "actual": "real window", "A": "scenario A, best value",
          "B": "scenario B, most points"}
    order_w = ("standing still", "actual", "A", "B")
    lab = f"{SEASON}-{str(SEASON + 1)[-2:]}"
    real_u = int(u["real_pts"].iloc[0])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 5.6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    fig.patch.set_facecolor(surface)
    for x_ in (ax, bx):
        x_.set_facecolor(surface)
        for sp in ("top", "right", "left"):
            x_.spines[sp].set_visible(False)
        x_.spines["bottom"].set_color(grid)
        x_.tick_params(colors=muted)
        x_.axvline(real_u, color=ink, lw=1, ls="--")
    for w in order_w:
        rr = r.copy()
        rr[UNITED] += windows[w]
        rr = rr - rr.mean()
        pts, _, order = simulate(rr, fx, k, h, a, sd, n=8000)
        x = pts[:, order.index(UNITED)]
        # smoothed (kernel density) instead of a step histogram, points are whole numbers so bins looked jagged
        grid_x = np.linspace(10, 110, 400)
        y = gaussian_kde(x, bw_method=0.25)(grid_x)
        ax.plot(grid_x, y, lw=2, color=cols[w], label=f"{nm[w]} ({x.mean():.0f} pts)")
        ax.fill_between(grid_x, y, color=cols[w], alpha=0.08)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.text(real_u + 0.7, ax.get_ylim()[1] * 0.95, f"real {lab}: {real_u}", color=ink, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"Pre-season odds on 1 Sept {SEASON}: United's {lab} points under each window",
                 loc="left", color=ink, fontsize=10.5)
    # hindsight rows (united_case.py): range = planned minutes vs minutes capped at what he really played
    hj = json.loads((PROCESSED_DIR / "united_case.json").read_text()).get("hindsight", {})
    for i, w in enumerate(order_w):
        if w not in hj:
            continue
        lo, hi = hj[w]["hindsight_pts_low"], hj[w]["hindsight_pts_high"]
        yy = len(order_w) - 1 - i
        bx.plot([lo, hi], [yy, yy], color=cols[w], lw=7, solid_capstyle="round", zorder=3)
        bx.scatter([lo, hi], [yy, yy], color=cols[w], s=45, zorder=4)     # so a zero-width range still shows
        bx.text(max(hi, lo) + 1.2, yy, f"{nm[w]}  {lo:.0f}-{hi:.0f}" if round(lo) != round(hi) else f"{nm[w]}  {lo:.0f}",
                va="center", color=cols[w], fontsize=8)
    bx.set_yticks([])
    bx.set_ylim(-0.7, len(order_w) - 0.3)
    bx.set_title(f"Hindsight: the same windows with every player's real {lab} value", loc="left", color=ink, fontsize=10.5)
    bx.set_xlabel(f"United league points, {lab}", color=muted)
    bx.set_xlim(15, 105)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "season_sim.png", facecolor=surface)
    print(f"\nsaved -> {PROCESSED_DIR / 'season_sim.parquet'}, figures/season_sim.png")
