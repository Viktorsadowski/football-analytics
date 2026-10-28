#!/usr/bin/env python3
"""
Step 7: what a point is worth, two prices (same idea as the two prices in the nba-moneyball case).

1. What a point EARNS, Premier League only (the United case is there)
   money by finishing position, in euros:
     merit payments     GBP 3.1m per place (2023-24, UK + international merit, Planet Football)
     Europe next season 1-4 Champions League, 5-6 Europa League, 7 Conference League
                        prize money only, English clubs 2024-25: CL GBP 65-97m (75m used), EL ~15m, ECL ~10m
                        (Planet Football / UEFA). matchday + commercial uplift from the CL is real but an
                        estimate, so it's a sensitivity (CL_UPLIFT), not the base
     relegation         18-20 lose ~GBP 110m: ~92m a year gap to the Championship, minus parachutes 49/40/17m,
                        and a fair chance of coming straight back up (The ESK)
   points -> position: every EPL season 2014-15 to 2023-24, a team with X points slotted into that table,
   averaged over the 10 seasons. then smoothed with the uncertainty of a projection (sd PTS_SD), since a
   club buys expected points, not points. euro per point = slope of that curve at the team's level.

2. What the market PAYS per point, all five leagues
   transfermarkt value at the start of s+1 against projected PAR for s+1 (war.py), with age and position,
   per league. the market value is the price of the whole player (contract years, resale), PAR is one season,
   so this is "euros per projected point this season" as the market prices it. market_value.py does the
   multi-season comparison per player.

Output
  data/processed/eur_per_point.parquet    EPL: points, expected money, euro per extra point (base + uplift)
  data/processed/market_price.json        market fit per league
  figures/eur_per_point.png

  python src/price.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ROOT, SEASONS
from tm_match import start_values

GBP_EUR = 1.18             # 2024 average
MERIT_PER_PLACE = 3.1      # GBP m
EUROPE = {"CL": 75.0, "EL": 15.0, "ECL": 10.0}   # GBP m, prize money
CL_UPLIFT = 30.0           # GBP m on top for CL, matchday + commercial, sensitivity only
RELEGATION = 110.0         # GBP m
PTS_SD = 9.0               # sd of final points around a projection (war.py team check: MAE 8)


def money_by_position(uplift: bool = False) -> pd.Series:
    """euros (m) per finishing position, relative to 20th place without relegation"""
    pos = np.arange(1, 21)
    m = (20 - pos) * MERIT_PER_PLACE
    cl = EUROPE["CL"] + (CL_UPLIFT if uplift else 0)
    m = m + np.select([pos <= 4, pos <= 6, pos == 7], [cl, EUROPE["EL"], EUROPE["ECL"]], 0)
    m = m - np.where(pos >= 18, RELEGATION, 0)
    return pd.Series(m * GBP_EUR, index=pos)


def epl_tables() -> list:
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    m = m[m["league"] == "EPL"]
    pts_h = np.select([m["goals_h"] > m["goals_a"], m["goals_h"] == m["goals_a"]], [3, 1], 0)
    pts_a = np.select([m["goals_a"] > m["goals_h"], m["goals_h"] == m["goals_a"]], [3, 1], 0)
    t = pd.concat([pd.DataFrame({"season": m["season"], "team": m["home_team"], "pts": pts_h}),
                   pd.DataFrame({"season": m["season"], "team": m["away_team"], "pts": pts_a})])
    t = t.groupby(["season", "team"])["pts"].sum().reset_index()
    return [g["pts"].to_numpy() for _, g in t.groupby("season")]


def money_curve(tables: list, money: pd.Series) -> pd.DataFrame:
    """expected money for a team finishing on X points, slotted into each past table in turn.
    he replaces the team closest to him on points (so there are still 20), ties split half above half below"""
    xs = np.arange(10, 101)
    out = []
    for x in xs:
        vals = []
        for pts in tables:
            others = np.delete(pts, np.argmin(np.abs(pts - x)))
            above = (others > x).sum() + 0.5 * (others == x).sum()
            p = int(np.clip(np.round(above + 1), 1, 20))
            vals.append(money[p])
        out.append(np.mean(vals))
    return pd.DataFrame({"pts": xs, "money": out})


def smooth_expected(curve: pd.DataFrame) -> pd.DataFrame:
    """expected money for a projection of X points (final points ~ Normal(X, PTS_SD)), and its slope"""
    xs = curve["pts"].to_numpy()
    w = np.exp(-0.5 * ((xs[None, :] - xs[:, None]) / PTS_SD) ** 2)
    w /= w.sum(1, keepdims=True)
    e = w @ curve["money"].to_numpy()
    return pd.DataFrame({"pts": xs, "exp_money": e, "eur_per_pt": np.gradient(e, xs)})


def market_fit() -> dict:
    """log market value ~ PAR, PAR^2, age, age^2, position group, per league. returns the implied euros per
    projected point at a typical regular (PAR = median of 1500+ projected minutes)"""
    from sklearn.linear_model import LinearRegression
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    rows = []
    for s in list(SEASONS)[:-1]:
        mv = start_values(s + 1)
        d = war[war["season"] == s].copy()
        d["mv"] = d["player_id"].map(mv)
        rows.append(d)
    d = pd.concat(rows)
    d = d[(d["mv"] > 0) & d["age"].notna() & (d["proj_min"] >= 900)]
    out = {}
    for lg, g in d.groupby("league"):
        X = pd.DataFrame({"par": g["par"], "par2": g["par"] ** 2, "age": g["age"], "age2": g["age"] ** 2,
                          "season": g["season"]})
        for p in ("GK", "DEF", "MID", "ATT"):
            X[f"g_{p}"] = (g["grp"] == p).astype(float)
        y = np.log(g["mv"])
        m = LinearRegression().fit(X, y)
        r2 = m.score(X, y)
        # euros per point at a typical regular: d(value)/d(par) = value * d(log value)/d(par)
        reg = g[g["proj_min"] >= 1500]
        par0 = reg["par"].median()
        mv0 = reg["mv"].median()
        slope = m.coef_[0] + 2 * m.coef_[1] * par0
        out[lg] = {"r2": round(r2, 3), "n": int(len(g)), "par_median": round(par0, 2),
                   "mv_median_m": round(mv0 / 1e6, 1), "eur_per_par_m": round(mv0 * slope / 1e6, 2),
                   "pct_value_per_par": round(100 * slope, 1),
                   "corr_par_logmv": round(np.corrcoef(g["par"], y)[0, 1], 2)}
    return out


def plot(base: pd.DataFrame, up: pd.DataFrame, raw: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted)
        ax.set_xlabel("Projected points", color=muted)
    a = axes[0]
    a.plot(raw["pts"], raw["money"], color=muted, lw=1, ls=":", label="Finishing on exactly X points")
    a.plot(base["pts"], base["exp_money"], color="#2a78d6", lw=2.2, label="Projected X points, prize money")
    a.plot(up["pts"], up["exp_money"], color="#eb6834", lw=1.6, ls="--", label="With CL matchday/commercial uplift")
    a.set_ylabel("Money vs 20th place (EUR m)", color=muted)
    a.set_title("What a Premier League season earns", loc="left", color=ink, fontsize=11)
    a.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    b = axes[1]
    b.plot(base["pts"], base["eur_per_pt"], color="#2a78d6", lw=2.2, label="Prize money")
    b.plot(up["pts"], up["eur_per_pt"], color="#eb6834", lw=1.6, ls="--", label="With CL uplift")
    b.set_ylabel("EUR m per extra projected point", color=muted)
    b.set_title("What the next point is worth", loc="left", color=ink, fontsize=11)
    b.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper right")
    for ax in axes:
        ax.set_xlim(20, 95)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "eur_per_point.png", facecolor=surface)


if __name__ == "__main__":
    tables = epl_tables()
    th = pd.DataFrame([np.sort(t)[::-1][[0, 3, 4, 6, 16, 17]] for t in tables],
                      columns=["1st", "4th", "5th", "7th", "17th", "18th"])
    print("EPL points needed, 2014-15 to 2023-24 (mean / min / max):")
    print("  " + "   ".join(f"{c} {th[c].mean():.0f} ({th[c].min()}-{th[c].max()})" for c in th.columns))

    raw = money_curve(tables, money_by_position())
    base = smooth_expected(raw)
    up = smooth_expected(money_curve(tables, money_by_position(uplift=True)))
    out = base.merge(up.rename(columns={"exp_money": "exp_money_uplift", "eur_per_pt": "eur_per_pt_uplift"}), on="pts")
    out.to_parquet(PROCESSED_DIR / "eur_per_point.parquet", index=False)

    print("\neuros per extra projected point (EUR m), prize money / with CL uplift:")
    for p in (30, 40, 50, 55, 60, 65, 70, 75, 80, 90):
        r = out[out["pts"] == p].iloc[0]
        print(f"  at {p} points: {r['eur_per_pt']:5.2f} / {r['eur_per_pt_uplift']:5.2f}")
    mid = out[out["pts"].between(35, 85)]
    print(f"  average over 35-85 points: {mid['eur_per_pt'].mean():.2f} / {mid['eur_per_pt_uplift'].mean():.2f}")

    mk = market_fit()
    print("\nwhat the market pays (transfermarkt value at start of s+1 vs projected PAR, 900+ projected min):")
    for lg, r in mk.items():
        print(f"  {lg:<10} r(PAR, log value) {r['corr_par_logmv']:.2f}, R2 {r['r2']:.2f}: a typical regular "
              f"(PAR {r['par_median']}, value EUR {r['mv_median_m']}m) costs +{r['pct_value_per_par']}% "
              f"= EUR {r['eur_per_par_m']}m per extra projected point  (n {r['n']:,})")
    (PROCESSED_DIR / "market_price.json").write_text(json.dumps(mk, indent=1))
    plot(base, up, raw)
    print(f"\nsaved -> {PROCESSED_DIR / 'eur_per_point.parquet'}, market_price.json, figures/eur_per_point.png")
