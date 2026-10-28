#!/usr/bin/env python3
"""
Step 6: points above replacement (PAR), projected one season ahead. Football version of war.py in
nba-moneyball, in league points instead of wins.

  PAR = (value - replacement) * projected minutes / 90 * points per npxG

  value            as of the end of season s, for season s+1:
                     priored RAPM (O + D, npxG per 90)
                   + keeper shot-stopping from gk.py (npxG faced - non-pen goals conceded per 90, shrunk)
                   + aging to s+1 with his position group's curve (aging.py)
                   + league strength (league_strength.py, if it's been run): his league's offset / 11, so all five
                     leagues are on the Premier League's scale. replacement level is then one pool too
                   then calibrated on team results: the ridge shrinks everything towards 0, so it's stretched
                   by however much it takes to predict next season's team npxG differences. only the
                   RAPM + aging part, shot-stopping doesn't move npxG so it isn't stretched
  replacement      what a team gets for next to nothing: minute-weighted value of players who were valued in
                   the bottom 20% of their league (transfermarkt, start of s+1) and then played s+1.
                   per position group, keepers are on their own scale (offense pinned, shot-stopping added)
  minutes          projected share of his team's minutes (availability.py) x a full next season.
                   without availability.parquet: 50/30/20 blend of his last 3 seasons' shares
  healthy          with injury.py's risk: par_healthy = same share of minutes when fit, no injury games,
                   inj_cost = par_healthy - par = points his injury risk is expected to cost
  points per npxG  fitted on team-seasons: points per match ~ npxG difference per match

Output
  data/processed/war.parquet        one row per player per "as of" season s, projecting s+1
  data/processed/war_params.json    calibration, replacement levels, points per npxG

  python src/war.py
"""

import json

import numpy as np
import pandas as pd

from aging import BROAD, LAG
from box_prior import POS
from config import PROCESSED_DIR, SEASONS, season_label
from rapm import keepers
from tm_match import birth_dates, start_values

WEIGHTS = (0.5, 0.3, 0.2)
CHEAP_Q = 0.20             # bottom 20% of market values = "next to nothing"


# ── basics ────────────────────────────────────────────────────────────────────

def lineups_season() -> pd.DataFrame:
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet")
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season", "league"])
    return lu.merge(m, on="match_id")


def team_seasons() -> pd.DataFrame:
    """points, matches and npxG difference per team-season"""
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    sh = pd.read_parquet(PROCESSED_DIR / "shots.parquet")
    npx = sh[sh["is_pen"] == 0].groupby(["match_id", "h_a"])["xg"].sum().unstack(fill_value=0)
    m = m.join(npx, on="match_id").fillna({"h": 0, "a": 0})
    pts_h = np.select([m["goals_h"] > m["goals_a"], m["goals_h"] == m["goals_a"]], [3, 1], 0)
    pts_a = np.select([m["goals_a"] > m["goals_h"], m["goals_h"] == m["goals_a"]], [3, 1], 0)
    t = pd.concat([pd.DataFrame({"season": m["season"], "league": m["league"], "team_id": m["home_id"],
                                 "pts": pts_h, "npxgd": m["h"] - m["a"]}),
                   pd.DataFrame({"season": m["season"], "league": m["league"], "team_id": m["away_id"],
                                 "pts": pts_a, "npxgd": m["a"] - m["h"]})])
    return t.groupby(["season", "league", "team_id"]).agg(matches=("pts", "size"), pts=("pts", "sum"),
                                                          npxgd=("npxgd", "mean")).reset_index()


def points_per_npxg(ts: pd.DataFrame) -> float:
    # points per match = c + b * npxG difference per match, team-seasons weighted by matches
    return np.polyfit(ts["npxgd"], ts["pts"] / ts["matches"], 1, w=ts["matches"])[0]


# ── aging ─────────────────────────────────────────────────────────────────────

def age_step(curve: pd.DataFrame, grp: pd.Series, age_from: pd.Series, age_to: pd.Series) -> np.ndarray:
    """curve value at age_to minus at age_from, linear between the integer ages"""
    out = np.zeros(len(grp))
    for g in grp.dropna().unique():
        c = curve[curve["group"] == g].sort_values("age")
        idx = (grp == g).to_numpy()
        f = np.interp(age_from[idx], c["age"], c["curve"])
        t = np.interp(age_to[idx], c["age"], c["curve"])
        out[idx] = np.nan_to_num(t - f)
    return out


# ── minutes ───────────────────────────────────────────────────────────────────

def minute_shares(lu: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """share of his team's available minutes per season (all teams he played for that season)"""
    p = lu.groupby(["season", "player_id", "team_id"])["minutes"].sum().reset_index()
    p = p.merge(ts[["season", "team_id", "matches"]], on=["season", "team_id"])
    p["avail"] = p["matches"] * 90
    return p.groupby(["season", "player_id"]).agg(minutes=("minutes", "sum"), avail=("avail", "sum")).reset_index() \
        .assign(share=lambda d: d["minutes"] / d["avail"])


def projected_share(sh: pd.DataFrame, as_of: int) -> pd.Series:
    # 50/30/20 over the last 3 seasons. a season he wasn't in the data counts as 0 if he was there before,
    # otherwise the weights are renormalised over the seasons he was around for
    first = sh.groupby("player_id")["season"].min()
    parts = []
    for lag, w in enumerate(WEIGHTS):
        s = sh[sh["season"] == as_of - lag].set_index("player_id")["share"]
        parts.append((lag, w, s))
    ids = parts[0][2].index
    num, den = pd.Series(0.0, index=ids), pd.Series(0.0, index=ids)
    for lag, w, s in parts:
        here = first.reindex(ids) <= as_of - lag
        num += w * s.reindex(ids).fillna(0) * here
        den += w * here
    return num / den


# ── shrink ────────────────────────────────────────────────────────────────────

def shrink(v: pd.DataFrame):
    """val_cal pulled towards its league-season-position mean by how far it actually carries into next season:
    val = mean + b(m) x (val - mean), b(m) = c x m / (m + k), m = his minutes over s-2..s. c and k fitted by least
    squares on deviation as of s vs deviation as of s+1 (players with 900+ minutes in s+1), only pairs that end by
    the as-of season, so it's out of sample. the first season has no pairs and borrows the second's fit.
    first version had 5 minutes bands, but the lowest band (0-1500, b 0.59) still let 337-minute players through
    as top targets in united_case.py. the curve goes towards 0 for them.
    the values pool seasons (rapm window), so the two deviations share data and b is if anything too high"""
    from scipy.optimize import least_squares
    d = v[["season", "player_id", "league", "grp", "minutes", "val_cal"]].copy()
    w = d["minutes"].clip(lower=1)
    d["center"] = (d["val_cal"] * w).groupby([d["season"], d["league"], d["grp"]]).transform("sum") / \
        w.groupby([d["season"], d["league"], d["grp"]]).transform("sum")
    d["dev"] = d["val_cal"] - d["center"]
    mins = d.groupby(["player_id", "season"])["minutes"].sum().unstack(fill_value=0)
    d["m3"] = [mins.loc[p, [c for c in mins.columns if s - 2 <= c <= s]].sum() for p, s in zip(d["player_id"], d["season"])]
    nxt = d[["season", "player_id", "dev", "minutes"]].assign(season=d["season"] - 1) \
        .groupby(["season", "player_id"]).agg(dev1=("dev", "mean"), min1=("minutes", "sum")).reset_index()
    pairs = d.merge(nxt, on=["season", "player_id"])
    pairs = pairs[pairs["min1"] >= 900]
    fits, out = {}, d["val_cal"].copy()
    for s in SEASONS:
        pr = pairs[pairs["season"] + 1 <= max(s, min(SEASONS) + 1)]
        m, x, y = pr["m3"].to_numpy(), pr["dev"].to_numpy(), pr["dev1"].to_numpy()
        r = least_squares(lambda p: p[0] * m / (m + p[1]) * x - y, [0.9, 500.0], bounds=([0, 0], [1, 1e5]))
        c, k = r.x
        fits[s] = {"c": c, "k": k}
        idx = d["season"] == s
        out[idx] = d.loc[idx, "center"] + c * d.loc[idx, "m3"] / (d.loc[idx, "m3"] + k) * d.loc[idx, "dev"]
    return out, fits


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lu = lineups_season()
    ts = team_seasons()
    ppx = points_per_npxg(ts)
    print(f"points per npxG: {ppx:.3f} per match (+1 npxG difference per match = {ppx * 38:.0f} points "
          f"over 38 games, fitted on {len(ts)} team-seasons)")

    gks = pd.read_parquet(PROCESSED_DIR / "gk_saves.parquet")   # gk.py

    # ── values as of each season ──
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    curve = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    dob = birth_dates()
    v = rapm[["season", "player_id", "name", "team", "league", "position", "minutes", "o_rapm", "d_rapm",
              "rapm"]].copy()
    v = v.merge(gks[["season", "player_id", "save90"]], on=["season", "player_id"], how="left")
    v["save90"] = v["save90"].fillna(0)
    # usual group: his position that season, else his most common one
    v["grp"] = v["position"].map(POS).map(BROAD)
    v["grp"] = v["grp"].fillna(v.groupby("player_id")["grp"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else None)).fillna("MID")
    v.loc[v["player_id"].isin(keepers()), "grp"] = "GK"
    mid = pd.to_datetime((v["season"] + 1).astype(str) + "-01-01")
    v["age"] = (mid - v["player_id"].map(dob)).dt.days / 365.25
    # the rating describes him ~0.57 years ago (pooled window), the projection is for a full year later
    step = age_step(curve, v["grp"], v["age"] - LAG, v["age"] + 1)
    v["age_adj"] = np.where(v["age"].notna(), step, 0.0)
    # xG part (what the calibration below is fitted on) and the shot-stopping part kept apart
    v["val_xg"] = v["rapm"] + v["age_adj"]
    v["val"] = v["val_xg"] + v["save90"]

    # ── calibration on team results ──
    # season t-1 values x season t minutes -> team npxG difference per match in t, regress the real one on it.
    # out of sample (t-1 values predicting t). newcomers to the big five get what newcomers actually posted
    sh = minute_shares(lu, ts)
    first = rapm.sort_values("season").drop_duplicates("player_id")
    first = first[first["season"] > min(SEASONS)]
    newcomer = np.average(first["rapm"], weights=first["minutes"])
    rows = []
    for t in list(SEASONS)[1:]:
        tp = lu[lu["season"] == t].groupby(["team_id", "player_id"])["minutes"].sum().reset_index()
        prev = v[v["season"] == t - 1].set_index("player_id")["val_xg"]
        tp["val"] = tp["player_id"].map(prev).fillna(newcomer)
        tp["new"] = ~tp["player_id"].isin(prev.index)
        g = tp.assign(x=tp["val"] * tp["minutes"] / 90).groupby("team_id")
        real = ts[ts["season"] == t].set_index("team_id")
        rows.append(pd.DataFrame({"season": t, "pred": g["x"].sum() / real["matches"],
                                  "new_share": g.apply(lambda d: np.average(d["new"], weights=d["minutes"]),
                                                       include_groups=False),
                                  "real": real["npxgd"], "pts": real["pts"], "matches": real["matches"],
                                  "league": real["league"]}))
    chk = pd.concat(rows).dropna()
    # league mean taken out on both sides, the values are relative to the league already
    for c in ("pred", "real"):
        chk[c + "_c"] = chk[c] - chk.groupby(["season", "league"])[c].transform("mean")
    a = np.sum(chk["pred_c"] * chk["real_c"]) / np.sum(chk["pred_c"] ** 2)
    r = np.corrcoef(chk["pred_c"], chk["real_c"])[0, 1]
    print(f"calibration: val_cal = {a:.2f} * val   (team npxG difference r = {r:.2f}, {len(chk)} team-seasons, "
          f"newcomers at {newcomer:+.3f})")
    # saves don't move npxG, so they don't get the stretch, they're already in goals per 90
    v["val_cal"] = a * v["val_xg"] + v["save90"]
    # league strength (league_strength.py): values so far are vs the player's own league, this puts all five on
    # the Premier League's scale. after the calibration on purpose, that one is fitted within leagues
    lpath = PROCESSED_DIR / "league_strength.parquet"
    if lpath.exists():
        ls = pd.read_parquet(lpath).set_index(["season", "league"])["per_player"]
        v["league_adj"] = [ls.get((s, lg), 0.0) for s, lg in zip(v["season"], v["league"])]
        v["val_cal"] += v["league_adj"]
        print("league strength: on (league_strength.py), per player " + ", ".join(
            f"{lg} {ls.get((max(SEASONS), lg), 0):+.3f}" for lg in ("EPL", "La liga", "Bundesliga", "Serie A", "Ligue 1")))
    else:
        v["league_adj"] = 0.0
    newcomer_cal = a * newcomer
    # shrink: values are too spread out for individual players (next season's value moves only ~65-90% as far
    # from average as this one says, less the fewer minutes behind it). found in united_case.py, where picking
    # the best of 100+ targets picked the noisiest ones. see shrink()
    v["val_cal_raw"] = v["val_cal"]
    v["val_cal"], slopes = shrink(v)
    f = slopes[max(SEASONS)]
    print(f"shrink to league/position mean: b = {f['c']:.2f} x m / (m + {f['k']:.0f}), m = minutes over 3 seasons "
          f"(latest as-of). 500 min -> {f['c'] * 500 / (500 + f['k']):.2f}, 3000 -> {f['c'] * 3000 / (3000 + f['k']):.2f}, "
          f"8000 -> {f['c'] * 8000 / (8000 + f['k']):.2f}")

    pts_pred = chk.groupby(["season", "league"])["pts"].transform("mean") + a * chk["pred_c"] * ppx * chk["matches"]
    print(f"team points check (last season's values, this season's minutes): MAE "
          f"{(pts_pred - chk['pts']).abs().mean():.1f} points, r = {np.corrcoef(pts_pred, chk['pts'])[0, 1]:.2f}")

    # ── replacement level ──
    # value as of s for players who were cheap at the start of s+1 and then played s+1
    pool = []
    for s in list(SEASONS)[:-1]:
        mv = start_values(s + 1)
        nxt = lu[lu["season"] == s + 1].groupby(["player_id", "league"])["minutes"].sum().reset_index() \
            .sort_values("minutes").drop_duplicates("player_id", keep="last")
        nxt["mv"] = nxt["player_id"].map(mv)
        nxt = nxt.dropna(subset=["mv"])
        cut = nxt.groupby("league")["mv"].transform(lambda x: x.quantile(CHEAP_Q))
        cheap = nxt[nxt["mv"] <= cut]
        p = cheap.merge(v[v["season"] == s][["player_id", "grp", "val_cal"]], on="player_id")
        p["season"] = s
        pool.append(p)
    pool = pd.concat(pool, ignore_index=True)
    repl = pool.groupby("grp").apply(lambda g: np.average(g["val_cal"], weights=g["minutes"]), include_groups=False)
    print("replacement level (npxG per 90, calibrated), from players in the bottom 20% of market values:")
    print("  " + "   ".join(f"{g} {repl[g]:+.3f} (n {int((pool['grp'] == g).sum())})" for g in repl.index))

    # ── projections, as of the end of each season, for the next one ──
    apath = PROCESSED_DIR / "availability.parquet"
    avail = pd.read_parquet(apath) if apath.exists() else None
    rpath = PROCESSED_DIR / "injury_risk.parquet"
    risk = pd.read_parquet(rpath) if rpath.exists() else None
    print("\nminutes from:", "availability model (availability.py)" if avail is not None
          else "50/30/20 blend of past shares (run availability.py)")
    out = []
    for s in SEASONS:
        d = v[v["season"] == s].copy()
        if avail is not None:
            d["proj_share"] = d["player_id"].map(avail[avail["season"] == s].set_index("player_id")["proj_share"])
        else:
            d["proj_share"] = d["player_id"].map(projected_share(sh, s))
        lg_games = ts[ts["season"] == s].groupby("league")["matches"].median()
        # next season's length per league: 38, Bundesliga 34, Ligue 1 34 from 2023-24
        full = d["league"].map(lg_games).fillna(38)
        d["proj_min"] = d["proj_share"] * full * 90
        d["repl"] = d["grp"].map(repl)
        d["par"] = (d["val_cal"] - d["repl"]) * d["proj_min"] / 90 * ppx
        d["par90"] = (d["val_cal"] - d["repl"]) * ppx
        # healthy version, like war_healthy in nba-moneyball: same share of minutes when fit, no injury games.
        # share when fit = projected share / (1 - expected injured share), capped at 1
        if risk is not None:
            r = risk[risk["season"] == s].set_index("player_id")["exp_inj_share"]
            d["exp_inj_share"] = d["player_id"].map(r).fillna(0)
            d["healthy_min"] = (d["proj_share"] / (1 - d["exp_inj_share"]).clip(lower=0.2)).clip(upper=1) * full * 90
            d["par_healthy"] = (d["val_cal"] - d["repl"]) * d["healthy_min"] / 90 * ppx
            d["inj_cost"] = d["par_healthy"] - d["par"]
        out.append(d)
    war = pd.concat(out, ignore_index=True)
    war.to_parquet(PROCESSED_DIR / "war.parquet", index=False)
    params = dict(a=a, ppx=ppx, newcomer=newcomer, newcomer_cal=newcomer_cal, repl=repl.to_dict(),
                  cheap_q=CHEAP_Q)
    (PROCESSED_DIR / "war_params.json").write_text(json.dumps(params, indent=1))

    # a whole league's worth of players above replacement, per team: sanity check on the level
    last = war[war["season"] == max(SEASONS)]
    per_team = last.groupby("league")["par"].sum() / last.groupby("league")["team"].nunique()
    print("\nPAR per team (sum over all players, projected), by league: "
          + ", ".join(f"{k} {v:.0f}" for k, v in per_team.items()))
    show = last.sort_values("par", ascending=False)
    cols = ["name", "team", "grp", "age", "val_cal", "save90", "proj_min", "par"]
    if "inj_cost" in show:
        cols += ["par_healthy", "inj_cost"]
    print(f"\nprojected {season_label(max(SEASONS) + 1)} points above replacement, top 20:")
    print(show[cols].head(20).round(2).to_string(index=False))
    if "inj_cost" in show:
        top = show[show["par_healthy"] >= 2].assign(pct=lambda x: 100 * x["inj_cost"] / x["par_healthy"])
        print(f"\nbiggest share of value lost to injury risk (2+ PAR healthy):")
        print(top.sort_values("pct", ascending=False)[cols + ["exp_inj_share", "pct"]].head(10).round(2)
              .to_string(index=False))
    gkshow = last[last["grp"] == "GK"].sort_values("par", ascending=False)
    print("\nkeepers, top 8:")
    print(gkshow[["name", "team", "age", "rapm", "save90", "val_cal", "proj_min", "par"]].head(8).round(3)
          .to_string(index=False))
    print(f"\nsaved -> {PROCESSED_DIR / 'war.parquet'}, war_params.json")
