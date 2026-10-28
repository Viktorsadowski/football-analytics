#!/usr/bin/env python3
"""
Step 9c: league strength from European games. Every rating so far is relative to the player's own league
(RAPM is fitted per league, see rapm.py), so a Ligue 1 player and a Premier League player with the same
value aren't the same player. the only direct link between the leagues is when their clubs play each other.

Model, every CL / EL / Conference League game (qualifiers too) between two big five clubs, 2015-16 on:
  goal difference = home + beta * (S_home - S_away) + (L_home - L_away)
    S       squad strength vs its own league's average (forecast_backtest.our_strength: squad on 1 September,
            projected minutes, values as of the end of the season before). npxG per match
    L       league offset = how much better the average team in that league is than the average Premier
            League team. EPL = 0. pooled over seasons with a straight-line drift per league
            (L = a + b * (season - 2019)), ~100 games a season is too few for five free offsets every year
    beta    fitted too, should be ~1 if the values are on the right scale
  finals are on neutral ground, no home term there
Per player the offset is L / 11 (a team is 11 players' worth of value, see the backtest).

Second version, the one that's used: L = scale x (league mean of elo.py's ratings on 1 August - the EPL's).
one fitted parameter instead of eight, and the elo already holds years of European games. it won the held-out
check below (6 of 9 seasons vs 4 of 9 for the free offsets), so per_player in the output is from the elo version.

Checks, pass marks set before running:
  1. held-out seasons   offsets fitted without season t predict t's European games better (lower MSE on goal
                        difference) than no offsets, in most seasons and overall
  2. movers             players who switch leagues: change in value from s to s+2 (s+2's window has mostly new-
                        league seasons) against the change the offsets predict, age taken out. slope > 0 and
                        roughly in line, an independent check (no European games in it)
  3. elo                league averages of elo.py's ratings in the same order

Output: data/processed/league_strength.parquet  season, league, offset + per_player (elo version, used),
        offset_free (the 8-parameter version, for the report)

  python src/league_strength.py
"""

import json

import numpy as np
import pandas as pd

import forecast_backtest as fb
from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from tm_match import LEAGUE_TM, match_clubs

TM = RAW_DIR / "transfermarkt"
EURO = ["CL", "EL", "UCOL", "CLQ", "ELQ", "ECLQ"]
LEAGUES = ["EPL", "La liga", "Bundesliga", "Serie A", "Ligue 1"]
MID = 2019
RIDGE = 1.0   # light pull of the drift terms towards 0


def design(d: pd.DataFrame):
    """columns: home, beta, then a_l and b_l for the 4 non-EPL leagues"""
    others = LEAGUES[1:]
    X = np.zeros((len(d), 2 + 2 * len(others)))
    X[:, 0] = d["home"]
    X[:, 1] = d["s_h"] - d["s_a"]
    for i, lg in enumerate(others):
        h, a = (d["lg_h"] == lg).astype(float), (d["lg_a"] == lg).astype(float)
        X[:, 2 + i] = h - a
        X[:, 2 + len(others) + i] = (h - a) * (d["season"] - MID)
    return X


def fit(d: pd.DataFrame, offsets: bool = True) -> np.ndarray:
    X = design(d)
    if not offsets:
        X = X[:, :2]
    y = d["gd"].to_numpy()
    pen = np.zeros(X.shape[1])
    if offsets:
        pen[2 + 4:] = RIDGE * len(d) / 100
    return np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)


def offsets_from(b: np.ndarray, season: int) -> dict:
    out = {"EPL": 0.0}
    for i, lg in enumerate(LEAGUES[1:]):
        out[lg] = b[2 + i] + b[2 + 4 + i] * (season - MID)
    return out


if __name__ == "__main__":
    act = fb.actual()
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    avail = pd.read_parquet(PROCESSED_DIR / "availability.parquet")
    params = json.loads((PROCESSED_DIR / "war_params.json").read_text())

    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    m["tm_comp"] = m["league"].map(LEAGUE_TM)
    g = pd.read_csv(TM / "games.csv", usecols=["game_id", "competition_id", "season", "date", "round", "home_club_id",
                                               "away_club_id", "home_club_goals", "away_club_goals"])
    cmap = match_clubs(m, g[g["competition_id"].isin(LEAGUE_TM.values())]).set_index(["season", "us_team"])["tm_club"]
    tm_to_us = {(s, v): k for (s, k), v in cmap.items()}

    gl = pd.read_csv(TM / "game_lineups.csv", usecols=["game_id", "player_id", "club_id"])
    gd_ = pd.read_csv(TM / "games.csv", usecols=["game_id", "date", "competition_type"])
    gl = gl.merge(gd_[gd_["competition_type"] != "national_team_competition"][["game_id", "date"]], on="game_id")
    gl["date"] = pd.to_datetime(gl["date"])
    fb._LINEUPS = gl
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    fb._TRANSFERS = tr.dropna(subset=["transfer_date"])

    # squad strength per team-season (vs own league average)
    st = pd.concat([fb.our_strength(t, cmap, war, avail, params, act) for t in fb.TARGETS], ignore_index=True)
    S = st.set_index(["season", "team_id"])["strength"]
    LG = st.set_index(["season", "team_id"])["league"]

    eu = g[g["competition_id"].isin(EURO) & g["season"].isin(fb.TARGETS)].dropna(subset=["home_club_goals"]).copy()
    eu["h"] = [tm_to_us.get((s, c)) for s, c in zip(eu["season"], eu["home_club_id"])]
    eu["a"] = [tm_to_us.get((s, c)) for s, c in zip(eu["season"], eu["away_club_id"])]
    eu = eu.dropna(subset=["h", "a"])
    eu["s_h"] = [S.get((s, h), np.nan) for s, h in zip(eu["season"], eu["h"])]
    eu["s_a"] = [S.get((s, a), np.nan) for s, a in zip(eu["season"], eu["a"])]
    eu["lg_h"] = [LG.get((s, h)) for s, h in zip(eu["season"], eu["h"])]
    eu["lg_a"] = [LG.get((s, a)) for s, a in zip(eu["season"], eu["a"])]
    eu = eu.dropna(subset=["s_h", "s_a", "lg_h", "lg_a"])
    eu = eu[eu["lg_h"] != eu["lg_a"]]    # same-league ties say nothing about league strength
    eu["home"] = (~eu["round"].fillna("").str.contains("Final", case=False) |
                  eu["round"].fillna("").str.contains("Quarter|Semi|1/8|1/4", case=False)).astype(float)
    eu["gd"] = eu["home_club_goals"] - eu["away_club_goals"]
    print(f"European games between clubs from two different big five leagues: {len(eu):,} "
          f"({len(eu) / eu['season'].nunique():.0f} a season)")

    # league mean elo on 1 August of that season, vs the EPL. built from earlier games only, so out of sample
    elo = pd.read_parquet(PROCESSED_DIR / "elo_aug1.parquet")
    st["tm"] = [cmap.get((s, t)) for s, t in zip(st["season"], st["team_id"])]
    ei = elo.set_index(["season", "club_id"])["elo"]
    st["elo"] = [ei.get((s, c), np.nan) for s, c in zip(st["season"], st["tm"])]
    le = st.groupby(["season", "league"])["elo"].mean().unstack()
    le = le.sub(le["EPL"], axis=0)
    eu["e_h"] = [le.loc[s, l] for s, l in zip(eu["season"], eu["lg_h"])]
    eu["e_a"] = [le.loc[s, l] for s, l in zip(eu["season"], eu["lg_a"])]

    b = fit(eu)
    print(f"  home {b[0]:+.2f} goals, beta {b[1]:.2f} (1 = our squad values are on the right scale)")

    # 1. held-out seasons
    print("\n1. held-out seasons, MSE of goal difference (everything fitted without that season):")
    print("   free = 8 offset parameters from the European games, elo = league mean elo x one fitted scale")
    tot = {"none": 0.0, "free": 0.0, "elo": 0.0}
    wins = {"free": 0, "elo": 0}
    ks = []
    for t in sorted(eu["season"].unique()):
        tr_, te = eu[eu["season"] != t], eu[eu["season"] == t]
        bo, bn = fit(tr_), fit(tr_, offsets=False)
        Xe_tr = np.column_stack([tr_["home"], tr_["s_h"] - tr_["s_a"], tr_["e_h"] - tr_["e_a"]])
        Xe_te = np.column_stack([te["home"], te["s_h"] - te["s_a"], te["e_h"] - te["e_a"]])
        be = np.linalg.lstsq(Xe_tr, tr_["gd"], rcond=None)[0]
        ks.append(be[2])
        e = {"free": np.mean((te["gd"] - design(te) @ bo) ** 2),
             "none": np.mean((te["gd"] - design(te)[:, :2] @ bn) ** 2),
             "elo": np.mean((te["gd"] - Xe_te @ be) ** 2)}
        for k in tot:
            tot[k] += e[k] * len(te)
        for k in wins:
            wins[k] += e[k] < e["none"]
        print(f"  {season_label(t)}: none {e['none']:.3f}   free {e['free']:.3f}   elo {e['elo']:.3f}  ({len(te)} games)")
    n_s = eu["season"].nunique()
    print(f"  overall: none {tot['none'] / len(eu):.3f}   free {tot['free'] / len(eu):.3f} (better in {wins['free']} of "
          f"{n_s})   elo {tot['elo'] / len(eu):.3f} (better in {wins['elo']} of {n_s})")
    Xe = np.column_stack([eu["home"], eu["s_h"] - eu["s_a"], eu["e_h"] - eu["e_a"]])
    be_all = np.linalg.lstsq(Xe, eu["gd"], rcond=None)[0]
    print(f"  elo scale: {100 * be_all[2]:.2f} goals per match per 100 elo points of league average "
          f"(held-out fits {100 * min(ks):.2f} to {100 * max(ks):.2f})")

    # offsets per season, team level and per player. seasons before 2015 (no elo burn-in for the squads)
    # get 2015-16's, the window values from those seasons hardly matter downstream
    rows = []
    for t in list(SEASONS):
        tt = max(t, int(le.index.min()))
        for lg, v in offsets_from(b, t).items():
            e = be_all[2] * le.loc[tt, lg]
            rows.append({"season": t, "league": lg, "offset": e, "per_player": e / 11, "offset_free": v})
    off = pd.DataFrame(rows)
    print("\nleague offset vs the Premier League (npxG per match, team level), 2015-16 / 2019-20 / 2023-24, "
          "elo version (used) | free version:")
    for lg in LEAGUES:
        o = off[off["league"] == lg].set_index("season")
        print(f"  {lg:<10} {o.loc[2015, 'offset']:+.2f} / {o.loc[2019, 'offset']:+.2f} / {o.loc[2023, 'offset']:+.2f}"
              f"   |   {o.loc[2015, 'offset_free']:+.2f} / {o.loc[2019, 'offset_free']:+.2f} / "
              f"{o.loc[2023, 'offset_free']:+.2f}   per player (elo, 2023-24) {o.loc[2023, 'per_player']:+.3f}")

    # 2. movers: value change s -> s+2 for players who switched league, vs what the offsets say
    w = war[["season", "player_id", "league", "val_cal", "age", "proj_min"]]
    a2 = w.merge(w.assign(season=w["season"] - 2), on=["season", "player_id"], suffixes=("", "_2"))
    a2 = a2[(a2["proj_min"] >= 900) & a2["age"].notna()]
    o = off.assign(pp_free=off["offset_free"] / 11).set_index(["season", "league"])["pp_free"]
    # a player's value is vs his own league, so moving to a league that's stronger by x should LOWER his value
    # by x/11: predicted change = old league offset - new league offset
    a2["moved"] = a2["league"] != a2["league_2"]
    a2["delta"] = a2["val_cal_2"] - a2["val_cal"]
    mv = a2[a2["moved"]]
    print(f"\n2. movers: {len(mv):,} players switched big five league between s and s+2 "
          f"(value at s and at s+2, s+2's window still has some of the old league in it, so under 1 is expected)")
    # elo version of the per-player offset, scale from the fit above
    elo_pp = {(s, lg): be_all[2] * le.loc[s, lg] / 11 for s in le.index for lg in LEAGUES}
    for name, src in (("free offsets", lambda s, l: o.get((s, l), 0)), ("elo offsets", lambda s, l: elo_pp.get((s, l), 0))):
        a2["pred"] = [src(s, l1) - src(min(s + 2, max(le.index)), l2) for s, l1, l2 in
                      zip(a2["season"], a2["league"], a2["league_2"])]
        # his value at s in: players move after unusual seasons, and regression to the mean would look like a
        # league effect otherwise (moving up after a career year, then 'dropping')
        X = np.column_stack([np.ones(len(a2)), a2["age"], a2["age"] ** 2, a2["val_cal"], a2["pred"]])
        coef = np.linalg.lstsq(X, a2["delta"], rcond=None)[0]
        res = a2["delta"] - X[:, :4] @ coef[:4]
        to_strong = res[a2["moved"] & (a2["pred"] < 0)].mean()
        to_weak = res[a2["moved"] & (a2["pred"] > 0)].mean()
        print(f"  {name}: slope {coef[4]:+.2f} (age and value at s taken out), to a stronger league {to_strong:+.3f}, "
              f"to a weaker league {to_weak:+.3f} npxG per 90 vs what age and value predict")

    # 3. elo
    print("\n3. elo league averages vs the Premier League (elo points), 2015-16 / 2019-20 / 2023-24:")
    for lg in LEAGUES:
        print(f"  {lg:<10} {le.loc[2015, lg]:+.0f} / {le.loc[2019, lg]:+.0f} / {le.loc[2023, lg]:+.0f}")

    off.to_parquet(PROCESSED_DIR / "league_strength.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'league_strength.parquet'}")
