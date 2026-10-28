#!/usr/bin/env python3
"""
Step 9a: our own Elo, a benchmark for the August forecasts. ClubElo itself wasn't reachable (API down, and
its robots.txt says no to bots), so this is the same kind of rating built from results we already have:
every transfermarkt game since 2012-13, league + cups + Europe. the European games are what tie the leagues
together.

  expected   E = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400))
  update     R += K * G * (result - E), result 1 / 0.5 / 0, G = goal difference multiplier (World Football
             Elo style: 1, 1.5, then (11 + gd) / 8)
  new teams  a club coming up into a big five league gets the average rating of the clubs that went down
             from it the season before (the second tiers aren't in the data, so that's the best guess)
  start      everyone at 1500 in July 2012, two seasons of burn-in before the first forecast (2015-16)

Output: data/processed/elo_aug1.parquet  season, tm club_id, elo on 1 August of that season

  python src/elo.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR

TM = RAW_DIR / "transfermarkt"
BIG5 = ["GB1", "ES1", "L1", "IT1", "FR1"]
K, HFA, START = 20.0, 65.0, 1500.0


def goal_mult(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


if __name__ == "__main__":
    g = pd.read_csv(TM / "games.csv", usecols=["game_id", "competition_id", "competition_type", "season", "date",
                                               "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"])
    g["date"] = pd.to_datetime(g["date"])
    g = g[(g["competition_type"] != "national_team_competition") & (g["date"] >= "2012-07-01")]
    g = g.dropna(subset=["home_club_goals", "away_club_goals"]).sort_values(["date", "game_id"])
    lg = g[g["competition_id"].isin(BIG5)]
    members = lg.groupby(["competition_id", "season"]).apply(
        lambda d: set(d["home_club_id"]) | set(d["away_club_id"]), include_groups=False)

    R = {}
    snaps = []
    seasons = sorted(lg["season"].unique())
    by_season = {s: d for s, d in g.groupby("season")}
    for s in seasons:
        # promoted clubs: rating of last season's relegated clubs from the same league
        if s - 1 in seasons:
            for comp in BIG5:
                now, before = members.get((comp, s), set()), members.get((comp, s - 1), set())
                down = [R[c] for c in before - now if c in R]
                if down:
                    for c in now - before:
                        R[c] = float(np.mean(down))
        snaps += [{"season": s, "club_id": c, "elo": R.get(c, START)} for c in set().union(
            *[members.get((comp, s), set()) for comp in BIG5])]
        d = by_season.get(s)
        if d is None:
            continue
        for h, a, hg, ag in zip(d["home_club_id"], d["away_club_id"], d["home_club_goals"], d["away_club_goals"]):
            rh, ra = R.get(h, START), R.get(a, START)
            e = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
            res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = K * goal_mult(hg - ag) * (res - e)
            R[h], R[a] = rh + delta, ra - delta

    out = pd.DataFrame(snaps)
    out.to_parquet(PROCESSED_DIR / "elo_aug1.parquet", index=False)
    # quick look: top of the table as of 1 August 2023
    names = pd.read_csv(TM / "clubs.csv", usecols=["club_id", "name"]).set_index("club_id")["name"]
    top = out[out["season"] == 2023].sort_values("elo", ascending=False).head(10)
    print("Elo on 1 August 2023, top 10:")
    print("  " + ", ".join(f"{names.get(c, c)} {e:.0f}" for c, e in zip(top["club_id"], top["elo"])))
    print(f"saved -> {PROCESSED_DIR / 'elo_aug1.parquet'} ({len(out):,} club-seasons)")
