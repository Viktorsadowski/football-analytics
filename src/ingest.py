#!/usr/bin/env python3
"""
Step 1: raw understat dumps -> clean parquet files in data/processed.

  matches.parquet   one row per match: league, season, date, teams, goals, xG, shots, ppda
  lineups.parquet   one row per player per match: minutes, position, subs, cards, per-match box stats
  shots.parquet     one row per shot: minute, xG, situation, shooter, which side

Raw files expected in data/raw/understat (third-party dumps of understat.com, see README):
  general_game_stats.csv, lineup_stats.csv, game_events.csv

  python src/ingest.py
"""

import pandas as pd

from config import LEAGUES, PROCESSED_DIR, SEASONS, UNDERSTAT_DIR, season_label


def load_matches() -> pd.DataFrame:
    g = pd.read_csv(UNDERSTAT_DIR / "general_game_stats.csv")
    g = g.rename(columns={"id": "match_id", "h_id": "home_id", "a_id": "away_id", "team_h": "home_team",
                          "team_a": "away_team", "h_xg": "xg_h", "a_xg": "xg_a", "h_goals": "goals_h",
                          "a_goals": "goals_a", "h_shot": "shots_h", "a_shot": "shots_a",
                          "h_shotOnTarget": "sot_h", "a_shotOnTarget": "sot_a", "h_deep": "deep_h",
                          "a_deep": "deep_a", "h_ppda": "ppda_h", "a_ppda": "ppda_a",
                          # understat's own pre-computed win/draw/loss probabilities from xG, kept for reference
                          "h_w": "us_pw", "h_d": "us_pd", "h_l": "us_pl"})
    g["date"] = pd.to_datetime(g["date"])
    g = g[g["league"].isin(LEAGUES) & g["season"].isin(SEASONS)]
    return g.drop(columns=["fid", "league_id"]).sort_values("date").reset_index(drop=True)


def load_lineups(match_ids) -> pd.DataFrame:
    l = pd.read_csv(UNDERSTAT_DIR / "lineup_stats.csv")
    l = l[l["match_id"].isin(match_ids)].rename(columns={"time": "minutes", "xG": "xg", "xA": "xa",
                                                         "xGChain": "xg_chain", "xGBuildup": "xg_buildup",
                                                         "positionOrder": "pos_order"})
    l["starter"] = l["position"] != "Sub"
    # one match (23028) has its whole roster in there twice
    l = l.drop_duplicates(["match_id", "player_id"])
    return l.reset_index(drop=True)


def load_shots(match_ids) -> pd.DataFrame:
    e = pd.read_csv(UNDERSTAT_DIR / "game_events.csv")
    e = e[e["match_id"].isin(match_ids)]
    e = e.rename(columns={"xG": "xg", "shotType": "shot_type", "lastAction": "last_action",
                          "player_assisted": "assister", "id": "shot_id"})
    # own goals are listed as a "shot" by the defender, credited to the wrong side for xG. drop them,
    # the goals target in segments.py gets them from the match score instead
    e = e[e["result"] != "OwnGoal"]
    # a handful of minutes are garbage (up to 333). nothing real goes past ~100 in a league game
    e = e[e["minute"] <= 105]
    e["is_goal"] = (e["result"] == "Goal").astype(int)
    e["is_pen"] = (e["situation"] == "Penalty").astype(int)
    keep = ["shot_id", "match_id", "minute", "h_a", "player_id", "player", "xg", "is_goal", "is_pen",
            "situation", "shot_type", "last_action", "assister", "X", "Y"]
    return e[keep].reset_index(drop=True)


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m = load_matches()
    l = load_lineups(m["match_id"])
    s = load_shots(m["match_id"])

    print("matches per league and season:")
    print(m.pivot_table(index="season", columns="league", values="match_id", aggfunc="size")
          .rename(index=season_label).to_string())
    # quick sanity: shot xG summed per match should be close to the match xG understat reports
    sx = s.groupby(["match_id", "h_a"])["xg"].sum().unstack()
    chk = m.set_index("match_id")[["xg_h", "xg_a"]].join(sx)
    print(f"\nshot xG vs match xG, mean abs diff: home {(chk['h'] - chk['xg_h']).abs().mean():.3f}"
          f"  away {(chk['a'] - chk['xg_a']).abs().mean():.3f}")
    print(f"lineup rows {len(l):,}, shots {len(s):,}, matches {len(m):,}")

    m.to_parquet(PROCESSED_DIR / "matches.parquet", index=False)
    l.to_parquet(PROCESSED_DIR / "lineups.parquet", index=False)
    s.to_parquet(PROCESSED_DIR / "shots.parquet", index=False)
    print(f"saved -> {PROCESSED_DIR}")
