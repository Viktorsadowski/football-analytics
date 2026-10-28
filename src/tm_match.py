#!/usr/bin/env python3
"""
Step 4: link understat players to transfermarkt players, so we get birth dates, market values and fees.

Done through the games both sources have, names alone are too messy (Rodri, Brahim Diaz vs Brahim Díaz...):
  1. clubs    understat team -> transfermarkt club, per league-season, by which club plays at home on the
              same dates (counted over the whole season, so a few timezone date shifts don't matter)
  2. games    (season, home club, away club) is unique within a league season, so games link on that
  3. players  per club-season, every understat player vs every transfermarkt player of that club:
                overlap   shared games / the smaller of the two game counts
                mins      share of shared games where the minutes agree within 3
                name      name similarity, accents and case stripped
              best one-to-one pairs (hungarian) on overlap * mins + name, kept if the pair holds up
  4. one id   across seasons a player gets the transfermarkt id he was linked to most often

Manual fixes go in manual/id_fixes.csv (us_id, tm_id), they win over everything.
Output: data/processed/id_map.parquet  us_id, tm_id, names, seasons linked, how sure (min overlap, name)

  python src/tm_match.py
"""

import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS

TM_DIR = RAW_DIR / "transfermarkt"
LEAGUE_TM = {"EPL": "GB1", "La liga": "ES1", "Bundesliga": "L1", "Serie A": "IT1", "Ligue 1": "FR1"}
FIXES = ROOT / "manual" / "id_fixes.csv"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalpha() else " " for c in s.lower()).split())


def name_sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    full = SequenceMatcher(None, a, b).ratio()
    # nicknames and missing first names: a shared token of 4+ letters counts as a decent match on its own
    ta, tb = set(a.split()), set(b.split())
    tok = 0.8 if any(len(t) >= 4 for t in ta & tb) else 0.0
    return max(full, tok)


# ── 1. clubs ──────────────────────────────────────────────────────────────────

def match_clubs(us: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    us = us.assign(day=us["date"].dt.normalize())
    tm = tm.assign(day=pd.to_datetime(tm["date"]))
    j = us.merge(tm, left_on=["tm_comp", "season", "day"], right_on=["competition_id", "season", "day"])
    pairs = pd.concat([j[["season", "home_id", "home_club_id"]].set_axis(["season", "us_team", "tm_club"], axis=1),
                       j[["season", "away_id", "away_club_id"]].set_axis(["season", "us_team", "tm_club"], axis=1)])
    cnt = pairs.value_counts().rename("n").reset_index().sort_values("n", ascending=False)
    best = cnt.drop_duplicates(["season", "us_team"])
    # share of that team's games where the club guess plays on the same day, should be ~1 everywhere
    games = pd.concat([us[["season", "home_id"]].set_axis(["season", "us_team"], axis=1),
                       us[["season", "away_id"]].set_axis(["season", "us_team"], axis=1)]).value_counts()
    best = best.join(games.rename("tot"), on=["season", "us_team"])
    best["share"] = best["n"] / best["tot"]
    return best


# ── 3. players in one club-season ─────────────────────────────────────────────

def match_club_season(u: pd.DataFrame, t: pd.DataFrame) -> list:
    """u: understat appearances (us_id, name, tm_game, minutes), t: transfermarkt (tm_id, name, tm_game, minutes)"""
    ug = u.groupby("us_id")["tm_game"].apply(set)
    tg = t.groupby("tm_id")["tm_game"].apply(set)
    un = u.groupby("us_id")["name"].first()
    tn = t.groupby("tm_id")["name"].first()
    umin = u.set_index(["us_id", "tm_game"])["minutes"]
    tmin = t.set_index(["tm_id", "tm_game"])["minutes"]
    U, T = list(ug.index), list(tg.index)
    score = np.zeros((len(U), len(T)))
    info = {}
    for i, a in enumerate(U):
        for k, b in enumerate(T):
            shared = ug[a] & tg[b]
            if not shared:
                continue
            ov = len(shared) / min(len(ug[a]), len(tg[b]))
            mins = np.mean([abs(umin[(a, g)] - tmin[(b, g)]) <= 3 for g in shared])
            ns = name_sim(un[a], tn[b])
            score[i, k] = ov * mins + ns
            info[(i, k)] = (ov, mins, ns, len(shared))
    rows = []
    ri, ci = linear_sum_assignment(-score)
    for i, k in zip(ri, ci):
        if (i, k) not in info:
            continue
        ov, mins, ns, sh = info[(i, k)]
        # good name and real overlap, or near-perfect overlap with matching minutes even if the name is off.
        # games-only needs 8+ shared games, with 1-2 games it linked Macías to Cucurella and the like
        if (ns >= 0.5 and ov >= 0.5) or (ov >= 0.8 and mins >= 0.8 and sh >= 8):
            rows.append((U[i], T[k], un[U[i]], tn[T[k]], ov, mins, ns, sh))
    return rows


# ── helpers for the later steps (after id_map.parquet exists) ────────────────

_VALS = None


def us_to_tm() -> pd.Series:
    """us_id -> tm_id (one tm id per understat id, the duplicate understat ids both point at it)"""
    return pd.read_parquet(PROCESSED_DIR / "id_map.parquet").drop_duplicates("us_id").set_index("us_id")["tm_id"]


def birth_dates() -> pd.Series:
    """us_id -> date of birth"""
    dob = pd.read_csv(TM_DIR / "players.csv", usecols=["player_id", "date_of_birth"])
    dob = pd.to_datetime(dob.set_index("player_id")["date_of_birth"], errors="coerce")
    return us_to_tm().map(dob)


def start_values(season: int) -> pd.Series:
    """us_id -> latest transfermarkt market value (eur) on or before 1 Sept of that season"""
    global _VALS
    if _VALS is None:
        _VALS = pd.read_csv(TM_DIR / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
        _VALS["date"] = pd.to_datetime(_VALS["date"])
    v = _VALS[_VALS["date"] <= f"{season}-09-01"].sort_values("date").drop_duplicates("player_id", keep="last")
    return us_to_tm().map(v.set_index("player_id")["market_value_in_eur"]).dropna()


if __name__ == "__main__":
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["match_id", "player_id", "player", "minutes",
                                                                    "team_id"])
    m["tm_comp"] = m["league"].map(LEAGUE_TM)
    g = pd.read_csv(TM_DIR / "games.csv", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                                   "away_club_id"])
    g = g[g["competition_id"].isin(LEAGUE_TM.values()) & g["season"].isin(SEASONS)]

    # 1. clubs
    clubs = match_clubs(m, g)
    print(f"clubs: {len(clubs)} team-seasons, median share of games agreeing {clubs['share'].median():.2f}, "
          f"lowest {clubs['share'].min():.2f}")
    cmap = clubs.set_index(["season", "us_team"])["tm_club"]

    # 2. games
    m["tm_home"] = [cmap.get((s, t)) for s, t in zip(m["season"], m["home_id"])]
    m["tm_away"] = [cmap.get((s, t)) for s, t in zip(m["season"], m["away_id"])]
    gm = m.merge(g, left_on=["season", "tm_home", "tm_away"], right_on=["season", "home_club_id", "away_club_id"])
    print(f"games: {len(gm):,} of {len(m):,} understat matches linked")
    gid = gm.set_index("match_id")["game_id"]

    # 3. players
    a = pd.read_csv(TM_DIR / "appearances.csv", usecols=["game_id", "player_id", "player_club_id", "player_name",
                                                         "minutes_played"])
    a = a[a["game_id"].isin(gid.values) & (a["minutes_played"] > 0)]
    a = a.merge(g[["game_id", "season"]], on="game_id")
    u = lu[lu["match_id"].isin(gid.index)].copy()
    u["tm_game"] = u["match_id"].map(gid)
    u = u.merge(m[["match_id", "season"]], on="match_id")
    u["tm_club"] = [cmap.get((s, t)) for s, t in zip(u["season"], u["team_id"])]
    u = u.rename(columns={"player_id": "us_id", "player": "name"})
    a = a.rename(columns={"player_id": "tm_id", "player_name": "name", "minutes_played": "minutes",
                          "game_id": "tm_game", "player_club_id": "tm_club"})

    rows = []
    a_g = dict(tuple(a.groupby(["season", "tm_club"])))
    for key, uu in u.groupby(["season", "tm_club"]):
        if key in a_g:
            rows += [(key[0], *r) for r in match_club_season(uu, a_g[key])]
    links = pd.DataFrame(rows, columns=["season", "us_id", "tm_id", "us_name", "tm_name", "overlap", "mins_agree",
                                        "name_sim", "shared"])

    # 4. one tm id per understat id
    votes = links.groupby(["us_id", "tm_id"]).agg(seasons=("season", "size"), us_name=("us_name", "first"),
                                                 tm_name=("tm_name", "first"), min_overlap=("overlap", "min"),
                                                 name_sim=("name_sim", "max")).reset_index()
    votes = votes.sort_values("seasons", ascending=False)
    conflicts = votes[votes.duplicated("us_id", keep=False)]
    idmap = votes.drop_duplicates("us_id")
    if FIXES.exists():
        fx = pd.read_csv(FIXES)
        idmap = idmap[~idmap["us_id"].isin(fx["us_id"])]
        idmap = pd.concat([idmap, fx.assign(seasons=0, manual=True)], ignore_index=True)
    # the other way round: two understat ids on one tm id. either understat has the same guy twice (same name,
    # fine, both keep the link) or a surname-only match (Lluis López -> Diego López). keep the best name only
    best_ns = idmap.groupby("tm_id")["name_sim"].transform("max")
    idmap = idmap[(idmap["name_sim"] >= best_ns) | idmap.get("manual", pd.Series(False, index=idmap.index)).fillna(False)]
    dup_tm = idmap[idmap.duplicated("tm_id", keep=False)]

    # how much of the playing time is covered, that's what matters downstream
    pm = lu.merge(m[["match_id", "season"]], on="match_id").groupby("player_id")["minutes"].sum()
    cov = pm[pm.index.isin(idmap["us_id"])].sum() / pm.sum()
    print(f"players: {len(idmap):,} of {lu['player_id'].nunique():,} understat players linked, "
          f"{cov:.2%} of all minutes")
    print(f"  {len(conflicts['us_id'].unique())} understat ids linked to more than one tm id (majority kept), "
          f"{len(dup_tm)} rows where one tm id has two understat ids")
    # biggest unlinked players, the ones worth a manual fix
    miss = pm[~pm.index.isin(idmap["us_id"])].sort_values(ascending=False).head(10)
    names = lu.groupby("player_id")["player"].first()
    print("  most minutes without a link:", ", ".join(f"{names[i]} ({v:,})" for i, v in miss.items()))
    weird = idmap[idmap["name_sim"] < 0.5].sort_values("seasons", ascending=False)
    print(f"  linked on games alone (name_sim < 0.5): {len(weird)}, e.g.")
    print(weird[["us_name", "tm_name", "seasons", "min_overlap", "name_sim"]].head(12).round(2).to_string(index=False))

    idmap.to_parquet(PROCESSED_DIR / "id_map.parquet", index=False)
    links.to_parquet(PROCESSED_DIR / "id_links.parquet", index=False)
    print(f"saved -> {PROCESSED_DIR / 'id_map.parquet'} (+ id_links.parquet, per season)")
