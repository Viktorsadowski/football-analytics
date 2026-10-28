#!/usr/bin/env python3
"""
Step 5c: defensive and passing stats from the two full seasons of free event data.
  Wyscout public dataset (Pappalardo et al. 2019, CC BY 4.0): all five leagues, 2017-18
  StatsBomb open data (credit StatsBomb): all five leagues, 2015-16 (pulled by statsbomb_pull.py)

Per player-season, per 90: passes, pass%, progressive passes, key passes, tackles (defending duels) + won,
interceptions, clearances, aerial duels + won, dribbles + completed, fouls. StatsBomb also has pressures,
recoveries, blocks, dribbled past and carries, Wyscout doesn't, so those stay StatsBomb-only.

Linking to our ids:
  Wyscout     players.json has birth dates -> transfermarkt players on date of birth + name similarity
              -> understat through id_map
  StatsBomb   no birth dates. teams linked through the matches (date + score) to understat, then players by
              name within team-season (nickname first, it's the shirt name), like tm_match.py

Output: data/processed/event_stats.parquet  season, player_id (understat), source, minutes, *_90 columns

  python src/event_stats.py
"""

import json
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR
from tm_match import us_to_tm

WY = RAW_DIR / "wyscout"
SB = RAW_DIR / "statsbomb"
LEAGUE_FILES = {"England": "EPL", "Spain": "La liga", "Germany": "Bundesliga", "Italy": "Serie A", "France": "Ligue 1"}
STATS = ["passes", "passes_cmp", "prog_passes", "key_passes", "tackles", "tackles_won", "interceptions",
         "clearances", "aerials", "aerials_won", "dribbles", "dribbles_cmp", "fouls"]
SB_ONLY = ["pressures", "recoveries", "blocks", "dribbled_past", "carries"]


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalpha() else " " for c in s).split())


def sim(a, b) -> float:
    a, b = norm(a), norm(b)
    tok = 0.8 if any(len(t) >= 4 for t in set(a.split()) & set(b.split())) else 0.0
    return max(SequenceMatcher(None, a, b).ratio(), tok)


# ── wyscout ───────────────────────────────────────────────────────────────────

def wyscout() -> pd.DataFrame:
    rows, mins = [], {}
    for f, lg in LEAGUE_FILES.items():
        for m in json.load(open(WY / "matches" / f"matches_{f}.json")):
            for td in m["teamsData"].values():
                fm = td["formation"]
                # a few matches have the string "null" in there instead of a list
                sl = fm.get("substitutions") if isinstance(fm.get("substitutions"), list) else []
                subs = {x["playerOut"]: x["minute"] for x in sl}
                ins = {x["playerIn"]: x["minute"] for x in sl}
                for p in fm.get("lineup") if isinstance(fm.get("lineup"), list) else []:
                    pid = p["playerId"]
                    mins[pid] = mins.get(pid, 0) + subs.get(pid, 90)
                for pid, mi in ins.items():
                    mins[pid] = mins.get(pid, 0) + (subs.get(pid, 90) - mi if pid in subs else 90 - mi)
        ev = json.load(open(WY / "events" / f"events_{f}.json"))
        for e in ev:
            tags = {t["id"] for t in e["tags"]}
            r = {"wy_id": e["playerId"], "league": lg}
            sub, evn = e["subEventId"], e["eventId"]
            if evn == 8:
                r["passes"] = 1
                if 1801 in tags:
                    r["passes_cmp"] = 1
                    pos = e["positions"]
                    if len(pos) == 2:
                        x0, x1 = pos[0]["x"], pos[1]["x"]
                        if (100 - x1) <= 0.75 * (100 - x0) and x1 > 50:
                            r["prog_passes"] = 1
                if 302 in tags:
                    r["key_passes"] = 1
            elif sub == 12:
                r["tackles"] = 1
                if 703 in tags:
                    r["tackles_won"] = 1
            elif sub == 10:
                r["aerials"] = 1
                if 703 in tags:
                    r["aerials_won"] = 1
            elif sub == 11 and tags & {503, 504}:
                r["dribbles"] = 1
                if 703 in tags:
                    r["dribbles_cmp"] = 1
            elif sub == 71:
                r["clearances"] = 1
            elif evn == 2:
                r["fouls"] = 1
            if 1401 in tags:
                r["interceptions"] = 1
            if len(r) > 2:
                rows.append(r)
    d = pd.DataFrame(rows).fillna(0)
    d = d.groupby(["wy_id", "league"]).sum().reset_index()
    d["minutes"] = d["wy_id"].map(mins).fillna(0)
    # a player with two clubs in the season: one row per league is fine, minutes are per player though,
    # so keep the league where he had the most events
    d = d.sort_values("passes").drop_duplicates("wy_id", keep="last")

    # link: wyscout birth date + name -> transfermarkt -> understat
    pl = pd.DataFrame(json.load(open(WY / "players.json")))
    pl["dob"] = pd.to_datetime(pl["birthDate"], errors="coerce")
    pl["wy_name"] = (pl["firstName"] + " " + pl["lastName"]).map(lambda s: s.encode().decode("unicode_escape")
                                                                 if "\\u" in s else s)
    tm = pd.read_csv(RAW_DIR / "transfermarkt" / "players.csv", usecols=["player_id", "name", "date_of_birth"])
    tm["dob"] = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.normalize()
    j = pl[["wyId", "wy_name", "shortName", "dob"]].merge(tm, on="dob")
    j["s"] = [max(sim(a, b), sim(c, b)) for a, c, b in zip(j["wy_name"], j["shortName"], j["name"])]
    j = j[j["s"] >= 0.6].sort_values("s").drop_duplicates("wyId", keep="last")
    us = {v: k for k, v in us_to_tm().items()}
    j["player_id"] = j["player_id"].map(us)
    link = j.dropna(subset=["player_id"]).set_index("wyId")["player_id"].astype(int)
    d["player_id"] = d["wy_id"].map(link)
    print(f"wyscout 2017-18: {len(d):,} players with events, {d['player_id'].notna().mean():.0%} linked "
          f"({d.loc[d['player_id'].notna(), 'minutes'].sum() / d['minutes'].sum():.0%} of minutes)")
    d = d.dropna(subset=["player_id"])
    d["season"], d["source"] = 2017, "wyscout"
    return d.drop(columns=["wy_id"])


# ── statsbomb ─────────────────────────────────────────────────────────────────

def statsbomb() -> pd.DataFrame | None:
    f = SB / "sb_2015_player_match.parquet"
    if not f.exists():
        print("statsbomb: not pulled yet (python src/statsbomb_pull.py), skipped")
        return None
    pm = pd.read_parquet(f)
    sm = pd.read_parquet(SB / "sb_2015_matches.parquet")
    um = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    um = um[um["season"] == 2015].assign(day=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
    j = sm.merge(um, left_on=["date", "hg", "ag"], right_on=["day", "goals_h", "goals_a"])
    # statsbomb team name -> understat team id, by votes over the season
    tv = pd.concat([j[["home", "home_id"]].set_axis(["t", "id"], axis=1), j[["away", "away_id"]].set_axis(["t", "id"], axis=1)])
    team = tv.value_counts().reset_index().drop_duplicates("t").set_index("t")["id"]
    pm = pm.merge(sm[["match_id", "league"]], on="match_id")
    pm["team_id"] = pm["team"].map(team)
    agg = pm.groupby(["sb_player_id", "team_id", "league"]).agg(
        name=("name", "first"), nickname=("nickname", "first"), minutes=("minutes", "sum"),
        **{c: (c, "sum") for c in STATS + SB_ONLY if c in pm}).reset_index()

    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["match_id", "player_id", "player", "team_id", "minutes"])
    lu = lu.merge(um[["match_id"]], on="match_id")
    roster = lu.groupby(["team_id", "player_id"]).agg(name=("player", "first"), us_min=("minutes", "sum")).reset_index()
    # one-to-one per team (hungarian), like tm_match.py. first version took the best name for every statsbomb
    # player on its own, and several of them landed on the same understat player (5,700 minutes in a season)
    from scipy.optimize import linear_sum_assignment
    links = []
    for tid, g in agg.groupby("team_id"):
        r = roster[roster["team_id"] == tid].reset_index(drop=True)
        g = g.reset_index(drop=True)
        S = np.zeros((len(g), len(r)))
        for i, row in g.iterrows():
            for k, q in r.iterrows():
                ns = max(sim(row["nickname"], q["name"]) if row["nickname"] else 0, sim(row["name"], q["name"]))
                ok = abs(row["minutes"] - q["us_min"]) <= max(150, 0.15 * q["us_min"])
                S[i, k] = ns + (0.5 if ok else 0)
        ri, ci = linear_sum_assignment(-S)
        for i, k in zip(ri, ci):
            # decent name and the minutes agree
            if S[i, k] >= 1.1:
                links.append((g.loc[i, "sb_player_id"], tid, r.loc[k, "player_id"]))
    lk = pd.DataFrame(links, columns=["sb_player_id", "team_id", "player_id"])
    d = agg.merge(lk, on=["sb_player_id", "team_id"])
    print(f"statsbomb 2015-16: {len(agg):,} player-clubs, {len(d) / len(agg):.0%} linked "
          f"({d['minutes'].sum() / agg['minutes'].sum():.0%} of minutes)")
    d = d.groupby(["player_id", "league"])[["minutes"] + [c for c in STATS + SB_ONLY if c in d]].sum().reset_index()
    d = d.sort_values("minutes").drop_duplicates("player_id", keep="last")
    d["season"], d["source"] = 2015, "statsbomb"
    return d


if __name__ == "__main__":
    parts = [wyscout()]
    s = statsbomb()
    if s is not None:
        parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    for c in STATS + SB_ONLY:
        if c in d:
            d[f"{c}_90"] = 90 * d[c] / d["minutes"].clip(lower=1)
    d["pass_pct"] = d["passes_cmp"] / d["passes"].clip(lower=1)
    d["tackle_pct"] = d["tackles_won"] / d["tackles"].clip(lower=1)
    d["aerial_pct"] = d["aerials_won"] / d["aerials"].clip(lower=1)
    big = d[d["minutes"] >= 900]
    print("\nper 90, players with 900+ min (median):")
    print(big.groupby("source")[[f"{c}_90" for c in STATS]].median().round(2).T.to_string())
    d.to_parquet(PROCESSED_DIR / "event_stats.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'event_stats.parquet'}")
