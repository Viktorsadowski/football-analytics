#!/usr/bin/env python3
"""
Step 5c (pull): StatsBomb open data, 2015-16, all big five leagues. Free for research, credit StatsBomb.
Raw events are ~4 MB a match, so this keeps only counts per player per match and throws the events away.

Per player per match: minutes (from the lineups, position from/to), passes + completed, progressive passes
(moves the ball 25%+ of the way to the goal), key passes, tackles + won, interceptions, clearances, blocks,
aerial duels + won, pressures, ball recoveries, dribbles + completed, fouls committed, dribbled past, carries.

Output: data/raw/statsbomb/sb_2015_player_match.parquet, sb_2015_matches.parquet

  python src/statsbomb_pull.py
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import RAW_DIR

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPS = {2: "EPL", 11: "La liga", 9: "Bundesliga", 12: "Serie A", 7: "Ligue 1"}
SEASON_ID = 27   # 2015/2016
OUT = RAW_DIR / "statsbomb"


def get(url):
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            continue
    return None


def minutes(lineup):
    """player_id -> minutes, from each player's position spells (from / to as mm:ss, to=None = to the end)"""
    out = {}
    for team in lineup:
        for p in team["lineup"]:
            m = 0.0
            for pos in p.get("positions", []):
                a = sum(int(x) * f for x, f in zip(pos["from"].split(":"), (1, 1 / 60)))
                b = pos.get("to")
                b = 95.0 if b is None else sum(int(x) * f for x, f in zip(b.split(":"), (1, 1 / 60)))
                m += max(b - a, 0)
            if m > 0:
                out[p["player_id"]] = (p["player_name"], p.get("player_nickname"), team["team_name"], min(m, 95))
    return out


def one_match(mid):
    ev = get(f"{BASE}/events/{mid}.json")
    lu = get(f"{BASE}/lineups/{mid}.json")
    if ev is None or lu is None:
        return []
    mins = minutes(lu)
    c = {}

    def add(pid, k, v=1):
        c.setdefault(pid, {}).setdefault(k, 0)
        c[pid][k] += v

    for e in ev:
        p = e.get("player", {}).get("id")
        if p is None:
            continue
        t = e["type"]["name"]
        if t == "Pass":
            ps = e["pass"]
            add(p, "passes")
            if "outcome" not in ps:          # no outcome = completed in statsbomb
                add(p, "passes_cmp")
                x0, x1 = e["location"][0], ps["end_location"][0]
                if (120 - x1) <= 0.75 * (120 - x0) and x1 > 60:
                    add(p, "prog_passes")
            if ps.get("shot_assist") or ps.get("goal_assist"):
                add(p, "key_passes")
        elif t == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Tackle":
            add(p, "tackles")
            if e["duel"].get("outcome", {}).get("name") in ("Won", "Success", "Success In Play", "Success Out"):
                add(p, "tackles_won")
        elif t == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Aerial Lost":
            add(p, "aerials")
        elif t == "Interception":
            add(p, "interceptions")
        elif t == "Clearance":
            add(p, "clearances")
            if e.get("clearance", {}).get("aerial_won"):
                add(p, "aerials")
                add(p, "aerials_won")
        elif t == "Block":
            add(p, "blocks")
        elif t == "Pressure":
            add(p, "pressures")
        elif t == "Ball Recovery":
            add(p, "recoveries")
        elif t == "Dribble":
            add(p, "dribbles")
            if e.get("dribble", {}).get("outcome", {}).get("name") == "Complete":
                add(p, "dribbles_cmp")
        elif t == "Foul Committed":
            add(p, "fouls")
        elif t == "Dribbled Past":
            add(p, "dribbled_past")
        elif t == "Carry":
            add(p, "carries")
        # aerial won flags sit on several event types
        for k in ("pass", "shot", "miscontrol"):
            if e.get(k, {}).get("aerial_won"):
                add(p, "aerials")
                add(p, "aerials_won")
    rows = []
    for pid, (name, nick, team, m) in mins.items():
        r = {"match_id": mid, "sb_player_id": pid, "name": name, "nickname": nick, "team": team, "minutes": m}
        r.update(c.get(pid, {}))
        rows.append(r)
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    matches = []
    for cid, lg in COMPS.items():
        for m in get(f"{BASE}/matches/{cid}/{SEASON_ID}.json") or []:
            matches.append({"match_id": m["match_id"], "league": lg, "date": m["match_date"],
                            "home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"],
                            "hg": m["home_score"], "ag": m["away_score"]})
    mdf = pd.DataFrame(matches)
    mdf.to_parquet(OUT / "sb_2015_matches.parquet", index=False)
    print(f"{len(mdf):,} matches", mdf["league"].value_counts().to_dict(), flush=True)
    rows = []
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(one_match, mdf["match_id"])):
            rows += r
            if i % 100 == 0:
                print(f"  {i:,} matches done", flush=True)
    pm = pd.DataFrame(rows)
    pm.to_pickle(OUT / "sb_2015_player_match.pkl")      # backup first, a failing save shouldn't cost the download
    # counts missing = 0, but only the counts (text columns like nickname stay text)
    num = [c for c in pm.columns if c not in ("name", "nickname", "team")]
    pm[num] = pm[num].fillna(0)
    pm["nickname"] = pm["nickname"].fillna("").astype(str)
    pm.to_parquet(OUT / "sb_2015_player_match.parquet", index=False)
    print(f"saved {len(pm):,} player-matches -> {OUT}")
