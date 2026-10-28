#!/usr/bin/env python3
"""
Step 6b: injuries, inferred from matchday squads. the transfermarkt dump has no injury table, so this
works out who was out from who wasn't in the squad (starting XI + bench), in every competition.

A spell = 2+ club games in a row where he's not in the matchday squad, with a squad game at the same club on
both sides (so the end of a stint, a transfer or a loan never counts). Only for regulars: in the squad for 60%+
and starting 40%+ of his last 20 club games there (fewer if he only just joined, at least 3), otherwise "not
in the squad" is just not being picked.
Games where transfermarkt has no lineup for his club are unknown: skipped, they don't count either way.

Removed:
  suspension      a red / second yellow in the game before, or a league yellow that was his 5th, 10th, 15th
                  (Ligue 1: from the 3rd on, they ban on 3 in 10 games), and the spell is 2-3 games -> a ban
  AFCON / Asian   players from CAF / AFC countries, club games during the tournament (+ camp) are unknown
  Cup
  loans, moves    any transfer record (loans too) dated inside the spell, or a squad game for another club
                  inside it. a loan to a league transfermarkt doesn't track would otherwise look like an injury

Kept as features, for the confidence score and for checking:
  left_early      in the last game before, started and played under 60 min, or came on and played under 15
  ret_start       came back straight into the starting XI (dropped players tend to come back on the bench)
  games, days     length in club games missed and in days (last squad game -> first squad game back)

Validation (the point of this script, see the README / report):
  1. recall    manual/known_injuries.csv, documented injuries (club statements, press). found = a spell that
               starts within 1 club game of the documented date and covers it
  2. precision manual/injury_sample.csv, a random sample of detected spells, written once (seed fixed) to be
               checked by hand against news reports. the 'verdict' column is filled in by hand
  3. sanity    lengths, age, month, spells per team-season

Output: data/processed/injuries.parquet  one row per spell (transfermarkt player id, club, dates, length, flags)

  python src/inferred_injuries.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT

TM = RAW_DIR / "transfermarkt"
MANUAL = ROOT / "manual"
BIG5 = ["GB1", "ES1", "L1", "IT1", "FR1"]
MIN_GAMES = 2
# "regular" check over his last 20 known club games at this club (fewer if he joined recently, min 3).
# first version was 5 of the last 6 in the squad + 3 starts: missed injuries that came right after another
# absence or a summer (Neuer, Rodri, Sané...), which is exactly when injuries cluster. games inside a spell
# found earlier are left out of the 20, being injured shouldn't count against being a regular (Jota 2022)
PRIOR_N, PRIOR_MIN, PRIOR_SQUAD, PRIOR_START = 20, 3, 0.6, 0.4
SAMPLE_N = 40

# tournament windows (start - 10 days for the camp, end + 3 days), winter ones only, summer ones have no club games
AFCON = [("2015-01-07", "2015-02-11"), ("2017-01-04", "2017-02-08"), ("2021-12-30", "2022-02-09"),
         ("2024-01-03", "2024-02-14"), ("2025-12-11", "2026-01-21")]
ASIAN = [("2014-12-30", "2015-02-03"), ("2018-12-26", "2019-02-04"), ("2024-01-02", "2024-02-13")]
CAF = {"Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde",
       "Central African Republic", "Chad", "Comoros", "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea",
       "Eritrea", "Eswatini", "Ethiopia", "Gabon", "The Gambia", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
       "Cote d'Ivoire", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
       "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
       "Sao Tome and Principe", "Senegal", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
       "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"}
AFC = {"Australia", "Bahrain", "China", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Korea, South",
       "South Korea", "Kuwait", "Lebanon", "Malaysia", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria",
       "Tajikistan", "Thailand", "United Arab Emirates", "Uzbekistan", "Vietnam", "Kyrgyzstan", "Hong Kong"}


def load():
    g = pd.read_csv(TM / "games.csv", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                               "away_club_id"])
    g["date"] = pd.to_datetime(g["date"])
    gl = pd.read_csv(TM / "game_lineups.csv", usecols=["game_id", "player_id", "club_id", "type"])
    gl["start"] = gl["type"] == "starting_lineup"
    # club-seasons in a big five league, from the games (clubs.csv only has the current league)
    lg = g[g["competition_id"].isin(BIG5)]
    big = set(zip(lg["season"], lg["home_club_id"])) | set(zip(lg["season"], lg["away_club_id"]))
    return g, gl, big


def club_schedules(g: pd.DataFrame, gl: pd.DataFrame) -> dict:
    """club -> its games that have a lineup for that club, sorted by date (game_id, date, season, competition)"""
    have = gl[["game_id", "club_id"]].drop_duplicates()
    s = have.merge(g[["game_id", "date", "season", "competition_id"]], on="game_id").sort_values(["date", "game_id"])
    return {c: d[["game_id", "date", "season", "competition_id"]].reset_index(drop=True) for c, d in s.groupby("club_id")}


def bans(g: pd.DataFrame) -> set:
    """(player_id, game_id) of games where the card would get him banned for the next one(s)"""
    e = pd.read_csv(TM / "game_events.csv", usecols=["game_id", "type", "player_id", "description"])
    e = e[e["type"] == "Cards"].merge(g[["game_id", "competition_id"]], on="game_id")
    d = e["description"].fillna("")
    red = d.str.contains("Red card|Second yellow")
    n = pd.to_numeric(d.str.extract(r"^(\d+)\. Yellow")[0], errors="coerce")
    yel = d.str.contains("Yellow card") & ~red
    lg = e["competition_id"].isin(BIG5)
    acc = yel & lg & (((n % 5 == 0) & (e["competition_id"] != "FR1")) | ((n >= 3) & (e["competition_id"] == "FR1")))
    hit = e[red | acc]
    return set(zip(hit["player_id"], hit["game_id"]))


def tournament_days(players: pd.DataFrame) -> dict:
    """player_id -> list of (start, end) windows where his club games don't count"""
    nat = players.set_index("player_id")["country_of_citizenship"]
    out = {}
    for pid, c in nat.items():
        w = (AFCON if c in CAF else []) + (ASIAN if c in AFC else [])
        if w:
            out[pid] = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in w]
    return out


def detect(g, gl, big, sched, banned, tourn, apps) -> pd.DataFrame:
    gl = gl.merge(g[["game_id", "date", "season"]], on="game_id")
    gl = gl[[(s, c) in big for s, c in zip(gl["season"], gl["club_id"])]]
    gl = gl.sort_values(["player_id", "date"])
    mins = apps.set_index(["player_id", "game_id"])["minutes_played"]
    rows = []
    for (pid, club), d in gl.groupby(["player_id", "club_id"]):
        sc = sched.get(club)
        if sc is None or len(d) < PRIOR_N + 1:
            continue
        in_sq = sc["game_id"].isin(set(d["game_id"])).to_numpy()
        started = sc["game_id"].isin(set(d.loc[d["start"], "game_id"])).to_numpy()
        # unknown games: tournament windows for his country
        unknown = np.zeros(len(sc), bool)
        for a, b in tourn.get(pid, []):
            unknown |= sc["date"].between(a, b).to_numpy()
        # only between his first and last squad game at this club. loans out and back are taken out after
        idx = np.flatnonzero(in_sq)
        lo, hi = idx.min(), idx.max()
        # games inside a spell found earlier don't count against him in the regular check
        injured = np.zeros(len(sc), bool)
        i = lo
        while i < hi:
            if in_sq[i] or unknown[i]:
                i += 1
                continue
            j = i
            while j <= hi and not in_sq[j]:
                j += 1
            # spell = games i..j-1, minus the unknown ones
            miss = [k for k in range(i, j) if not unknown[k]]
            if len(miss) >= MIN_GAMES and j <= hi:
                before = [k for k in range(i - 1, lo - 1, -1) if not unknown[k] and not injured[k]][:PRIOR_N]
                ok = (len(before) >= PRIOR_MIN and in_sq[before].mean() >= PRIOR_SQUAD
                      and started[before].mean() >= PRIOR_START)
                if ok:
                    injured[miss] = True
                    last = [k for k in range(i - 1, lo - 1, -1) if not unknown[k]][0]
                    gid_last = sc.loc[last, "game_id"]
                    m_last = mins.get((pid, gid_last), np.nan)
                    st_last = started[last]
                    rows.append({
                        "player_id": pid, "club_id": club, "season": sc.loc[i, "season"],
                        "last_game": gid_last, "last_date": sc.loc[last, "date"], "start_date": sc.loc[i, "date"],
                        "first_back": sc.loc[j, "date"],
                        "games": len(miss), "days": (sc.loc[j, "date"] - sc.loc[last, "date"]).days,
                        "league_games": int(sc.loc[miss, "competition_id"].isin(BIG5).sum()),
                        "banned": (pid, gid_last) in banned,
                        "left_early": bool((st_last and m_last < 60) or (not st_last and m_last < 15)),
                        "ret_start": bool(started[j]),
                    })
            i = j
    return pd.DataFrame(rows)


def recall(sp: pd.DataFrame, file: str = "known_injuries.csv") -> pd.DataFrame:
    """known injuries vs detected spells. found = a spell of his where the injury date falls between his last
    squad game before it and the first game he missed (a week of slack each side, ~1 club game)"""
    k = pd.read_csv(MANUAL / file, parse_dates=["injury_date"])
    out = []
    for r in k.itertuples():
        s = sp[sp["player_id"] == r.tm_id]
        hit = s[(s["last_date"] - pd.Timedelta(days=7) <= r.injury_date) &
                (r.injury_date <= s["start_date"] + pd.Timedelta(days=7))]
        if len(hit):
            h = hit.iloc[0]
            out.append({"player": r.player, "injury_date": r.injury_date.date(), "found": True,
                        "spell_start": h["start_date"].date(), "games": h["games"], "days": h["days"]})
        else:
            out.append({"player": r.player, "injury_date": r.injury_date.date(), "found": False})
    return pd.DataFrame(out)


if __name__ == "__main__":
    g, gl, big = load()
    sched = club_schedules(g, gl)
    players = pd.read_csv(TM / "players.csv", usecols=["player_id", "name", "date_of_birth", "country_of_citizenship",
                                                        "position"])
    apps = pd.read_csv(TM / "appearances.csv", usecols=["player_id", "game_id", "minutes_played"])
    sp = detect(g, gl, big, sched, bans(g), tournament_days(players), apps)
    n0 = len(sp)
    sp = sp[~(sp["banned"] & (sp["games"] <= 3))]
    n1 = len(sp)
    # loans / moves inside the spell
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    sp = sp.reset_index(drop=True)
    j = sp.reset_index().merge(tr, on="player_id")
    hit = j[(j["transfer_date"] > j["start_date"] - pd.Timedelta(days=3)) & (j["transfer_date"] < j["first_back"])]["index"]
    gd = gl.merge(g[["game_id", "date"]], on="game_id")
    j = sp.reset_index().merge(gd[["player_id", "club_id", "date"]].rename(columns={"club_id": "c2"}), on="player_id")
    hit2 = j[(j["c2"] != j["club_id"]) & (j["date"] >= j["start_date"]) & (j["date"] < j["first_back"])]["index"]
    sp = sp.drop(index=set(hit) | set(hit2))
    print(f"spells: {n0:,} found, {n0 - n1:,} dropped as bans, {n1 - len(sp):,} as loans / moves, {len(sp):,} left")
    sp = sp.merge(players[["player_id", "name", "date_of_birth", "position"]], on="player_id", how="left")
    sp["age"] = (sp["start_date"] - pd.to_datetime(sp["date_of_birth"], errors="coerce")).dt.days / 365.25
    sp.to_parquet(PROCESSED_DIR / "injuries.parquet", index=False)
    print(f"saved -> {PROCESSED_DIR / 'injuries.parquet'}")

    # the first list was used to fix the rules (see PRIOR_N notes), so it's in-sample now. the holdout list
    # was written down after the rules were frozen and before running it, that's the honest number
    for f, label in (("known_injuries.csv", "rules were tuned on this one"), ("known_injuries_holdout.csv", "holdout")):
        rc = recall(sp, f)
        print(f"\n1. recall, {f} ({label}): {rc['found'].sum()} of {len(rc)} ({rc['found'].mean():.0%}), pass mark 85%")
        print(rc.to_string(index=False))

    # 2. precision: random sample to check by hand, written once so the draw can't be redone until it looks good
    out = MANUAL / "injury_sample.csv"
    if not out.exists():
        clubs = pd.read_csv(TM / "clubs.csv", usecols=["club_id", "name"]).set_index("club_id")["name"]
        smp = sp.sample(SAMPLE_N, random_state=42)[["player_id", "name", "club_id", "last_date", "start_date",
                                                     "first_back", "games", "days", "left_early", "ret_start"]]
        smp["club"] = smp["club_id"].map(clubs)
        smp["verdict"] = ""   # injury / illness / not injury / unclear, filled in by hand
        smp["source"] = ""
        smp.to_csv(out, index=False)
        print(f"\n2. precision sample written -> {out}, fill in verdict + source by hand")
    else:
        smp = pd.read_csv(out, dtype={"verdict": str, "source": str})
        # unclear = no source covers that date (mostly before 2016), left out of the rate but reported
        done = smp[smp["verdict"].fillna("").isin(["injury", "illness", "not injury"])]
        if len(done):
            good = done["verdict"].isin(["injury", "illness"]).mean()
            print(f"\n2. precision on the hand-checked sample: {good:.0%} of {len(done)} decided were real injuries / "
                  f"illness (pass mark 80%), {len(smp) - len(done)} unclear")
            print(smp["verdict"].value_counts().to_string())
