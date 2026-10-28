#!/usr/bin/env python3
"""
Step 9b: August forecasts, every season 2015-16 to 2023-24, against simple and market benchmarks.
Same question as forecast_backtest.py in nba-moneyball: do the player values add up to good team forecasts,
and is the forecast too compressed (the NBA one was, we fixed it with a stretch).

Our forecast for season t, using only what was known at the end of t-1:
  squad      who's at the club on 1 September t (window shut): his latest squad game or transfer before then,
             whichever is later (transfermarkt)
  minutes    availability.py's projected share of team minutes (injury risk in), shares scaled so a team
             has 11 players' worth. any minutes left over go to "newcomers" (players with no big five
             history) at the newcomer value from war.py
  strength   sum of val_cal x share = projected npxG difference per match, league mean taken out
  points     league mean + ppx x strength x matches (war.py's points per npxG, nothing fitted on t)

Benchmarks, all turned into points the same way: a line fitted on the seasons BEFORE t only
  last       last season's points per game, promoted clubs at what promoted clubs usually get
  elo        our Elo on 1 August (elo.py)
  market     team ratings fitted on Pinnacle odds of the first 3 matchdays (the nearest free stand-in for
             pre-season odds, football has no free season-points totals like the NBA win totals)
  closing    the same on the closing odds of the whole season. uses in-season information, so it's a
             ceiling to compare against, not a fair competitor
  ours, fit  our strength through the same fitted line, to see what the line alone does for us

Scored on 2016-17 to 2023-24 (the fitted lines need at least one earlier season).
Stretch check: slope of actual points on our predicted points (out of sample). 1 = right spread, > 1 =
too compressed.

Output: data/processed/forecast_backtest.parquet  one row per team-season, all predictions + actual

  python src/forecast_backtest.py
"""

import io
import json
import zipfile

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from tm_match import LEAGUE_TM, match_clubs, us_to_tm

TM = RAW_DIR / "transfermarkt"
ODDS = RAW_DIR / "odds"
FD = {"EPL": "E0", "La liga": "SP1", "Bundesliga": "D1", "Serie A": "I1", "Ligue 1": "F1"}
TARGETS = list(SEASONS)[1:]          # 2015-16 on
SCORED = list(SEASONS)[2:]           # 2016-17 on
MARKET_ROUNDS = 3


# ── actual results ────────────────────────────────────────────────────────────

def actual() -> pd.DataFrame:
    from war import team_seasons
    ts = team_seasons()
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    names = pd.concat([m.set_index("home_id")["home_team"], m.set_index("away_id")["away_team"]])
    ts["team"] = ts["team_id"].map(names[~names.index.duplicated()])
    return ts


# ── squads on 1 September ─────────────────────────────────────────────────────

def squads(t: int, cmap: pd.Series) -> pd.DataFrame:
    """understat player_id -> understat team_id for season t, as of 1 September t"""
    cut = pd.Timestamp(f"{t}-09-01")
    tm_to_us = {v: k for (s, k), v in cmap.items() if s == t}       # tm club -> understat team (season t)
    gl = _LINEUPS
    sq = gl[gl["date"] <= cut].sort_values("date").drop_duplicates("player_id", keep="last")[["player_id", "club_id", "date"]]
    tr = _TRANSFERS[_TRANSFERS["transfer_date"] <= cut].sort_values("transfer_date").drop_duplicates("player_id", keep="last")
    j = sq.merge(tr[["player_id", "to_club_id", "transfer_date"]], on="player_id", how="outer")
    later = j["transfer_date"].notna() & (j["date"].isna() | (j["transfer_date"] > j["date"]))
    j["club"] = np.where(later, j["to_club_id"], j["club_id"])
    # only players seen in the year before (squad or transfer), not someone who retired years ago
    recent = (j["date"] >= cut - pd.Timedelta(days=400)) | (j["transfer_date"] >= cut - pd.Timedelta(days=400))
    j = j[recent]
    j["team_id"] = j["club"].map(tm_to_us)
    j = j.dropna(subset=["team_id"])
    us = {v: k for k, v in us_to_tm().items()}
    j["us_id"] = j["player_id"].map(us)
    return j.dropna(subset=["us_id"])[["us_id", "team_id"]].astype(int)


def our_strength(t: int, cmap, war, avail, params, act) -> pd.DataFrame:
    sq = squads(t, cmap)
    v = war[war["season"] == t - 1].set_index("player_id")
    a = avail[avail["season"] == t - 1].set_index("player_id")["proj_share"]
    sq["val"] = sq["us_id"].map(v["val_cal"])
    sq["share"] = sq["us_id"].map(a)
    sq = sq.dropna(subset=["val", "share"])
    teams = act[act["season"] == t][["team_id", "league", "matches", "pts", "team"]]
    rows = []
    for r in teams.itertuples():
        p = sq[sq["team_id"] == r.team_id]
        tot = p["share"].sum()
        scale = min(1.0, 11 / tot) if tot > 0 else 1.0
        known = (p["val"] * p["share"] * scale).sum()
        rest = max(0.0, 11 - tot * scale)
        rows.append({"season": t, "team_id": r.team_id, "team": r.team, "league": r.league, "matches": r.matches,
                     "pts": r.pts, "strength": known + rest * params["newcomer_cal"], "known_share": tot * scale / 11,
                     "n_players": len(p)})
    d = pd.DataFrame(rows)
    d["strength"] -= d.groupby("league")["strength"].transform("mean")
    lg_mean = d.groupby("league")["pts"].transform("mean")   # league mean is fixed by the schedule, not a forecast
    d["ours"] = lg_mean + params["ppx"] * d["strength"] * d["matches"]
    return d


# ── odds ──────────────────────────────────────────────────────────────────────

def odds_ratings(t: int, act: pd.DataFrame, rounds: int | None, closing: bool) -> pd.Series:
    """team rating (expected points per match vs an average team) from match odds, fitted by least squares:
    3 * (p_home - p_away) = c + r_home - r_away. understat team_id -> rating"""
    z = zipfile.ZipFile(ODDS / f"{str(t)[2:]}{str(t + 1)[2:]}.zip")
    out = {}
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    for lg, code in FD.items():
        d = pd.read_csv(io.BytesIO(z.read(f"{code}.csv")), encoding="latin-1")
        # 16/08/14 in the old files, 11/08/2023 in the new ones
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
        cols = ["PSCH", "PSCD", "PSCA"] if closing else ["PSH", "PSD", "PSA"]
        d = d.dropna(subset=cols + ["FTHG", "FTAG"]).copy()
        # football-data names -> understat team ids, by matching the games (date and score)
        um = m[(m["league"] == lg) & (m["season"] == t)].assign(day=lambda x: x["date"].dt.normalize())
        j = d.merge(um, left_on=["Date", "FTHG", "FTAG"], right_on=["day", "goals_h", "goals_a"])
        name = pd.concat([j[["HomeTeam", "home_id"]].set_axis(["n", "id"], axis=1),
                          j[["AwayTeam", "away_id"]].set_axis(["n", "id"], axis=1)]).value_counts().reset_index()
        name = name.drop_duplicates("n").set_index("n")["id"]
        d["h"], d["a"] = d["HomeTeam"].map(name), d["AwayTeam"].map(name)
        d = d.dropna(subset=["h", "a"]).sort_values("Date")
        if rounds is not None:
            # first N games of every team
            seen, keep = {}, []
            for i, r in d.iterrows():
                if seen.get(r["h"], 0) < rounds or seen.get(r["a"], 0) < rounds:
                    keep.append(i)
                seen[r["h"]] = seen.get(r["h"], 0) + 1
                seen[r["a"]] = seen.get(r["a"], 0) + 1
            d = d.loc[keep]
        inv = 1 / d[cols]
        p = inv.div(inv.sum(axis=1), axis=0)       # margin taken out
        y = 3 * (p[cols[0]] - p[cols[2]]).to_numpy()
        teams = sorted(set(d["h"]) | set(d["a"]))
        idx = {tm: i for i, tm in enumerate(teams)}
        X = np.zeros((len(d), len(teams) + 1))
        X[np.arange(len(d)), [idx[x] for x in d["h"]]] = 1
        X[np.arange(len(d)), [idx[x] for x in d["a"]]] = -1
        X[:, -1] = 1
        # small ridge, with 3 games a team it's barely identified otherwise
        lam = 0.5 if rounds is not None else 0.01
        A = X.T @ X + lam * np.diag(np.r_[np.ones(len(teams)), 0])
        b = np.linalg.solve(A, X.T @ y)
        for tm, i in idx.items():
            out[int(tm)] = b[i]
    return pd.Series(out)


# ── benchmarks through a fitted line ──────────────────────────────────────────

def fitted(d: pd.DataFrame, col: str) -> pd.Series:
    """points ~ league mean + b * (x - league mean of x), b fitted on the seasons before t"""
    out = pd.Series(np.nan, index=d.index)
    xc = d[col] - d.groupby(["season", "league"])[col].transform("mean")
    yc = d["pts"] - d.groupby(["season", "league"])["pts"].transform("mean")
    for t in SCORED:
        tr = (d["season"] < t) & xc.notna()
        b = np.sum(xc[tr] * yc[tr]) / np.sum(xc[tr] ** 2)
        te = d["season"] == t
        out[te] = d.loc[te].groupby("league")["pts"].transform("mean") + b * xc[te]
    return out


def score(d: pd.DataFrame, col: str) -> dict:
    s = d[d["season"].isin(SCORED) & d[col].notna()]
    e = s[col] - s["pts"]
    pc = s[col] - s.groupby(["season", "league"])["pts"].transform("mean")
    ac = s["pts"] - s.groupby(["season", "league"])["pts"].transform("mean")
    return {"mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()), "r": np.corrcoef(s[col], s["pts"])[0, 1],
            "slope": np.sum(pc * ac) / np.sum(pc ** 2), "n": len(s)}


_LINEUPS = None
_TRANSFERS = None

if __name__ == "__main__":
    act = actual()
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    avail = pd.read_parquet(PROCESSED_DIR / "availability.parquet")
    params = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    elo = pd.read_parquet(PROCESSED_DIR / "elo_aug1.parquet")

    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    m["tm_comp"] = m["league"].map(LEAGUE_TM)
    g = pd.read_csv(TM / "games.csv", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                               "away_club_id"])
    g = g[g["competition_id"].isin(LEAGUE_TM.values())]
    clubs = match_clubs(m, g)
    cmap = clubs.set_index(["season", "us_team"])["tm_club"]

    gl = pd.read_csv(TM / "game_lineups.csv", usecols=["game_id", "player_id", "club_id"])
    gd = pd.read_csv(TM / "games.csv", usecols=["game_id", "date", "competition_type"])
    gd = gd[gd["competition_type"] != "national_team_competition"]
    gl = gl.merge(gd[["game_id", "date"]], on="game_id")
    gl["date"] = pd.to_datetime(gl["date"])
    _LINEUPS = gl
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    _TRANSFERS = tr.dropna(subset=["transfer_date"])

    rows = []
    for t in TARGETS:
        d = our_strength(t, cmap, war, avail, params, act)
        # last season's points per game, promoted clubs get nan here and are filled below
        prev = act[act["season"] == t - 1].set_index("team_id")
        d["last_ppg"] = d["team_id"].map(prev["pts"] / prev["matches"])
        d["promoted"] = d["last_ppg"].isna()
        d["elo"] = [elo.set_index(["season", "club_id"])["elo"].get((t, cmap.get((t, tid))), np.nan)
                    for tid in d["team_id"]]
        d["mkt"] = d["team_id"].map(odds_ratings(t, act, MARKET_ROUNDS, closing=False))
        d["close"] = d["team_id"].map(odds_ratings(t, act, None, closing=True))
        rows.append(d)
        print(f"  {season_label(t)}: {len(d)} teams, our squads cover {d['known_share'].mean():.0%} of minutes with "
              f"big five history")
    d = pd.concat(rows, ignore_index=True)
    # promoted clubs: what promoted clubs got before (points per game), from earlier seasons only
    for t in TARGETS:
        past = d[(d["season"] < t) & d["promoted"]]
        val = (past["pts"] / past["matches"]).mean() if len(past) else 1.0
        d.loc[(d["season"] == t) & d["promoted"], "last_ppg"] = val
    d["last_ppg"] = d["last_ppg"] * d["matches"]

    d["p_last"] = fitted(d, "last_ppg")
    d["p_elo"] = fitted(d, "elo")
    d["p_mkt"] = fitted(d, "mkt")
    d["p_close"] = fitted(d, "close")
    d["p_ours_fit"] = fitted(d, "strength")
    d["p_blend"] = 0.5 * d["p_ours_fit"] + 0.5 * d["p_mkt"]

    print(f"\nAugust forecasts, {season_label(SCORED[0])} to {season_label(SCORED[-1])}, points per team-season:")
    print(f"  {'method':<36} {'MAE':>5} {'RMSE':>6} {'r':>5} {'slope':>6}")
    for col, name in (("ours", "ours (no fit on t, war.py's ppx)"), ("p_ours_fit", "ours through a fitted line"),
                      ("p_last", "last season, fitted"), ("p_elo", "Elo 1 August, fitted"),
                      ("p_mkt", f"market, first {MARKET_ROUNDS} matchdays, fitted"),
                      ("p_blend", "50/50 ours + market"), ("p_close", "closing odds all season (ceiling)")):
        s = score(d, col)
        print(f"  {name:<36} {s['mae']:5.1f} {s['rmse']:6.1f} {s['r']:5.2f} {s['slope']:6.2f}")
    print("  slope = actual on predicted, league means out. 1 = right spread, > 1 = forecast too compressed")

    per = d[d["season"].isin(SCORED)].groupby("league").apply(
        lambda x: pd.Series({c: (x[c] - x["pts"]).abs().mean() for c in ("ours", "p_last", "p_elo", "p_mkt")}),
        include_groups=False)
    print("\nMAE by league:")
    print(per.round(1).to_string())
    d.to_parquet(PROCESSED_DIR / "forecast_backtest.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'forecast_backtest.parquet'}")
