#!/usr/bin/env python3
"""
Step 6c: injury risk, the share of next season's league games a player is expected to miss injured.
Built on the spells from inferred_injuries.py (validated there, see the notes at the top of that file).

Per player-season (transfermarkt seasons = start year, same as understat):
  inj_share   league games he missed injured / his club's league games. a spell's league games are put in the
              season they happen in, so a spring ACL counts in both seasons
  spells      number of spells starting that season, long = any spell of 10+ games
  ongoing     injured at the last league game of the season (carries into the next one, the strongest signal)

Model: next season's inj_share from age, position group, inj_share in s, s-1, s-2, spells, long injury in the
last 2 seasons, ongoing, share of team minutes in s (load), rapm. gradient boosting, rolling out of sample like
availability.py. two baselines: the average for his age band, and last season's share pulled halfway to it.

Only regulars get spells detected (see inferred_injuries.py), so for fringe players this is a lower bound.

Output: data/processed/injury_risk.parquet  season (as of), player_id (understat), exp_inj_share, + history
        availability.py and war.py pick it up if it's there.

  python src/injury.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from aging import BROAD
from box_prior import POS
from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from inferred_injuries import BIG5
from rapm import keepers
from tm_match import birth_dates, us_to_tm

TM = RAW_DIR / "transfermarkt"
LONG = 10
FEATS = ["age", "inj0", "inj1", "inj2", "spells0", "long2", "ongoing", "share0", "rapm",
         "g_GK", "g_DEF", "g_MID", "g_ATT"]


def league_games_missed() -> pd.DataFrame:
    """per (tm player, season): league games missed injured, spells started, long spells, ongoing at season end"""
    sp = pd.read_parquet(PROCESSED_DIR / "injuries.parquet")
    g = pd.read_csv(TM / "games.csv", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                               "away_club_id"])
    g["date"] = pd.to_datetime(g["date"])
    lg = g[g["competition_id"].isin(BIG5)]
    club_games = pd.concat([lg[["season", "date", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                            lg[["season", "date", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    cg = {c: d.sort_values("date") for c, d in club_games.groupby("club_id")}
    last_game = club_games.groupby(["club_id", "season"])["date"].max()

    rows = []
    for r in sp.itertuples():
        d = cg.get(r.club_id)
        if d is None:
            continue
        miss = d[(d["date"] >= r.start_date) & (d["date"] < r.first_back)]
        for s, n in miss.groupby("season").size().items():
            end = last_game.get((r.club_id, s))
            rows.append({"tm_id": r.player_id, "season": s, "club_id": r.club_id, "lg_missed": n,
                         "started": int(r.season == s), "long": int(r.games >= LONG and r.season == s),
                         "ongoing": int(end is not None and r.start_date <= end < r.first_back)})
    t = pd.DataFrame(rows)
    out = t.groupby(["tm_id", "season"]).agg(lg_missed=("lg_missed", "sum"), spells=("started", "sum"),
                                             long=("long", "max"), ongoing=("ongoing", "max")).reset_index()
    # his club's league games that season, for the share (the club he was injured at, the main one if two)
    n = club_games.groupby(["club_id", "season"]).size()
    main = t.sort_values("lg_missed").drop_duplicates(["tm_id", "season"], keep="last").set_index(["tm_id", "season"])["club_id"]
    out["club_games"] = [n.get((main.get((p, s)), s), 38) for p, s in zip(out["tm_id"], out["season"])]
    out["inj_share"] = (out["lg_missed"] / out["club_games"]).clip(upper=1)
    return out


def at_big5(season: int) -> set:
    """tm ids in a big five club's matchday squad at least once in that season (any competition)"""
    g = pd.read_csv(TM / "games.csv", usecols=["game_id", "competition_id", "season", "home_club_id", "away_club_id"])
    lg = g[g["competition_id"].isin(BIG5) & (g["season"] == season)]
    clubs = set(lg["home_club_id"]) | set(lg["away_club_id"])
    gl = pd.read_csv(TM / "game_lineups.csv", usecols=["game_id", "player_id", "club_id"])
    gl = gl[gl["club_id"].isin(clubs) & gl["game_id"].isin(g.loc[g["season"] == season, "game_id"])]
    return set(gl["player_id"])


def features() -> pd.DataFrame:
    inj = league_games_missed().set_index(["tm_id", "season"])
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")[["season", "player_id", "rapm", "position", "minutes"]]
    tm = us_to_tm()
    dob = birth_dates()
    from war import lineups_season, minute_shares, team_seasons
    shares = minute_shares(lineups_season(), team_seasons()).set_index(["season", "player_id"])["share"]

    rows = []
    for s in SEASONS:
        d = rapm[rapm["season"] == s].copy()
        d["tm_id"] = d["player_id"].map(tm)
        d = d.dropna(subset=["tm_id"])
        d["tm_id"] = d["tm_id"].astype(int)

        def get(col, lag):
            return [inj[col].get((p, s - lag), 0) for p in d["tm_id"]]
        for lag in range(3):
            d[f"inj{lag}"] = get("inj_share", lag)
        d["spells0"] = get("spells", 0)
        d["long2"] = np.maximum(get("long", 0), get("long", 1))
        d["ongoing"] = get("ongoing", 0)
        d["share0"] = [shares.get((s, p), 0) for p in d["player_id"]]
        d["age"] = (pd.Timestamp(f"{s + 1}-01-01") - d["player_id"].map(dob)).dt.days / 365.25
        grp = d["position"].map(POS).map(BROAD)
        grp[d["player_id"].isin(keepers())] = "GK"
        for gname in ("GK", "DEF", "MID", "ATT"):
            d[f"g_{gname}"] = (grp == gname).astype(float)
        d["target"] = [inj["inj_share"].get((p, s + 1), 0) for p in d["tm_id"]]
        d["season"] = s
        rows.append(d)
    f = pd.concat(rows, ignore_index=True)
    # target only for players still at a big five club next season (transfermarkt squads, so a player injured
    # the whole next season still counts, he'd drop out of the understat lineups)
    f["has_next"] = False
    for s in SEASONS:
        nxt = at_big5(s + 1)
        idx = f["season"] == s
        f.loc[idx, "has_next"] = f.loc[idx, "tm_id"].isin(nxt)
    return f


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=80, random_state=0)


if __name__ == "__main__":
    f = features()
    lab = f[f["has_next"]].copy()
    lab["age_band"] = pd.cut(lab["age"], [0, 21, 24, 27, 30, 33, 50])

    # what the data says before any model: injured share next season by age, and does it repeat
    print("injured share of league games next season, by age band (regulars and fringe together):")
    print("  " + "   ".join(f"{b}: {v:.3f}" for b, v in lab.groupby("age_band", observed=True)["target"].mean().items()))
    print(f"  r(injured share s, s+1) = {np.corrcoef(lab['inj0'], lab['target'])[0, 1]:.2f}, "
          f"ongoing at season end -> next season {lab.loc[lab['ongoing'] == 1, 'target'].mean():.3f} vs "
          f"{lab.loc[lab['ongoing'] == 0, 'target'].mean():.3f}")

    res = []
    for t in list(SEASONS)[4:]:
        tr = lab[lab["season"] + 1 < t]
        te = lab[lab["season"] + 1 == t].copy()
        m = model().fit(tr[FEATS], tr["target"])
        te["pred"] = np.clip(m.predict(te[FEATS]), 0, 1)
        band = tr.groupby("age_band", observed=True)["target"].mean()
        te["base_age"] = te["age_band"].map(band).astype(float).fillna(tr["target"].mean())
        te["base_last"] = 0.5 * te["inj0"] + 0.5 * te["base_age"]
        res.append(te)
    res = pd.concat(res)
    print(f"\nout of sample, target seasons {season_label(res['season'].min() + 1)} to "
          f"{season_label(res['season'].max() + 1)} ({len(res):,} player-seasons):")
    for name, col in (("model", "pred"), ("age average", "base_age"), ("last season + age, 50/50", "base_last")):
        e = res[col] - res["target"]
        print(f"  {name:<26} MAE {e.abs().mean():.4f}   RMSE {np.sqrt((e ** 2).mean()):.4f}   "
              f"= {38 * e.abs().mean():.2f} league games off per player")
    # calibration: predicted vs actual by decile of the prediction
    res["dec"] = pd.qcut(res["pred"], 10, labels=False, duplicates="drop")
    cal = res.groupby("dec").agg(pred=("pred", "mean"), actual=("target", "mean"))
    print("  calibration by decile (predicted / actual, in league games of 38):")
    print("  " + "  ".join(f"{38 * a:.1f}/{38 * b:.1f}" for a, b in zip(cal["pred"], cal["actual"])))

    m = model().fit(lab[FEATS], lab["target"])
    f["exp_inj_share"] = np.clip(m.predict(f[FEATS]), 0, 1)
    f[["season", "player_id", "tm_id", "exp_inj_share", "inj0", "inj1", "inj2", "spells0", "long2", "ongoing"]] \
        .to_parquet(PROCESSED_DIR / "injury_risk.parquet", index=False)
    show = f[(f["season"] == max(SEASONS)) & (f["share0"] >= 0.5)].sort_values("exp_inj_share", ascending=False)
    names = pd.read_parquet(PROCESSED_DIR / "rapm.parquet").set_index(["season", "player_id"])["name"]
    show["name"] = [names.get((s, p)) for s, p in zip(show["season"], show["player_id"])]
    print(f"\nhighest injury risk for {season_label(max(SEASONS) + 1)}, regulars (50%+ share):")
    print(show[["name", "age", "inj0", "inj1", "ongoing", "exp_inj_share"]].head(10).round(2).to_string(index=False))
    print(f"\nsaved -> {PROCESSED_DIR / 'injury_risk.parquet'}")
