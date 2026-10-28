#!/usr/bin/env python3
"""
Step 6a: availability, how much of next season will he actually play. Share of his team's league minutes.

The transfermarkt dump has no injury table, so injuries, rotation and coach's decision are all in one number
here. what we can do is predict it better than war.py's fallback (50/30/20 blend of the last 3 shares):

  features   share in s, s-1, s-2 (0 if he was in the data before but not that season), how many seasons he's
             been around, age, position group, priored rapm in s (better players get picked), share of his
             games he started, share of his team's matches he appeared in
  model      gradient boosting, target = share in s+1, only players who are in the big five in s+1
             (leaving the big five is a different question, the case study handles squads directly)
             + injury.py's expected injured share and injury history, if injury_risk.parquet exists
  test       rolling: for every target season t, fit on targets before t only, predict t. vs the blend
             (and vs the same model without the injury features, to see what they add)

Output: data/processed/availability.parquet  season (as of), player_id, proj_share, blend_share
war.py picks it up automatically if it's there.

  python src/availability.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from aging import BROAD
from box_prior import POS
from config import PROCESSED_DIR, SEASONS, season_label
from rapm import keepers
from tm_match import birth_dates
from war import lineups_season, minute_shares, projected_share, team_seasons

FEATS = ["sh0", "sh1", "sh2", "around", "age", "rapm", "start_share", "app_share",
         "g_GK", "g_DEF", "g_MID", "g_ATT"]
# from injury.py, if it's been run: expected injured share next season + the injury history behind it
INJ_FEATS = ["exp_inj_share", "inj0", "inj1", "ongoing", "long2"]


def features(lu: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    sh = minute_shares(lu, ts)
    first = sh.groupby("player_id")["season"].min()
    wide = sh.pivot(index="player_id", columns="season", values="share")
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")[["season", "player_id", "rapm", "position"]]
    dob = birth_dates()

    # starts and appearances vs his team's matches
    app = lu.groupby(["season", "player_id"]).agg(games=("match_id", "size"), starts=("starter", "sum"))
    tm_games = sh.set_index(["season", "player_id"])["avail"] / 90

    rows = []
    for s in SEASONS:
        ids = sh.loc[sh["season"] == s, "player_id"]
        d = pd.DataFrame({"season": s, "player_id": ids.to_numpy()})
        for lag in range(3):
            col = wide.get(s - lag)
            v = d["player_id"].map(col) if col is not None else pd.Series(np.nan, index=d.index)
            # around but didn't play that season -> 0. not yet around -> stays NaN (the model handles it)
            around = d["player_id"].map(first) <= s - lag
            d[f"sh{lag}"] = np.where(around, v.fillna(0), np.nan)
        d["around"] = s - d["player_id"].map(first) + 1
        d["age"] = (pd.Timestamp(f"{s + 1}-01-01") - d["player_id"].map(dob)).dt.days / 365.25
        r = rapm[rapm["season"] == s].set_index("player_id")
        d["rapm"] = d["player_id"].map(r["rapm"])
        grp = d["player_id"].map(r["position"]).map(POS).map(BROAD)
        grp[d["player_id"].isin(keepers())] = "GK"
        for g in ("GK", "DEF", "MID", "ATT"):
            d[f"g_{g}"] = (grp == g).astype(float)
        a = app.loc[s].reindex(d["player_id"])
        d["start_share"] = (a["starts"] / a["games"]).to_numpy()
        d["app_share"] = (a["games"].to_numpy() / tm_games.loc[s].reindex(d["player_id"]).to_numpy())
        d["blend"] = d["player_id"].map(projected_share(sh, s)).to_numpy()
        # target: his share next season, if he's in the big five then
        nxt = wide.get(s + 1)
        d["target"] = d["player_id"].map(nxt) if nxt is not None else np.nan
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=50, random_state=0)


if __name__ == "__main__":
    lu = lineups_season()
    ts = team_seasons()
    f = features(lu, ts)
    ipath = PROCESSED_DIR / "injury_risk.parquet"
    feats = FEATS
    if ipath.exists():
        ir = pd.read_parquet(ipath)[["season", "player_id"] + INJ_FEATS]
        f = f.merge(ir, on=["season", "player_id"], how="left")
        feats = FEATS + INJ_FEATS
    lab = f[f["target"].notna()]

    # rolling out-of-sample test. sh2 only exists from 2016-17 on (data starts 2014-15), so the first
    # target season is 2018-19, trained on as-of seasons up to 2016-17
    res = []
    for t in list(SEASONS)[4:]:
        tr = lab[lab["season"] + 1 < t]
        te = lab[lab["season"] + 1 == t].copy()
        m = model().fit(tr[feats], tr["target"])
        te["pred"] = np.clip(m.predict(te[feats]), 0, 1)
        if feats is not FEATS:
            m0 = model().fit(tr[FEATS], tr["target"])
            te["pred_noinj"] = np.clip(m0.predict(te[FEATS]), 0, 1)
        res.append(te)
    res = pd.concat(res)
    print("next-season share of team minutes, out of sample (target seasons "
          f"{season_label(res['season'].min() + 1)} to {season_label(res['season'].max() + 1)}):")
    for name, sub in (("all players", res), ("regulars (50%+ share in s)", res[res["sh0"] >= 0.5])):
        mae_m = (sub["pred"] - sub["target"]).abs().mean()
        mae_b = (sub["blend"] - sub["target"]).abs().mean()
        print(f"  {name:<28} model MAE {mae_m:.3f}   blend MAE {mae_b:.3f}   ({len(sub):,} player-seasons, "
              f"mean actual {sub['target'].mean():.3f}, model {sub['pred'].mean():.3f}, blend {sub['blend'].mean():.3f})")
    if "pred_noinj" in res:
        for name, sub in (("all players", res), ("regulars", res[res["sh0"] >= 0.5])):
            print(f"  {name}: without the injury features MAE {(sub['pred_noinj'] - sub['target']).abs().mean():.3f}")
    # in minutes over a 38-game season, that's what PAR uses
    print(f"  = about {38 * 90 * (res['pred'] - res['target']).abs().mean():.0f} vs "
          f"{38 * 90 * (res['blend'] - res['target']).abs().mean():.0f} minutes off per player")

    # where the blend goes wrong most: old regulars
    old = res[(res["age"] >= 31) & (res["sh0"] >= 0.5)]
    print(f"  31+ regulars: model {(old['pred'] - old['target']).mean():+.3f} bias, blend {(old['blend'] - old['target']).mean():+.3f}")

    # final: fit on everything with a target, predict every as-of season (incl. the last one, for 2024-25)
    m = model().fit(lab[feats], lab["target"])
    f["proj_share"] = np.clip(m.predict(f[feats]), 0, 1)
    f.rename(columns={"blend": "blend_share"})[["season", "player_id", "proj_share", "blend_share"]] \
        .to_parquet(PROCESSED_DIR / "availability.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'availability.parquet'}")
