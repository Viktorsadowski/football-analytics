#!/usr/bin/env python3
"""
Step 5a: keeper shot-stopping. RAPM is on npxG, and xG is measured before the shot, so a keeper who saves
what he "shouldn't" gets nothing for it there. this is that missing part, in goals per 90.

Per keeper-season, pooled over the rapm window (s, s-1, s-2 with 1, .5, .25):
  who's in goal   the starting keeper until his minutes run out (sub or red), the sub keeper after that
  raw rate        (npxG faced - non-pen goals conceded) per 90, minus the average keeper
  noise           sum xg(1-xg) of the shots faced (binomial variance), per 90^2
  signal          variance of the raw rates across keepers (900+ min) - average noise (method of moments)
  save90          raw * signal / (signal + noise)   (empirical bayes shrinkage towards the average keeper)

Check: save90 at s vs the raw rate at s+3. windows don't overlap, so it's an honest test that it repeats.

Output: data/processed/gk_saves.parquet  season, player_id, npxg_faced, goals, minutes, raw90, save90

  python src/gk.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, SEASONS
from rapm import WINDOW, keepers


def shots_faced(lu: pd.DataFrame) -> pd.DataFrame:
    sh = pd.read_parquet(PROCESSED_DIR / "shots.parquet")
    sh = sh[sh["is_pen"] == 0].rename(columns={"player_id": "shooter"})
    k = lu[lu["player_id"].isin(keepers())]
    st = k[k["starter"]].drop_duplicates(["match_id", "h_a"])[["match_id", "h_a", "player_id", "minutes",
                                                               "roster_in", "red_card"]]
    sb = k[~k["starter"]].drop_duplicates(["match_id", "h_a"])[["match_id", "h_a", "player_id"]] \
        .rename(columns={"player_id": "sub_gk"})
    st = st.merge(sb, on=["match_id", "h_a"], how="left").rename(columns={"h_a": "def_side"})
    # shots by the attacking side are faced by the other side's keeper
    sh["def_side"] = np.where(sh["h_a"] == "h", "a", "h")
    sh = sh.merge(st, on=["match_id", "def_side"])
    gone = ((sh["roster_in"] > 0) | (sh["red_card"] > 0)) & (sh["minute"] > sh["minutes"]) & sh["sub_gk"].notna()
    sh["gk"] = np.where(gone, sh["sub_gk"], sh["player_id"]).astype(int)
    return sh.merge(lu[["match_id", "season"]].drop_duplicates(), on="match_id")


def gk_table(lu: pd.DataFrame) -> pd.DataFrame:
    sh = shots_faced(lu)
    sh["noise"] = sh["xg"] * (1 - sh["xg"])
    per = sh.groupby(["season", "gk"]).agg(npxg_faced=("xg", "sum"), goals=("is_goal", "sum"),
                                           noise=("noise", "sum"), shots=("xg", "size")).reset_index()
    mins = lu[lu["player_id"].isin(keepers())].groupby(["season", "player_id"])["minutes"].sum()
    per["minutes"] = [mins.get((s, g), 0) for s, g in zip(per["season"], per["gk"])]
    per = per[per["minutes"] > 0]

    out = []
    for s in SEASONS:
        w = per[per["season"].isin([s - b for b in WINDOW])].copy()
        w["sw"] = (s - w["season"]).map(WINDOW)
        g = w.assign(**{c: w[c] * w["sw"] for c in ("npxg_faced", "goals", "noise", "minutes")}) \
            .groupby("gk")[["npxg_faced", "goals", "noise", "minutes"]].sum()
        g = g[g["minutes"] > 0]
        g["raw90"] = 90 * (g["npxg_faced"] - g["goals"]) / g["minutes"]
        # xG models aren't perfectly calibrated on keepers, so relative to the average keeper
        mu = (g["npxg_faced"] - g["goals"]).sum() / g["minutes"].sum() * 90
        g["raw90"] -= mu
        g["noise90"] = g["noise"] * (90 / g["minutes"]) ** 2
        big = g[g["minutes"] >= 900]
        signal = max(np.average(big["raw90"] ** 2, weights=big["minutes"])
                     - np.average(big["noise90"], weights=big["minutes"]), 1e-6)
        g["save90"] = g["raw90"] * signal / (signal + g["noise90"])
        g["season"] = s
        g["signal_sd"] = np.sqrt(signal)
        out.append(g.reset_index().rename(columns={"gk": "player_id"}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet")
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season"])
    lu = lu.merge(m, on="match_id")
    gks = gk_table(lu)
    yy = gks.merge(gks.assign(season=gks["season"] - 3)[["season", "player_id", "raw90"]],
                   on=["season", "player_id"], suffixes=("", "_later"))
    yy = yy[yy["minutes"] >= 1500]
    print(f"keeper shot-stopping: true spread (sd) {gks['signal_sd'].median():.3f} goals per 90 "
          f"(~{gks['signal_sd'].median() * 38:.1f} goals a season)")
    print(f"  shrunk at s vs raw at s+3: r = {np.corrcoef(yy['save90'], yy['raw90_later'])[0, 1]:.2f} "
          f"({len(yy)} keeper pairs, 1500+ weighted min)")
    big = gks[(gks["season"] == max(SEASONS)) & (gks["minutes"] >= 1500)].sort_values("save90", ascending=False)
    names = lu.groupby("player_id")["player"].last()
    big["name"] = big["player_id"].map(names)
    print("\nbest and worst shot-stoppers as of the last season:")
    print(pd.concat([big.head(5), big.tail(3)])[["name", "minutes", "raw90", "save90"]].round(3).to_string(index=False))
    gks.to_parquet(PROCESSED_DIR / "gk_saves.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'gk_saves.parquet'}")
