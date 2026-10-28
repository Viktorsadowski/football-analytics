#!/usr/bin/env python3
"""
Step 5d: what do defensive and passing stats add, by position. The two seasons with full event data
(event_stats.py: StatsBomb 2015-16, Wyscout 2017-18) against plain RAPM (rapm_plain.parquet, so the prior
can't leak into the target).

  1. which stats matter  per position group (DEF, MID, ATT), weighted ridge of plain O-RAPM and D-RAPM on
                         standardised per 90 stats: the understat box stats we already use + the event stats.
                         coefficients per position = what a stat is worth for that position
  2. do they add         fitted on one event season, tested on the other (2015-16 <-> 2017-18), R2 with and
                         without the event stats. the out-of-season version is what a prior needs
  3. accuracy            the real test: box_prior.py's priored RAPM with a prior that has the event stats
                         (fitted on the OTHER event season) vs the current prior, forward test on the season after
                         (2016-17 and 2018-19). pass mark set before running: better in both seasons

Only common stats in 1-3 (both sources have them): passes, pass%, progressive passes, tackles, tackles won,
interceptions, clearances, aerial duels won + %, completed dribbles, fouls.
The two sources count differently (wyscout calls every defending duel a tackle: 6 per 90, statsbomb ~2), so
every event stat is turned into a z-score within its own season (= source) first, among players with 900+ min.
without that, fitting on one season and testing on the other compares two different scales (first run: R2 -1.9)

  python src/defense_study.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from box_prior import FEATS, box_features, fit_season_prior, forward_error, fit_prior
from config import PROCESSED_DIR
from rapm import load

EV = ["passes_90", "pass_pct", "prog_passes_90", "tackles_90", "tackles_won_90", "interceptions_90",
      "clearances_90", "aerials_won_90", "aerial_pct", "dribbles_cmp_90", "fouls_90"]
GROUPS = {"DEF": ["CB", "FB"], "MID": ["DM", "CM"], "ATT": ["W", "AM", "FW"]}
MIN_MIN = 900
SEASONS_EV = [2015, 2017]


def zscore(ev: pd.DataFrame) -> pd.DataFrame:
    """event stats as z-scores within season (= source), mean / sd from players with MIN_MIN+ minutes"""
    ev = ev.copy()
    for s, g in ev.groupby("season"):
        ref = g[g["minutes"] >= MIN_MIN]
        for c in EV:
            ev.loc[g.index, c] = (g[c] - ref[c].mean()) / ref[c].std()
    return ev


def table() -> pd.DataFrame:
    plain = pd.read_parquet(PROCESSED_DIR / "rapm_plain.parquet")
    f = box_features(plain)
    ev = zscore(pd.read_parquet(PROCESSED_DIR / "event_stats.parquet"))
    d = f.merge(ev[["season", "player_id"] + EV], on=["season", "player_id"], how="inner")
    d["g"] = d["grp"].map({p: g for g, ps in GROUPS.items() for p in ps})
    return d[d["minutes"] >= MIN_MIN].dropna(subset=["g", "o_rapm"])


def fit_eval(tr, te, feats, target):
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    w = tr["minutes"].clip(upper=3000)
    m = Ridge(alpha=5.0).fit((tr[feats] - mu) / sd, tr[target], sample_weight=w)
    p = m.predict((te[feats] - mu) / sd)
    wt = te["minutes"].clip(upper=3000)
    ss = np.sum(wt * (te[target] - np.average(te[target], weights=wt)) ** 2)
    return 1 - np.sum(wt * (te[target] - p) ** 2) / ss, pd.Series(m.coef_, index=feats)


if __name__ == "__main__":
    d = table()
    print(f"players with event stats and {MIN_MIN}+ min: " +
          ", ".join(f"{s}: {n}" for s, n in d.groupby("season").size().items()))
    box = [c for c in FEATS if not c.startswith("pos_")]

    print("\n1-2. out of season R2 (fit on one event season, test on the other, averaged both ways)")
    coefs = {}
    for g in GROUPS:
        for target in ("o_rapm", "d_rapm"):
            r_box, r_all, cs = [], [], []
            for a, b in ((2015, 2017), (2017, 2015)):
                tr, te = d[(d["season"] == a) & (d["g"] == g)], d[(d["season"] == b) & (d["g"] == g)]
                r_box.append(fit_eval(tr, te, box, target)[0])
                r, c = fit_eval(tr, te, box + EV, target)
                r_all.append(r)
                cs.append(c)
            coefs[(g, target)] = pd.concat(cs, axis=1).mean(axis=1)
            print(f"  {g} {target[0].upper()}-RAPM: box stats {np.mean(r_box):.3f} -> with event stats {np.mean(r_all):.3f}")

    print("\nwhat a stat is worth, standardised ridge coefficient (1 sd of the stat, npxG per 90), both seasons:")
    show = EV + ["npxg90", "xa90", "xg_chain90", "xg_buildup90", "key_passes90"]
    tab = pd.DataFrame({f"{g} {t[0].upper()}": coefs[(g, t)].reindex(show) for g in GROUPS for t in ("o_rapm", "d_rapm")})
    print((1000 * tab).round(1).to_string())
    print("  (x 1000, so 5.0 = 0.005 npxG per 90 for one sd more of that stat)")

    # 3. the accuracy test: priored RAPM with the event stats in the prior
    print("\n3. forward test, current prior vs a prior with the event stats (fitted on the other event season)")
    seg, sp = load()
    plain = pd.read_parquet(PROCESSED_DIR / "rapm_plain.parquet")
    f = box_features(plain)
    base_prior = fit_prior(f)
    ev = zscore(pd.read_parquet(PROCESSED_DIR / "event_stats.parquet"))
    full = f.merge(ev[["season", "player_id"] + EV], on=["season", "player_id"], how="left")
    feats_all = FEATS + EV
    for s in SEASONS_EV:
        other = [x for x in SEASONS_EV if x != s][0]
        tr = full[(full["season"] == other) & (full["minutes"] >= MIN_MIN)].dropna(subset=EV + ["o_rapm"])
        te = full[full["season"] == s]
        has = te[EV].notna().all(axis=1)
        mu, sd = tr[feats_all].mean(), tr[feats_all].std().replace(0, 1)
        pr = base_prior[base_prior["season"].between(s - 2, s)].copy()
        new = pr.copy()
        for side in ("o", "d"):
            m = Ridge(alpha=1.0).fit((tr[feats_all] - mu) / sd, tr[f"{side}_rapm"], sample_weight=tr["minutes"].clip(upper=3000))
            p = pd.Series(m.predict((te.loc[has, feats_all] - mu) / sd), index=te.loc[has, "player_id"])
            idx = (new["season"] == s) & new["player_id"].isin(p.index)
            new.loc[idx, f"{side}_prior"] = new.loc[idx, "player_id"].map(p).to_numpy()
        res = {}
        for name, prior in (("current prior", pr), ("prior + event stats", new)):
            v = fit_season_prior(seg, sp, s, prior)
            res[name] = forward_error(v, "o_rapm", "d_rapm", seg, sp, seasons=[s])
        print(f"  {s}-{str(s + 1)[-2:]} values -> {s + 1}-{str(s + 2)[-2:]}: current {res['current prior']['gain']:.3%} "
              f"explained, with event stats {res['prior + event stats']['gain']:.3%} "
              f"({has.sum():,} players had event stats)")
