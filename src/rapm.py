#!/usr/bin/env python3
"""
Step 3: RAPM, offense and defense, non-penalty xG per 90. The football version of rapm.py in nba-moneyball.

Every segment becomes two rows, one per team on offense:
  y      = npxG per 90 for the attacking side in that segment, minus the league-season average
           (leagues barely share players, so each league's own level is taken out before the fit)
  x      = +1 for every attacking player (offense columns), +1 for every defender (defense columns),
           a home-offense flag, and man advantage (attackers on the pitch - defenders, after reds)
  weight = segment minutes x season weight

Pooled window: values for season s are fitted on s, s-1 and s-2 with weights 1, 0.5, 0.25.
A season of football has way fewer segments per player than an NBA season has stints, one season alone
is too thin. The first two seasons (2014-15, 2015-16) just use what there is.

Ridge, lambda by cross-validation with whole matches held out (both rows of a segment, and all segments
of a match, land in the same fold). Everyone stays in the regression, minutes filters are only for looking.

Sign convention in the output: positive is good for both.
  o_rapm  npxG per 90 the team creates on top of average, with him on the pitch
  d_rapm  npxG per 90 the team concedes less with him on the pitch
  rapm    o_rapm + d_rapm

  python src/rapm.py
  python src/rapm.py --seasons 2023 --alpha 3000     # skip the CV, fixed lambda
"""

import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from config import PROCESSED_DIR, SEASONS, season_label

ALPHAS = [1000, 2000, 3000, 5000, 7500, 10000, 20000]
WINDOW = {0: 1.0, 1: 0.5, 2: 0.25}   # seasons back -> weight
# keepers play every minute, so plain RAPM dumps team-level attack on them (Neuer, Ederson, Bravo all looked
# like elite attackers). their offense columns get scaled by this, a column scaled by c costs 1/c^2 more
# penalty, so 0.1 = 100x the lambda. offense stays at the prior (~0 for keepers), defense stays free.
# tested: forward test the same with and without (3.548% vs 3.546%), top keeper O went 0.129 -> 0.055
GK_SCALE = 0.1


# ── design matrix ─────────────────────────────────────────────────────────────

def load():
    seg = pd.read_parquet(PROCESSED_DIR / "segments.parquet")
    sp = pd.read_parquet(PROCESSED_DIR / "seg_players.parquet")
    return seg, sp


_KEEPERS = None


def keepers() -> set:
    """players who started more than half their games in goal, over the whole data"""
    global _KEEPERS
    if _KEEPERS is None:
        lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["player_id", "position", "starter"])
        share = lu[lu["starter"]].groupby("player_id")["position"].agg(lambda s: (s == "GK").mean())
        _KEEPERS = set(share[share > 0.5].index)
    return _KEEPERS


def build_rows(seg: pd.DataFrame, sp: pd.DataFrame, target: str = "npxg"):
    """seg = the segments in the window, with a 'sw' season-weight column.
    returns X, y, w, groups, players, scale. X has the keeper offense columns scaled by GK_SCALE, so the real
    coefficients are fitted coef * scale (and a prior goes in as b0 / scale)"""
    seg = seg[seg["dur"] > 0].reset_index(drop=True)
    row_of = pd.Series(np.arange(len(seg)), index=seg["seg_id"])
    sp = sp[sp["seg_id"].isin(row_of.index)]
    players = np.sort(sp["player_id"].unique())
    col = pd.Series(np.arange(len(players)), index=players)
    n, n_p = len(seg), len(players)

    r = row_of[sp["seg_id"]].to_numpy()
    c = col[sp["player_id"]].to_numpy()
    home = (sp["side"] == "h").to_numpy()

    # rows 0..n-1 = home attacking, n..2n-1 = away attacking
    # home attacking: home players in offense cols, away players in defense cols. flipped for the away rows
    rr = np.concatenate([r[home], r[~home], r[~home] + n, r[home] + n])
    cc = np.concatenate([c[home], c[~home] + n_p, c[~home], c[home] + n_p])
    X_pl = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(2 * n, 2 * n_p))

    is_home = np.concatenate([np.ones(n), np.zeros(n)])
    adv = np.concatenate([seg["n_h"] - seg["n_a"], seg["n_a"] - seg["n_h"]]).astype(float)
    X = sparse.hstack([X_pl, sparse.csr_matrix(np.column_stack([is_home, adv]))]).tocsr()
    scale = np.ones(X.shape[1])
    scale[:n_p][np.isin(players, list(keepers()))] = GK_SCALE
    X = X.multiply(scale).tocsr()

    y = np.concatenate([90 * seg[f"{target}_h"] / seg["dur"], 90 * seg[f"{target}_a"] / seg["dur"]])
    w = np.concatenate([(seg["dur"] * seg["sw"]).to_numpy()] * 2)
    # league-season level out, so the ridge doesn't have to soak it up through the players
    ls = np.concatenate([seg["league"] + seg["season"].astype(str)] * 2)
    lvl = pd.Series(y * w).groupby(ls).sum() / pd.Series(w).groupby(ls).sum()
    y = y - lvl[ls].to_numpy()
    groups = np.concatenate([seg["match_id"].to_numpy()] * 2)
    return X, y, w, groups, players, scale


# ── lambda by cross-validation ────────────────────────────────────────────────

def pick_alpha(X, y, w, groups) -> tuple[float, dict]:
    scores = {}
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    for a in ALPHAS:
        errs, wsum = 0.0, 0.0
        for tr, te in folds:
            m = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr])
            errs += np.sum(w[te] * (y[te] - m.predict(X[te])) ** 2)
            wsum += w[te].sum()
        scores[a] = errs / wsum
    return min(scores, key=scores.get), scores


# ── one season ────────────────────────────────────────────────────────────────

def window(seg: pd.DataFrame, year: int) -> pd.DataFrame:
    back = year - seg["season"]
    win = seg[back.isin(list(WINDOW))].copy()
    win["sw"] = (year - win["season"]).map(WINDOW)
    return win


def player_info(year: int) -> pd.DataFrame:
    """name, minutes in season s, main team (most minutes) and league"""
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["match_id", "player_id", "player", "minutes",
                                                                    "team_id", "position", "starter"])
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season", "league", "home_id",
                                                                   "home_team", "away_id", "away_team"])
    lu = lu.merge(m[["match_id", "season", "league"]], on="match_id")
    lu = lu[lu["season"] == year]
    names = pd.concat([m.set_index("home_id")["home_team"], m.set_index("away_id")["away_team"]])
    names = names[~names.index.duplicated()]
    tm = lu.groupby(["player_id", "team_id", "league"])["minutes"].sum().reset_index() \
        .sort_values("minutes").drop_duplicates("player_id", keep="last").set_index("player_id")
    out = lu.groupby("player_id").agg(name=("player", "last"), minutes=("minutes", "sum"), games=("match_id", "size"))
    out["team"] = tm["team_id"].map(names)
    out["league"] = tm["league"]
    # usual starting position, subs are only listed as "Sub"
    pos = lu[lu["starter"]].groupby("player_id")["position"].agg(lambda s: s.value_counts().index[0])
    out["position"] = pos
    return out


def fit_season(seg, sp, year: int, alpha: float | None, target: str = "npxg") -> pd.DataFrame:
    X, y, w, groups, players, scale = build_rows(window(seg, year), sp, target)
    if alpha is None:
        alpha, scores = pick_alpha(X, y, w, groups)
        base = np.sum(w * y ** 2) / w.sum()   # y is already centered, so 0 is the "league average" guess
        edge = " (edge of the grid!)" if alpha in (ALPHAS[0], ALPHAS[-1]) else ""
        print(f"\n{season_label(year)}: lambda {alpha}{edge}  cv mse {scores[alpha]:.4f} (baseline {base:.4f})")
    else:
        print(f"\n{season_label(year)}: lambda {alpha} (fixed)")

    m = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    n_p = len(players)
    coef = m.coef_ * scale
    print(f"  home: {coef[-2]:+.3f} {target}/90   man advantage: {coef[-1]:+.3f} per extra player")

    out = pd.DataFrame({"season": year, "player_id": players,
                        "o_rapm": coef[:n_p], "d_rapm": -coef[n_p:2 * n_p]})
    out["rapm"] = out["o_rapm"] + out["d_rapm"]
    out["lambda"] = alpha
    out = out.join(player_info(year), on="player_id")
    # players only in the earlier seasons of the window have no row in season s, they're not his values
    out = out[out["minutes"].notna()]

    show = out[out["minutes"] >= 1500].sort_values("rapm", ascending=False)
    print("  top 10 (1500+ min):")
    print(show[["name", "team", "position", "minutes", "o_rapm", "d_rapm", "rapm"]].head(10).round(3)
          .to_string(index=False))
    return out


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    seg, sp = load()
    res = pd.concat([fit_season(seg, sp, y, args.alpha) for y in args.seasons], ignore_index=True)
    out = PROCESSED_DIR / "rapm.parquet"
    # rerunning a few seasons only replaces those, keeps the rest of the file
    if out.exists():
        old = pd.read_parquet(out)
        res = pd.concat([old[~old["season"].isin(args.seasons)], res], ignore_index=True)
    res.sort_values(["season", "rapm"], ascending=[True, False]).to_parquet(out, index=False)
    print(f"\nsaved {len(res):,} player-seasons -> {out}")
