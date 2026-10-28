#!/usr/bin/env python3
"""
Step 8: what is every player worth, against what the market says. Football version of surplus.py in
nba-moneyball, without contracts and wages (not in the public data, see the notes at the bottom).

  model value  as of the end of season s (= the summer window before s+1):
                 year 1   war.py's projected PAR for s+1 (expected injuries in, availability.py minutes)
                 year k   same player moved k-1 more years along his group's aging curve, same minutes share,
                          floored at 0 (a club can always bench him). floored as E[max(PAR, 0)] with the projection
                          spread (sd grows with sqrt(k), fitted on how much values move season to season), so an
                          uncertain young player keeps his upside, like the option years in nba-moneyball.
                          times the chance he's still playing in the
                          big five k-1 years on (survival(), by age and quality). without it the model gave a
                          40-year-old (Dante) four full seasons and called him the most underpriced player
               sum over HORIZON seasons of PAR x euros per point (price.py, EPL prize money, average over
               35-85 points), discounted DISCOUNT a year. this is the value to an average Premier League
               club of his points. for players in other leagues it's his points against HIS league's
               replacement level (no league strength yet, see config/aging notes), so cross-league
               comparisons are rough
  market value transfermarkt value at the start of s+1 (1 Sept)
  gap          log((model + 1m) / (market + 1m)). above 0 = the model likes him more than the market does.
               the 1m on both sides keeps players worth ~0 from blowing up the log

Test (the point of it): if the model knows something the market doesn't, the market should move towards
the model. gap at s vs the change in log market value from the start of s+1 to the start of s+2, with age
and the market value level taken out (young players rise anyway, expensive ones regress). out of sample by
construction, the gap only uses data up to s.

Output: data/processed/market_value.parquet  one row per player per as-of season

  python src/market_value.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, SEASONS, season_label
from tm_match import start_values
from war import age_step

HORIZON = 4
DISCOUNT = 0.08
PAD = 1e6


def survival(war: pd.DataFrame, horizon: int = HORIZON) -> dict:
    """k -> logistic fit of P(still in the big five data k seasons later) on age, age^2, PAR. players whose club
    went down the season after are left out (a buyer takes him out of that club, relegation isn't decline).
    horizon > 4 for transfer_value.py, contracts run up to 6 years"""
    from sklearn.linear_model import LogisticRegression
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["season", "home_team"])
    teams = m.groupby("season")["home_team"].apply(set)
    seen = set(zip(war["season"], war["player_id"]))
    base = war[war["age"].notna() & (war["proj_min"] >= 900)].copy()
    base = base[[t in teams.get(s + 1, set()) for s, t in zip(base["season"], base["team"])]]
    fits = {}
    for k in range(1, horizon):
        b = base[base["season"] + k <= max(SEASONS)]
        y = np.array([(s + k, p) in seen for s, p in zip(b["season"], b["player_id"])])
        X = np.column_stack([b["age"], b["age"] ** 2, b["par"]])
        fits[k] = LogisticRegression(C=10, max_iter=1000).fit(X, y)
    return fits


def surv_prob(fits, k, age, par) -> np.ndarray:
    if k == 0:
        return np.ones(len(age))
    X = np.column_stack([age, age ** 2, par])
    return fits[k].predict_proba(np.nan_to_num(X, nan=27.0))[:, 1]


def eur_per_point() -> float:
    e = pd.read_parquet(PROCESSED_DIR / "eur_per_point.parquet")
    return float(e[e["pts"].between(35, 85)]["eur_per_pt"].mean())


def value_sd(war: pd.DataFrame) -> float:
    """sd of next season's calibrated value around this season's projection, regulars only (npxG per 90)"""
    a = war[["season", "player_id", "val_cal", "proj_min"]]
    b = war[["season", "player_id", "val_cal"]].assign(season=war["season"] - 1)
    j = a.merge(b, on=["season", "player_id"], suffixes=("", "_next"))
    j = j[j["proj_min"] >= 1500]
    return float((j["val_cal_next"] - j["val_cal"]).std())


def floored(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """E[max(X, 0)] for X ~ Normal(mu, sd)"""
    from scipy.stats import norm
    z = mu / np.maximum(sd, 1e-9)
    return mu * norm.cdf(z) + sd * norm.pdf(z)


def model_values(war: pd.DataFrame, curve: pd.DataFrame, a: float, ppx: float, eur: float, fits,
                 sd_val: float) -> pd.DataFrame:
    d = war.copy()
    total = np.zeros(len(d))
    for k in range(1, HORIZON + 1):
        if k == 1:
            par = d["par"].to_numpy()
        else:
            # more aging on top of what war.py already did (that took him to age + 1)
            step = age_step(curve, d["grp"], d["age"] + 1, d["age"] + k)
            val = d["val_cal"] + a * np.where(d["age"].notna(), step, 0.0)
            par = ((val - d["repl"]) * d["proj_min"] / 90 * ppx).to_numpy()
        d[f"surv_y{k}"] = surv_prob(fits, k - 1, d["age"].to_numpy(), d["par"].to_numpy())
        sd = sd_val * np.sqrt(k) * d["proj_min"].fillna(0).to_numpy() / 90 * ppx
        d[f"par_y{k}"] = floored(np.nan_to_num(par), sd) * d[f"surv_y{k}"]
        total += d[f"par_y{k}"] / (1 + DISCOUNT) ** (k - 1)
    d["pv_par"] = total
    d["model_eur"] = total * eur * 1e6
    return d


if __name__ == "__main__":
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    curve = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    params = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    eur = eur_per_point()
    print(f"euros per point: {eur:.2f}m (EPL prize money, average 35-85 points), horizon {HORIZON} seasons, "
          f"discount {DISCOUNT:.0%}")

    fits = survival(war)
    for age in (24, 30, 33, 36, 39):
        pr = [surv_prob(fits, k, np.array([age]), np.array([3.0]))[0] for k in range(1, HORIZON)]
        print(f"  still in the big five 1/2/3 years on, age {age}, PAR 3: " + " / ".join(f"{p:.2f}" for p in pr))
    sd_val = value_sd(war)
    print(f"  season-to-season sd of value: {sd_val:.3f} npxG per 90 (regulars), grows with sqrt(years)")
    d = model_values(war, curve, params["a"], params["ppx"], eur, fits, sd_val)
    mv = {s: start_values(s) for s in list(SEASONS) + [max(SEASONS) + 1, max(SEASONS) + 2]}
    d["mv"] = [mv[s + 1].get(p, np.nan) for s, p in zip(d["season"], d["player_id"])]
    d["mv_next"] = [mv[s + 2].get(p, np.nan) if (s + 2) in mv else np.nan for s, p in zip(d["season"], d["player_id"])]
    ok = d["mv"] > 0
    d["gap"] = np.where(ok, np.log((d["model_eur"].where(ok) + PAD) / (d["mv"].where(ok) + PAD)), np.nan)

    # ── the test: does the market move towards the model ──
    t = d[ok & (d["mv_next"] > 0) & d["age"].notna() & (d["proj_min"] >= 900)].copy()
    t["dlog"] = np.log(t["mv_next"] / t["mv"])
    X = np.column_stack([np.ones(len(t)), t["gap"], t["age"], t["age"] ** 2, np.log(t["mv"])])
    b, *_ = np.linalg.lstsq(X, t["dlog"], rcond=None)
    # same without the gap, how much does it add
    X0 = X[:, [0, 2, 3, 4]]
    b0, *_ = np.linalg.lstsq(X0, t["dlog"], rcond=None)
    r2 = 1 - np.sum((t["dlog"] - X @ b) ** 2) / np.sum((t["dlog"] - t["dlog"].mean()) ** 2)
    r20 = 1 - np.sum((t["dlog"] - X0 @ b0) ** 2) / np.sum((t["dlog"] - t["dlog"].mean()) ** 2)
    # per season, so one season can't carry it
    per = {}
    for s, g in t.groupby("season"):
        Xs = np.column_stack([np.ones(len(g)), g["gap"], g["age"], g["age"] ** 2, np.log(g["mv"])])
        per[s] = np.linalg.lstsq(Xs, g["dlog"], rcond=None)[0][1]
    print(f"\ndoes the market move towards the model? {len(t):,} player-seasons (900+ projected min)")
    print(f"  next year's change in log market value on the gap: {b[1]:+.3f} per unit of gap "
          f"(R2 {r20:.3f} -> {r2:.3f} with the gap)")
    print(f"  per as-of season: " + ", ".join(f"{season_label(s)} {v:+.3f}" for s, v in per.items()))
    q = pd.qcut(t["gap"], 5, labels=["most overpriced", "2", "3", "4", "most underpriced"])
    resid = t["dlog"] - X0 @ b0
    print("  market value change next year by gap quintile (age and level taken out):")
    print("  " + "   ".join(f"{k}: {100 * (np.exp(v) - 1):+.1f}%" for k, v in resid.groupby(q, observed=True).mean().items()))

    d.to_parquet(PROCESSED_DIR / "market_value.parquet", index=False)

    last = d[(d["season"] == max(SEASONS)) & ok & (d["proj_min"] >= 1500)].copy()
    cols = ["name", "team", "grp", "age", "par", "pv_par", "model_eur", "mv", "gap"]
    fmt = last.assign(model_eur=last["model_eur"] / 1e6, mv=last["mv"] / 1e6)
    for lg in ("EPL", None):
        sub = fmt if lg is None else fmt[fmt["league"] == lg]
        label = "all five leagues" if lg is None else lg
        print(f"\nsummer {max(SEASONS) + 1}, {label}, most underpriced by the model (EUR m, 1500+ projected min):")
        print(sub.sort_values("gap", ascending=False)[cols].head(12).round(2).to_string(index=False))
        print(f"most overpriced:")
        print(sub.sort_values("gap")[cols].head(8).round(2).to_string(index=False))
    print(f"\nsaved -> {PROCESSED_DIR / 'market_value.parquet'}")
    print("\nnote: no wages in here. a high-gap player can still be a bad deal if his wages eat the gap, and the")
    print("market value is the price of the player, not of his salary. the report says this next to every list")
