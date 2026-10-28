#!/usr/bin/env python3
"""
Step 10: what a player is worth to a Premier League buyer once wages, contract length and resale are in. the
nba-moneyball surplus.py idea: value = what he gives you - what he costs you, over the deal.

  contract     L years by age at signing (contracts.parquet, wages.py). he's sold with one year left, so H = L - 1
               seasons at the buyer (min 1)
  contribution year k: projected PAR, aged along his group's curve, floored + survival like market_value.py,
               x euros per point (EPL prize money). minutes = a role share of the season x (1 - expected injury share),
               or his own projected minutes if no role is given
  wages        signing wage from wages.py's player model (market value now, age, buyer's elo, EPL), flat over the
               contract, + employer NI. minus what a replacement-level player would cost the same club, since PAR is
               already over replacement and you'd have to pay someone to play there anyway
  resale       end of year H: what the market would pay, from a fit of log market value on PAR, age, position
               (EPL players), at his projected PAR and age then, x survival
  fee          for a target: what the market charges, fit of log fee on log market value, age and an EPL buyer
               dummy (big five summer moves 2015-23). for a real deal the actual fee
  npv          sum over k of discount^(k-1) x (contribution_k - wage_k) + discount^H x resale
  surplus      npv - fee. max fee = npv

Check (pass mark set before running): EPL summer signings 2015-20, predicted surplus vs realised surplus over
3 seasons (realised PAR at whatever big five club, model wages, market value 3 years on as the resale, actual fee).
rank correlation has to beat the old no-wage version (market_value.py model value - fee) and plain mv - fee.
the realised side uses model wages too, so the wage part isn't tested by this, only the contribution/resale part
and how they're weighed against the fee

Output: data/processed/transfer_value.parquet  as-of 2023 (= summer 2024), every player, United as the buyer

  python src/transfer_value.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS
from market_value import DISCOUNT, eur_per_point, floored, surv_prob, survival, value_sd
from tm_match import start_values, us_to_tm
from war import age_step
from wages import player_wage

TM = RAW_DIR / "transfermarkt"
EMPLOYER_NI = 0.138          # uk employer national insurance on top of gross wages
FULL = 38 * 90
UNITED = 985
MAX_YEARS = 6


def load():
    mv = pd.read_parquet(PROCESSED_DIR / "market_value.parquet")
    curve = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    params = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    wf = json.loads((PROCESSED_DIR / "wage_fit.json").read_text())
    con = pd.read_parquet(PROCESSED_DIR / "contracts.parquet")
    elo = pd.read_parquet(PROCESSED_DIR / "elo_aug1.parquet").set_index(["season", "club_id"])["elo"]
    return mv, curve, params, wf, con, elo


def contract_years(con, age) -> np.ndarray:
    return np.clip(np.round(np.interp(age, con["age"], con["years_fit"])), 1, MAX_YEARS).astype(int)


# ── market side: fees and resale prices ───────────────────────────────────────

def fee_model():
    """log fee ~ log mv (just before the move) + age + EPL buyer. big five summer moves with a fee, 2015-2023"""
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id", "transfer_fee"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    tr = tr[(tr["transfer_fee"] > 0) & tr["transfer_date"].dt.month.between(5, 9)
            & tr["transfer_date"].dt.year.between(2015, 2023)]
    clubs = pd.read_csv(TM / "clubs.csv", usecols=["club_id", "domestic_competition_id"]).set_index("club_id")["domestic_competition_id"]
    tr["to_lg"] = tr["to_club_id"].map(clubs)
    tr = tr[tr["to_lg"].isin(["GB1", "ES1", "L1", "IT1", "FR1"])]
    v = pd.read_csv(TM / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"])
    j = pd.merge_asof(tr.sort_values("transfer_date"), v.sort_values("date"), left_on="transfer_date",
                      right_on="date", by="player_id", direction="backward")
    dob = pd.to_datetime(pd.read_csv(TM / "players.csv", usecols=["player_id", "date_of_birth"])
                         .set_index("player_id")["date_of_birth"], errors="coerce")
    j["age"] = (j["transfer_date"] - j["player_id"].map(dob)).dt.days / 365.25
    j = j.dropna(subset=["market_value_in_eur", "age"])
    j = j[j["market_value_in_eur"] >= 1e6]
    X = np.column_stack([np.ones(len(j)), np.log(j["market_value_in_eur"]), j["age"] - 27, (j["to_lg"] == "GB1")])
    y = np.log(j["transfer_fee"])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    rms = float(np.sqrt(np.mean((y - X @ b) ** 2)))
    return b, rms, len(j)


def fee_hat(b, mv, age, epl=True):
    # age clipped to 18-34, the fit is thin outside and a 16-year-old got a 200m+ fee
    return np.exp(b[0] + b[1] * np.log(np.clip(mv, 1e6, None)) + b[2] * (np.clip(age, 18, 34) - 27) + b[3] * epl)


def resale_model(mv: pd.DataFrame):
    """log mv ~ par, par^2, age, age^2, group. EPL player-seasons, mv at the start of the season the par is for"""
    d = mv[(mv["league"] == "EPL") & (mv["mv"] > 0) & mv["age"].notna() & (mv["proj_min"] >= 900)]
    X, cols = resale_X(d)
    b = np.linalg.lstsq(X, np.log(d["mv"]), rcond=None)[0]
    return b, cols


def resale_X(d):
    cols = [np.ones(len(d)), d["par"], d["par"] ** 2, d["age"] - 27, (d["age"] - 27) ** 2]
    cols += [(d["grp"] == g).astype(float) for g in ("DEF", "MID", "ATT")]
    return np.column_stack(cols), None


def resale_hat(b, par, age, grp):
    d = pd.DataFrame({"par": par, "age": age, "grp": grp})
    return np.exp(resale_X(d)[0] @ b)


# ── the valuation ─────────────────────────────────────────────────────────────

def value(d: pd.DataFrame, buyer_elo, ctx, role_share=None, fee=None, sell=True, years=None, wage=None,
          repl_wage=True) -> pd.DataFrame:
    """d = market_value.parquet rows (as of season s). role_share: share of the season's minutes he'd get at the buyer
    (None = his own projection). fee: actual fee (None = what the market would charge).
    years / wage: for players already at the club (united_case.py), years left on the deal and the wage he's on,
    instead of a new contract. with 1 year left there's no resale, he walks"""
    curve, params, wf, con, fits, sd_val, eur, fb, rb, repl_mv = (ctx[k] for k in
        ("curve", "params", "wf", "con", "fits", "sd_val", "eur", "fee_b", "resale_b", "repl_mv"))
    a, ppx = params["a"], params["ppx"]
    d = d.copy()
    age = d["age"].fillna(27).to_numpy()
    L = contract_years(con, age) if years is None else np.asarray(years, dtype=int)
    H = np.maximum(L - 1, 1) if sell else L
    no_resale = (L <= 1) | (not sell)
    if role_share is None:
        mins = d["proj_min"].fillna(0).to_numpy()
    else:
        mins = np.asarray(role_share, dtype=float) * FULL * (1 - d["exp_inj_share"].fillna(0.08).to_numpy())
    buyer_elo = np.broadcast_to(np.asarray(buyer_elo, dtype=float), len(d))
    # wage: signing wage at the buyer, minus a replacement player's wage at the same club
    w0 = player_wage(wf["player"], d["mv"].fillna(1e6).to_numpy(), age, buyer_elo) * (1 + EMPLOYER_NI)
    if wage is not None:
        w0 = np.where(np.isnan(wage), w0, np.asarray(wage, dtype=float) * (1 + EMPLOYER_NI))
    wr = player_wage(wf["player"], repl_mv, 27, buyer_elo) * (1 + EMPLOYER_NI)
    if not repl_wage:
        # squad version (united_case.py): full wage, the replacement body is charged there only when a slot is empty
        wr = np.zeros(len(d))
    # can go below 0: a kid on a small deal is cheaper than the replacement you'd otherwise pay
    net_wage = w0 - wr

    contrib = np.zeros(len(d))
    wages = np.zeros(len(d))
    resale = np.zeros(len(d))
    par1 = d["par"].to_numpy() if role_share is None else \
        ((d["val_cal"] - d["repl"]).to_numpy() * mins / 90 * ppx)
    for k in range(1, MAX_YEARS + 1):
        step = age_step(curve, d["grp"], d["age"] + 1, d["age"] + k) if k > 1 else np.zeros(len(d))
        val = d["val_cal"].to_numpy() + a * np.where(d["age"].notna(), step, 0.0)
        par = (val - d["repl"].to_numpy()) * mins / 90 * ppx
        sd = sd_val * np.sqrt(k) * mins / 90 * ppx
        surv = surv_prob(fits, k - 1, age, par1)
        pk = floored(np.nan_to_num(par), sd) * surv
        on = k <= H
        disc = 1 / (1 + DISCOUNT) ** (k - 1)
        contrib += np.where(on, pk * eur * 1e6 * disc, 0)
        wages += np.where(on, net_wage * disc, 0)
        # resale at the end of year H, priced on the next season's projected PAR and his age then
        at = k == H + 1
        if at.any():
            rs = resale_hat(rb, np.nan_to_num(par), age + H, d["grp"].fillna("MID")) * surv
            resale += np.where(at & ~no_resale, rs / (1 + DISCOUNT) ** H, 0)
    d["years"], d["held"], d["role_min"] = L, H, mins
    d["wage"], d["wage_repl"] = w0, wr
    d["contrib_pv"], d["wages_pv"], d["resale_pv"] = contrib, wages, resale
    d["npv"] = contrib - wages + resale
    d["fee"] = fee_hat(fb, d["mv"].fillna(1e6), age) if fee is None else fee
    d["surplus"] = d["npv"] - d["fee"]
    return d


def context(mv, curve, params, wf, con):
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    fb, frms, fn = fee_model()
    rb, _ = resale_model(mv)
    # replacement player's market value: bottom 20% of EPL regulars (same cut as war.py's replacement pool)
    e = mv[(mv["league"] == "EPL") & (mv["proj_min"] >= 900) & (mv["mv"] > 0)]
    repl_mv = float(e[e["season"] == e["season"].max()]["mv"].quantile(params["cheap_q"]))
    return {"curve": curve, "params": params, "wf": wf, "con": con, "fits": survival(war, MAX_YEARS + 1),
            "sd_val": value_sd(war), "eur": eur_per_point(), "fee_b": fb, "fee_rms": frms, "fee_n": fn,
            "resale_b": rb, "repl_mv": repl_mv}


# ── backtest ──────────────────────────────────────────────────────────────────

def backtest(mv, ctx, elo):
    """EPL summer signings 2015-20: predicted surplus (as of the summer) vs realised over 3 seasons"""
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet").set_index(["season", "player_id"])
    params, eur = ctx["params"], ctx["eur"]
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["match_id", "player_id", "minutes"])
    mm = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season"])
    mins = lu.merge(mm, on="match_id").groupby(["season", "player_id"])["minutes"].sum()
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id", "transfer_fee"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    clubs = pd.read_csv(TM / "clubs.csv", usecols=["club_id", "domestic_competition_id"]).set_index("club_id")["domestic_competition_id"]
    tr = tr[(tr["transfer_fee"] > 0) & tr["transfer_date"].dt.month.between(5, 9)
            & (tr["to_club_id"].map(clubs) == "GB1")]
    tr["season"] = tr["transfer_date"].dt.year
    tr = tr[tr["season"].between(min(SEASONS) + 1, max(SEASONS) - 3)]
    us = {v: k for k, v in us_to_tm().items()}
    tr["us_id"] = tr["player_id"].map(us)
    tr = tr.dropna(subset=["us_id"]).astype({"us_id": int})
    mvi = mv.set_index(["season", "player_id"])
    rows = []
    for r in tr.itertuples():
        if (r.season - 1, r.us_id) not in mvi.index:
            continue
        rows.append(mvi.loc[[(r.season - 1, r.us_id)]].assign(fee_actual=r.transfer_fee, buy_season=r.season,
                                                                 elo_b=elo.get((r.season, r.to_club_id), 1800.0)))
    b = pd.concat(rows).reset_index()
    # held fixed at 3 seasons on both sides, so predicted and realised cover the same window
    ctx3 = dict(ctx, con=pd.DataFrame({"age": [15, 45], "years_fit": [4, 4]}))     # L = 4, H = 3
    p = value(b, b["elo_b"].to_numpy(), ctx3, fee=b["fee_actual"].to_numpy())
    real = np.zeros(len(b))
    for k in range(3):
        disc = 1 / (1 + DISCOUNT) ** k
        for i, (s, pid) in enumerate(zip(b["buy_season"] + k, b["player_id"])):
            if (s, pid) in war.index:
                w = war.loc[(s, pid)]
                real[i] += max((w["val_cal"] - w["repl"]) * mins.get((s, pid), 0) / 90 * params["ppx"], 0) * eur * 1e6 * disc
    end_mv = np.array([start_values(s + 3).get(pid, 0.0) for s, pid in zip(b["buy_season"], b["player_id"])])
    b["real_surplus"] = real - p["wages_pv"] + end_mv / (1 + DISCOUNT) ** 3 - b["fee_actual"]
    b["pred_surplus"] = p["surplus"]
    b["old_surplus"] = b["model_eur"] - b["fee_actual"]
    b["naive"] = b["mv"] - b["fee_actual"]
    return b


if __name__ == "__main__":
    mv, curve, params, wf, con, elo = load()
    ctx = context(mv, curve, params, wf, con)
    fb = ctx["fee_b"]
    print(f"fee model ({ctx['fee_n']:,} big five summer moves 2015-23): fee ~ mv^{fb[1]:.2f}, "
          f"x{np.exp(fb[2]):.3f} per year of age, EPL buyer x{np.exp(fb[3]):.2f}, rms {ctx['fee_rms']:.2f}")
    print(f"  a 24-year-old valued at 40m costs {fee_hat(fb, 40e6, 24) / 1e6:.1f}m from an EPL club")
    print(f"replacement player's market value {ctx['repl_mv'] / 1e6:.1f}m, euros per point {ctx['eur']:.2f}m, "
          f"discount {DISCOUNT:.0%}")

    print("\nbacktest, EPL summer signings 2015-20 (3 seasons each):")
    bt = backtest(mv, ctx, elo)
    from scipy.stats import spearmanr
    res = {c: spearmanr(bt[c], bt["real_surplus"]).statistic for c in ("pred_surplus", "old_surplus", "naive")}
    print(f"  {len(bt)} deals. rank correlation with realised surplus: new {res['pred_surplus']:.3f}, "
          f"old no-wage {res['old_surplus']:.3f}, mv - fee {res['naive']:.3f}")
    print(f"  pass mark (new beats both): {'pass' if res['pred_surplus'] > max(res['old_surplus'], res['naive']) else 'FAIL'}")
    q = pd.qcut(bt["pred_surplus"], 5, labels=False)
    print("  realised surplus by predicted quintile (eur m, median): " +
          " / ".join(f"{v / 1e6:+.1f}" for v in bt.groupby(q)["real_surplus"].median()))

    # summer 2024, united as the buyer, everyone on his own projected minutes
    s = max(SEASONS)
    cur = mv[mv["season"] == s]
    u_elo = elo.get((s + 1, UNITED), elo.get((s, UNITED)))
    out = value(cur, u_elo, ctx)
    out["buyer_elo"] = u_elo
    out.to_parquet(PROCESSED_DIR / "transfer_value.parquet", index=False)
    show = out[(out["proj_min"] >= 1500) & (out["mv"] > 0)].copy()
    cols = ["name", "team", "grp", "age", "mv", "fee", "wage", "contrib_pv", "wages_pv", "resale_pv", "npv", "surplus"]
    fmt = show[cols].copy()
    for c in cols[4:]:
        fmt[c] = fmt[c] / 1e6
    print(f"\nsummer {s + 1}, value to united (elo {u_elo:.0f}), own projected minutes, eur m. best surplus:")
    print(fmt.sort_values("surplus", ascending=False).head(15).round(1).to_string(index=False))
    print("worst:")
    print(fmt.sort_values("surplus").head(8).round(1).to_string(index=False))
    print(f"\nsaved -> {PROCESSED_DIR / 'transfer_value.parquet'}")
