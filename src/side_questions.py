#!/usr/bin/env python3
"""
Step 10: the side questions, all on the same values (war.py, market_value.py), summer windows 2015 to 2024.

  1. positions   at the same model value, does the market pay more for some positions? (the NBA version was
                 three-point shooters). log(market / model) by position group, regulars only, age and league
                 taken out
  2. age         same thing by age: does the market discount age more than performance does?
  3. big fees    every transfer into a big five club with a fee, summers 2015 to 2023:
                 fee against the model's value before the move, and against what he actually gave the buyer
                 the season after (realised PAR x euros per point). by fee band. and does the model pick the
                 better deals: deals where the model's value beat the fee vs the rest, inside each fee band.
                 realised = his value for the season he played for the buyer, which is a pooled window with
                 his last two seasons before the move in it (1 / .5 / .25), so it isn't fully independent of the
                 projection. the report says so
  4. one good    does the market overreact to one good season? the market value change over a season against
     season      the next year's change (age and level out). negative = jumps get partly given back. and does
                 the model's gap (market_value.py) already see it coming

Output: figures/side_*.png, data/processed/side_questions.json

  python src/side_questions.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label
from tm_match import start_values, us_to_tm

TM = RAW_DIR / "transfermarkt"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def resid_gap(d: pd.DataFrame, by: str) -> pd.Series:
    """mean of log(market / model) by group, with league and season means taken out (and age, unless by=age)"""
    y = np.log((d["mv"] + 1e6) / (d["model_eur"] + 1e6))
    X = [pd.get_dummies(d["league"], drop_first=True, dtype=float), pd.get_dummies(d["season"], drop_first=True, dtype=float)]
    if by != "age_band":
        X.append(pd.DataFrame({"age": d["age"], "age2": d["age"] ** 2}))
    X = pd.concat(X, axis=1)
    X.insert(0, "c", 1.0)
    b = np.linalg.lstsq(X.to_numpy(float), y.to_numpy(), rcond=None)[0]
    r = y - X.to_numpy(float) @ b
    out = r.groupby(d[by], observed=True).mean()
    return out - out.mean()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (ROOT / "figures").mkdir(exist_ok=True)
    res = {}

    mv = pd.read_parquet(PROCESSED_DIR / "market_value.parquet")
    reg = mv[(mv["proj_min"] >= 1500) & (mv["mv"] > 0) & mv["age"].notna()].copy()

    # ── 1. positions ──
    pos = resid_gap(reg, "grp")
    print("1. market price vs model value by position, regulars, age / league / season taken out")
    print("   " + "   ".join(f"{g}: {100 * (np.exp(v) - 1):+.0f}%" for g, v in pos.sort_values().items()))
    res["position_premium_pct"] = {g: round(100 * (np.exp(v) - 1), 1) for g, v in pos.items()}
    # within attack: forwards vs wingers / 10s, the 'strikers' question
    reg["role"] = np.where(reg["position"].isin(["FW"]), "striker", reg["grp"].replace({"ATT": "winger / 10"}))
    role = resid_gap(reg, "role")
    print("   strikers on their own: " + "   ".join(f"{g}: {100 * (np.exp(v) - 1):+.0f}%" for g, v in role.sort_values().items()))
    res["role_premium_pct"] = {g: round(100 * (np.exp(v) - 1), 1) for g, v in role.items()}

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    style(ax, fig)
    r = role.sort_values()
    vals = 100 * (np.exp(r) - 1)
    ax.barh(r.index, vals, color=[ORANGE if v > 0 else BLUE for v in vals], zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.0f}%", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Market value vs model value, same age, league and season", color=MUTED)
    ax.set_title("What the market pays extra for, by role", loc="left", color=INK, fontsize=11)
    ax.set_xlim(min(vals.min() * 1.4, -10), max(vals.max() * 1.4, 10))
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "side_positions.png", facecolor=SURFACE)

    # ── 2. age ──
    reg["age_band"] = pd.cut(reg["age"], [17, 21, 23, 25, 27, 29, 31, 33, 41],
                             labels=["18-21", "22-23", "24-25", "26-27", "28-29", "30-31", "32-33", "34+"])
    ag = resid_gap(reg, "age_band")
    print("\n2. market price vs model value by age (league / season taken out, NOT age):")
    print("   " + "   ".join(f"{g}: {100 * (np.exp(v) - 1):+.0f}%" for g, v in ag.items()))
    res["age_premium_pct"] = {str(g): round(100 * (np.exp(v) - 1), 1) for g, v in ag.items()}
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    style(ax, fig)
    vals = 100 * (np.exp(ag) - 1)
    ax.bar(range(len(ag)), vals, color=[ORANGE if v > 0 else BLUE for v in vals], zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.0f}%", (i, v), xytext=(0, 4 if v >= 0 else -11), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(ag)), ag.index)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Age", color=MUTED)
    ax.set_ylabel("Market vs model value", color=MUTED)
    ax.set_title("The market pays for youth, and discounts age harder than performance does",
                 loc="left", color=INK, fontsize=11)
    ax.set_ylim(vals.min() * 1.3, vals.max() * 1.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "side_age.png", facecolor=SURFACE)

    # ── 3. big fees ──
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    params = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    from market_value import eur_per_point
    eur = eur_per_point()
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id", "transfer_fee",
                                                    "from_club_name", "to_club_name", "player_name"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    tr = tr[(tr["transfer_fee"] > 0)]
    # summer window of year y = before 1 Oct y, feeds season y
    tr["season"] = tr["transfer_date"].dt.year.where(tr["transfer_date"].dt.month <= 9, np.nan)
    tr = tr.dropna(subset=["season"])
    tr["season"] = tr["season"].astype(int)
    us = {v: k for k, v in us_to_tm().items()}
    tr["us_id"] = tr["player_id"].map(us)
    tr = tr.dropna(subset=["us_id"])
    tr["us_id"] = tr["us_id"].astype(int)
    tr = tr[tr["season"].between(min(SEASONS) + 1, max(SEASONS))]

    mvv = mv.set_index(["season", "player_id"])
    w = war.set_index(["season", "player_id"])
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["match_id", "player_id", "minutes"])
    mm = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season"])
    mins = lu.merge(mm, on="match_id").groupby(["season", "player_id"])["minutes"].sum()
    rows = []
    for r in tr.itertuples():
        before = (r.season - 1, r.us_id)
        after = (r.season, r.us_id)        # the season he played for the buyer, its value as of that season
        if before not in mvv.index:
            continue
        b = mvv.loc[before]
        # realised: value fitted on the season he played + his actual minutes, vs replacement
        if after in w.index:
            a = w.loc[after]
            real = max((a["val_cal"] - a["repl"]) * mins.get(after, 0) / 90 * params["ppx"], 0)
        else:
            real = 0.0     # not in the big five data at all = no value to a big five buyer that season
        rows.append({"season": r.season, "name": b["name"], "from": r.from_club_name, "to": r.to_club_name,
                     "fee": r.transfer_fee, "model_eur": b["model_eur"], "par_proj": max(b["par"], 0),
                     "par_real": real, "age": b["age"]})
    f = pd.DataFrame(rows)
    f["band"] = pd.cut(f["fee"] / 1e6, [0, 5, 15, 30, 60, 300], labels=["<5", "5-15", "15-30", "30-60", "60+"])
    f["real_eur_y1"] = f["par_real"] * eur * 1e6
    g = f.groupby("band", observed=True).agg(n=("fee", "size"), fee=("fee", "median"), model=("model_eur", "median"),
                                             par_proj=("par_proj", "mean"), par_real=("par_real", "mean"))
    g["fee_per_real_pt"] = f.groupby("band", observed=True).apply(lambda x: x["fee"].sum() / max(x["par_real"].sum(), 1e-9) / 1e6,
                                                                  include_groups=False)
    print(f"\n3. transfers with a fee into the big five, summers {f['season'].min()} to {f['season'].max()} "
          f"({len(f):,} with a value before the move)")
    print(f"   {'fee band EUR m':<15}{'n':>5}{'median fee':>12}{'median model':>14}{'PAR proj':>10}{'PAR real':>10}{'fee per real point':>20}")
    for band, x in g.iterrows():
        print(f"   {band:<15}{x['n']:>5.0f}{x['fee'] / 1e6:>12.1f}{x['model'] / 1e6:>14.1f}{x['par_proj']:>10.2f}"
              f"{x['par_real']:>10.2f}{x['fee_per_real_pt']:>20.1f}")
    # does the model pick the better deals: per euro of fee, realised year-1 points
    f["model_likes"] = f["model_eur"] > f["fee"]
    for k, x in f.groupby("model_likes"):
        print(f"   model value {'above' if k else 'below'} the fee: {len(x):,} deals, realised points per EUR 10m of fee "
              f"{x['par_real'].sum() / (x['fee'].sum() / 1e7):.2f}")
    # cheap deals deliver more per euro anyway, and "model above fee" is mostly cheap deals. so the fair version
    # is inside each fee band
    print("   same thing inside each fee band (realised points per EUR 10m, model above / below the fee):")
    for band, x in f.groupby("band", observed=True):
        hi, lo = x[x["model_likes"]], x[~x["model_likes"]]
        if len(hi) >= 10 and len(lo) >= 10:
            print(f"     {band:<6} {hi['par_real'].sum() / (hi['fee'].sum() / 1e7):.2f} ({len(hi)}) / "
                  f"{lo['par_real'].sum() / (lo['fee'].sum() / 1e7):.2f} ({len(lo)})")
    big = f[f["fee"] >= 30e6]
    print(f"   30m+ deals: projected {big['par_proj'].mean():.2f} PAR, realised {big['par_real'].mean():.2f}, "
          f"r(projected, realised) {np.corrcoef(big['par_proj'], big['par_real'])[0, 1]:.2f}")
    res["fees"] = g.reset_index().assign(band=lambda x: x["band"].astype(str)).to_dict("records")
    f.to_parquet(PROCESSED_DIR / "side_transfers.parquet", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    style(ax, fig)
    ax.bar(range(len(g)), g["fee_per_real_pt"], color=BLUE, zorder=3)
    for i, v in enumerate(g["fee_per_real_pt"]):
        ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
    # reference: what 4 seasons of a point are worth in EPL prize money
    # in a legend, a label on the line sat on top of the 5-15m bar's number
    ax.axhline(eur * 4, color=ORANGE, lw=1.2, ls="--", zorder=4, label="4 seasons of prize money per point")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.set_xticks(range(len(g)), [f"EUR {b}m" for b in g.index])
    ax.set_ylabel("Fee per point delivered in year 1 (EUR m)", color=MUTED)
    ax.set_title("Bigger fees buy points at a higher price", loc="left", color=INK, fontsize=11)
    ax.set_ylim(0, g["fee_per_real_pt"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "side_fees.png", facecolor=SURFACE)

    # ── 4. one good season ──
    vals = {s: start_values(s) for s in range(min(SEASONS), max(SEASONS) + 3)}
    rows = []
    for s in SEASONS:
        for pid in mv.loc[mv["season"] == s, "player_id"]:
            v0, v1, v2 = vals[s].get(pid), vals[s + 1].get(pid), vals[s + 2].get(pid)
            if v0 and v1 and v2 and v0 > 0 and v1 > 0 and v2 > 0:
                rows.append((s, pid, np.log(v1 / v0), np.log(v2 / v1)))
    o = pd.DataFrame(rows, columns=["season", "player_id", "d_this", "d_next"]).merge(
        mv[["season", "player_id", "age", "mv", "gap", "proj_min"]], on=["season", "player_id"])
    o = o[(o["proj_min"] >= 900) & o["age"].notna()]
    X = np.column_stack([np.ones(len(o)), o["d_this"], o["age"], o["age"] ** 2, np.log(o["mv"])])
    b1 = np.linalg.lstsq(X, o["d_next"], rcond=None)[0]
    X2 = np.column_stack([X, o["gap"]])
    b2 = np.linalg.lstsq(X2, o["d_next"], rcond=None)[0]
    print(f"\n4. market value change over a season vs the next season's change ({len(o):,} player-seasons, age and "
          f"level taken out)")
    print(f"   next change on this change: {b1[1]:+.3f} (negative = part of a jump is given back the year after)")
    print(f"   with the model's gap added: this change {b2[1]:+.3f}, gap {b2[5]:+.3f}")
    jump = o[o["d_this"] > np.log(1.5)]
    print(f"   players whose value went up 50%+ in a season: next year {100 * (np.exp(jump['d_next'].mean()) - 1):+.1f}% "
          f"on average vs {100 * (np.exp(o['d_next'].mean()) - 1):+.1f}% for everyone ({len(jump):,} cases)")
    res["overreaction"] = {"coef_this": round(b1[1], 3), "coef_this_with_gap": round(b2[1], 3), "coef_gap": round(b2[5], 3)}

    (PROCESSED_DIR / "side_questions.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nsaved -> figures/side_positions.png, side_age.png, side_fees.png, {PROCESSED_DIR / 'side_questions.json'}")
