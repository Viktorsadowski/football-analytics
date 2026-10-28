#!/usr/bin/env python3
"""
Step 5: aging curves. How much do O and D RAPM change with age, overall and per position group.
Same method as aging.py in nba-moneyball:

  fixed effects  rapm_it = talent_i + age_effect(age_it), weighted by minutes. each player's own level is
                 soaked up by talent_i, what's left is the age shape. main estimate.
  delta method   average change from age a to a+1, players with 900+ min in both seasons, weighted by the
                 smaller of the two. biased by survivors, so only a sanity check

Both smoothed with a weighted quadratic and set to 0 at the peak.

Ages: transfermarkt birth date (through id_map), age on 1 January of the season (= mid-season).
One football thing: rapm for season s is pooled over s, s-1, s-2 (weights 1, .5, .25), so the rating is
of a player on average 0.57 years younger than his age in s. the curve is fitted on that effective age,
otherwise it comes out shifted about half a year late.

Groups (box_prior.POS, merged so each has enough players): GK, DEF (CB, FB), MID (DM, CM), ATT (W, AM, FW)
Keepers: d_rapm gets the shot-stopping from gk.py added (save90). their offense is pinned to the box prior,
and the prior climbs with minutes per game / starts (old keepers who keep their place have lots), so keeper
"offense" rose with age for no real reason and dragged the peak to 35-37. the keeper curve is defense +
shot-stopping only, offense flat, and shrunk towards the DEF total curve (only ~600 keepers):
  y - DEF(age) = talent_i + dev(age), ridge on dev. dev columns scaled by c (penalty 1/c^2),
  c picked by 5-fold CV on held-out keeper-seasons. c -> 0 = DEF curve, c large = keepers on their own

Survivorship (ghost seasons). the fixed effects handle "only good players last" (everyone is compared to
himself), but not "players who fall off a cliff stop playing", so the late decline comes out too flat.
a player 30+ with 900+ min in season s who is gone from the big five in s+1 gets a ghost season s+1:
  his club got relegated                                       no ghost, he's just in the 2nd tier now (half
                                                               of all the cases, they'd swamp the rest)
  moved to PO1, NL1, BE1, TR1 (strong) or SA1, MLS1 (money)   no ghost, he left while still good
  retired, without club, any other league, or no transfer     ghost at replacement level of his group
  keeper with no transfer (still at the club, lost his place) ghost GK_BELOW under keeper replacement
  keeper still in the data but under 450 min after 900+       his s+1 season set to that same level
replacement level here = minute-weighted o/d rapm of players in the bottom 20% of market values (same pool
as war.py, but on the raw rapm scale). ghosts weigh GHOST_MIN minutes. s+1 has to be inside the data, so
2023-24 can't produce ghosts.

Output
  data/processed/aging_curve.parquet   group, age, o_curve, d_curve, curve (npxG per 90, relative to peak)
  figures/aging_curve.png              all players, offense / defense / total
  figures/aging_by_position.png        total curve per group

  python src/aging.py
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from box_prior import POS
from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS
from rapm import WINDOW, keepers
from tm_match import start_values, us_to_tm

AGE_MIN, AGE_MAX = 18, 37   # few players outside this, they get clipped into the ends
DELTA_MIN = 900
LAG = sum(k * w for k, w in WINDOW.items()) / sum(WINDOW.values())   # 0.57, see above
BROAD = {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "CM": "MID", "W": "ATT", "AM": "ATT", "FW": "ATT"}
GROUPS = ["ALL", "GK", "DEF", "MID", "ATT"]
GHOST_AGE, GHOST_FROM, GHOST_MIN = 30, 900, 900
GK_BELOW = 0.01          # npxG per 90 under keeper replacement for a keeper who lost his place
GK_BENCH_MIN = 450
NO_GHOST = {"PO1", "NL1", "BE1", "TR1", "SA1", "MLS1"}
CHEAP_Q = 0.20


def load() -> pd.DataFrame:
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    gks = pd.read_parquet(PROCESSED_DIR / "gk_saves.parquet")[["season", "player_id", "save90"]]
    rapm = rapm.merge(gks, on=["season", "player_id"], how="left")
    rapm["d_rapm"] += rapm["save90"].fillna(0)
    rapm["rapm"] = rapm["o_rapm"] + rapm["d_rapm"]
    idmap = pd.read_parquet(PROCESSED_DIR / "id_map.parquet")[["us_id", "tm_id"]]
    dob = pd.read_csv(RAW_DIR / "transfermarkt" / "players.csv", usecols=["player_id", "date_of_birth"])
    dob["date_of_birth"] = pd.to_datetime(dob["date_of_birth"], errors="coerce")
    d = rapm.merge(idmap, left_on="player_id", right_on="us_id") \
        .merge(dob.rename(columns={"player_id": "tm_id"}), on="tm_id")
    d = d[d["date_of_birth"].notna() & (d["minutes"] > 0)].copy()
    mid = pd.to_datetime((d["season"] + 1).astype(str) + "-01-01")
    d["age_real"] = (mid - d["date_of_birth"]).dt.days / 365.25
    d["age"] = (d["age_real"] - LAG).round().clip(AGE_MIN, AGE_MAX).astype(int)
    # group from his usual position in that season (rapm.player_info), subs-only seasons get his most common one
    d["grp"] = d["position"].map(POS).map(BROAD)
    d["grp"] = d["grp"].fillna(d.groupby("player_id")["grp"].transform(lambda s: s.mode().iloc[0] if s.notna().any()
                                                                       else None))
    d.loc[d["player_id"].isin(keepers()), "grp"] = "GK"
    d["ghost"] = ""
    return d


# ── ghost seasons ─────────────────────────────────────────────────────────────

def replacement(d: pd.DataFrame) -> pd.DataFrame:
    """o/d rapm per group of players in the bottom 20% of market values at the start of the next season"""
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet", columns=["match_id", "player_id", "minutes"])
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season", "league"])
    lu = lu.merge(m, on="match_id")
    pool = []
    for s in list(SEASONS)[:-1]:
        mv = start_values(s + 1)
        nxt = lu[lu["season"] == s + 1].groupby(["player_id", "league"])["minutes"].sum().reset_index() \
            .sort_values("minutes").drop_duplicates("player_id", keep="last")
        nxt["mv"] = nxt["player_id"].map(mv)
        nxt = nxt.dropna(subset=["mv"])
        cheap = nxt[nxt["mv"] <= nxt.groupby("league")["mv"].transform(lambda x: x.quantile(CHEAP_Q))]
        pool.append(cheap[["player_id", "minutes"]].merge(d[d["season"] == s][["player_id", "grp", "o_rapm",
                                                                                   "d_rapm"]], on="player_id"))
    pool = pd.concat(pool)
    return pool.groupby("grp").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["minutes"])
                                                         for c in ("o_rapm", "d_rapm")}), include_groups=False)


def add_ghosts(d: pd.DataFrame) -> pd.DataFrame:
    repl = replacement(d)
    tr = pd.read_csv(RAW_DIR / "transfermarkt" / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id",
                                                                           "to_club_name"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    comp = pd.read_csv(RAW_DIR / "transfermarkt" / "clubs.csv", usecols=["club_id", "domestic_competition_id"]) \
        .set_index("club_id")["domestic_competition_id"]
    tm = us_to_tm()
    last = max(SEASONS)
    seen = set(zip(d["season"], d["player_id"]))
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["season", "home_team"])
    teams = m.groupby("season")["home_team"].apply(set)

    cand = d[(d["age_real"] >= GHOST_AGE) & (d["minutes"] >= GHOST_FROM) & (d["season"] < last)]
    ghosts, cats = [], {}
    for r in cand.itertuples():
        nxt = (r.season + 1, r.player_id)
        is_gk = r.grp == "GK"
        if nxt in seen:
            # keeper still around but benched: his real s+1 row gets overwritten below
            continue
        if r.team not in teams[r.season + 1]:
            cats["club relegated"] = cats.get("club relegated", 0) + 1
            continue
        # first move after season s ended (from June s+1) and before s+1 ended
        t = tr[(tr["player_id"] == tm.get(r.player_id)) & tr["transfer_date"].between(
            pd.Timestamp(f"{r.season + 1}-06-01"), pd.Timestamp(f"{r.season + 2}-05-31"))].sort_values("transfer_date")
        if len(t):
            league = comp.get(t["to_club_id"].iloc[0])
            if league in NO_GHOST:
                cats["moved, strong or money league"] = cats.get("moved, strong or money league", 0) + 1
                continue
            cat, below = "left: retired / lower league / other", 0.0
        else:
            cat, below = ("keeper, no transfer (lost his place)", GK_BELOW) if is_gk else ("no transfer, didn't play", 0.0)
        cats[cat] = cats.get(cat, 0) + 1
        rp = repl.loc[r.grp]
        ghosts.append({"season": r.season + 1, "player_id": r.player_id, "grp": r.grp, "minutes": GHOST_MIN,
                       "o_rapm": rp["o_rapm"], "d_rapm": rp["d_rapm"] - below, "age_real": r.age_real + 1,
                       "ghost": cat})
    g = pd.DataFrame(ghosts)
    g["rapm"] = g["o_rapm"] + g["d_rapm"]
    g["age"] = (g["age_real"] - LAG).round().clip(AGE_MIN, AGE_MAX).astype(int)

    # keepers still in the data but benched after a 900+ season
    prev = d[(d["grp"] == "GK") & (d["minutes"] >= GHOST_FROM) & (d["age_real"] >= GHOST_AGE)][["season", "player_id"]]
    prev = prev.assign(season=prev["season"] + 1)
    bench = d.merge(prev, on=["season", "player_id"]).query("minutes < @GK_BENCH_MIN")
    idx = d.set_index(["season", "player_id"]).index.isin(list(zip(bench["season"], bench["player_id"])))
    rp = repl.loc["GK"]
    d.loc[idx, "o_rapm"] = rp["o_rapm"]
    d.loc[idx, "d_rapm"] = rp["d_rapm"] - GK_BELOW
    d.loc[idx, "rapm"] = d.loc[idx, "o_rapm"] + d.loc[idx, "d_rapm"]
    d.loc[idx, "minutes"] = GHOST_MIN
    d.loc[idx, "ghost"] = "keeper benched (<450 min)"
    cats["keeper benched (<450 min)"] = int(idx.sum())

    print("ghost seasons (30+, 900+ min, then gone):")
    for k, v in cats.items():
        print(f"  {k:<36} {v}")
    print("  replacement level used (raw rapm): " + ", ".join(f"{k} {v:+.3f}" for k, v in repl["o_rapm"].add(
        repl["d_rapm"]).items()))
    return pd.concat([d, g], ignore_index=True)


# ── fixed effects ─────────────────────────────────────────────────────────────

def fixed_effects(d: pd.DataFrame, col: str) -> pd.Series:
    pids = {p: i for i, p in enumerate(d["player_id"].unique())}
    ages = list(range(AGE_MIN, AGE_MAX + 1))
    n, n_p = len(d), len(pids)
    rows = np.arange(n)
    p_idx = d["player_id"].map(pids).to_numpy()
    a_idx = n_p + (d["age"].to_numpy() - AGE_MIN)
    X = sparse.csr_matrix((np.ones(2 * n), (np.concatenate([rows, rows]), np.concatenate([p_idx, a_idx]))),
                          shape=(n, n_p + len(ages)))
    # light ridge so a 30-minute guy doesn't get a crazy talent term, the age terms have
    # hundreds of rows each so the penalty barely touches them
    m = Ridge(alpha=1.0).fit(X, d[col].to_numpy(), sample_weight=d["minutes"].to_numpy() / 1000)
    return pd.Series(m.coef_[n_p:], index=ages)


# ── delta method ──────────────────────────────────────────────────────────────

def delta_method(d: pd.DataFrame, col: str) -> pd.Series:
    a = d[["player_id", "season", "age", "minutes", col]]
    b = a.drop(columns="age").assign(season=a["season"] - 1)  # next season, shifted back so it lines up
    p = a.merge(b, on=["player_id", "season"], suffixes=("", "_next"))
    p = p[(p["minutes"] >= DELTA_MIN) & (p["minutes_next"] >= DELTA_MIN)]
    p["w"] = np.minimum(p["minutes"], p["minutes_next"])
    p["delta"] = p[f"{col}_next"] - p[col]
    step = p.groupby("age").apply(lambda g: np.average(g["delta"], weights=g["w"]), include_groups=False)
    step = step.reindex(range(AGE_MIN, AGE_MAX)).fillna(0)
    # value at age a = sum of steps before a
    # note: next season's rating shares two of its three window seasons with this one, so the deltas are
    # damped, this one is flatter than the real curve. it's only checked for the shape and the peak
    return pd.Series(np.concatenate([[0], step.cumsum().to_numpy()]), index=range(AGE_MIN, AGE_MAX + 1))


# ── smoothing ─────────────────────────────────────────────────────────────────

def smooth(raw: pd.Series, weight: pd.Series) -> pd.Series:
    ages = raw.index.to_numpy()
    coef = np.polyfit(ages, raw.to_numpy(), deg=2, w=np.sqrt(weight.reindex(ages).fillna(1).to_numpy()))
    fit = np.polyval(coef, ages)
    return pd.Series(fit - fit.max(), index=ages)


def curves(d: pd.DataFrame, grp: str) -> pd.DataFrame:
    mins = d.groupby("age")["minutes"].sum()
    fe = {c: fixed_effects(d, c) for c in ("o_rapm", "d_rapm")}
    c = pd.DataFrame({"group": grp, "age": range(AGE_MIN, AGE_MAX + 1)})
    c["o_curve"] = smooth(fe["o_rapm"], mins).to_numpy()
    c["d_curve"] = smooth(fe["d_rapm"], mins).to_numpy()
    # total = O + D, re-zeroed at its own peak (O and D don't peak at the same age)
    tot = c["o_curve"] + c["d_curve"]
    c["curve"] = tot - tot.max()
    c["raw_total"] = (fe["o_rapm"] + fe["d_rapm"]).to_numpy()
    c["minutes"] = c["age"].map(mins).fillna(0).astype(int)
    return c


def gk_curve(g: pd.DataFrame, def_curve: pd.DataFrame) -> pd.DataFrame:
    """keeper curve, defense + saves only, shrunk towards the DEF curve (see top)"""
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    g = g.reset_index(drop=True)
    base = g["age"].map(def_curve.set_index("age")["curve"]).to_numpy()
    y = g["d_rapm"].to_numpy() - base
    w = g["minutes"].to_numpy() / 1000
    pids = {p: i for i, p in enumerate(g["player_id"].unique())}
    n, n_p = len(g), len(pids)
    rows = np.arange(n)
    p_idx = g["player_id"].map(pids).to_numpy()
    a_idx = n_p + (g["age"].to_numpy() - AGE_MIN)

    def design(c):
        return sparse.csr_matrix((np.concatenate([np.ones(n), np.full(n, c)]),
                                  (np.concatenate([rows, rows]), np.concatenate([p_idx, a_idx]))),
                                 shape=(n, n_p + len(ages)))

    scores = {}
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        X = design(c)
        err = 0.0
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = Ridge(alpha=1.0).fit(X[tr], y[tr], sample_weight=w[tr])
            err += np.sum(w[te] * (y[te] - m.predict(X[te])) ** 2)
        scores[c] = err / w.sum()
    best = min(scores, key=scores.get)
    out = {}
    for name, c in (("shrunk", best), ("own", 3.0)):
        m = Ridge(alpha=1.0).fit(design(c), y, sample_weight=w)
        dev = pd.Series(m.coef_[n_p:] * c, index=ages)
        raw = def_curve.set_index("age")["curve"].reindex(ages).to_numpy() + dev.to_numpy()
        out[name] = smooth(pd.Series(raw, index=ages), g.groupby("age")["minutes"].sum())
    print(f"  keeper curve: CV picks c = {best} ({', '.join(f'{k}: {v:.5f}' for k, v in scores.items())})")
    print(f"    own peak {int(out['own'].idxmax())}, shrunk peak {int(out['shrunk'].idxmax())}, "
          f"DEF peak {int(def_curve.loc[def_curve['curve'].idxmax(), 'age'])}")
    c = pd.DataFrame({"group": "GK", "age": ages, "o_curve": 0.0, "d_curve": out["shrunk"].to_numpy()})
    c["curve"] = c["d_curve"] - c["d_curve"].max()
    c["raw_total"] = (out["own"]).to_numpy()
    c["minutes"] = c["age"].map(g.groupby("age")["minutes"].sum()).fillna(0).astype(int)
    return c


# ── plots ─────────────────────────────────────────────────────────────────────

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = {"GK": "#7a5af5", "DEF": "#eb6834", "MID": "#1f9e73", "ATT": "#2a78d6"}


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("Age", color=MUTED)
    ax.set_ylabel("RAPM vs peak (npxG per 90)", color=MUTED)
    ax.set_xlim(AGE_MIN - 0.5, AGE_MAX + 0.5)
    ax.set_xticks(range(AGE_MIN, AGE_MAX + 1, 2))


def plot_all(c: pd.DataFrame, delta_total: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    _style(ax, fig)
    a = c["age"]
    ax.plot(a, c["o_curve"], color="#2a78d6", lw=2, label="Offense")
    ax.plot(a, c["d_curve"], color="#eb6834", lw=2, label="Defense")
    ax.plot(a, c["curve"], color=INK, lw=2.4, label="Total")
    ax.scatter(a, c["raw_total"] - c["raw_total"].max(), s=22, color=INK, alpha=0.35, zorder=3,
               label="Total, unsmoothed")
    ax.plot(delta_total.index, delta_total - delta_total.max(), color=MUTED, lw=1.2, ls="--",
            label="Total, delta method")
    peak = int(a[c["curve"].idxmax()])
    ax.axvline(peak, color=GRID, lw=1, zorder=0)
    ax.annotate(f"peak {peak}", (peak, 0), xytext=(4, 6), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_title("Aging curve, big five leagues, 2014-15 to 2023-24", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "aging_curve.png", facecolor=SURFACE)


def plot_groups(cs: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    _style(ax, fig)
    for g in GROUPS[1:]:
        c = cs[cs["group"] == g]
        peak = int(c.loc[c["curve"].idxmax(), "age"])
        ax.plot(c["age"], c["curve"], color=COLORS[g], lw=2.2, label=f"{g} (peak {peak})")
        ax.scatter([peak], [0], color=COLORS[g], s=26, zorder=3)
    ax.set_title("Aging curve by position, total RAPM", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "aging_by_position.png", facecolor=SURFACE)


if __name__ == "__main__":
    d0 = load()
    print(f"{len(d0):,} player-seasons with a birth date, {d0['player_id'].nunique():,} players "
          f"(effective age = age mid-season - {LAG:.2f})\n")
    d = add_ghosts(d0.copy())
    print()

    out = [curves(d, "ALL")]
    for g in ("DEF", "MID", "ATT"):
        out.append(curves(d[d["grp"] == g], g))
    out.append(gk_curve(d[d["grp"] == "GK"], out[1]))
    cs = pd.concat(out, ignore_index=True)

    delta_total = delta_method(d, "rapm")
    allc = cs[cs["group"] == "ALL"].reset_index(drop=True)
    print(allc.assign(delta_method=(delta_total - delta_total.max()).to_numpy())
          [["age", "minutes", "o_curve", "d_curve", "curve", "delta_method"]].round(3).to_string(index=False))
    print()
    # without ghosts, to see what the survivorship fix does
    no = [curves(d0, "ALL")] + [curves(d0[d0["grp"] == g], g) for g in ("DEF", "MID", "ATT")]
    no = pd.concat(no + [gk_curve(d0[d0["grp"] == "GK"], no[1])], ignore_index=True)
    for g in GROUPS:
        c0 = no[no["group"] == g]
        print(f"  {g:<4} without ghosts: peak {int(c0.loc[c0['curve'].idxmax(), 'age'])}, "
              f"at 33 {c0.loc[c0['age'] == 33, 'curve'].iloc[0]:+.3f}, at 35 {c0.loc[c0['age'] == 35, 'curve'].iloc[0]:+.3f}")
    print()
    for g in GROUPS:
        c = cs[cs["group"] == g]
        pk = {k: int(c.loc[c[k].idxmax(), "age"]) for k in ("o_curve", "d_curve", "curve")}
        at = lambda age: c.loc[c["age"] == age, "curve"].iloc[0]
        print(f"  {g:<4} peak O {pk['o_curve']}, D {pk['d_curve']}, total {pk['curve']}   "
              f"vs peak at 21 {at(21):+.3f}, at 30 {at(30):+.3f}, at 33 {at(33):+.3f}, at 35 {at(35):+.3f}")

    cs.drop(columns="raw_total").to_parquet(PROCESSED_DIR / "aging_curve.parquet", index=False)
    plot_all(allc, delta_total)
    plot_groups(cs)
    print(f"\nsaved -> {PROCESSED_DIR / 'aging_curve.parquet'}, figures/aging_curve.png, figures/aging_by_position.png")
