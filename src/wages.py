#!/usr/bin/env python3
"""
Step 9: a wage proxy, since player wages aren't in any source we're allowed to use in bulk (FBref / Capology terms).

  1. player model     manual/reported_wages.csv, 59 wages reported in the press (Gazzetta via Calcio e Finanza, Sport
                      Bild, L'Équipe, The Athletic, The Times, ...), one source url per row, spread over the five
                      leagues, ages and club strength. everything converted to gross euros a year. the 4 la liga rows
                      are flagged low quality (unnamed source or net/gross unclear) and left out of the fit
                      fit: log wage = a + b log(mv) + league + c (age - 27) + d (club elo - 1800) / 100
                      mv = transfermarkt value on 1 Sept of the wage season, so mostly the value when he signed (most
                      reports are about a new deal). a term stays only if it lowers the leave one out error
  2. club bills       manual/wage_bills.csv, Premier League clubs from published annual accounts (total staff costs),
                      126 club-seasons 2014-15 to 2022-23. fitted on its own first (wage bill on the squad's market
                      values), then used as a check on the player model: the player model summed over each squad
                      against the accounts. the gap is non-playing staff, bonuses, players without league minutes
  3. contract length  by age at signing, from transfermarkt: big five paid signings 2019-24 where the player is still
                      at the club in the snapshot, contract end - signing date

first version split the club bills over the squad by market value with an age term. club totals are dominated by
the established players, so it couldn't see how a squad splits between young and old (age effect 0.057 a year vs
0.091 from the reported wages). the club fit stays in as a check

Output: data/processed/wage_fit.json (player model + club fit), data/processed/contracts.parquet (age -> years)

  python src/wages.py
"""

import json
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from config import PROCESSED_DIR, RAW_DIR, ROOT
from price import GBP_EUR
from tm_match import start_values, us_to_tm

TM = RAW_DIR / "transfermarkt"
AGE0, ELO0 = 27, 1800
LEAGUES_FE = ["Serie A", "Bundesliga", "Ligue 1", "La liga"]   # EPL is the base
# reported amounts -> gross eur a year. spanish net -> gross with a rough 1.9 (top tax rate ~47%), only on a low quality row
CONV = {"gbp_week": 52 * GBP_EUR, "eur_year_gross": 1, "eur_month_gross": 12, "eur_year_net": 1.9, "eur_year_unknown": 1}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(s.split())


# ── 1. player model on the reported wages ─────────────────────────────────────

def reported() -> pd.DataFrame:
    """reported wages with tm id, mv on 1 sept, age, club elo"""
    r = pd.read_csv(ROOT / "manual" / "reported_wages.csv")
    pl = pd.read_csv(TM / "players.csv", usecols=["player_id", "name", "date_of_birth"])
    pl["n"] = pl["name"].map(norm)
    ap = pd.read_csv(TM / "appearances.csv", usecols=["player_id", "player_club_id", "date"])
    d = pd.to_datetime(ap["date"])
    ap["season"] = d.dt.year - (d.dt.month < 7)
    played = set(zip(ap["player_id"], ap["player_club_id"], ap["season"]))
    # name match, and he has to have played for that club that season (kills the namesakes)
    ids = []
    for _, x in r.iterrows():
        c = pl[pl["n"] == norm(x["player"])]
        if c.empty:
            c = pl[pl["n"].str.contains(norm(x["player"]).split()[-1], regex=False)]
        c = c[[(p, x["tm_club_id"], x["season"]) in played for p in c["player_id"]]]
        ids.append(c["player_id"].iloc[0] if len(c) == 1 else np.nan)
    r["tm_id"] = ids
    if r["tm_id"].isna().any():
        print("  not linked:", r.loc[r["tm_id"].isna(), "player"].tolist())
    r = r.dropna(subset=["tm_id"]).astype({"tm_id": int})
    r["wage"] = r["amount"] * r["unit"].map(CONV)

    v = pd.read_csv(TM / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"])
    r["d"] = pd.to_datetime(r["season"].astype(str) + "-09-01")
    q = pd.merge_asof(r.sort_values("d"), v.sort_values("date").rename(columns={"player_id": "tm_id"}),
                      left_on="d", right_on="date", by="tm_id", direction="backward")
    # no value before 1 sept (rico lewis): his first one
    first = v.sort_values("date").drop_duplicates("player_id").set_index("player_id")["market_value_in_eur"]
    q["mv"] = q["market_value_in_eur"].fillna(q["tm_id"].map(first))
    dob = pd.to_datetime(pl.set_index("player_id")["date_of_birth"], errors="coerce")
    q["age"] = (q["d"] - q["tm_id"].map(dob)).dt.days / 365.25
    elo = pd.read_parquet(PROCESSED_DIR / "elo_aug1.parquet").set_index(["season", "club_id"])["elo"]
    q["elo"] = [elo.get((s, c), np.nan) for s, c in zip(q["season"], q["tm_club_id"])]
    return q.drop(columns=["date", "market_value_in_eur"])


def design(d: pd.DataFrame, terms) -> np.ndarray:
    cols = [np.ones(len(d)), np.log(d["mv"].clip(lower=1e5))]
    if "league" in terms:
        cols += [(d["league"] == lg).astype(float) for lg in LEAGUES_FE]
    if "age" in terms:
        cols.append(d["age"] - AGE0)
    if "elo" in terms:
        cols.append((d["elo"] - ELO0) / 100)
    return np.column_stack(cols)


def loo(d, terms) -> float:
    y, A = np.log(d["wage"]).to_numpy(), design(d, terms)
    e = []
    for i in range(len(d)):
        m = np.arange(len(d)) != i
        b = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
        e.append(y[i] - A[i] @ b)
    return float(np.sqrt(np.mean(np.square(e))))


def player_wage(pm: dict, mv, age, elo, league="EPL") -> np.ndarray:
    """gross eur a year for a contract signed now. pm = wage_fit.json["player"]"""
    lg = pm["league"].get(league, 0.0) if isinstance(league, str) else np.array([pm["league"].get(x, 0.0) for x in league])
    return np.exp(pm["a"] + pm["b"] * np.log(np.clip(mv, 1e5, None)) + lg + pm["c"] * (np.asarray(age) - AGE0)
                  + pm["d"] * (np.asarray(elo) - ELO0) / 100)


# ── 2. club bills ─────────────────────────────────────────────────────────────

def squads() -> pd.DataFrame:
    """one row per player per EPL club-season with minutes, mv on 1 Sept and age"""
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet",
                          columns=["season", "player_id", "name", "team", "league", "minutes", "age"])
    d = war[(war["league"] == "EPL") & (war["minutes"] > 0)].copy()
    d["mv"] = np.nan
    for s, g in d.groupby("season"):
        d.loc[g.index, "mv"] = g["player_id"].map(start_values(s)).to_numpy()
    d["age"] = d["age"].fillna(d["age"].median())
    return d


def bills() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "manual" / "wage_bills.csv")
    b["wages_eur"] = b["wages_gbp_m"] * GBP_EUR * 1e6
    return b


def club_predict(p, sq: pd.DataFrame) -> pd.Series:
    a, tr, beta, g = p
    w = np.exp(a + tr * (sq["season"] - 2019)) * sq["mv"].clip(lower=1e5) ** beta * np.exp(g * (sq["age"] - AGE0))
    return w.groupby([sq["season"], sq["team"]]).sum()


def club_fit(sq, b):
    y = np.log(b.set_index(["season", "team"])["wages_eur"])

    def res(x):
        return np.log(club_predict(x, sq).reindex(y.index).to_numpy()) - y.to_numpy()

    r = least_squares(res, [np.log(0.1), 0.0, 1.0, 0.0])
    return tuple(r.x), float(np.sqrt(np.mean(r.fun ** 2)))


def squad_elo(sq: pd.DataFrame) -> pd.Series:
    """club elo on 1 aug for each squad row: understat team -> the tm club most of its players played league games for"""
    ap = pd.read_csv(TM / "appearances.csv", usecols=["player_id", "player_club_id", "date", "competition_id"])
    ap = ap[ap["competition_id"] == "GB1"]
    d = pd.to_datetime(ap["date"])
    ap["season"] = d.dt.year - (d.dt.month < 7)
    cl = ap.groupby(["season", "player_id"])["player_club_id"].agg(lambda x: x.mode().iloc[0])
    tm = sq["player_id"].map(us_to_tm())
    club = pd.Series([cl.get((s, t), np.nan) for s, t in zip(sq["season"], tm)], index=sq.index)
    team_club = club.groupby([sq["season"], sq["team"]]).agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    elo = pd.read_parquet(PROCESSED_DIR / "elo_aug1.parquet").set_index(["season", "club_id"])["elo"]
    return pd.Series([elo.get((s, team_club.get((s, t))), np.nan) for s, t in zip(sq["season"], sq["team"])], index=sq.index)


# ── 3. contract length ────────────────────────────────────────────────────────

def contracts():
    """contract length (years) by age at signing. big five paid signings 2019-2024 still at the club in the snapshot"""
    pl = pd.read_csv(TM / "players.csv", usecols=["player_id", "date_of_birth", "contract_expiration_date",
                                                  "current_club_id", "current_club_domestic_competition_id"])
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "to_club_id", "transfer_fee"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"])
    snap = pd.to_datetime(pd.read_csv(TM / "player_valuations.csv", usecols=["date"])["date"]).max()
    tr = tr[tr["transfer_date"] <= snap]
    # last move per player = the contract he's on now
    last = tr.sort_values("transfer_date").drop_duplicates("player_id", keep="last")
    d = last.merge(pl, on="player_id")
    d = d[(d["to_club_id"] == d["current_club_id"]) & d["current_club_domestic_competition_id"].isin(["GB1", "ES1", "L1", "IT1", "FR1"])]
    # paid moves only, free agents / loans returning have odd lengths
    d = d[(d["transfer_fee"] > 0) & d["transfer_date"].between("2019-06-01", "2024-09-30")]
    d["end"] = pd.to_datetime(d["contract_expiration_date"], errors="coerce")
    d["years"] = (d["end"] - d["transfer_date"]).dt.days / 365.25
    d["age"] = (d["transfer_date"] - pd.to_datetime(d["date_of_birth"], errors="coerce")).dt.days / 365.25
    d = d.dropna(subset=["years", "age"])
    d = d[d["years"].between(0.5, 7)]
    d["age_b"] = d["age"].round().clip(18, 34)
    c = d.groupby("age_b")["years"].agg(["median", "mean", "size"]).reset_index().rename(columns={"age_b": "age"})
    # smooth: years on age, quadratic weighted by n, so thin ages don't jump around
    w = np.polyfit(c["age"], c["mean"], 2, w=np.sqrt(c["size"]))
    c["years_fit"] = np.polyval(w, c["age"])
    return c, d


if __name__ == "__main__":
    # 1. player model
    r = reported()
    ok = r[r["quality"] == "ok"].reset_index(drop=True)
    print(f"reported wages: {len(r)} ({len(ok)} used, la liga rows low quality) - "
          + ", ".join(f"{lg} {n}" for lg, n in ok["league"].value_counts().items()))
    print(f"  ages {ok['age'].min():.0f}-{ok['age'].max():.0f}, club elo {ok['elo'].min():.0f}-{ok['elo'].max():.0f}")
    print("\nleave one out rms error, log wage:")
    res = {}
    for t in [("league",), ("league", "age"), ("league", "elo"), ("league", "age", "elo")]:
        res[t] = loo(ok, t)
        print(f"  log mv + {' + '.join(t):20s} {res[t]:.3f}")
    best = min(res, key=res.get)
    b = np.linalg.lstsq(design(ok, best), np.log(ok["wage"]), rcond=None)[0]
    pm = {"a": b[0], "b": b[1], "league": {"EPL": 0.0}, "c": 0.0, "d": 0.0, "terms": list(best), "loo_rms": res[best]}
    k = 2
    if "league" in best:
        pm["league"].update({lg: b[k + i] for i, lg in enumerate(LEAGUES_FE)})
        k += len(LEAGUES_FE)
    if "age" in best:
        pm["c"] = b[k]
        k += 1
    if "elo" in best:
        pm["d"] = b[k]
    # la liga isn't in the fit, give it the bundesliga level (closest by club revenues), flagged in the report
    pm["league"]["La liga"] = pm["league"]["Bundesliga"]
    print(f"\nkept: {' + '.join(best)}. wage ~ mv^{pm['b']:.2f}, x{np.exp(pm['c']):.3f} per year of age, "
          f"x{np.exp(pm['d']):.2f} per 100 club elo")
    print("  league levels vs EPL: " + ", ".join(f"{lg} x{np.exp(v):.2f}" for lg, v in pm["league"].items() if lg != "EPL"))
    print(f"  same mv: a 31-year-old costs {np.exp(4 * pm['c']):.2f}x a 27-year-old, a 21-year-old {np.exp(-6 * pm['c']):.2f}x")
    low = r[r["quality"] == "low"]
    print("\nla liga rows (not fitted), reported vs model, eur m:")
    for _, x in low.iterrows():
        p = player_wage(pm, x["mv"], x["age"], x["elo"], "La liga")
        print(f"  {x['player']:20s} {x['wage'] / 1e6:5.1f} vs {p / 1e6:5.1f}")

    # 2. club bills: own fit, then as a check on the player model
    sq = squads().dropna(subset=["mv"])
    bl = bills()
    have = sq.groupby(["season", "team"]).size()
    bl = bl[[k in have.index for k in zip(bl["season"], bl["team"])]]
    cp, cr = club_fit(sq, bl)
    print(f"\nclub bills ({len(bl)} EPL club-seasons): bill ~ sum mv^{cp[2]:.2f} x exp({cp[3]:.3f} x (age - 27)), "
          f"rms {cr:.3f}")
    sq["elo"] = squad_elo(sq)
    sq["pw"] = player_wage(pm, sq["mv"], sq["age"], sq["elo"].fillna(ELO0))
    s = sq.groupby(["season", "team"])["pw"].sum()
    bl["share"] = [s.get(k, np.nan) / w for k, w in zip(zip(bl["season"], bl["team"]), bl["wages_eur"])]
    print(f"player model summed over the squad / accounts wage bill: median {bl['share'].median():.2f} "
          f"(sd of log {np.log(bl['share']).std():.2f})")
    print("  by season: " + ", ".join(f"{k} {v:.2f}" for k, v in bl.groupby("season")["share"].median().items()))
    print("  the rest is staff, bonuses, players without league minutes. rising over time = market values inflating "
          "faster than wages")

    # 3. contracts
    c, raw = contracts()
    print(f"\ncontract length at signing, {len(raw):,} paid big five signings 2019-24 still at the club:")
    print(c.round(2).to_string(index=False))

    json.dump({"player": pm, "club": dict(zip(["a", "trend", "beta", "g"], cp)), "club_rms": cr,
               "squad_share_median": float(bl["share"].median())},
              open(PROCESSED_DIR / "wage_fit.json", "w"), indent=1)
    c.to_parquet(PROCESSED_DIR / "contracts.parquet", index=False)
    r.to_parquet(PROCESSED_DIR / "reported_wages.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'wage_fit.json'}, contracts.parquet, reported_wages.parquet")
