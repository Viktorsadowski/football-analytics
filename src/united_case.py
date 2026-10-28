#!/usr/bin/env python3
"""
Step 11: the case. Manchester United, summer 2024, the window treated as still open. what should they have done,
against what they did, both graded on 2024-25.

  squad        united's players before the window: everyone who played for them in 2023-24 + loan returns, minus
               contracts that ran out / loans that ended (Varane, Martial, Amrabat, ...). from transfermarkt
  rules        25-man senior squad (U21 = born 2003 or later, not counted), at least 8 homegrown in the 25.
               homegrown approximated from transfer history: 3+ years at English / Welsh clubs between 15 and 21
  minutes      4-2-3-1: GK, LB, CB x2, RB, DM, CM, LW, AM, RW, FW. a player can fill his main position (transfermarkt)
               and any other he started 20%+ of his games in over 2021-24 (transfermarkt lineups), no penalty there
               since his value was measured with those games in. minutes go to the best (player, position) pairs by
               value, each capped at 92% of the season x (1 - expected injury share), next man up covers the rest.
               a group that runs out of players gets replacement level (0 PAR), i.e. a cheap body
  values       transfer_value.py at united's elo. buys: new contract, market fee (mv on 1 June 2024, before the
               window moves it). incumbents: years left on their deal and their wage (reported where we have it,
               else the wage model at the value he had when he joined). sales: 0.8 x market value (forced sale)
  budget       EUR 150m net on fees. the real net spend was lower (see the printout), and PSR was tight, so this is
               on the generous side. wages aren't in the budget, they're in the value
  objectives   A  max window value (squad npv after - before - fees + sales) with 2024-25 points >= standing still
               B  max 2024-25 points above replacement with window value >= 0
               greedy best-improvement over buy / sell / swap moves, both start from the pre-window squad
  graded       on 2024-25 with what we have after the window: league minutes (transfermarkt appearances) and the
               market value change to 1 Sept 2025. understat stops at 2023-24 so there's no rapm for 2024-25,
               the grade is the market's verdict + whether they played, not the model's

Output: data/processed/united_case.json, data/processed/united_case_moves.parquet

  python src/united_case.py
"""

import json
import re

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT
from tm_match import us_to_tm
from transfer_value import FULL, UNITED, context, fee_hat, load, value

TM = RAW_DIR / "transfermarkt"
AS_OF = 2023                         # market_value rows as of the end of 2023-24
PRICE_DATE = "2024-06-01"
WIN_FROM, WIN_TO = "2024-07-01", "2024-09-15"
BUDGET = 150e6
FORCED = 0.8
SLOTS = {"GK": 1, "LB": 1, "CB": 2, "RB": 1, "DM": 1, "CM": 1, "LW": 1, "AM": 1, "RW": 1, "FW": 1}
SECOND_SHARE, SECOND_MIN = 0.20, 5      # share of his starts / starts, for a second position to count
CAP = 0.92
MAX_SENIOR, MIN_HG = 25, 8
U21_BORN = "2003-01-01"
SUB = {"Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "LB", "Right-Back": "RB", "Defensive Midfield": "DM",
       "Central Midfield": "CM", "Attacking Midfield": "AM", "Left Winger": "LW", "Right Winger": "RW",
       "Left Midfield": "LW", "Right Midfield": "RW", "Centre-Forward": "FW", "Second Striker": "FW", "Sweeper": "CB"}
GRP = {"GK": "GK", "LB": "DEF", "CB": "DEF", "RB": "DEF", "DM": "MID", "CM": "MID", "LW": "ATT", "AM": "ATT",
       "RW": "ATT", "FW": "ATT"}
GRID = np.linspace(0, 1, 11)
POOL_PER_SLOT = 10
SURPLUS = 0.5          # seller rule: below this share of his club's minutes he's surplus there, sellable whoever owns him
TRACK = 1500           # big five league minutes over the last 3 seasons. below it: not a target, and not a first choice
                       # at united (max half the minutes). war.py's shrink still left 337-minute players (girona's solis)
                       # as top targets, the shrink is fitted on players who did play the season after so it's too kind
                       # to short records. a scouting rule, not a fit
MAX_MOVES = 6           # buys + sales + swaps. a real window is a handful of deals, without a cap A sold half the squad


# ── data ──────────────────────────────────────────────────────────────────────

def tm_tables():
    pl = pd.read_csv(TM / "players.csv", usecols=["player_id", "name", "date_of_birth", "sub_position",
                                                  "contract_expiration_date", "current_club_id"])
    pl["dob"] = pd.to_datetime(pl["date_of_birth"], errors="coerce")
    tr = pd.read_csv(TM / "transfers.csv", usecols=["player_id", "transfer_date", "from_club_id", "to_club_id",
                                                    "from_club_name", "to_club_name", "transfer_fee", "player_name"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    v = pd.read_csv(TM / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"])
    ap = pd.read_csv(TM / "appearances.csv", usecols=["player_id", "player_club_id", "date", "competition_id",
                                                      "minutes_played"])
    ap["date"] = pd.to_datetime(ap["date"])
    return pl, tr, v, ap


def value_on(v, ids, date) -> pd.Series:
    x = v[(v["date"] <= date) & v["player_id"].isin(ids)].sort_values("date").drop_duplicates("player_id", keep="last")
    return x.set_index("player_id")["market_value_in_eur"]


def english_names(tr) -> set:
    """names transfermarkt uses for english clubs (premier league club ids in clubs.csv, youth teams stripped)"""
    clubs = pd.read_csv(TM / "clubs.csv", usecols=["club_id", "domestic_competition_id"])
    gb = set(clubs.loc[clubs["domestic_competition_id"] == "GB1", "club_id"])
    names = set(tr.loc[tr["to_club_id"].isin(gb), "to_club_name"]) | set(tr.loc[tr["from_club_id"].isin(gb), "from_club_name"])
    return {strip_youth(n) for n in names}


def strip_youth(n) -> str:
    return re.sub(r"\s+(U\d\d|Youth|Academy|B)$", "", str(n)).strip()


def homegrown(pids, pl, tr) -> pd.Series:
    """3+ years at english clubs between his 15th and 21st birthday, from the transfer timeline. before his first
    recorded move he was at that move's from-club"""
    eng = english_names(tr)
    dob = pl.set_index("player_id")["dob"]
    out = {}
    for p in pids:
        t = tr[tr["player_id"] == p].dropna(subset=["transfer_date"]).sort_values("transfer_date")
        b = dob.get(p)
        if pd.isna(b):
            out[p] = False
            continue
        a, z = b + pd.DateOffset(years=15), b + pd.DateOffset(years=21)
        if t.empty:
            out[p] = False
            continue
        spells = [(pd.Timestamp("1900-01-01"), t["transfer_date"].iloc[0], t["from_club_name"].iloc[0])]
        for i in range(len(t)):
            end = t["transfer_date"].iloc[i + 1] if i + 1 < len(t) else pd.Timestamp("2100-01-01")
            spells.append((t["transfer_date"].iloc[i], end, t["to_club_name"].iloc[i]))
        days = sum(max((min(e, z) - max(s, a)).days, 0) for s, e, c in spells if strip_youth(c) in eng)
        out[p] = days >= 3 * 365
    return pd.Series(out)


def pre_window(pl, tr, ap) -> pd.DataFrame:
    """united's players when the window opens: his last move up to 30 June 2024 was to united, and he's been
    around the first team (league / cup minutes 2021-24, or joined since 2021). contracts that ran out on 1 July
    (to Without Club, or a free) are gone before the window. first version took everyone who played for united in
    2023-24 and got reguilón (a loan, back at spurs) and missed malacia (injured all season)"""
    pre = tr[tr["transfer_date"] <= "2024-06-30"].sort_values("transfer_date").drop_duplicates("player_id", keep="last")
    ids = set(pre.loc[pre["to_club_id"] == UNITED, "player_id"])
    recent = set(ap.loc[(ap["player_club_id"] == UNITED) & (ap["date"] >= "2021-07-01"), "player_id"])
    recent |= set(tr.loc[(tr["to_club_id"] == UNITED) & tr["transfer_date"].between("2021-07-01", "2024-06-30"), "player_id"])
    ids &= recent
    expiry = tr[(tr["from_club_id"] == UNITED) & (tr["transfer_date"] == WIN_FROM) & (tr["transfer_fee"].fillna(0) <= 0)]
    ids -= set(expiry["player_id"])
    sq = pl[pl["player_id"].isin(ids)].copy()
    sq["slot"] = sq["sub_position"].map(SUB).fillna("CM")
    return sq


def actual_window(tr):
    w = tr[tr["transfer_date"].between(WIN_FROM, WIN_TO)]
    buys = w[(w["to_club_id"] == UNITED) & (w["transfer_fee"] > 0)]
    sales = w[(w["from_club_id"] == UNITED) & ~w["to_club_name"].str.contains("Man Utd", na=False)]
    # contracts out on 1 july aren't window moves, pre_window() already dropped them
    sales = sales[~((sales["transfer_date"] == WIN_FROM) & (sales["transfer_fee"].fillna(0) <= 0))]
    return buys, sales


# ── model inputs per player ───────────────────────────────────────────────────

def model_rows(tm_ids, mv_rows):
    """market_value rows (as of AS_OF) by tm id. players we have no row for (youth, outside the big five) get a
    replacement-level row: val = repl, so 0 PAR whatever they play"""
    us = us_to_tm()
    t2u = {t: u for u, t in us.items()}
    cur = mv_rows[mv_rows["season"] == AS_OF].set_index("player_id")
    rows = []
    for t in tm_ids:
        u = t2u.get(t)
        if u is not None and u in cur.index:
            r = cur.loc[u].copy()
            if isinstance(r, pd.DataFrame):
                r = r.sort_values("proj_min").iloc[-1]
        else:
            r = pd.Series({"val_cal": np.nan, "repl": np.nan, "grp": None, "age": np.nan, "par": 0.0,
                           "proj_min": 0.0, "exp_inj_share": 0.08, "mv": np.nan, "name": None})
        r["tm_id"] = t
        rows.append(r)
    d = pd.DataFrame(rows).reset_index(drop=True)
    return d


def npv_grid(d, ctx, elo, years=None, wage=None, fee=None):
    """npv at each role share on GRID (0..1), so the search can interpolate instead of re-running value()"""
    out = np.zeros((len(d), len(GRID)))
    par1 = np.zeros((len(d), len(GRID)))
    for j, g in enumerate(GRID):
        x = value(d, elo, ctx, role_share=np.full(len(d), g), years=years, wage=wage, fee=fee, repl_wage=False)
        out[:, j] = x["npv"].to_numpy()
        par1[:, j] = ((d["val_cal"] - d["repl"]).to_numpy() * x["role_min"].to_numpy() / 90 * ctx["params"]["ppx"])
    return out, par1


def positions(ids) -> pd.Series:
    """tm id -> positions he can fill: main one + any he started 20%+ (and 5+) of his games in, 2021-24"""
    g = pd.read_csv(TM / "game_lineups.csv", usecols=["date", "player_id", "type", "position"])
    g = g[(g["type"] == "starting_lineup") & g["date"].between("2021-07-01", "2024-06-30")]
    if ids is not None:
        g = g[g["player_id"].isin(set(ids))]
    g["slot"] = g["position"].map(SUB)
    c = g.dropna(subset=["slot"]).groupby(["player_id", "slot"]).size()
    sh = c / c.groupby(level=0).transform("sum")
    ok = c[(sh >= SECOND_SHARE) & (c >= SECOND_MIN)]
    return ok.reset_index().groupby("player_id")["slot"].agg(set)


def add_slots(df, pos):
    df["slots"] = [tuple(sorted({m} | pos.get(t, set()))) for m, t in zip(df["slot"], df["tm_id"])]
    return df


def track_minutes(ap) -> pd.Series:
    """tm id -> big five league minutes 2021-22 to 2023-24"""
    x = ap[ap["competition_id"].isin(["GB1", "ES1", "L1", "IT1", "FR1"]) & ap["date"].between("2021-07-01", "2024-06-30")]
    return x.groupby("player_id")["minutes_played"].sum()


# ── squad evaluation ──────────────────────────────────────────────────────────

def allocate(sq: pd.DataFrame, detail=False):
    """role share (of a full season, before injuries) per player + seasons' worth of minutes nobody covers, by slot.
    greedy over (player, position) pairs, best value first. a player's minutes are shared over his positions"""
    inj = sq["exp_inj_share"].to_numpy()
    cap = CAP * FULL * (1 - inj)
    cap = np.where(sq["m3"].to_numpy() < TRACK, np.minimum(cap, 0.5 * FULL * (1 - inj)), cap)
    left = {k: n * FULL for k, n in SLOTS.items()}
    val = sq["val_cal"].to_numpy()
    main = sq["slot"].to_numpy()
    # main position first on ties, so a CB isn't moved to RB for nothing
    pairs = sorted(((val[i], k == main[i], i, k) for i, ks in enumerate(sq["slots"]) for k in ks if k in left),
                   reverse=True)
    used = np.zeros(len(sq))
    who = []
    for _, _, i, k in pairs:
        m = min(cap[i] - used[i], left[k])
        if m > 0:
            used[i] += m
            left[k] -= m
            who.append((k, sq["name"].iloc[i], m, sq["tm_id"].iloc[i], val[i]))
    if detail:
        return pd.DataFrame(who, columns=["pos", "name", "minutes", "tm_id", "val_cal"])
    share = used / (FULL * (1 - inj))
    short = {k: max(v, 0) / FULL for k, v in left.items()}
    return share, short


REPL_COST = None      # pv of a replacement body for one slot-season, set in main (3 seasons, a cheap squad player
                      # gets signed and stays a while)


def evaluate(sq: pd.DataFrame):
    share, short = allocate(sq)
    npv = np.array([np.interp(s, GRID, g) for s, g in zip(share, sq["grid"])])
    par1 = np.array([np.interp(s, GRID, g) for s, g in zip(share, sq["pgrid"])])
    return share, float(npv.sum()) - sum(short.values()) * REPL_COST, float(par1.sum())


def depth(sq) -> str:
    """who plays where, minutes per position, best first"""
    w = allocate(sq, detail=True)
    order = list(SLOTS)
    w["o"] = w["pos"].map(order.index)
    return "\n".join(f"    {k:3s} " + ", ".join(f"{n} {m:.0f}" for n, m in zip(g["name"], g["minutes"]))
                     for k, g in w.sort_values(["o", "minutes"], ascending=[True, False]).groupby("pos", sort=False))


def rules_ok(sq: pd.DataFrame) -> bool:
    senior = sq[~sq["u21"]]
    return len(senior) <= MAX_SENIOR and (~senior["hg"]).sum() <= MAX_SENIOR - MIN_HG


def search(base, pool, objective, base_npv, base_pts, max_moves=MAX_MOVES):
    """greedy best-improvement. state = squad frame + fees paid + sales received"""
    sq, fees, sales, log = base.copy(), 0.0, 0.0, []

    def score(s, f, sl):
        _, npv, pts = evaluate(s)
        wv = npv - base_npv - f + sl
        ok = (f - sl) <= BUDGET and rules_ok(s)
        if objective == "A":
            ok = ok and pts >= base_pts - 1e-9
            return (wv if ok else -np.inf), wv, pts
        ok = ok and wv >= 0
        return (pts if ok else -np.inf), wv, pts

    cur, _, _ = score(sq, fees, sales)
    if not np.isfinite(cur):
        cur = -1e18          # the pre-window squad can break the 25 rule (loan returns), moves have to fix it
    for _ in range(max_moves):
        best = None
        out_cands = list(sq.index)
        in_cands = [i for i in pool.index if pool.loc[i, "tm_id"] not in set(sq["tm_id"])]
        moves = [("sell", o, None) for o in out_cands] + [("buy", None, i) for i in in_cands]
        moves += [("swap", o, i) for o in out_cands for i in in_cands
                  if set(sq.loc[o, "slots"]) & set(pool.loc[i, "slots"])]
        for kind, o, i in moves:
            s, f, sl = sq, fees, sales
            if o is not None:
                sl = sl + sq.loc[o, "sale"]
                s = s.drop(index=o)
            if i is not None:
                f = f + pool.loc[i, "fee"]
                s = pd.concat([s, pool.loc[[i]]])
            sc, wv, pts = score(s, f, sl)
            if sc > cur + 1e-6 and (best is None or sc > best[0]):
                best = (sc, kind, o, i, s, f, sl, wv, pts)
        if best is None:
            break
        cur, kind, o, i, sq, fees, sales, wv, pts = best
        log.append({"move": kind, "out": base.loc[o, "name"] if o is not None and o in base.index else
                    (None if o is None else "(bought earlier)"),
                    "in": pool.loc[i, "name"] if i is not None else None, "net_spend": fees - sales,
                    "window_value": wv, "points": pts})
    return sq, fees, sales, log


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mv, curve, params, wf, con, elo = load()
    ctx = context(mv, curve, params, wf, con)
    u_elo = float(elo.get((AS_OF + 1, UNITED)))
    from transfer_value import EMPLOYER_NI
    from wages import player_wage as _pw
    from market_value import DISCOUNT
    REPL_COST = float(_pw(wf["player"], ctx["repl_mv"], 27, u_elo)) * (1 + EMPLOYER_NI) * sum(1 / (1 + DISCOUNT) ** k for k in range(3))
    pl, tr, v, ap = tm_tables()

    # ── united before the window ──
    pre = pre_window(pl, tr, ap)
    d = model_rows(pre["player_id"], mv)
    d = d.drop(columns=[c for c in ("name",) if c in d]).merge(
        pre[["player_id", "name", "dob", "slot", "contract_expiration_date", "current_club_id"]],
        left_on="tm_id", right_on="player_id")
    d["age"] = d["age"].fillna((pd.Timestamp("2024-07-01") - d["dob"]).dt.days / 365.25)
    d["mv"] = d["tm_id"].map(value_on(v, d["tm_id"], PRICE_DATE)).fillna(5e5)
    d["val_cal"] = d["val_cal"].fillna(d["repl"])
    rep = GRP
    d["grp"] = d["grp"].fillna(d["slot"].map(rep))
    d["repl"] = d["repl"].fillna(d["grp"].map(params["repl"]))
    d["val_cal"] = d["val_cal"].fillna(d["repl"])
    d["u21"] = d["dob"] >= U21_BORN
    d["hg"] = d["tm_id"].map(homegrown(d["tm_id"], pl, tr)).fillna(False)
    pos = positions(None)
    d = add_slots(d, pos)

    # years left: transfermarkt's contract end if he's still there in the snapshot, else join + contract length
    end = pd.to_datetime(d["contract_expiration_date"], errors="coerce")
    # join = his last real move to united before the window: a fee, or up from the academy. loan returns
    # don't start a new contract (sancho came back on 30 june 2024, his deal is from 2021)
    j = tr[(tr["to_club_id"] == UNITED) & (tr["transfer_date"] <= "2024-06-30")
           & ((tr["transfer_fee"] > 0) | tr["from_club_name"].str.contains("Man Utd", na=False))]
    j0 = tr[(tr["to_club_id"] == UNITED) & (tr["transfer_date"] <= "2024-06-30")].groupby("player_id")["transfer_date"].min()
    join = j.groupby("player_id")["transfer_date"].max().reindex(j0.index).fillna(j0)
    # 1. manual/united_contracts_2024.csv: expiry as of summer 2024 from press lists (source per row)
    # 2. else the snapshot's contract end, if he never moved after the window (the snapshot runs to 2026, a later
    #    move = a different contract) and it's no more than 6.5 years after he joined (longer = renewed later)
    # 3. else join + typical contract length at his age then
    moved = set(tr.loc[tr["transfer_date"] > "2024-06-30", "player_id"])
    jd = d["tm_id"].map(join)
    ok_end = (d["current_club_id"] == UNITED) & (end > "2024-07-01") & ~d["tm_id"].isin(moved) \
        & ((end - jd).dt.days / 365.25 <= 6.5)
    yl = ((end - pd.Timestamp("2024-07-01")).dt.days / 365.25).round()
    from transfer_value import contract_years
    age_join = ((jd - d["dob"]).dt.days / 365.25).fillna(d["age"])
    fallback = ((jd + pd.to_timedelta(contract_years(con, age_join) * 365.25, unit="D") - pd.Timestamp("2024-07-01"))
                .dt.days / 365.25).round()
    known = pd.read_csv(ROOT / "manual" / "united_contracts_2024.csv").set_index("player")["expiry_year"] - 2024
    d["years_left"] = np.clip(d["name"].map(known).fillna(pd.Series(np.where(ok_end, yl, fallback.fillna(2)),
                                                                     index=d.index)), 1, 6).astype(int)
    d["years_src"] = np.where(d["name"].isin(known.index), "press", np.where(ok_end, "snapshot", "join+typical"))
    # wage: reported (press, wages.py) where we have one, else the model at his value when he joined
    rw = pd.read_parquet(PROCESSED_DIR / "reported_wages.parquet")
    rw = rw[rw["tm_club_id"] == UNITED].sort_values("season").drop_duplicates("tm_id", keep="last").set_index("tm_id")["wage"]
    mv_join = pd.Series([value_on(v, [t], dt).get(t, np.nan) if pd.notna(dt) else np.nan
                         for t, dt in zip(d["tm_id"], jd)], index=d.index)
    from wages import player_wage
    w_model = player_wage(wf["player"], mv_join.fillna(d["mv"]).to_numpy(), age_join.to_numpy(), u_elo)
    d["wage_now"] = d["tm_id"].map(rw).fillna(pd.Series(w_model, index=d.index))
    d["wage_src"] = np.where(d["tm_id"].isin(rw.index), "reported", "model")
    d["sale"] = FORCED * d["mv"]
    d["m3"] = d["tm_id"].map(track_minutes(ap)).fillna(0)
    g, pg = npv_grid(d, ctx, u_elo, years=d["years_left"].to_numpy(), wage=d["wage_now"].to_numpy(dtype=float),
                     fee=np.zeros(len(d)))
    d["grid"], d["pgrid"] = list(g), list(pg)
    d.index = [f"u{i}" for i in range(len(d))]
    share0, npv0, pts0 = evaluate(d)
    d["share0"] = share0
    d["npv0"] = [np.interp(x, GRID, g) / 1e6 for x, g in zip(share0, d["grid"])]

    print(f"united before the window: {len(d)} players ({(~d['u21']).sum()} senior, "
          f"{(d['hg'] & ~d['u21']).sum()} homegrown senior), elo {u_elo:.0f}")
    show = d.assign(mv=d["mv"] / 1e6, wage=d["wage_now"] / 1e6)[
        ["name", "slot", "slots", "age", "u21", "hg", "mv", "wage", "wage_src", "years_left", "years_src", "m3", "val_cal", "share0", "npv0"]]
    print(show.sort_values(["slot", "val_cal"], ascending=[True, False]).round(2).to_string(index=False))
    print(f"standing still: {pts0:.1f} points above replacement in 2024-25, squad npv {npv0 / 1e6:.0f}m")

    # ── targets ──
    cur = mv[(mv["season"] == AS_OF) & (mv["proj_min"] >= 900) & mv["age"].between(17, 31)].copy()
    t2u = us_to_tm()
    cur["tm_id"] = cur["player_id"].map(t2u)
    cur = cur.dropna(subset=["tm_id"]).astype({"tm_id": int})
    cur = cur[~cur["tm_id"].isin(d["tm_id"])].drop_duplicates("tm_id")
    cur = cur.merge(pl[["player_id", "dob", "sub_position"]].rename(columns={"player_id": "tm_id"}), on="tm_id")
    cur["m3"] = cur["tm_id"].map(track_minutes(ap)).fillna(0)
    cur = cur[cur["m3"] >= TRACK]
    # seller rule: clubs above united (elo 1 aug 2024) don't sell a starter to them. so a target has to be at a
    # club rated below united, or be surplus where he is: under half the minutes, or out on loan in 2023-24
    # (then his owner didn't want him). the owner = his last move before the window (a loan ending 30 june
    # counts, eric garcia belongs to barcelona again). removed kane, kimmich, kim, pedri, havertz, brahim from B
    own = tr[tr["transfer_date"] <= "2024-06-30"].sort_values("transfer_date").drop_duplicates("player_id", keep="last")
    cur["owner"] = cur["tm_id"].map(own.set_index("player_id")["to_club_id"])
    cur["owner_elo"] = [elo.get((AS_OF + 1, c), np.nan) for c in cur["owner"]]
    played = ap[ap["date"].between("2023-07-01", "2024-06-30") & ap["competition_id"].isin(["GB1", "ES1", "L1", "IT1", "FR1"])]
    main_club = played.groupby(["player_id", "player_club_id"])["minutes_played"].sum().reset_index() \
        .sort_values("minutes_played").drop_duplicates("player_id", keep="last").set_index("player_id")["player_club_id"]
    cur["loaned"] = cur["tm_id"].map(main_club).ne(cur["owner"]) & cur["tm_id"].map(main_club).notna()
    # share when fit, so an injured starter (pedri) isn't counted as surplus
    fit_share = cur["proj_share"] / (1 - cur["exp_inj_share"].fillna(0)).clip(lower=0.2)
    cur["surplus"] = (fit_share < SURPLUS) | cur["loaned"]
    cur["sellable"] = (cur["owner_elo"].fillna(0) < u_elo) | cur["surplus"]
    print(f"\nseller rule: {(~cur['sellable']).sum()} of {len(cur)} targets out (club above united and a starter there), "
          f"e.g. {', '.join(cur.loc[~cur['sellable']].nlargest(8, 'val_cal')['name'])}")
    cur = cur[cur["sellable"]]
    cur["slot"] = cur["sub_position"].map(SUB).fillna("CM")
    cur["mv"] = cur["tm_id"].map(value_on(v, cur["tm_id"], PRICE_DATE))
    cur = cur.dropna(subset=["mv"])
    cur = cur[cur["mv"] > 0]
    cur["fee"] = fee_hat(ctx["fee_b"], cur["mv"], cur["age"])
    cur["u21"] = cur["dob"] >= U21_BORN
    # starter-role surplus for the shortlist: best per slot by surplus and by value (so B has stars to pick from)
    st = value(cur, u_elo, ctx, role_share=np.full(len(cur), 0.85))
    cur["surplus_starter"] = st["surplus"].to_numpy()
    pick = []
    for slot, gq in cur.groupby("slot"):
        pick += list(gq.nlargest(POOL_PER_SLOT, "surplus_starter").index)
        pick += list(gq[gq["fee"] <= BUDGET].nlargest(POOL_PER_SLOT // 2, "val_cal").index)
    pool = add_slots(cur.loc[sorted(set(pick))].copy(), pos)
    pool["hg"] = pool["tm_id"].map(homegrown(pool["tm_id"], pl, tr)).fillna(False)
    pg_, ppg = npv_grid(pool, ctx, u_elo, fee=np.zeros(len(pool)))
    pool["grid"], pool["pgrid"] = list(pg_), list(ppg)
    pool["sale"] = FORCED * pool["mv"]
    pool.index = [f"t{i}" for i in range(len(pool))]
    print(f"\nshortlist: {len(pool)} targets ({', '.join(f'{k} {n}' for k, n in pool['slot'].value_counts().items())})")

    # ── the two windows ──
    res = {}
    for obj in ("A", "B"):
        sq, fees, sales, log = search(d, pool, obj, npv0, pts0)
        share, npv, pts = evaluate(sq)
        res[obj] = {"squad": sq.assign(share=share), "fees": fees, "sales": sales, "log": log,
                    "window_value": npv - npv0 - fees + sales, "points": pts}
        res[obj]["xi"] = depth(sq)
        print(f"\nobjective {obj}: {pts:.1f} points (standing still {pts0:.1f}), window value "
              f"{(npv - npv0 - fees + sales) / 1e6:+.0f}m, net spend {(fees - sales) / 1e6:.0f}m")
        for m in log:
            print(f"  {m['move']:5s} out: {str(m['out']):24s} in: {str(m['in']):24s} net spend {m['net_spend'] / 1e6:6.0f}m "
                  f"value {m['window_value'] / 1e6:+6.0f}m points {m['points']:.1f}")

    for obj in ("A", "B"):
        print(f"\nobjective {obj}, minutes by position:\n{res[obj]['xi']}")
    print(f"\nstanding still, minutes by position:\n{depth(d)}")

    # ── what they actually did, same model ──
    buys, sales_act = actual_window(tr)
    ab = model_rows(buys["player_id"], mv).drop(columns=["name"], errors="ignore").merge(
        pl[["player_id", "name", "dob", "sub_position"]], left_on="tm_id", right_on="player_id")
    ab["slot"] = ab["sub_position"].map(SUB).fillna("CM")
    ab = add_slots(ab, pos)
    ab["mv"] = ab["tm_id"].map(value_on(v, ab["tm_id"], PRICE_DATE))
    ab["u21"] = ab["dob"] >= U21_BORN
    ab["m3"] = ab["tm_id"].map(track_minutes(ap)).fillna(0)
    ab["hg"] = False
    ab["fee"] = ab["tm_id"].map(buys.set_index("player_id")["transfer_fee"])
    ab["repl"] = ab["repl"].fillna(ab["slot"].map(rep).map(params["repl"]))
    ab["val_cal"] = ab["val_cal"].fillna(ab["repl"])
    ab["grp"] = ab["grp"].fillna(ab["slot"].map(rep))
    ab["age"] = ab["age"].fillna((pd.Timestamp("2024-07-01") - ab["dob"]).dt.days / 365.25)
    g, pg = npv_grid(ab, ctx, u_elo, fee=np.zeros(len(ab)))
    ab["grid"], ab["pgrid"] = list(g), list(pg)
    ab.index = [f"a{i}" for i in range(len(ab))]
    sold_ids = set(sales_act["player_id"])
    act = pd.concat([d[~d["tm_id"].isin(sold_ids)], ab])
    sale_money = sales_act["transfer_fee"].fillna(0).sum()
    share, npv, pts = evaluate(act)
    act_value = npv - npv0 - ab["fee"].sum() + sale_money
    print(f"\nactual window: bought {', '.join(ab['name'])} for {ab['fee'].sum() / 1e6:.0f}m, sold / loaned "
          f"{', '.join(sales_act['player_name'])} for {sale_money / 1e6:.0f}m, net {(ab['fee'].sum() - sale_money) / 1e6:.0f}m")
    print(f"  same model: {pts:.1f} points (standing still {pts0:.1f}), window value {act_value / 1e6:+.0f}m, "
          f"squad rules {'ok' if rules_ok(act) else 'broken'}")
    print(f"  minutes by position:\n{depth(act)}")
    res["actual"] = {"squad": act.assign(share=share), "fees": ab["fee"].sum(), "sales": sale_money,
                     "window_value": act_value, "points": pts}

    # ── graded on 2024-25 ──
    l24 = ap[ap["date"].between("2024-08-01", "2025-06-30") & ap["competition_id"].isin(["GB1", "ES1", "L1", "IT1", "FR1"])]
    mins24 = l24.groupby("player_id")["minutes_played"].sum()
    mv25 = value_on(v, None if False else v["player_id"].unique(), "2025-09-01")
    rows = []
    for name, r in res.items():
        sq = r["squad"]
        newb = sq[~sq["tm_id"].isin(d["tm_id"])]
        out = d[~d["tm_id"].isin(sq["tm_id"])]
        for _, x in newb.iterrows():
            rows.append({"window": name, "dir": "in", "name": x["name"], "fee": x.get("fee", np.nan),
                         "mv_jun24": x["mv"], "mv_sep25": mv25.get(x["tm_id"], np.nan), "min_24_25": mins24.get(x["tm_id"], 0)})
        for _, x in out.iterrows():
            rows.append({"window": name, "dir": "out", "name": x["name"], "fee": np.nan,
                         "mv_jun24": x["mv"], "mv_sep25": mv25.get(x["tm_id"], np.nan), "min_24_25": mins24.get(x["tm_id"], 0)})
    gr = pd.DataFrame(rows)
    gr["mv_change"] = gr["mv_sep25"] - gr["mv_jun24"]
    print("\ngraded on 2024-25 (league minutes anywhere in the big five, market value 1 June 2024 -> 1 Sept 2025, eur m):")
    for w, x in gr.groupby("window"):
        inn, out = x[x["dir"] == "in"], x[x["dir"] == "out"]
        print(f"  {w}: in {len(inn)}, avg {inn['min_24_25'].mean():.0f} league min, mv change {inn['mv_change'].sum() / 1e6:+.0f}m "
              f"| out {len(out)}, mv change {out['mv_change'].sum() / 1e6:+.0f}m")
        print(x.assign(fee=x["fee"] / 1e6, mv_jun24=x["mv_jun24"] / 1e6, mv_sep25=x["mv_sep25"] / 1e6)
              [["dir", "name", "fee", "mv_jun24", "mv_sep25", "min_24_25"]].round(1).to_string(index=False))

    # ── sensitivity: CL uplift on euros per point ──
    e = pd.read_parquet(PROCESSED_DIR / "eur_per_point.parquet")
    e = e[e["pts"].between(35, 85)]
    print(f"\nsensitivity: euros per point with the CL matchday/commercial uplift {e['eur_per_pt_uplift'].mean():.2f}m "
          f"vs {e['eur_per_pt'].mean():.2f}m base. contribution scales with it, so every window's value moves by "
          f"{e['eur_per_pt_uplift'].mean() / e['eur_per_pt'].mean() - 1:+.0%} of its contribution part")

    out = {k: {kk: (vv if not isinstance(vv, (pd.DataFrame,)) else None) for kk, vv in r.items() if kk != "squad"}
           for k, r in res.items()}
    out["standing_still"] = {"points": pts0, "npv": npv0}
    json.dump(out, open(PROCESSED_DIR / "united_case.json", "w"), indent=1, default=float)
    gr.to_parquet(PROCESSED_DIR / "united_case_moves.parquet", index=False)
    # depth charts per window, for season_sim.py (united's strength under each window)
    dc = [allocate(r["squad"], detail=True).assign(window=k) for k, r in res.items()]
    dc.append(allocate(d, detail=True).assign(window="standing still"))
    pd.concat(dc, ignore_index=True).to_parquet(PROCESSED_DIR / "united_case_depth.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'united_case.json'}, united_case_moves.parquet")
