#!/usr/bin/env python3
"""
Step 3b: RAPM with a box-score prior, and the forward test. Same three parts as box_prior.py in nba-moneyball.

  1. box prior     per 90 stats from understat (npxG, shots, xA, key passes, xGChain, xGBuildup, cards),
                   minutes per game, share of games started, position group -> O-RAPM and D-RAPM, weighted
                   ridge on plain RAPM. NO market value in here, or the market comparison later is circular.
                   no age yet either, that needs the transfermarkt birth dates (tm_match.py)
                   out of season: plain RAPM of season t is fitted on t, t-1, t-2 (rapm.py window), so for
                   season s the prior weights never see any t from s to s+3. that way the prior never saw
                   the season it's used for, or the season the forward test predicts
  2. priored ridge  minimize  |y - X b|^2 + lambda |b - b_prior|^2, same window and CV as rapm.py
  3. forward test  values from season s -> npxG per 90 in every segment of season s+1, minute-weighted
                   error. plain vs box prior only vs RAPM with prior

rapm_plain.parquet keeps the plain version, rapm.parquet becomes the priored one (same columns
+ o_prior, d_prior, prior=True), so everything after this picks it up without changes.

  python src/rapm.py          # plain first
  python src/box_prior.py     # then this
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import PROCESSED_DIR, SEASONS, season_label
from rapm import build_rows, load, pick_alpha, player_info, window

STATS = ["npxg", "shots", "xa", "key_passes", "xg_chain", "xg_buildup", "cards"]
GHOST_MIN = 450       # per 90 rates of low-minute guys pulled towards the league rate, 5 games of nothing special
MIN_FIT_MIN = 900     # only players with real minutes teach the prior
BLOCK = 4             # prior for season s never fitted on plain RAPM of s..s+3 (see 1. above)

# understat position codes -> groups. DML/DMR are wing-backs, they go with the full-backs
POS = {"GK": "GK", "DC": "CB", "DL": "FB", "DR": "FB", "DML": "FB", "DMR": "FB", "DMC": "DM", "MC": "CM",
       "ML": "W", "MR": "W", "AML": "W", "AMR": "W", "FWL": "W", "FWR": "W", "AMC": "AM", "FW": "FW"}
GROUPS = ["GK", "CB", "FB", "DM", "CM", "W", "AM", "FW"]


# ── 1. box prior ──────────────────────────────────────────────────────────────

def box_features(plain: pd.DataFrame) -> pd.DataFrame:
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet")
    sh = pd.read_parquet(PROCESSED_DIR / "shots.parquet")
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet", columns=["match_id", "season"])
    lu = lu.merge(m, on="match_id")
    lu["cards"] = lu["yellow_card"] + 3 * lu["red_card"]
    # npxG straight from the shots, lineups only have xG with penalties in
    npx = sh[sh["is_pen"] == 0].merge(m, on="match_id").groupby(["season", "player_id"])["xg"].sum().rename("npxg")

    g = lu.groupby(["season", "player_id"]).agg(
        minutes=("minutes", "sum"), games=("match_id", "size"), starts=("starter", "sum"),
        shots=("shots", "sum"), xa=("xa", "sum"), key_passes=("key_passes", "sum"),
        xg_chain=("xg_chain", "sum"), xg_buildup=("xg_buildup", "sum"), cards=("cards", "sum"))
    g = g.join(npx).fillna({"npxg": 0}).reset_index()

    for s in STATS:
        lg = g.groupby("season")[s].transform("sum") / g.groupby("season")["minutes"].transform("sum")
        g[f"{s}90"] = 90 * (g[s] + lg * GHOST_MIN) / (g["minutes"] + GHOST_MIN)
    g["mpg"] = g["minutes"] / g["games"]
    g["start_share"] = g["starts"] / g["games"]

    pos = lu[lu["starter"]].assign(grp=lambda d: d["position"].map(POS)) \
        .groupby(["season", "player_id"])["grp"].agg(lambda s: s.value_counts().index[0])
    g = g.join(pos, on=["season", "player_id"])
    g["grp"] = g["grp"].fillna("SUB")   # never started that season
    for p in GROUPS:
        g[f"pos_{p}"] = (g["grp"] == p).astype(float)

    r = plain[["season", "player_id", "o_rapm", "d_rapm"]]
    return g.merge(r, on=["season", "player_id"], how="left")


FEATS = [f"{s}90" for s in STATS] + ["mpg", "start_share"] + [f"pos_{p}" for p in GROUPS]


def fit_prior(f: pd.DataFrame) -> pd.DataFrame:
    """o_prior, d_prior per player-season, weights never fitted on s..s+3"""
    out = []
    for s in SEASONS:
        tr = f[~f["season"].between(s, s + BLOCK - 1) & (f["minutes"] >= MIN_FIT_MIN) & f["o_rapm"].notna()]
        mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
        w = tr["minutes"].clip(upper=3000)
        te = f[f["season"] == s]
        res = te[["season", "player_id"]].copy()
        for side in ("o", "d"):
            m = Ridge(alpha=1.0).fit((tr[FEATS] - mu) / sd, tr[f"{side}_rapm"], sample_weight=w)
            res[f"{side}_prior"] = m.predict((te[FEATS] - mu) / sd)
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ── 2. priored ridge ──────────────────────────────────────────────────────────

def fit_season_prior(seg, sp, year: int, prior: pd.DataFrame) -> pd.DataFrame:
    X, y, w, groups, players, scale = build_rows(window(seg, year), sp)
    n_p = len(players)
    # players in the window but not in season s get their newest prior from inside the window
    pr = prior[prior["season"].between(year - 2, year)].sort_values("season") \
        .drop_duplicates("player_id", keep="last").set_index("player_id").reindex(players).fillna(0)
    # defense columns are xG allowed in the regression, so the prior goes in with a minus. home/adv: 0
    b0 = np.concatenate([pr["o_prior"].to_numpy(), -pr["d_prior"].to_numpy(), [0.0, 0.0]])
    # X has the keeper offense columns scaled (rapm.GK_SCALE), so the prior goes in as b0 / scale
    y_off = y - X @ (b0 / scale)
    alpha, scores = pick_alpha(X, y_off, w, groups)
    m = Ridge(alpha=alpha).fit(X, y_off, sample_weight=w)
    coef = m.coef_ * scale + b0
    out = pd.DataFrame({"season": year, "player_id": players, "o_rapm": coef[:n_p], "d_rapm": -coef[n_p:2 * n_p]})
    out["rapm"] = out["o_rapm"] + out["d_rapm"]
    out["lambda"] = alpha
    # the season-s prior itself (not the window fallback), that's what forward test "box only" uses
    own = prior[prior["season"] == year].set_index("player_id")
    out["o_prior"] = out["player_id"].map(own["o_prior"])
    out["d_prior"] = out["player_id"].map(own["d_prior"])
    out = out.join(player_info(year), on="player_id")
    out = out[out["minutes"].notna()]
    print(f"  {season_label(year)}: lambda {alpha:g}  cv mse {scores[alpha]:.4f}")
    return out


# ── 3. forward test ───────────────────────────────────────────────────────────

def forward_error(values: pd.DataFrame, col_o: str, col_d: str, seg, sp, seasons=None) -> dict:
    """season s values -> season s+1 segments. unknown players = 0. league level, home and man advantage
    fitted from s+1 itself on the residual, same for every method, so only the player values make the
    difference"""
    num, den, base_num, per = 0.0, 0.0, 0.0, {}
    sp_side = sp.merge(seg[["seg_id", "season"]], on="seg_id")
    for s in (seasons if seasons is not None else list(SEASONS)[:-1]):
        v = values[values["season"] == s].set_index("player_id")
        st = seg[(seg["season"] == s + 1) & (seg["dur"] > 0)].set_index("seg_id")
        p = sp_side[sp_side["season"] == s + 1]
        p = p.assign(o=p["player_id"].map(v[col_o]).fillna(0), d=p["player_id"].map(v[col_d]).fillna(0))
        agg = p.pivot_table(index="seg_id", columns="side", values=["o", "d"], aggfunc="sum").fillna(0)
        agg = agg.reindex(st.index).fillna(0)
        # home attacking: + home O - away D. away attacking: + away O - home D
        pred = np.concatenate([agg[("o", "h")] - agg[("d", "a")], agg[("o", "a")] - agg[("d", "h")]])
        yy = np.concatenate([90 * st["npxg_h"] / st["dur"], 90 * st["npxg_a"] / st["dur"]])
        ww = np.concatenate([st["dur"]] * 2).astype(float)
        home = np.concatenate([np.ones(len(st)), np.zeros(len(st))])
        adv = np.concatenate([st["n_h"] - st["n_a"], st["n_a"] - st["n_h"]]).astype(float)
        lg = pd.get_dummies(np.concatenate([st["league"]] * 2)).to_numpy(float)
        Z = np.column_stack([lg, home, adv])
        sw = np.sqrt(ww)[:, None]
        b = np.linalg.lstsq(Z * sw, (yy - pred) * sw[:, 0], rcond=None)[0]
        err = yy - pred - Z @ b
        b0 = np.linalg.lstsq(Z * sw, yy * sw[:, 0], rcond=None)[0]
        per[s] = np.sum(ww * err ** 2) / ww.sum()
        num += np.sum(ww * err ** 2)
        base_num += np.sum(ww * (yy - Z @ b0) ** 2)
        den += ww.sum()
    return {"mse": num / den, "base": base_num / den, "gain": 1 - num / base_num, "per": per}


if __name__ == "__main__":
    cur = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    plain_path = PROCESSED_DIR / "rapm_plain.parquet"
    # rapm.py always writes the plain version into rapm.parquet. if that's what's there, it's the fresh one
    if "prior" not in cur.columns or not cur["prior"].any():
        cur.to_parquet(plain_path, index=False)
    plain = pd.read_parquet(plain_path)
    seg, sp = load()

    f = box_features(plain)
    prior = fit_prior(f)
    fit = f.merge(prior, on=["season", "player_id"])
    big = fit[(fit["minutes"] >= MIN_FIT_MIN) & fit["o_rapm"].notna()]
    for side in ("o", "d"):
        r = np.corrcoef(big[f"{side}_prior"], big[f"{side}_rapm"])[0, 1]
        print(f"box prior vs plain {side.upper()}-RAPM (out of season, {MIN_FIT_MIN}+ min): r = {r:.2f}")

    print("\npriored RAPM per season:")
    res = pd.concat([fit_season_prior(seg, sp, y, prior) for y in SEASONS], ignore_index=True)
    res["prior"] = True

    print("\nforward test: season s values -> npxG per 90 in every segment of season s+1")
    tests = {"plain RAPM": (plain, "o_rapm", "d_rapm"),
             "box prior only": (res, "o_prior", "d_prior"),
             "RAPM with box prior": (res, "o_rapm", "d_rapm")}
    errs = {}
    for name, (v, co, cd) in tests.items():
        e = errs[name] = forward_error(v, co, cd, seg, sp)
        print(f"  {name:<20} mse {e['mse']:.4f}   (level-only baseline {e['base']:.4f}, explains {e['gain']:.3%})")
    # segment outcomes are mostly noise, so the share explained is small. what counts: better than plain, every season?
    wins = sum(errs["RAPM with box prior"]["per"][s] < errs["plain RAPM"]["per"][s] for s in errs["plain RAPM"]["per"])
    print(f"  prior beats plain in {wins} of {len(errs['plain RAPM']['per'])} seasons, "
          f"{errs['RAPM with box prior']['gain'] / errs['plain RAPM']['gain'] - 1:+.0%} more of the variance explained")

    show = res[(res["season"] == max(SEASONS)) & (res["minutes"] >= 1500)].sort_values("rapm", ascending=False)
    print(f"\ntop 15 {season_label(max(SEASONS))} with the prior (1500+ min):")
    print(show[["name", "team", "position", "minutes", "o_prior", "d_prior", "o_rapm", "d_rapm", "rapm"]]
          .head(15).round(3).to_string(index=False))
    res.sort_values(["season", "rapm"], ascending=[True, False]).to_parquet(PROCESSED_DIR / "rapm.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'rapm.parquet'} (plain kept in rapm_plain.parquet)")
