#!/usr/bin/env python3
"""
Step 2: split every match into segments, a new segment at every sub and every red card.
Same idea as the stints in nba-moneyball, just with 11 (or fewer) a side.

How the on-pitch time is read from the understat rosters (minutes are capped at 90):
  starter, not subbed off, no red   on from 0 to the end
  starter, subbed off or red        on from 0 to his minutes
  sub, normal                       on from 90 - minutes to the end. snapped to the exit minute of the
                                    subbed-off teammate closest to that (within SNAP minutes), because
                                    the two rows are often a minute apart (61 + 30), which would leave
                                    a fake 1-minute segment with 12 players
  sub who got subbed off or a red   (rare, ~0.5% of subs) the minute he came on is unknown from his own row.
                                    he gets the exit minute of a subbed-off teammate that no normal sub
                                    filled, then is on for his minutes from there
"the end" is 90 + STOPPAGE, so stoppage-time shots (minute 91, 92...) land in the last segment.

Output, per segment (both sides in one row):
  segments.parquet      seg_id, match_id, season, league, start, end, dur, home/away id,
                        n_h, n_a (players on the pitch, < 11 after a red), xg, npxg, goals, shots per side
  seg_players.parquet   seg_id, player_id, side (h/a)

  python src/segments.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, STOPPAGE

END = 90 + STOPPAGE
SNAP = 3


def on_pitch(rows: list) -> tuple[list, int]:
    """rows = (player_id, starter, minutes, roster_in, red_card) for one team in one match.
    returns (player_id, on, off) per player, and how many subs needed the fallback"""
    out, fallback = [], 0
    free = []  # minutes where a subbed-off player left, a sub should come on at each
    for pid, starter, mins, rin, red in rows:
        if starter:
            out.append((pid, 0, mins if (rin > 0 or red > 0) else END))
            if rin > 0:
                free.append(mins)

    normal = [r for r in rows if not r[1] and r[3] == 0 and r[4] == 0]
    special = [r for r in rows if not r[1] and not (r[3] == 0 and r[4] == 0)]
    for pid, _, mins, _, _ in normal:
        on = 90 - mins
        if free:
            # take the closest free exit, so whatever is left over is for the special subs
            k = int(np.argmin([abs(on - f) for f in free]))
            if abs(on - free[k]) <= SNAP:
                on = free[k]
            free.pop(k)
        out.append((pid, on, END))
    for pid, _, mins, _, _ in sorted(special, key=lambda r: -r[2]):
        if free:
            on = free.pop(0)
        else:
            on, fallback = 90 - mins, fallback + 1
        out.append((pid, on, min(on + mins, END)))
    return out, fallback


def split_match(mid, pitch: dict, shots: dict) -> tuple[list, list]:
    """pitch[side] from on_pitch, shots[side] = (minute, xg, is_pen, is_goal) arrays"""
    cuts = sorted({t for side in pitch for _, a, b in pitch[side] for t in (a, b) if 0 < t < END})
    edges = np.array([0, *cuts, END])
    nseg = len(edges) - 1

    segs = [{"seg_id": f"{mid}_{i}", "match_id": mid, "start": edges[i], "end": edges[i + 1],
             "dur": edges[i + 1] - edges[i]} for i in range(nseg)]
    players = []
    for side in ("h", "a"):
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            on = [p for p, x, y in pitch[side] if x <= a and y >= b]
            players += [(segs[i]["seg_id"], p, side) for p in on]
            segs[i][f"n_{side}"] = len(on)
        minute, xg, pen, goal = shots[side]
        # which segment each shot is in. minute past the last cut -> last segment (stoppage included)
        k = np.clip(np.searchsorted(edges, minute, side="right") - 1, 0, nseg - 1)
        for name, v in (("xg", xg), ("npxg", xg * (1 - pen)), ("goals", goal), ("shots", np.ones_like(xg))):
            tot = np.bincount(k, weights=v, minlength=nseg)
            for i in range(nseg):
                segs[i][f"{name}_{side}"] = tot[i]
    return segs, players


if __name__ == "__main__":
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    lu = pd.read_parquet(PROCESSED_DIR / "lineups.parquet")
    sh = pd.read_parquet(PROCESSED_DIR / "shots.parquet")
    # plain python tuples per match and side, pandas filtering inside the loop was way too slow
    lu_rows = {}
    for t in lu[["match_id", "h_a", "player_id", "starter", "minutes", "roster_in", "red_card"]].itertuples(index=False):
        lu_rows.setdefault((t[0], t[1]), []).append(t[2:])
    sh_arr = {k: (g["minute"].to_numpy(), g["xg"].to_numpy(), g["is_pen"].to_numpy(), g["is_goal"].to_numpy())
              for k, g in sh.groupby(["match_id", "h_a"])}
    none = (np.array([]), np.array([]), np.array([]), np.array([]))

    segs, players, fallback = [], [], 0
    for mid in m["match_id"]:
        pitch = {}
        for side in ("h", "a"):
            pitch[side], fb = on_pitch(lu_rows.get((mid, side), []))
            fallback += fb
        s, p = split_match(mid, pitch, {side: sh_arr.get((mid, side), none) for side in ("h", "a")})
        segs += s
        players += p

    seg = pd.DataFrame(segs).merge(m[["match_id", "season", "league", "home_id", "away_id"]], on="match_id")
    sp = pd.DataFrame(players, columns=["seg_id", "player_id", "side"])

    # sanity checks: nothing lost on the way, and the player counts make sense
    print(f"segments {len(seg):,} ({len(seg) / len(m):.1f} per match), median length {seg['dur'].median():.0f} min")
    print(f"shot xG kept in segments {seg[['xg_h', 'xg_a']].sum().sum() / sh['xg'].sum():.4f} of all shot xG")
    n = pd.concat([seg["n_h"], seg["n_a"]]).value_counts().sort_index()
    print("players on the pitch per side (segment count):", n.to_dict())
    print(f"subs placed by the fallback: {fallback} of {(~lu['starter']).sum():,}")
    # player minutes from segments vs the roster minutes (capped at 90 there)
    pm = sp.merge(seg[["seg_id", "start", "end"]], on="seg_id")
    pm["mins"] = pm["end"].clip(upper=90) - pm["start"].clip(upper=90)
    got = pm.groupby([pm["seg_id"].str.split("_").str[0].astype(int), "player_id"])["mins"].sum()
    want = lu.set_index(["match_id", "player_id"])["minutes"]
    j = pd.concat([got.rename("seg"), want.rename("roster")], axis=1).fillna(0)
    print(f"player-match minutes, segments vs roster: mean abs diff {(j['seg'] - j['roster']).abs().mean():.2f} min, "
          f"{((j['seg'] - j['roster']).abs() > 2).mean():.2%} off by more than 2")

    seg.to_parquet(PROCESSED_DIR / "segments.parquet", index=False)
    sp.to_parquet(PROCESSED_DIR / "seg_players.parquet", index=False)
    print(f"saved -> {PROCESSED_DIR / 'segments.parquet'}, seg_players.parquet")
