#!/usr/bin/env python3
"""
Builds report/united_case_summer_2023.pdf: the united case on top of the main report (same look).

Numbers in the text are from the October 2026 run (run_all.py, single-threaded). If it gets rerun the text has to
be updated by hand, the figures and the tables come along (src/united_case.py, src/season_sim.py outputs).

  python report/make_case_figures.py
  python report/build_case.py
"""

import json
import sys

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Spacer

from style import (FIG, HERE, MARGIN, RFIG, ROOT, A4, P, abs_h, abstract, box, bullets, byline, figure, h1, h2,
                   glue_headings, page_deco, subtitle, table, title)

sys.path.insert(0, str(ROOT / "src"))
from config import PROCESSED_DIR  # noqa: E402

OUT = HERE / "united_case_summer_2023.pdf"
uc = json.loads((PROCESSED_DIR / "united_case.json").read_text())
depth = pd.read_parquet(PROCESSED_DIR / "united_case_depth.parquet")
moves = pd.read_parquet(PROCESSED_DIR / "united_case_moves.parquet")
sim = pd.read_parquet(PROCESSED_DIR / "season_sim.parquet")

POS = ["GK", "LB", "CB", "RB", "DM", "CM", "LW", "AM", "RW", "FW"]
WIN = {"standing still": "Standing still", "actual": "Real window", "A": "Scenario A: best value",
       "B": "Scenario B: most points"}


def depth_table(cap):
    rows = [["", *[WIN[w] for w in ("standing still", "actual", "A", "B")]]]
    for p in POS:
        row = [p]
        for w in ("standing still", "actual", "A", "B"):
            d = depth[(depth["window"] == w) & (depth["pos"] == p)].sort_values("minutes", ascending=False)
            # the starter(s), the rest as backup in grey-ish brackets
            n = 2 if p == "CB" else 1
            first = ", ".join(d["name"].iloc[:n])
            rest = ", ".join(d["name"].iloc[n:])
            row.append(first + (f" <font color='#8a8984'>({rest})</font>" if rest else ""))
        rows.append(row)
    return table(rows, [1.0 * cm, 3.95 * cm, 3.95 * cm, 3.95 * cm, 3.95 * cm], cap)


def moves_table(w, cap):
    m = moves[moves["window"] == w]
    rows = [["", "Player", "Fee (EUR m)", "Projected value, 23-24", "Real value, 23-24", "League minutes, 23-24",
             "Market value (EUR m), Jun 23 → Sep 24"]]
    for _, r in m.iterrows():
        rows.append(["in" if r["dir"] == "in" else "out", r["name"],
                     f"{r['fee'] / 1e6:.0f}" if pd.notna(r["fee"]) else "",
                     f"{r['val_pre']:+.2f}" if pd.notna(r["val_pre"]) else "",
                     f"{r['val_real']:+.2f}" if pd.notna(r["val_real"]) else "no data",
                     f"{r['min_after']:,.0f}",
                     f"{r['mv_before'] / 1e6:.0f} → {r['mv_after'] / 1e6:.0f}" if pd.notna(r["mv_after"]) else ""])
    # totals: money in and out (sales per player aren't in the moves file, the window total is in united_case.json)
    paid, got = uc[w]["fees"] / 1e6, uc[w]["sales"] / 1e6
    for d, lab, money in (("in", "Signings, total paid", paid), ("out", "Sales, total received", got)):
        x = m[(m["dir"] == d) & m["mv_after"].notna()]
        b, a = x["mv_before"].sum() / 1e6, x["mv_after"].sum() / 1e6
        rows.append(["", lab, f"{money:.0f}", "", "", "", f"{b:.0f} → {a:.0f} ({a - b:+.0f})"])
    rows.append(["", "Net spend", f"{paid - got:+.0f}", "", "", "", ""])
    return table(rows, [0.9 * cm, 4.1 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 3.6 * cm], cap, totals=3)


s = []
s += [Spacer(1, 1.2 * cm),
      P("What should United have done in summer 2023?", title),
      P("Two alternative transfer windows from the model against the real one, decided on 2022-23 data and graded on 2023-24",
        subtitle),
      P("Viktor Sadowski · October 2026 · case study to What is a Premier League player worth? · code: "
        "football-analytics", byline),
      Spacer(1, 0.6 * cm)]

h = uc["hindsight"]
A, B, ACT = uc["A"], uc["B"], uc["actual"]


def net(w):
    return (w["fees"] - w["sales"]) / 1e6


s.append(box([
    P("Summary", abs_h),
    P(f"In summer 2023 Manchester United spent EUR 203m on fees for Mount, Onana, Højlund and Bayındır and took in "
      f"EUR 61m from sales, a net spend of EUR {uc['budget'] / 1e6:.0f}m. This case asks what the model in the main "
      "report would have done with the same net spend, the same squad rules and only the data available that "
      "summer, and then grades the answer on what the players really did in 2023-24.", abstract),
    P("The model searches for two alternative windows, called scenario A and scenario B. Both start from United's "
      "real squad in June 2023, make at most six signings or sales, and can only buy players a club would "
      "realistically sell to United. Scenario A, best value, looks for the window that is worth the most money over "
      "the players' contracts, as long as United are projected at least as many points as if they had done nothing. "
      "Scenario B, most points, looks for the window that gives the most projected 2023-24 points, as long as it "
      "doesn't lose money over the contracts.", abstract),
    P(f"Scenario B (Bruno Guimarães, Theo Hernández, Udogie, Botman, Solly March and Sørloth in; Varane, Martial, "
      f"Telles, Van de Beek, Malacia and Greenwood out, a net spend of EUR {net(B):.0f}m) was projected to give "
      "United an 87% chance of the top four, against 40% for the real window. Scored with every player's real "
      f"2023-24 numbers, it comes to {h['B']['hindsight_pts_low']:.0f}-{h['B']['hindsight_pts_high']:.0f} league "
      f"points, against the 60 United got. Scenario A sells for EUR {-net(A):.0f}m more than it buys and lands at "
      f"{h['A']['hindsight_pts_low']:.0f}-{h['A']['hindsight_pts_high']:.0f} points. The real window comes out "
      "level with doing nothing. The model would not have started Mount or Højlund over the players United already "
      "had, and in 2023-24 Mount played 514 league minutes and Højlund rated close to replacement level (explained "
      "below).", abstract),
]))

s.append(P("How to read the numbers", h2))
s.append(P("The case uses a few terms from the main report. They are explained here so the rest can be read on its "
           "own."))
s += bullets([
    "Value of a player: how much better his team does per 90 minutes when he is on the pitch, measured in "
    "non-penalty expected goals (npxG) for minus npxG against, compared with an average big-five player. It comes "
    "from RAPM, a regression on every stretch of play where the same 22 players were on the pitch. A value of +0.10 "
    "means his team creates 0.10 npxG more than it concedes per 90 minutes than with an average player. Over a full "
    "season (3,420 minutes) that is 3.8 npxG, which is worth about 2.8 league points.",
    "Replacement level: what a club gets from a player it can sign for next to nothing, measured on players in the "
    "bottom 20% of market values in their league. A whole team of replacement-level players would expect about 40 "
    "points in the Premier League, roughly 15th or 16th place and a few points above the relegation zone.",
    "Points above replacement: the league points a player adds over such a player in the minutes he is projected to "
    "play. For a squad it is the sum over the players. 60 points, United's real 2023-24 total, is about 7th place in "
    "an average season; 70 is usually 4th.",
    "Window value: the money a window adds over the contracts of the players involved, in EUR m. Each player's "
    "points are priced at EUR 4.1m a point (Premier League prize money per place), minus his wages, plus what he can "
    "be sold for later, and future years are discounted 8% a year. Fees paid and sales received are added, and "
    "standing still, doing nothing that summer, is the zero.",
    "Hindsight: the windows are chosen on data up to 2022-23. Afterwards every player gets the real value he had "
    "in 2023-24, and each window is scored again with those values."])

s.append(P("1. The setup", h1))
s.append(P("Everything the model uses is cut at the end of 2022-23: player values, injury risk, minutes, club Elo "
           "ratings, and the fitted models for fees, resale and how long a player stays at a club. The rules for both "
           "scenarios:"))
s += bullets([
    f"Budget: the net spend on fees can't be more than United's real net spend that summer, EUR "
    f"{uc['budget'] / 1e6:.0f}m, so the scenarios compete with the real window on the same money. Wages are not in "
    "the budget, they are part of the window value.",
    "Prices: buying a player costs what the market would charge, from a fee model fitted on big-five summer "
    "transfers up to 2022. It uses his Transfermarkt market value on 1 June 2023, his age and the extra a Premier "
    "League buyer pays. Selling a United player brings in 80% of his market value, since a club that has to sell "
    "rarely gets full price.",
    "Squad rules: at most 25 senior players, of which at least 8 homegrown (three years at English or Welsh clubs "
    "between age 15 and 21, approximated from transfer history). Players born in 2002 or later are under 21 and "
    "don't count towards the 25.",
    "Track record: a player needs at least 1,500 minutes in big-five league matches over the three seasons up to "
    "2022-23 to be a target. With fewer minutes his value is too uncertain to pay for.",
    "Sellers: a target must either play for a club with a lower Elo rating than United, or be surplus where he is "
    "(less than half of his club's minutes when fit, or out on loan). This removes 123 of the 1,405 targets with a "
    "track record, among them Pedri, Rodri, Upamecano, Nkunku and Araújo. The best 133 by position are kept as "
    "the shortlist.",
    "Minutes: United play a 4-2-3-1, split into ten positions (the two centre-backs share one). A player can fill his main position "
    "and any other he started at least 20% of his games in (and at least five games). The minutes of each position "
    "go to the best player first, capped at 92% of the season and his expected time out injured, and the next best "
    "covers the rest.",
    "Contracts and wages for United's own players: contract end dates as of summer 2023 from press lists, wages as "
    "reported in the press where there is a report (Casemiro, Bruno Fernandes, Rashford, Garnacho), else from the "
    "wage model in the main report.",
    "Search: at most six moves, each a signing, a sale or both. The search is greedy: each round it makes the single "
    "move that improves the scenario's goal the most, and it stops when no move helps."])
s.append(P("United before the window", h2))
s.append(P(f"Standing still, the model projected United's squad at {uc['standing_still']['points']:.1f} points "
           "above replacement for 2023-24, which the simulator in section 3 turns into 69 expected league points. That "
           "was too high, as section 4 shows, but it is the line both scenarios have to beat. The weak spots by the model were central "
           "midfield (Eriksen at 31 with Fred and nobody behind them), the striker position (Martial, often injured) "
           "and left-back cover behind Shaw, who missed half of most seasons. It rated Sancho and Antony as United's "
           "best wide players and Rashford lower than his reputation, from the 2022-23 on/off numbers (section 6 of "
           "the main report has the caveats)."))

s.append(P("2. The two scenarios", h1))
s.append(P("The two scenarios answer different questions. Scenario A asks how much money the window can make without "
           "making United worse on the pitch, which is the classic moneyball question. Scenario B asks how much better "
           "United can get on the pitch without the window losing money over the contracts. Table 1 has the money and "
           "the projected points side by side."))
rows = [["", "Fees paid (EUR m)", "Sales (EUR m)", "Net spend (EUR m)", "Window value (EUR m)",
         "Projected points above replacement"]]
rows.append(["Standing still", "0", "0", "0", "0", f"{uc['standing_still']['points']:.1f}"])
for k, w in (("actual", ACT), ("A", A), ("B", B)):
    rows.append([WIN[k], f"{w['fees'] / 1e6:.0f}", f"{w['sales'] / 1e6:.0f}", f"{net(w):+.0f}",
                 f"{w['window_value'] / 1e6:+.0f}", f"{w['points']:.1f}"])
s.append(table(rows, [3.9 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 3.7 * cm],
               "Table 1. The money and projected 2023-24 points of each window, as seen in summer 2023. A negative "
               "net spend means more came in from sales than went out on fees. Window value is against standing "
               "still."))
s.append(depth_table("Table 2. Who plays where in each window, starters first and backups in brackets, from the "
                     "minutes planned in summer 2023."))
s.append(P(f"Scenario A trades starters on big contracts for cheaper players the model rates as good or better: "
           "Barcola for Antony, Medina for Martínez, Udogie for Rashford (Udogie plays left-back, Sancho keeps the "
           "left wing), Omari for Varane, Kaboré for Wan-Bissaka and Sørloth for Martial. It brings in EUR "
           f"{-net(A):.0f}m more than it spends and saves a lot of wages, which is where most of its window value "
           "comes from. Scenario B fixes the spots the model saw as weak: Bruno Guimarães in central midfield, Theo "
           "Hernández at left-back with Udogie behind him, Botman next to Lindelöf, Solly March on the right and "
           "Sørloth up front. The real window spent most of its money on positions the model already had covered. "
           "Onana replaces Henderson in goal, a small step up. Mount and Højlund get no minutes in the model's plan at "
           "all: it rated Bruno Fernandes and Eriksen above Mount (+0.09), and Martial and Rashford above Højlund "
           "(+0.05), a 20-year-old striker it rated close to replacement level. Their EUR 148m in fees bought "
           "backups, which is why the real window projects slightly below standing still in Table 1."))

s.append(P("3. The pre-season odds", h1))
s.append(P("A season simulator plays the real 2023-24 fixture list 20,000 times. Every club's strength is its squad "
           "on 1 September 2023, player values times projected minutes. United's strength is moved by the difference "
           "between each window and the real one. In each simulated season every club's strength also gets a random "
           "error, sized to match how wrong the August forecast has been in earlier seasons: about 10.5 points per "
           "club, of which 7.4 is the luck of single matches. Tested on the real 2023-24 table, the forecast for all "
           "20 clubs missed by 8.4 points on average (correlation 0.88). It had United at 67 points with the real "
           "window, which was too high: they got 60."))
u = sim[sim["club"] == "Manchester United"].set_index("window")
rows = [["Window", "Expected points", "80% range (points)", "Average finish", "Top 4", "Title",
         "Points with real 2023-24 values"]]
for w in ("standing still", "actual", "A", "B"):
    r = u.loc[w]
    rows.append([WIN[w], f"{r['exp_pts']:.0f}", f"{r['p10']:.0f}-{r['p90']:.0f}", f"{r['exp_rank']:.1f}",
                 f"{100 * r['top4']:.0f}%", f"{100 * r['title']:.0f}%",
                 f"{h[w]['hindsight_pts_low']:.0f}-{h[w]['hindsight_pts_high']:.0f}"])
s.append(table(rows, [3.9 * cm, 1.9 * cm, 2.1 * cm, 1.8 * cm, 1.4 * cm, 1.4 * cm, 4.3 * cm],
               "Table 3. United's 2023-24 under each window. League points, chances from the simulator as seen on "
               "1 September 2023. The last column is the hindsight grade from section 4. 2023-24 had four Champions "
               "League places."))
s.append(figure(FIG / "season_sim.png", "Figure 1. Top: United's simulated 2023-24 league points under each window, "
                "as seen on 1 September 2023. Bottom: the same windows scored with every player's real 2023-24 "
                "value. The dashed line is the real 60."))

s.append(P("The curves in Figure 1 are United's points over the 20,000 simulated seasons. Scenario B's sits about 15 "
           "points to the right of the real window's, the difference between a 40% and an 87% chance of the top four. "
           "The bars below are the hindsight grade from section 4, a range because it is calculated two ways. The real "
           "window sits on 60 because every grade is measured from it."))

# united's real 2023-24 squad on its real league minutes, as an anchor for figure 2
war_ = pd.read_parquet(PROCESSED_DIR / "war.parquet")
prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
w23 = war_[war_["season"] == uc["decision"]].copy()
w23["rp"] = (prm["a"] * w23["rapm"] + w23["save90"].fillna(0) + w23["league_adj"].fillna(0) - w23["repl"]) \
    * w23["minutes"] / 90 * prm["ppx"]
pl23 = w23[w23["league"] == "EPL"].groupby("team")["rp"].sum().sort_values(ascending=False)
mu_par = pl23["Manchester United"]
mu_rank = list(pl23.index).index("Manchester United") + 1

s.append(P("4. With hindsight", h1))
s.append(P("2023-24 is in the data, so every player in every window has a real value for that season: the same RAPM "
           "measure, plus shot-stopping for keepers and the league strength adjustment, fitted at the end of 2023-24. "
           "Each window is then scored twice with those real values."))
s += bullets([
    "Planned minutes: every player gets the minutes the window planned for him in summer 2023. This is the "
    "optimistic end, since it assumes nobody missed more time than expected.",
    "Capped minutes: every player's minutes are capped at the league minutes he really played in 2023-24, anywhere "
    "in the big five, and the minutes he didn't play count as played by a replacement-level player. This is the "
    "pessimistic end. It counts every missed minute against the window, also when a player was dropped rather "
    "than injured (Sancho played 1,051 league minutes after falling out with Ten Hag), and it assumes United had "
    "nobody better than replacement level to fill in."])
s.append(P(f"For a sense of scale: United's real 2023-24 squad, every player on his real value and his real league "
           f"minutes for United, comes to {mu_par:.0f} points above replacement, {mu_rank}th of the 20 Premier League "
           f"squads. The line from the main report (about 40 points plus 0.96 per point above replacement) turns that "
           f"into {40.1 + 0.96 * mu_par:.0f} league points, against the {uc['real_points']} United really got. The capped "
           f"version of the real window ({h['actual']['real_par_real_min']:.0f}) is close to that, as it should be, "
           "since it is nearly the same squad on nearly the same minutes."))
s.append(figure(RFIG / "case_points.png", "Figure 2. League points above replacement level for 2023-24 per window: as "
                "projected in summer 2023 (dark), with the players' real 2023-24 values on the planned minutes "
                "(middle), and with their minutes capped at what they really played (light)."))
s.append(P(f"Every window drops from the projection to the real values, since the model projected United too high "
           f"overall: most of the squad rated lower in 2023-24 than on its 2022-23 numbers. The further drop to the "
           f"capped version is lost minutes. What matters for the grade is the order between the windows, and it "
           f"holds in both versions. Scenario B keeps "
           f"{h['B']['real_par_planned_min']:.0f} of its {h['B']['planned_par']:.0f} projected points above "
           f"replacement, the real window {h['actual']['real_par_planned_min']:.0f} of "
           f"{h['actual']['planned_par']:.0f}. In the capped version B loses another "
           f"{h['B']['real_par_planned_min'] - h['B']['real_par_real_min']:.0f}, mostly because Solly March was out "
           f"with a knee injury from October (571 league minutes) and Botman played 1,383."))
s.append(P(f"The hindsight points in Table 3 come from these bars. Each window's points are United's real "
           f"{uc['real_points']}, moved by its difference to the real window in the same version. For Scenario B: "
           f"{uc['real_points']} + ({h['B']['real_par_planned_min']:.0f} − {h['actual']['real_par_planned_min']:.0f}) "
           f"= {uc['real_points'] + h['B']['real_par_planned_min'] - h['actual']['real_par_planned_min']:.0f} with "
           f"planned minutes, and {uc['real_points']} + "
           f"({h['B']['real_par_real_min']:.0f} − {h['actual']['real_par_real_min']:.0f}) = "
           f"{uc['real_points'] + h['B']['real_par_real_min'] - h['actual']['real_par_real_min']:.0f} with capped minutes (rounded numbers in the sums). Measuring from the "
           "real window keeps whatever the values miss, the manager and the luck, the same for every window."))
s.append(figure(RFIG / "case_players.png", "Figure 3. Every signing in the three windows: projected value for 2023-24 "
                "made in summer 2023 against the real 2023-24 value, both in npxG difference per 90 against an average "
                "player. On the diagonal the projection was exactly right."))
s.append(P("Dots above the diagonal were better in 2023-24 than projected. Most of B's signings sit on or above it, "
           "for example Bruno Guimarães (+0.23 projected, +0.28 real) and Theo Hernández (+0.20, +0.26). Mount and "
           "Højlund fell short of projections that were already low."))
s.append(P("Tables 4 to 6 list every move, one row per player, signings (in) first and sales (out) after. Fees are "
           "what the window pays: the model's price for the scenarios, the real fee for the real window. Projected "
           "value is the model's number from summer 2023, real value the 2023-24 number, both in npxG difference per "
           "90 against an average player, so +0.10 is about 2.8 league points over a full season. \"no data\" means "
           "he played no big-five league minutes in 2023-24. League minutes and market value show what really "
           "happened to the player in 2023-24 at his real club, so for the scenarios they are what the player did "
           "elsewhere, not at United."))
s.append(moves_table("B", "Table 4. Scenario B, most points: signings (in) and sales (out)."))
s.append(P("In Table 4, the sales are mostly players who were leaving or barely played: Varane and Martial were in "
           "the last year of their contracts, Telles, Van de Beek and Malacia played 359 league minutes between "
           "them in 2023-24, and Greenwood was loaned out in reality too. B's money goes on three starters "
           "(Guimarães, Theo Hernández, Botman), with Udogie as left-back cover and March and Sørloth as cheaper "
           "starters on the right and up front."))
s.append(moves_table("A", "Table 5. Scenario A, best value: signings and sales."))
s.append(P("Table 5 shows where A's money comes from: it sells Rashford (EUR 80m market value), Antony (70m), "
           "Martínez (50m) and Varane (40m), and buys six players for EUR 105m in total. Antony rated much better "
           "in 2023-24 than A expected (+0.18 against +0.09), so that sale cost points. The window still makes money "
           "because the wages saved on the four are large."))
s.append(moves_table("actual", "Table 6. The real window. The loans in (Amrabat, Reguilón) and Evans on a free "
                               "aren't paid moves in Transfermarkt, so they are left out."))
s.append(P("Table 6 is the real window. United paid EUR 203m for four players, of whom only Onana and Højlund played "
           "a full season, and sold mostly fringe players for EUR 61m."))
s.append(P("The market's verdict a year later points the same way. By September 2024 the market value of Scenario A's "
           "signings had gone up by EUR 93m (Barcola alone from EUR 7m to 50m), Scenario B's by EUR 61m and the real "
           "window's by EUR 12m, with Mount down from EUR 65m to 35m. The players United really sold or loaned out "
           "lost EUR 43m in market value; the ones A and B would have sold lost EUR 72m to 92m, so the model sold "
           "at about the right time."))

s.append(P("5. What it says, and what it can't", h1))
s.append(P("The case supports two things. First, the model's values carry real information about the next season for "
           "the players it would have bought: Scenario B's signings rated as high or higher in 2023-24 as projected, "
           "and the real window's didn't. Second, with the same net spend and rules, Scenario B would have been worth "
           "about 17 league points more than the real window. That is roughly the gap between United's real 8th "
           "place and a top-four finish in 2023-24 (Aston Villa were 4th on 68)."))
s += bullets([
    "Hindsight scores what the players delivered, not the season. Fit, the manager, how a dressing room takes six new "
    "players and injuries spreading through the squad aren't in it, and United's real season had a lot of all four.",
    "The real 2023-24 value isn't purely 2023-24. RAPM pools a season with the three before at lower weights, so a "
    "player's earlier seasons still make up about a third of it.",
    "Scenario B buys four Premier League players (from Newcastle, Spurs and Brighton), Theo Hernández from Milan and "
    "Sørloth, who was on loan at Real Sociedad. The seller rule keeps it to clubs rated below United, but whether "
    "Newcastle would have sold Guimarães and Botman to a direct rival at the market price is a real question. A sale "
    "probability fitted on past transfers would make that uncertainty part of the value; here it is a yes/no rule.",
    "Scenario A is the textbook moneyball window, and its weakness shows: some of the cheap players it rates highly "
    "were noise (Omari +0.12 projected, +0.06 real). Maximising money alone keeps choosing uncertain bargains, while "
    "maximising points with a money floor gives the better window.",
    "The greedy search makes one move at a time, so it can miss a pair of moves that only works together. With six "
    "moves and 133 targets an exact search wasn't needed to make the point."])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title="What should United have done in summer 2023?",
                        author="Viktor Sadowski")
deco = page_deco("What should United have done in summer 2023?")
doc.build(glue_headings(s), onFirstPage=deco, onLaterPages=deco)
print(f"case -> {OUT}")
