#!/usr/bin/env python3
"""
Builds report/football_analytics_report.pdf from the text below + the figures.

Numbers in the text are from the October 2026 run (big five leagues, 2014-15 to 2023-24, run_all.py). If the
pipeline gets rerun on new data the text has to be updated by hand, the figures and the player table come along.

  python report/make_figures.py
  python report/build_report.py
"""

import sys

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Spacer

from style import (TEXT_W, FIG, HERE, MARGIN, RFIG, ROOT, A4, P, abs_h, abstract, box, bullets, byline, caption, figure, h1,
                   h2, glue_headings, mono, page_deco, subtitle, table, title)

sys.path.insert(0, str(ROOT / "src"))
from config import PROCESSED_DIR  # noqa: E402

OUT = HERE / "football_analytics_report.pdf"
CASE = "united_case_summer_2023.pdf"

s = []
s += [Spacer(1, 1.2 * cm),
      P("What is a Premier League player worth?", title),
      P("Pricing points, aging, injuries and wages with free data, big five leagues 2014-15 to 2023-24", subtitle),
      P("Viktor Sadowski · October 2026 · code: football-analytics", byline),
      Spacer(1, 0.6 * cm)]

s.append(box([
    P("Summary", abs_h),
    P("This paper prices every player in the big five European leagues in league points and money, and compares "
      "that with what the market pays: Transfermarkt values, transfer fees and wages. It uses only free public data: "
      "ten seasons of Understat shots and rosters split into 119,550 segments where the same 22 players were on the "
      "pitch, Transfermarkt squads, values and transfers, and two free event-data seasons. A player's impact is "
      "regularized adjusted plus-minus on non-penalty expected goals (npxG) with a box-score prior. It is projected "
      "forward with aging curves by position, keeper shot-stopping, league strength from European games and an "
      "injury model built from matchday squads, turned into points above replacement by calibrating on team results, "
      "and priced at what a point earns in Premier League prize money: about EUR 4.1m. A wage model fitted on 59 "
      "wages reported in the press, checked against 126 sets of club accounts, puts the cost side in.", abstract),
    P("Five findings matter for a recruitment department. The market pays a third more for strikers than their "
      "points are worth and a fifth less for defenders, and it pays for youth: 18-21-year-olds go for 39% more than "
      "the model says, players over 33 for half. When the model and the market disagree, the market moves towards "
      "the model the next year, in all ten seasons. Big fees buy points at a higher price, about EUR 35m per point "
      "delivered for EUR 60m+ signings against EUR 10m under EUR 5m, but the model picks the better deals inside "
      "every fee band. Adding wages, contract length and resale to a signing's value improves how well it ranks real "
      "Premier League signings (rank correlation 0.44 against 0.32 without). And defensive event stats, tackles, "
      "interceptions and clearances, add almost nothing to what xG data already says. In a case study on "
      "Manchester United's summer 2023 window, decided on data up to 2022-23 and graded on 2023-24, the window the "
      "model picks for the most points, with the same net spend on fees as United's real EUR 142m, would have been "
      "worth about 77-78 league points against the 60 United got. "
      "The case is a separate PDF in the same repository.", abstract),
]))

# 1
s.append(P("1. The idea", h1))
s.append(P("A club has a budget and wants points. The useful question about a player is how many points he adds per "
           "euro, against what else the money could buy. Baseball and basketball price players like this; football "
           "has the pieces, plus-minus models in research and crowd-sourced market values online, but they are rarely "
           "put together into one price with free data. This project does that, end to end, and is the sequel to "
           "nba-moneyball, which did the same for the NBA."))
s.append(P("The framework answers five questions:"))
s += bullets(["How many league points will a player add next season, and over a contract?",
              "How much of that is at risk because of age and injuries?",
              "What is a point worth, and what does a player cost in fees and wages?",
              "Where does the market misprice players?",
              "What would a club have gained by following it, tested on a real window?"])
s.append(P("Plus-minus in football has been done before. Sæbø and Hvattum (2015) and Kharrat, McHale and López Peña "
           "(2020) showed that plain regression fails because players share the pitch for 90 minutes with few "
           "substitutions, and that it only works with strong shrinkage or a prior. The new part here is the whole "
           "chain on public data in one place: xG plus-minus with a box-score prior, aging, availability, points "
           "above replacement in money, wages, and tests against the market's prices and the betting market."))

# 2
s.append(P("2. Data", h1))
s.append(table([
    ["Source", "Content", "Seasons"],
    ["Understat (third-party dumps, not scraped)", "shots with minute and xG, match rosters with minutes, positions "
     "and substitutions, player xA, xGChain, xGBuildup, key passes", "2014-15 to 2023-24"],
    ["transfermarkt-datasets (GitHub)", "games, lineups, appearances, market valuations, transfers with fees, "
     "contracts", "2012 to 2026"],
    ["StatsBomb open data (credit StatsBomb)", "full event data, big five leagues", "2015-16"],
    ["Wyscout public dataset, Pappalardo et al. 2019 (CC BY 4.0)", "full event data, big five leagues", "2017-18"],
    ["football-data.co.uk", "match results and Pinnacle odds, opening and closing", "2014-15 to 2023-24"],
    ["Club accounts, as reported by Swiss Ramble, Planet Football and others", "Premier League wage bills, 126 "
     "club-seasons", "2014-15 to 2022-23"],
    ["Press reports (Gazzetta via Calcio e Finanza, Sport Bild, L'Équipe, The Athletic, The Times, ...)",
     "59 player wages, one source per row", "2015-16 to 2024-25"],
    ["Premier League and UEFA", "prize money by finishing place, European payments, relegation", "current"],
], [5.6 * cm, 7.6 * cm, 3.6 * cm], "Table 1. Data sources. FBref and Capology were left out on their terms of use, "
   "including mirrors of their data on Kaggle. Every manual table (wages, contracts, known injuries) has a source "
   "per row in the repository."))
s.append(P("Understat covers 18,084 matches with 453,067 shots. Every match is split into segments at each "
           "substitution and red card: 119,550 segments, a median of 7 minutes, with the npxG for both sides. Players "
           "are linked to Transfermarkt through the games they shared (club and date) and their names, which links "
           "8,127 of 8,286 players and 99.91% of all minutes. Transfermarkt has no injury table, so injuries are "
           "inferred from matchday squads (section 3.6)."))

# 3
s.append(P("3. Method", h1))
s.append(figure(RFIG / "pipeline.png", "Figure 1. The pipeline. Blue: how good a player is. Orange: how much he "
                "plays. Grey: what he costs and what to do with it. Every box is one script in the code."))
s.append(P("3.1 Impact per 90 minutes (RAPM)", h2))
s.append(P("Each segment is a stretch of a match where the same 22 players are on the pitch. For each season a ridge "
           "regression explains the npxG per 90 in every segment with the players on the pitch. Each player has an "
           "offense and a defense column, each segment enters twice (once with each team attacking), and segments "
           "are weighted by their length. A home term and a man-advantage term take out red cards and home advantage (about +0.3 npxG per 90 at home, "
           "+0.4 per extra player). The penalty is picked by cross-validation with whole matches held out. To get "
           "enough data each fit pools the season with the three before at weights 1, 0.8, 0.6 and 0.4, tuned on the "
           "forward test below. Keepers get their offensive column scaled down, otherwise a keeper on a dominant team "
           "picked up his outfielders' attacking credit."))
s.append(P("A plain ridge shrinks everyone towards zero. It shrinks towards a box-score prior instead: a linear model "
           "per position group that predicts O-RAPM and D-RAPM from npxG, shots, xA, key passes, xGChain, xGBuildup "
           "and cards per 90, fitted out of season. The prior is good on offense (r 0.67 with plain O-RAPM in another "
           "season) and weak on defense (0.36), as expected without defensive data. The forward test uses values from "
           "season s to predict npxG in every segment of season s+1. There the prior version beats plain RAPM in 9 of 9 "
           "seasons. It explains 10% more of the predictable variance (3.57% against 3.25% of all variance, since segment "
           "npxG is mostly noise)."))
s.append(P("3.2 Keepers", h2))
s.append(P("npxG is measured before the shot, so a keeper who saves more than expected doesn't show up in RAPM. "
           "Shot-stopping is npxG faced minus non-penalty goals conceded per 90, shrunk with empirical Bayes. The true "
           "spread is 0.039 goals per 90, about 1.5 goals a season, and it carries over: a keeper's shrunk number "
           "correlates 0.23 with his raw number three seasons later. It is added to the keeper's value."))
s.append(P("3.3 Aging", h2))
s.append(P("Aging curves are fixed-effects regressions on RAPM (player talent plus one term per age), checked against "
           "the delta method and smoothed with a quadratic, per position group. Two corrections matter. Players who "
           "fall out of the data are the ones who declined, so players over 30 who vanish get a ghost season at "
           "replacement level, except when their club was relegated or they moved to a strong league. And the pooled "
           "window describes a player about a year before the season it is used for, which the age adjustment takes "
           "into account. The keeper curve is shrunk towards the defenders' curve by cross-validation, since keepers "
           "barely move RAPM. Peaks: attackers 27, midfielders 28, defenders and keepers 30."))
s.append(figure(FIG / "aging_by_position.png", "Figure 2. Aging by position group, total RAPM relative to the peak. "
                "The keeper curve is shrunk all the way to the defenders' (the cross-validation picked that), so the "
                "two lines lie on top of each other.", width=0.78 * TEXT_W))
s.append(P("3.4 League strength", h2))
s.append(P("RAPM compares a player with his own league. To put all five on one scale, 663 European games between "
           "clubs from two different big-five leagues show how the leagues compare. Two versions were tested on "
           "held-out seasons: free league offsets fitted on the European games, and offsets proportional to the "
           "league's average Elo (our own Elo, since ClubElo was unreachable). The Elo version predicts held-out "
           "European games better in 6 of 9 seasons and is used: 0.36 goals per match per 100 Elo points. Against the "
           "Premier League in 2023-24, La Liga is +0.04 npxG difference per match, the Bundesliga level, Serie A "
           "-0.25 and Ligue 1 -0.47."))
s.append(P("3.5 From npxG to points", h2))
s.append(P("Adding up each team's player values from season s, weighted by their minutes in season s+1, predicts the "
           "team's npxG difference per match in s+1 with r 0.84 over 880 team-seasons. The values are first stretched "
           "by 1.26, since the ridge shrinks too much at team level. One "
           "npxG of difference per match is worth 0.73 points per match, 28 points over a 38-game season. Then each "
           "player's value is pulled towards his league and position average, by how much of it has carried into the "
           "next season in the past. He keeps a share 0.89 × m / (m + 274) of his distance from the average, m being "
           "his minutes over the last three seasons, fitted only on earlier seasons. A player with 500 minutes keeps 57% of his distance from the average, one with "
           "3,000 keeps 81%. Without this the transfer search in the case study picked players with a few hundred "
           "minutes as its best targets. Replacement level is what clubs got from players in the bottom 20% of market "
           "values in their league, the kind of player a club can sign for next to nothing. On Premier League "
           "team-seasons, points come to about 40 plus 0.96 times the squad's points above replacement, so a team of "
           "such players would expect about 40 points, roughly 15th or 16th place. Then"))
s.append(P("PAR = (value − replacement) × projected minutes / 90 × 0.727", mono))
s.append(P("As a check, each team's points predicted from last season's values and this season's minutes miss by 8.0 "
           "points on average (r 0.81). This check knows who played; the real forecast, without that knowledge, is "
           "in section 4.6."))
s.append(P("3.6 Injuries and minutes", h2))
s.append(P("The Transfermarkt data has no injuries, so they are inferred: a regular who drops out of the matchday "
           "squad for a stretch, without a ban, a transfer, a national-team call-up or a tournament like the Africa "
           "Cup of Nations to explain it. That finds 19,773 spells. The rules were tuned on 36 known injuries (36 "
           "found) and fixed once on a first holdout of 11 (10 found). They were then tested on a second holdout of 13, "
           "written down before running (13 found, pass mark 85%). In a hand-checked sample of 40 spells, 86% of the 28 that could "
           "be decided were real injuries or illness (pass mark 80%). A gradient-boosting model on age and injury "
           "history predicts the share of next season's league games a player misses. It is off by 3.55 games per "
           "player, against 3.93 for the average by age. A second model predicts his share of team minutes, with the "
           "injury risk as an input. Its mean absolute error is 0.194, against 0.205 for a 50/30/20 blend of his past "
           "shares."))
s.append(P("3.7 The price of a point", h2))
s.append(P("Premier League prize money by finishing place includes merit payments, the European competitions a place "
           "leads to, and what relegation costs. It is turned into money by expected points, smoothed over how many "
           "points each place has needed. What one more projected point is worth depends on where a club is: most around 37 points "
           "(relegation) and around 66 (Europe). Averaged over 35 to 85 points it is EUR 4.06m, or EUR 4.72m with an "
           "estimate of the Champions League's matchday and commercial uplift, used as a sensitivity. A player's model "
           "value is his projected PAR over four seasons, aged and floored at zero (a club can always bench him), "
           "times the chance he is still in the big five, discounted 8% a year."))
s.append(figure(FIG / "eur_per_point.png", "Figure 3. Left: what a Premier League season earns against the "
                "bottom club, by projected points. Right: what the next projected point is worth."))
s.append(P("3.8 Wages", h2))
s.append(P("Player wages aren't in any source we can use in bulk, so they are modelled. The data is 59 wages reported "
           "in the press, each with a source, spread over the leagues, ages from 18 to 37 and club Elo from 1479 to "
           "2033. The model is a regression of log wage on log market value, league, age and club Elo. Every term had to lower the "
           "leave-one-out error to stay (0.74 with value and league only, 0.53 with all). A wage grows with market "
           "value to the power 0.56. At the same value it grows 9.5% for every year of age, since wages are set when a "
           "contract is signed and values move after, and 23% for every 100 Elo points of the club. As a check, the model summed "
           "over each Premier League squad comes to 65% of the wage bill in the club's published accounts, the rest "
           "being staff, bonuses and players without league minutes. Contract length by age at signing comes from "
           "Transfermarkt: about 5.2 years at 18-21, 4.1 at 31."))
s.append(P("3.9 What a signing is worth", h2))
s.append(P("A signing's value to a Premier League buyer is his contribution over the contract minus what he costs. Each "
           "year he is there counts his projected PAR times the price of a point, minus his signing wage (plus employer "
           "national insurance). At the end comes what the market would pay for him with a year left, and the fee is "
           "taken off. The fee is what the market charges: on 2,850 big-five summer moves a fee grows with market value to the power 0.91, "
           "falls 6.6% per year of age and is 1.48 times higher when the buyer is a Premier League club."))

# 4
s.append(P("4. Results", h1))
s.append(P("4.1 Who is valuable", h2))
war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
top = war[war["season"] == war["season"].max()].nlargest(12, "par")
rows = [["Player", "Club", "Age", "Value", "Proj. min", "PAR"]]
for _, r in top.iterrows():
    rows.append([r["name"], r["team"], f"{r['age']:.0f}", f"{r['val_cal']:+.2f}", f"{r['proj_min']:,.0f}",
                 f"{r['par']:.1f}"])
s.append(table(rows, [4.0 * cm, 4.2 * cm, 1.3 * cm, 1.6 * cm, 2.0 * cm, 1.4 * cm],
               "Table 2. Projected points above replacement for 2024-25, made at the end of 2023-24. Value is npxG "
               "difference per 90 against an average player in the Premier League, after aging, league strength and "
               "the shrink. The list is midfield-heavy: midfielders and centre-backs who are always on the pitch "
               "when their team controls games get most of the plus-minus credit."))
s.append(P("4.2 The model against the market", h2))
s.append(figure(RFIG / "market.png", "Figure 4. Model value (four seasons of projected points in money) against "
                "Transfermarkt's value, Premier League players with 1,500+ projected minutes, summer 2024. Above the "
                "line the model likes the player more than the market."))
s.append(P("The model and the market agree on the broad picture (r 0.61 between projected PAR and log market value "
           "in the Premier League) and disagree in patterned ways: the model likes older players and defenders who "
           "keep their team in control, the market likes young attackers. A typical Premier League regular costs about "
           "EUR 9.8m per extra projected point, more than twice what a point earns in prize money in a season, which is "
           "what you'd expect if values are paid for several seasons of points and for resale."))
s.append(P("The test that matters: if the model knows something the market doesn't, the market should move towards "
           "it. The gap between model and market at the end of a season predicts the change in market value over the "
           "next year, with age and value level taken out, in all ten seasons (+0.087 per unit of gap overall, "
           "between +0.04 and +0.11 per season). Players the model likes most gain 7.9% more than expected, the ones "
           "it likes least lose 6.8%."))
s.append(figure(RFIG / "market_test.png", "Figure 5. Next year's market value change by quintile of the model vs "
                "market gap. The market corrects towards the model."))
s.append(P("4.3 Are strikers overpaid?", h2))
s.append(P("With age, league and season taken out, the market pays 33% more for strikers than their projected points "
           "are worth, 9% more for wingers and number 10s, about the same for central midfielders, 13% less for "
           "keepers and 19% less for defenders. Part of it is what xG plus-minus can't see (a striker's finishing on "
           "top of his xG is real but not in npxG), but the gap is large, and it matches the market test: strikers "
           "the market rates far above the model tend to fall back."))
s.append(figure(FIG / "side_positions.png", "Figure 6. Market value against model value by role, same age, league "
                "and season."))
s.append(P("4.4 Age and big fees", h2))
s.append(P("The market pays for youth well beyond what aging curves justify: 39% over the model at 18-21, 22% at "
           "24-25, and it discounts age harder than performance falls, 28% under the model at 32-33 and half at 34+. "
           "Some of the youth premium is resale and upside the model floors away, so it isn't all mispricing."))
s.append(figure(FIG / "side_age.png", "Figure 7. Market value against model value by age, league and season taken "
                "out."))
s.append(P("For 2,517 transfers into the big five with a fee since 2015, the fee per point delivered in the first "
           "season rises with the fee: EUR 10.5m under EUR 5m, EUR 25m at EUR 15-30m and EUR 35m above EUR 60m. "
           "Expensive signings do deliver more (1.8 PAR projected and 2.5 delivered for the EUR 60m+ group), they are "
           "just a worse price per point. Inside every fee band, deals where the model's value was above the fee "
           "delivered two to five times as many points per euro as deals where it was below."))
s.append(figure(FIG / "side_fees.png", "Figure 8. Fee per point delivered in the first season, by fee band."))
s.append(P("Does the market overreact to one good season? Not in the sense of giving it back: a player whose value "
           "rose 50%+ in a season gains another 9.6% the year after on average, where the average player loses 11%. "
           "Values have momentum, and the model's gap adds to it."))
s.append(P("4.5 Do defensive stats help?", h2))
s.append(P("The prior has no defensive data, so the two free event-data seasons were used to test whether tackles, "
           "interceptions, clearances, aerial duels, passing and progressive passes add to it. Since StatsBomb and "
           "Wyscout count differently (Wyscout logs 6 tackles per 90 for a median regular, StatsBomb 1.7), every stat "
           "is a z-score within its own season. Fitted on one season and tested on the other, the event stats improve "
           "the out-of-season fit of D-RAPM only for midfielders (R² 0.10 to 0.13) and make it worse for defenders and "
           "attackers. In the forward test a prior with the event stats explains 4.15% against 4.03% in 2016-17 and "
           "3.61% against 3.60% in 2018-19. The useful defensive signal is already in xGChain and xGBuildup, which "
           "Understat has for every season; counting defensive actions measures how much a team defends more than "
           "how well."))
s.append(P("4.6 How good is the forecast?", h2))
s.append(P("Run the way it would have been every August since 2016, with squads as of 1 September and only earlier "
           "seasons, the forecast misses by 8.5 points a team, or 8.4 through a line fitted on earlier seasons. That "
           "ties our Elo (8.4) and beats last season's points (9.0). The betting market's ratings from the first three "
           "rounds are better (7.7), and a 50/50 blend of ours and the market's gets 7.8. The raw forecast is too "
           "compressed (slope 1.23 of actual on predicted), a side effect of the shrink, which the fitted line and the "
           "season simulator correct."))
s.append(figure(RFIG / "forecast.png", "Figure 9. August forecasts against the benchmarks, 2016-17 to 2023-24, all "
                "five leagues. The closing odds use information from the season itself and are a ceiling, not a "
                "competitor."))
s.append(P("4.7 Do wages, contracts and resale make the valuation better?", h2))
s.append(P("For 250 Premier League summer signings from 2015 to 2020, the predicted value of the deal (contribution "
           "minus wages over three seasons plus resale minus the fee) is compared with what happened: real PAR in "
           "money, the same wages, and the real market value three years on. The full valuation ranks the deals with "
           "a rank correlation of 0.44, against 0.32 without wages and resale and 0.09 for market value minus fee "
           "(pass mark: beat both). Deals in the best predicted fifth returned a median EUR 6.7m, the worst fifth lost "
           "EUR 14.8m. The realised side uses model wages too, so this tests contribution and resale against the fee, "
           "not the wage model itself."))
s.append(figure(RFIG / "transfer_test.png", "Figure 10. Left: realised surplus of Premier League signings by "
                "quintile of predicted surplus. Right: how well three valuations rank the same deals."))
s.append(P("4.8 Wages", h2))
s.append(figure(RFIG / "wages.png", "Figure 11. Reported wages against the wage model, each predicted with itself "
                "left out of the fit. A typical miss is 70% either way, about what you'd expect when the main input "
                "is a crowd-sourced market value and the reported wages are themselves estimates."))
s.append(P("At the same market value, age and club strength, a Serie A club pays 0.30 times what a Premier League club "
           "pays (part of it the Decreto Crescita tax break on gross wages), a Bundesliga club 0.58 and a Ligue 1 club "
           "0.64. The age term is the reason the wage model exists: a 21-year-old costs 0.58 times a 27-year-old with "
           "the same market value, a 31-year-old 1.44 times, so young players are cheaper than their value says and "
           "old players dearer."))

# 5
s.append(P("5. The case: Manchester United, summer 2023", h1))
s.append(P(f"The case study ({CASE}) puts the model to work on one window. Manchester United in summer 2023, "
           "decided on data up to 2022-23, with United's real net spend on fees that summer (EUR 142m) as the budget, sales at 80% "
           "of market value, the 25-man squad rule with 8 homegrown players, at most six moves, and only players a "
           "club would actually sell to United (a club rated below them, or a player surplus where he is). Two "
           "alternative windows are searched. Scenario A, best value, makes the most money over the players' "
           "contracts without United losing projected points. Scenario B, most points, gets the most projected "
           "2023-24 points without the window losing money over the contracts. Both are graded on 2023-24 with every player's real rating that season, and a season simulator "
           "gives the pre-season odds. Scenario B (Bruno Guimarães, Theo Hernández, Udogie, Botman, Solly March "
           "and Sørloth in; Varane, Martial, Telles, Van de Beek, Malacia and Greenwood out, a net spend of EUR 129m) would have been worth "
           "about 77-78 league points with the players' real 2023-24 ratings, against the 60 United got. The real window "
           "(Mount, Onana, Højlund, Bayındır) comes out level with standing still."))

# 6
s.append(P("6. Limitations", h1))
s += bullets([
    "RAPM on npxG is noisy and can't fully separate players who are always on the pitch together. The prior helps on "
    "offense; on defense the model mostly sees who was on the pitch when the team conceded fewer chances. Midfielders "
    "who play every minute for controlling teams get a lot of credit; wide forwards with low xG involvement get "
    "little. Some of the model's verdicts (Mainoo low, Sancho and Antony high in 2023-24) come from that.",
    "npxG leaves out finishing and shot-stopping above expectation for outfielders, and penalties. Finishing skill "
    "exists but is small and noisy per season.",
    "The Understat data comes from third-party dumps, not from Understat itself, since its robots.txt disallows "
    "crawling. Injuries are inferred, not recorded, and miss short knocks between matchdays.",
    "Wages are modelled from 59 reports with a typical miss of 70%. La Liga has no usable reported wages (the "
    "press mostly repeats Capology), so La Liga clubs get the Bundesliga's level, and the model underestimates Real "
    "Madrid and Barcelona stars.",
    "Market values and fees are Transfermarkt's. Fees don't include add-ons consistently, and loans with options "
    "to buy can't be told apart from plain loans.",
    "The backtests are mostly out of sample, but some pieces (the aging curves, the box-score prior's form, the "
    "window weights, the injury rules) were fitted or tuned on all seasons.",
    "The case study's greedy search can tip on a near tie; one run picked Barcola, another Ali-Cho as A's right "
    "winger. run_all.py runs single-threaded to make reruns identical."])

# appendix
s.append(P("Appendix A. Every check and its pass mark", h1))
s.append(table([
    ["Step", "Check", "Result"],
    ["Box-score prior", "forward test, segment npxG of next season, vs plain RAPM", "better in 9 of 9 seasons, +10% "
     "variance explained"],
    ["Injury inference", "recall on known injuries (pass 85%), precision on a hand-checked sample (pass 80%)",
     "36/36 tuning, 10/11 holdout 1, 13/13 holdout 2; 86% precision"],
    ["Injury risk", "games missed next season, out of sample, vs age average", "3.55 vs 3.93 games off"],
    ["Minutes", "share of team minutes next season, vs 50/30/20 blend", "MAE 0.194 vs 0.205"],
    ["Calibration", "team npxG difference next season from player values", "r 0.84, points MAE 8.0"],
    ["League strength", "held-out European games, Elo offsets vs none and free offsets", "Elo best in 6 of 9 seasons"],
    ["Shrink", "fitted on earlier seasons only, minutes curve", "0.57 at 500 min, 0.81 at 3,000"],
    ["Model vs market", "next year's market value change on the gap, every season", "positive in 10 of 10 seasons"],
    ["August forecast", "points per team, vs last season, Elo, odds", "8.4 vs 9.0, 8.4, 7.7"],
    ["Defensive event stats", "forward test with event stats in the prior", "+0.12 and +0.01 points of % explained"],
    ["Wage model", "leave-one-out error of log wage, each term must lower it", "0.53 vs 0.74"],
    ["Transfer valuation", "rank of realised surplus, 250 PL signings (beat both benchmarks)", "0.44 vs 0.32 and "
     "0.09"],
    ["United case", "real 2023-24 ratings on each window's depth chart", "scenario B (most points) 77-78 vs 60 real"],
], [3.2 * cm, 7.8 * cm, 5.8 * cm]))
s.append(P("Appendix B. Reproducing it", h1))
s.append(P("Everything runs from the repository with Python 3 (pandas, numpy, scipy, scikit-learn, matplotlib, "
           "pyarrow, reportlab). The raw data goes in data/raw (Understat dumps, transfermarkt-datasets, the Wyscout "
           "files from figshare, football-data.co.uk odds), then:"))
for line in ["python src/run_all.py        # the whole chain, about 15 minutes, logs in logs/",
             "python report/make_figures.py && python report/build_report.py",
             "python report/make_case_figures.py && python report/build_case.py"]:
    s.append(P(line, mono))
s.append(Spacer(1, 6))
s.append(P("Data: Understat via third-party dumps; transfermarkt-datasets by dcaribou (github.com/dcaribou/"
           "transfermarkt-datasets); StatsBomb open data (github.com/statsbomb/open-data); Wyscout public dataset, "
           "Pappalardo et al. 2019, figshare, CC BY 4.0; football-data.co.uk; wage and contract sources listed per row "
           "in manual/.", caption))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title="What is a Premier League player worth?", author="Viktor Sadowski")
deco = page_deco("What is a Premier League player worth?")
doc.build(glue_headings(s), onFirstPage=deco, onLaterPages=deco)
print(f"report -> {OUT}")
