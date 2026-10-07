"""
THE ARROWHEAD EFFECT — reproducible dataset builder
Python 3.10+

Downloads nflverse PBP seasons 2014–2026 directly from the documented release URLs,
then creates the research tables. It does NOT invent stadium noise measurements.

Usage:
    python build_arrowhead_dataset.py --start 2014 --end 2026 --out ./built

For a historical-only paper, use --end 2025.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz"

KEEP = [
"game_id","play_id","season","week","game_type","game_date","home_team","away_team",
"posteam","defteam","stadium","roof","surface","temp","wind","humidity",
"qtr","game_seconds_remaining","half_seconds_remaining","quarter_seconds_remaining",
"down","ydstogo","yardline_100","goal_to_go","score_differential",
"posteam_score","defteam_score","posteam_timeouts_remaining","defteam_timeouts_remaining",
"play_type","shotgun","no_huddle","qb_dropback","pass","rush","sack","scramble",
"yards_gained","air_yards","yards_after_catch","first_down","touchdown",
"interception","fumble","fumble_lost","penalty","penalty_team","penalty_type",
"penalty_yards","timeout","third_down","fourth_down","two_point_conv",
"epa","wpa","cpoe","success","desc",
"home_score","away_score","field_goal_result","kick_distance","return_yards","punt_blocked"
]

def load_year(year):
    url = BASE.format(year=year)
    df = pd.read_csv(url, low_memory=False)
    return df[[c for c in KEEP if c in df.columns]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--out", default="./built")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in range(args.start, args.end + 1):
        print("Loading", year)
        frames.append(load_year(year))

    pbp = pd.concat(frames, ignore_index=True)

    desc = pbp["desc"].fillna("")
    pbp["false_start"] = desc.str.contains(r"false start", case=False, regex=True).astype("int8")
    pbp["delay_of_game"] = desc.str.contains(r"delay of game", case=False, regex=True).astype("int8")

    pen = pbp["penalty"].fillna(0).astype(int).eq(1)
    pbp["offensive_penalty"] = (pen & pbp["penalty_team"].eq(pbp["posteam"])).astype("int8")
    pbp["defensive_penalty"] = (pen & pbp["penalty_team"].eq(pbp["defteam"])).astype("int8")

    pbp["red_zone"] = (pbp["yardline_100"].fillna(999) <= 20).astype("int8")
    pbp["late_clock"] = (pbp["game_seconds_remaining"].fillna(99999) <= 120).astype("int8")
    pbp["high_leverage"] = pbp.get("wpa", pd.Series(np.nan, index=pbp.index)).abs().fillna(0).ge(.03).astype("int8")
    pbp["visitor_offense"] = pbp["posteam"].eq(pbp["away_team"]).astype("int8")

    # Important: this is a normalized modeling index, NOT a measured dBA value.
    pbp["noise_index_observed"] = np.nan
    pbp["noise_index_counterfactual"] = 1.0

    pbp.to_parquet(out/"plays_2014_2026.parquet", index=False)

    # Game-level table
    g = pbp.groupby("game_id", as_index=False).agg(
        season=("season","first"), week=("week","first"), game_type=("game_type","first"),
        game_date=("game_date","first"), home_team=("home_team","first"),
        away_team=("away_team","first"), stadium=("stadium","first"),
        home_score=("home_score","last"), away_score=("away_score","last"),
        home_epa=("epa", lambda x: x[pbp.loc[x.index,"posteam"].eq(pbp.loc[x.index,"home_team"])].sum()),
        away_epa=("epa", lambda x: x[pbp.loc[x.index,"posteam"].eq(pbp.loc[x.index,"away_team"])].sum()),
        total_false_starts=("false_start","sum"),
        total_delay_of_game=("delay_of_game","sum"),
        total_offensive_penalties=("offensive_penalty","sum")
    )
    g["home_win"] = (g["home_score"] > g["away_score"]).astype(int)
    g.to_csv(out/"games.csv", index=False)

    print("Finished:", len(pbp), "plays;", len(g), "games")

if __name__ == "__main__":
    main()
