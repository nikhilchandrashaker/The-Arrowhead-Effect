"""
schema.py: one definition of the columns each pipeline stage needs.
Builder, exposure, fitter, and simulator all call require() so a missing column
fails immediately with a clear message instead of silently dropping rows.
Verified against nflverse play_by_play_2023.csv.gz (372 columns).
NOTE: nflverse uses `season_type` (REG/POST); the builder aliases it to `game_type`.
"""

# Raw nflverse columns the builder keeps (all verified present in the real files).
RAW_KEEP = [
    "game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "location", "stadium", "stadium_id", "roof", "surface", "temp", "wind",
    "posteam", "defteam", "qtr", "game_seconds_remaining", "half_seconds_remaining",
    "quarter_seconds_remaining", "down", "ydstogo", "yardline_100", "goal_to_go",
    "score_differential", "posteam_score", "defteam_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "play_type", "shotgun", "no_huddle", "qb_dropback", "pass", "rush", "sack", "scramble",
    "yards_gained", "air_yards", "yards_after_catch", "first_down", "touchdown",
    "interception", "fumble", "fumble_lost", "penalty", "penalty_team", "penalty_type",
    "penalty_yards", "timeout", "third_down_converted", "third_down_failed", "fourth_down_converted",
    "fourth_down_failed", "two_point_conv_result", "epa", "wpa", "cpoe", "success", "desc",
    "home_score", "away_score", "field_goal_result", "kick_distance", "return_yards", "punt_blocked",
    "passer_player_id", "passer_player_name",
]

DERIVED = ["game_type", "false_start", "delay_of_game", "offensive_penalty", "defensive_penalty",
           "other_off_penalty", "visitor_offense", "neutral", "venue", "red_zone", "late_clock", "high_leverage"]

FIT_REQUIRED = [
    "game_id", "play_id", "season", "game_type", "posteam", "defteam", "home_team", "away_team", "venue",
    "neutral", "visitor_offense", "down", "ydstogo", "yardline_100", "score_differential",
    "game_seconds_remaining", "shotgun", "no_huddle", "play_type", "penalty", "timeout",
    "false_start", "delay_of_game", "offensive_penalty", "defensive_penalty", "epa",
    "first_down", "touchdown",
]

SIM_REQUIRED = [
    "game_id", "play_id", "game_type", "season", "posteam", "defteam", "home_team", "away_team",
    "visitor_offense", "down", "ydstogo", "yardline_100", "play_type", "yards_gained", "interception",
    "fumble_lost", "penalty", "false_start", "delay_of_game", "offensive_penalty", "penalty_yards",
    "timeout", "epa", "game_seconds_remaining", "kick_distance", "return_yards", "field_goal_result",
    "punt_blocked",
]

EXPOSURE_GAMES_REQUIRED = ["game_id", "season", "game_type", "home_team", "away_team", "venue",
                           "home_score", "away_score", "neutral"]

# Franchise harmonization (QC: relocations).  game_id strings keep their original abbreviations.
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def require(df, cols, who):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{who}] required columns missing: {missing}")
    return df
