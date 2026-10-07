"""
Exposure-index calibration layer.

Exposure for a game = crowd_fill x stadium_intensity

  crowd_fill        in [0, 1]  attendance / capacity. Identified by the 2020
                               season (zero or limited fans) vs. normal seasons.
  stadium_intensity ~ prior    normalized so the Arrowhead anchor (142.2 dBA peak)
                               = 1.0. This is a MODELING INDEX, not measured dBA.

The main estimates (crowd presence effect) depend only on crowd_fill.
Stadium intensity is used for dose-response sensitivity and to scale the
conservative / central / aggressive scenarios. Its uncertainty is carried
forward as draws so the simulator can propagate it.

Inputs
  games.csv                   from build step (needs game_id, stadium, season, home_team)
  attendance.csv              REQUIRED, you must source it: game_id, attendance
                              (verify against nflverse schedules or team/NFL sources;
                              this script does not invent attendance)
  stadium_capacity.csv        REQUIRED: stadium, season, capacity
  stadium_intensity_prior.csv OPTIONAL: stadium, intensity_mean, intensity_sd, basis
Outputs
  games_exposure.csv          one row per game with fill, regime, intensity draws summary
"""
import argparse
import numpy as np
import pandas as pd

# Documented peak readings (noise_events.csv). Used only to anchor the top of the scale.
ANCHOR_STADIUM = "Arrowhead Stadium"

# ASSUMPTION (flagged, sensitivity-tested): stadiums with no documented reading
# get a wide prior on the 0-1 scale. Change via CLI; report results across values.
DEFAULT_PRIOR_MEAN = 0.6
DEFAULT_PRIOR_SD = 0.2


def regime(fill):
    if pd.isna(fill):
        return "unknown"
    if fill < 0.02:
        return "empty"
    if fill < 0.5:
        return "limited"
    return "full"


def build(games, attendance, capacity, prior=None,
          prior_mean=DEFAULT_PRIOR_MEAN, prior_sd=DEFAULT_PRIOR_SD):
    g = (games.merge(attendance[["game_id", "attendance"]], on="game_id", how="left")
              .merge(capacity, on=["stadium", "season"], how="left"))
    g["crowd_fill"] = (g["attendance"] / g["capacity"]).clip(0, 1)
    g["crowd_regime"] = g["crowd_fill"].map(regime)

    g["intensity_mean"] = prior_mean
    g["intensity_sd"] = prior_sd
    if prior is not None:
        g = g.drop(columns=["intensity_mean", "intensity_sd"]).merge(
            prior[["stadium", "intensity_mean", "intensity_sd"]], on="stadium", how="left")
        g["intensity_mean"] = g["intensity_mean"].fillna(prior_mean)
        g["intensity_sd"] = g["intensity_sd"].fillna(prior_sd)
    # Anchor is fixed by definition.
    a = g["stadium"].eq(ANCHOR_STADIUM)
    g.loc[a, ["intensity_mean", "intensity_sd"]] = [1.0, 0.0]

    g["exposure_mean"] = g["crowd_fill"] * g["intensity_mean"]
    g["intensity_z"] = (g["intensity_mean"] - g["intensity_mean"].mean()) / g["intensity_mean"].std()
    return g


def intensity_draws(stadiums, n=2000, seed=42):
    """Draws of stadium intensity (truncated to [0, 1.25]) for propagation."""
    rng = np.random.default_rng(seed)
    d = stadiums.drop_duplicates("stadium").set_index("stadium")
    out = {}
    for s, r in d.iterrows():
        out[s] = np.clip(rng.normal(r["intensity_mean"], r["intensity_sd"], n), 0, 1.25)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="built/games.csv")
    ap.add_argument("--attendance", default="attendance.csv")
    ap.add_argument("--capacity", default="stadium_capacity.csv")
    ap.add_argument("--prior", default=None)
    ap.add_argument("--prior-mean", type=float, default=DEFAULT_PRIOR_MEAN)
    ap.add_argument("--prior-sd", type=float, default=DEFAULT_PRIOR_SD)
    ap.add_argument("--out", default="built/games_exposure.csv")
    a = ap.parse_args()
    prior = pd.read_csv(a.prior) if a.prior else None
    g = build(pd.read_csv(a.games), pd.read_csv(a.attendance), pd.read_csv(a.capacity),
              prior, a.prior_mean, a.prior_sd)
    g.to_csv(a.out, index=False)
    print(g.groupby(["season", "crowd_regime"]).size().unstack(fill_value=0))
    missing = g["crowd_fill"].isna().sum()
    if missing:
        print(f"WARNING: {missing} games lack attendance/capacity; fix before modeling.")


if __name__ == "__main__":
    main()
