"""
fit_models.py: estimate crowd-presence / intensity effects on offensive outcomes.

Design (difference-in-differences):
  logit(Y) = a + b1*crowd_fill + b2*crowd_fill:visitor_offense + b3*visitor_offense
             + situation controls + FE(season, posteam, defteam, stadium)

  Season FE absorb common 2020 shocks, so identification comes from how the
  VISITOR-vs-HOME offense gap changes with crowd_fill.  Visitor effect of a full
  crowd vs. an empty one = b1 + b2.  Home effect = b1.
  Sensitivity (dose-response): adds crowd_fill:intensity_z (+ x visitor).

Outputs (outdir):
  coefficients.json   per-model exposure coefficients, covariance, n, cluster count
  draws.npz           multivariate-normal draws of exposure coefficients for the simulator
  placebo.csv         placebo-test estimates
  robustness.csv      excl-2014-record-game, regular-season-only, etc.

Notes
  * SEs are clustered by game_id.  QB/stadium random effects are approximated by
    fixed effects here; for the paper, refit M01/M03 hierarchically (bambi/brms).
  * Everything is labeled as a modeled index, never measured dBA.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

SITUATION = ("C(down) + ydstogo + yardline_100 + score_differential + "
             "game_seconds_remaining + shotgun + no_huddle")
FE = "C(season) + C(posteam) + C(defteam) + C(stadium)"
RECORD_GAME = "2014_04_NE_KC"   # Sept 29, 2014; verify id format against built data

EXPO = ["crowd_fill", "crowd_fill:visitor_offense"]
EXPO_INT = EXPO + ["crowd_fill:intensity_z", "crowd_fill:intensity_z:visitor_offense"]

MODELS = {
    # name: (outcome, family, extra RHS, filter)
    "M01_false_start":   ("false_start", "logit", "", "snap"),
    "M02_delay_of_game": ("delay_of_game", "logit", "", "snap"),
    "M03_epa":           ("epa", "ols", "", "snap"),
    "M04_third_down":    ("conversion", "logit", "", "third"),
    "M05_off_penalty":   ("offensive_penalty", "logit", "", "snap"),
    "M05b_other_off_pen": ("other_off_penalty", "logit", "", "snap"),
    "M06_epa_nonpen":    ("epa", "ols", "", "nonpen"),   # execution effect net of penalties (feeds simulator)
}


def prep(plays, exposure):
    d = plays.merge(exposure[["game_id", "crowd_fill", "intensity_z"]], on="game_id", how="inner")
    d = d.dropna(subset=["crowd_fill", "down", "ydstogo", "yardline_100", "posteam"])
    for c in ["false_start", "delay_of_game", "offensive_penalty", "defensive_penalty", "penalty", "timeout"]:
        d[c] = d[c].fillna(0)
    d["other_off_penalty"] = (d["offensive_penalty"] - d["false_start"] - d["delay_of_game"]).clip(lower=0)
    d["conversion"] = ((d["first_down"].fillna(0) == 1) | (d["touchdown"].fillna(0) == 1)).astype(int)
    return d


def subset(d, kind, regular_only=True):
    """
    snap   : every offensive snap attempt, INCLUDING 'no_play' rows (false starts / delay of game
             are no_play in nflverse). Excludes timeouts and non-snap rows (down is NaN).
    third  : 3rd-down pass/run attempts (replayed no_plays are not attempts).
    nonpen : pass/run with no penalty, so execution is not mixed with penalty yardage.
    """
    if regular_only:
        d = d[d["game_type"] == "REG"]
    d = d[d["down"].notna() & (d["timeout"] != 1)]
    if kind == "snap":
        return d[d["play_type"].isin(["pass", "run", "no_play"])]
    d = d[d["play_type"].isin(["pass", "run"])]
    if kind == "third":
        d = d[d["down"] == 3]
    elif kind == "nonpen":
        d = d[d["penalty"] == 0]
    return d


def fit(d, outcome, family, expo_terms, extra=""):
    rhs = " + ".join(expo_terms + ["visitor_offense", SITUATION, FE] + ([extra] if extra else []))
    f = f"{outcome} ~ {rhs}"
    if family == "logit":
        m = smf.glm(f, d, family=sm.families.Binomial())
    else:
        m = smf.ols(f, d)
    return m.fit(cov_type="cluster", cov_kwds={"groups": d["game_id"].astype("category").cat.codes})


def expo_summary(res, terms):
    keys = [k for k in res.params.index
            if any(set(k.split(":")) == set(t.split(":")) for t in terms)]
    return keys


def run(plays, exposure, outdir, ndraws=5000, seed=42):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d0 = prep(plays, exposure)
    coefs, draws, robust = {}, {}, []

    for name, (y, fam, extra, kind) in MODELS.items():
        d = subset(d0, kind)
        for tag, terms in [("", EXPO), ("_dose", EXPO_INT)]:
            res = fit(d, y, fam, terms, extra)
            ks = expo_summary(res, terms)
            b, V = res.params[ks], res.cov_params().loc[ks, ks]
            key = name + tag
            coefs[key] = {"terms": ks, "coef": b.tolist(), "se": res.bse[ks].tolist(),
                          "cov": V.values.tolist(), "n": int(res.nobs),
                          "n_games": int(d["game_id"].nunique()), "family": fam}
            draws[key] = rng.multivariate_normal(b.values, V.values, ndraws)
            print(key, dict(zip(ks, np.round(b.values, 4))))

        # ---- robustness on the primary spec only ----
        for label, dd in [
            ("excl_record_game", d[d["game_id"] != RECORD_GAME]),
            ("reg_plus_post", subset(d0, kind, regular_only=False)),
            ("excl_2020", d[d["season"] != 2020]),   # expect no crowd_fill identification; sanity
        ]:
            if dd["crowd_fill"].nunique() < 2:
                continue
            try:
                r = fit(dd, y, fam, EXPO, extra)
                for k in expo_summary(r, EXPO):
                    robust.append({"model": name, "spec": label, "term": k,
                                   "coef": r.params[k], "se": r.bse[k]})
            except Exception as e:  # collinearity when 2020 is dropped, etc.
                robust.append({"model": name, "spec": label, "term": "FAILED", "coef": np.nan, "se": str(e)[:80]})

    # ---- placebo 1: fake treatment, 2019 assigned a 2020-style fill pattern ----
    placebo = []
    pl = d0[d0["season"].isin([2018, 2019])].copy()
    if len(pl):
        base = exposure[exposure["season"] == 2020]["crowd_fill"].dropna().values
        if len(base):
            pg = pl[["game_id"]].drop_duplicates()
            pg["fake_fill"] = rng.choice(base, len(pg))
            pl = pl.merge(pg, on="game_id"); pl["crowd_fill"] = pl["fake_fill"]
            for name in ["M01_false_start", "M06_epa_nonpen"]:
                y, fam, extra, kind = MODELS[name]
                r = fit(subset(pl, kind), y, fam, EXPO, extra)
                for k in expo_summary(r, EXPO):
                    placebo.append({"test": "fake_2020_pattern_in_2018_19", "model": name,
                                    "term": k, "coef": r.params[k], "se": r.bse[k]})

    # ---- placebo 2: outcome that crowd noise should NOT drive (defensive penalties) ----
    d = subset(d0, "snap")
    r = fit(d, "defensive_penalty", "logit", EXPO)
    for k in expo_summary(r, EXPO):
        placebo.append({"test": "placebo_outcome_defensive_penalty", "model": "defensive_penalty",
                        "term": k, "coef": r.params[k], "se": r.bse[k]})

    (outdir / "coefficients.json").write_text(json.dumps(coefs, indent=2))
    np.savez(outdir / "draws.npz", seed=seed, **draws)
    pd.DataFrame(placebo).to_csv(outdir / "placebo.csv", index=False)
    pd.DataFrame(robust).to_csv(outdir / "robustness.csv", index=False)
    print("Wrote", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="built/plays_2014_2026.parquet")
    ap.add_argument("--exposure", default="built/games_exposure.csv")
    ap.add_argument("--out", default="fitted")
    ap.add_argument("--draws", type=int, default=5000)
    a = ap.parse_args()
    p = pd.read_parquet(a.plays) if a.plays.endswith(".parquet") else pd.read_csv(a.plays)
    run(p, pd.read_csv(a.exposure), a.out, a.draws)
