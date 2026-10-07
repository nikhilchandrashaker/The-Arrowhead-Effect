"""
simulate.py: state-dependent, sequential counterfactual NFL simulator.

Pipeline
  fitted effects (draws.npz)  ->  per-game scenario shifts  ->  play-by-play game engine
  ->  season (schedule, standings, seeds)  ->  playoffs  ->  Super Bowl
  Each simulated season uses ONE coefficient draw, so reported intervals contain
  parameter uncertainty + simulation noise.  --param-mode mean isolates the latter.

How the baseline works (no invented effect sizes)
  * Play outcomes are resampled from observed plays that share the same STATE:
    (offense venue home/away, down, distance bucket, field-position bucket).
    Baseline = observed world, so home/away execution gaps are already embedded.
  * Team strength: shrunken offense / defense EPA ratings converted to yards.
  * Pre-snap penalties (false start, delay, other offensive) are drawn explicitly from
    state-dependent base rates.

How the counterfactual works
  For each play, a scenario shifts log-odds / EPA by
      delta = effect(scenario crowd) - effect(observed crowd)
      effect = (b0 + b1*visitor) * fill + (b2 + b3*visitor) * fill * intensity_z
  with b drawn from the fitted model (dose-response specs).  Shifts feed
    false start  -> 5 yards, same down, longer distance   (M01)
    delay        -> 5 yards, same down                    (M02)
    other off. penalty -> ~10 yards                       (M05b)
    execution    -> yards-per-play shift = dEPA / (EPA per yard)  (M06, non-penalty plays)
  and propagate forward through down, distance, field position, clock, score,
  possession, and the schedule.  Third-down conversion is NOT shifted directly (it would
  double count M06 and the penalty channel); M04 is used as a validation target instead.

Known v1 simplifications (documented, not hidden)
  * Timeouts and hurry-up behavior are not modeled; the state vector has room for them.
  * No QB-level effects yet (team ratings only); QB resilience is the next module.
  * OT is sudden death (first score wins), 10 minutes, ties possible.
  * 2-point tries, onside kicks, defensive/return touchdowns, kickoff returns are not modeled.
  * Playoff tiebreaks are random; neutral-site Super Bowl has zero crowd shift.
  * Intensity above 1.0 (Extreme) extrapolates beyond the anchor; interpret accordingly.
"""
import argparse, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
@dataclass
class Scenario:
    name: str
    fill: float | None        # None => keep observed
    intensity: float | None   # None => keep observed; Arrowhead anchor = 1.0 (modeled index, not dBA)

SCENARIOS = [
    Scenario("Baseline", None, None),
    Scenario("No crowd", 0.0, 0.0),
    Scenario("Conservative Arrowhead", 1.0, 0.50),
    Scenario("Central Arrowhead", 1.0, 1.00),
    Scenario("Extreme Arrowhead", 1.0, 1.25),
]

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAMS = sorted(t for v in DIVISIONS.values() for t in v)
TIDX = {t: i for i, t in enumerate(TEAMS)}
CONF = {c: [t for d, v in DIVISIONS.items() if d.startswith(c) for t in v] for c in ("AFC", "NFC")}

MODEL_FOR = {"fs": "M01_false_start", "dl": "M02_delay_of_game",
             "op": "M05b_other_off_pen", "epa": "M06_epa_nonpen"}
SECS_FS, SECS_DL, SECS_OP = 6.0, 10.0, 8.0
PAT_PCT = 0.94

def sig(x): return 1.0 / (1.0 + np.exp(-x))
def lg(p): p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))

# ----------------------------------------------------------------------------- baseline tables
YTG_BINS = [3, 6, 10, 11, 16]
YL_BINS = [11, 21, 41, 61, 81]

def _key(vis, down, ytg, yl):
    return ((vis * 4 + (down - 1)) * 6 + np.digitize(ytg, YTG_BINS)) * 6 + np.digitize(yl, YL_BINS)

class Baseline:
    """Empirical state-conditional resampling tables built from the play table."""

    def __init__(self, plays, rating_seasons=None, shrink_n=400):
        p = plays.sort_values(["game_id", "play_id"]).copy()
        g = p.groupby("game_id")["game_seconds_remaining"]
        p["secs"] = (g.shift(0) - g.shift(-1)).clip(0, 45)
        p["secs"] = p["secs"].fillna(p["secs"].median())
        p = p[p["game_type"] == "REG"]
        for c in ["false_start", "delay_of_game", "offensive_penalty", "penalty", "timeout",
                  "interception", "fumble_lost", "punt_blocked"]:
            if c not in p: p[c] = 0
            p[c] = p[c].fillna(0)
        p = p[p["down"].notna() & (p["timeout"] != 1)].copy()
        p["vis"] = (p["posteam"] == p["away_team"]).astype(int)
        p["down"] = p["down"].astype(int)

        # --- non-penalty scrimmage outcomes
        sc = p[p["play_type"].isin(["pass", "run"]) & (p["penalty"] == 0)
               & p[["yards_gained", "ydstogo", "yardline_100"]].notna().all(axis=1)].copy()
        sc = sc[sc["down"].between(1, 4)]
        key = _key(sc["vis"].values, sc["down"].values, sc["ydstogo"].values, sc["yardline_100"].values)
        self.Y = sc["yards_gained"].values.astype(float)
        self.TO = ((sc["interception"] == 1) | (sc["fumble_lost"] == 1)).values.astype(np.int8)
        self.S = sc["secs"].values.astype(float)
        self.k = float(np.polyfit(sc["yards_gained"], sc["epa"].fillna(0), 1)[0])  # EPA per yard
        self._build_pools(key, sc)
        # --- penalty base rates [vis, down-1]
        snap = p[p["play_type"].isin(["pass", "run", "no_play"])]
        oth = (snap["offensive_penalty"] - snap["false_start"] - snap["delay_of_game"]).clip(lower=0)
        snap = snap.assign(other=oth)
        def rate(col):
            r = snap.groupby(["vis", "down"])[col].mean().reindex(
                pd.MultiIndex.from_product([[0, 1], [1, 2, 3, 4]]), fill_value=np.nan)
            return r.fillna(snap[col].mean()).values.reshape(2, 4)
        self.P_FS, self.P_DL, self.P_OP = rate("false_start"), rate("delay_of_game"), rate("other")
        opy = snap[(snap["other"] > 0) & snap["penalty_yards"].notna()]["penalty_yards"].abs() \
            if "penalty_yards" in snap else pd.Series(dtype=float)
        self.OP_YDS = opy.values if len(opy) else np.array([10.0])
        # --- special teams
        fgp = p[(p["play_type"] == "field_goal") & p["kick_distance"].notna()] if "kick_distance" in p else p.iloc[:0]
        self.FG_P = np.array([.99, .95, .85, .70, .55])
        if len(fgp) > 200:
            grp = np.digitize(fgp["kick_distance"], [30, 40, 50, 55])
            made = (fgp["field_goal_result"] == "made").astype(float)
            for gi in range(5):
                if (grp == gi).sum() > 20: self.FG_P[gi] = made[grp == gi].mean()
        pu = p[(p["play_type"] == "punt") & p["kick_distance"].notna() & (p["punt_blocked"] != 1)] if "kick_distance" in p else p.iloc[:0]
        self.PUNT_NET = (pu["kick_distance"] - pu["return_yards"].fillna(0)).values if len(pu) else np.array([40.0])
        # --- team ratings (EPA/play, shrunken), converted to yards
        rs = sc if rating_seasons is None else sc[sc["season"].isin(rating_seasons)]
        lm = rs["epa"].mean()
        def shrink(grp, sign):
            a = rs.groupby(grp)["epa"].agg(["mean", "count"])
            return sign * (a["mean"] - lm) * a["count"] / (a["count"] + shrink_n)
        off = shrink("posteam", 1.0).reindex(TEAMS).fillna(0.0)
        dfn = shrink("defteam", 1.0).reindex(TEAMS).fillna(0.0)   # EPA allowed above avg => bad defense => +yards
        self.off_shift, self.def_shift = off.values / self.k, dfn.values / self.k
        # --- observed targets for validation
        t3 = sc[sc["down"] == 3]
        self.obs = {"third_down_conv": float(((t3["yards_gained"] >= t3["ydstogo"])).mean()),
                    "false_start_rate": float(snap["false_start"].mean()),
                    "delay_rate": float(snap["delay_of_game"].mean())}

    def _build_pools(self, key, sc):
        nb = 2 * 4 * 6 * 6
        order = np.argsort(key, kind="stable")
        counts = np.bincount(key, minlength=nb)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pool = [order]
        extra_start = len(order)
        starts, counts = starts.copy(), counts.copy()
        vis, down = key // 144, (key // 36) % 4
        ytgb = (key // 6) % 6
        coarse = {1: ytgb + 6 * (down + 4 * vis), 2: down + 4 * vis}
        for b in np.flatnonzero(counts < 25):
            bv, bd, by = b // 144, (b // 36) % 4, (b // 6) % 6
            idx = np.flatnonzero(coarse[1] == by + 6 * (bd + 4 * bv))
            if len(idx) < 25: idx = np.flatnonzero(coarse[2] == bd + 4 * bv)
            if len(idx) == 0: idx = np.arange(len(key))
            starts[b], counts[b] = extra_start, len(idx)
            pool.append(idx); extra_start += len(idx)
        self.pool = np.concatenate(pool)
        self.starts, self.counts = starts, counts

    def sample(self, key, rng):
        j = self.starts[key] + (rng.random(len(key)) * self.counts[key]).astype(int)
        return self.pool[j]

# ----------------------------------------------------------------------------- scenario shifts
def load_draws(fit_dir):
    fit_dir = Path(fit_dir)
    meta = json.load(open(fit_dir / "coefficients.json"))
    z = np.load(fit_dir / "draws.npz")
    roles = {frozenset(["crowd_fill"]): 0, frozenset(["crowd_fill", "visitor_offense"]): 1,
             frozenset(["crowd_fill", "intensity_z"]): 2,
             frozenset(["crowd_fill", "visitor_offense", "intensity_z"]): 3}
    out = {}
    for key, m in meta.items():
        if not key.endswith("_dose"): continue
        cols = {roles[frozenset(t.split(":"))]: z[key][:, j] for j, t in enumerate(m["terms"])}
        out[key[:-5]] = np.stack([cols[i] for i in range(4)], axis=1)
    missing = [v for v in MODEL_FOR.values() if v not in out]
    if missing: raise SystemExit(f"fitted models missing from draws: {missing}; rerun fit_models.py")
    return out

def effect(B, vis, fill, z):
    return (B[:, 0] + B[:, 1] * vis) * fill + (B[:, 2] + B[:, 3] * vis) * fill * z

def make_deltas(draws, did, fill_o, int_o, scn, mu, sd):
    """Per-instance shifts, shape (n, 2): [:,0] home offense, [:,1] visiting offense."""
    fill_s = fill_o if scn.fill is None else np.full_like(fill_o, scn.fill)
    int_s = int_o if scn.intensity is None else np.full_like(int_o, scn.intensity)
    z_o, z_s = (int_o - mu) / sd, (int_s - mu) / sd
    out = {}
    for k, name in MODEL_FOR.items():
        B = draws[name][did]
        out[k] = np.stack([effect(B, v, fill_s, z_s) - effect(B, v, fill_o, z_o) for v in (0, 1)], axis=1)
    return out

# ----------------------------------------------------------------------------- game engine
def play_games(base, home, away, dl, rng, ot=True, max_steps=700):
    """Vectorized sequential game simulation. Returns (home_pts, away_pts, counters)."""
    n = len(home)
    frh = rng.random(n) < 0.5                       # home receives opening kickoff
    S = dict(ph=frh.copy(), yl=np.full(n, 75), down=np.ones(n, int), ytg=np.full(n, 10),
             t=np.full(n, 3600.0), sh=np.zeros(n, int), sa=np.zeros(n, int), ot=np.zeros(n, bool))
    C = np.zeros((n, 12), np.int32)                 # counters: k*2+vis; k: fs,dl,op,snaps,att3,conv3
    active = np.ones(n, bool)
    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0: break
        _step(S, C, idx, home, away, dl, rng, base, frh, active, ot)
    return S["sh"], S["sa"], C

def _step(S, C, idx, home, away, dl, rng, base, frh, active, ot_enabled):
    ph = S["ph"][idx]; vis = (~ph).astype(int)
    off = np.where(ph, home[idx], away[idx]); de = np.where(ph, away[idx], home[idx])
    d, y, l, t0 = S["down"][idx], S["ytg"][idx], S["yl"][idx], S["t"][idx]
    sh, sa, inot = S["sh"][idx].copy(), S["sa"][idx].copy(), S["ot"][idx]
    sd = np.where(ph, sh - sa, sa - sh)
    m = len(idx)
    is4 = d == 4
    go = is4 & (l > 35) & (((y <= 2) & (l <= 50)) | ((sd < 0) & (t0 < 400) & (y <= 8)))
    fgm = is4 & (l <= 35)
    pun = is4 & ~go & ~fgm
    scr = ~(fgm | pun)
    nl, nd, ny, nph = l.copy(), d.copy(), y.copy(), ph.copy()
    secs = np.zeros(m); p_off = np.zeros(m, int); p_def = np.zeros(m, int)
    kick = np.zeros(m, bool); scored = np.zeros(m, bool)

    def flip(mask, spot):                      # defense takes over; spot = their yardline_100
        spot = np.clip(spot, 1, 99)
        nph[mask] = ~ph[mask]; nd[mask] = 1; nl[mask] = spot; ny[mask] = np.minimum(10, spot)

    s = np.flatnonzero(scr)
    if s.size:
        v, ds, gi = vis[s], d[s] - 1, idx[s]
        pfs = sig(lg(base.P_FS[v, ds]) + dl["fs"][gi, v])
        pdl = sig(lg(base.P_DL[v, ds]) + dl["dl"][gi, v])
        pop = sig(lg(base.P_OP[v, ds]) + dl["op"][gi, v])
        u = rng.random(s.size)
        isfs = u < pfs
        isdl = ~isfs & (u < pfs + pdl)
        isop = ~isfs & ~isdl & (u < pfs + pdl + pop)
        pen = isfs | isdl | isop
        C[gi, 3 * 2 + v] += 1
        C[gi[isfs], 0 * 2 + v[isfs]] += 1; C[gi[isdl], 1 * 2 + v[isdl]] += 1; C[gi[isop], 2 * 2 + v[isop]] += 1
        # penalty replays the down
        L = np.where(isfs | isdl, 5, 0).astype(int)
        if isop.any(): L[isop] = rng.choice(base.OP_YDS, isop.sum()).astype(int)
        sp = s[pen]
        nl[sp] = np.minimum(l[sp] + L[pen], 99); ny[sp] = np.minimum(y[sp] + L[pen], nl[sp])
        secs[sp] = np.where(isfs[pen], SECS_FS, np.where(isdl[pen], SECS_DL, SECS_OP))
        # normal snap
        nm = ~pen; sn = s[nm]
        if sn.size:
            vn, dn, ln, yn, on, dn_ = v[nm], d[sn], l[sn], y[sn], off[sn], de[sn]
            r = base.sample(_key(vn, dn, yn, ln), rng)
            shift = base.off_shift[on] + base.def_shift[dn_] + dl["epa"][idx[sn], vn] / base.k
            yards = np.rint(base.Y[r] + shift).astype(int)
            to = base.TO[r] == 1
            secs[sn] = base.S[r]
            newl = ln - yards
            td = ~to & (newl <= 0)
            saf = ~to & (newl >= 100)
            third = dn == 3
            C[idx[sn][third], 4 * 2 + vn[third]] += 1
            C[idx[sn][third & (yards >= yn)], 5 * 2 + vn[third & (yards >= yn)]] += 1
            # outcomes
            pat = (rng.random(sn.size) < PAT_PCT).astype(int)
            p_off[sn[td]] = 6 + pat[td]; kick[sn[td]] = True; scored[sn[td]] = True
            p_def[sn[saf]] = 2
            flip(sn[saf], np.full(saf.sum(), 60))
            tm = sn[to & ~td]; flip(tm, 100 - np.clip(newl[to & ~td], 1, 99))
            cont = ~td & ~saf & ~to
            fd = cont & (yards >= yn)
            nl[sn[fd]] = newl[fd]; nd[sn[fd]] = 1; ny[sn[fd]] = np.minimum(10, newl[fd])
            nf = cont & ~fd
            nl[sn[nf]] = newl[nf]; nd[sn[nf]] = dn[nf] + 1; ny[sn[nf]] = yn[nf] - yards[nf]
            tod = nf & (dn == 4)                       # failed 4th down
            flip(sn[tod], 100 - np.clip(newl[tod], 1, 99))

    f = np.flatnonzero(fgm)
    if f.size:
        dist = l[f] + 17
        made = rng.random(f.size) < base.FG_P[np.digitize(dist, [30, 40, 50, 55])]
        secs[f] = 6.0
        p_off[f[made]] = 3; kick[f[made]] = True; scored[f[made]] = True
        flip(f[~made], np.minimum(80, 93 - l[f[~made]]))
    q = np.flatnonzero(pun)
    if q.size:
        net = rng.choice(base.PUNT_NET, q.size)
        bl = l[q] - net
        spot = np.where(bl <= 0, 80, np.clip(100 - bl, 1, 99)).astype(int)
        secs[q] = 10.0
        flip(q, spot)

    # kickoff after scores
    nph[kick] = ~ph[kick]; nl[kick] = 75; nd[kick] = 1; ny[kick] = 10
    sh = sh + np.where(ph, p_off, p_def); sa = sa + np.where(ph, p_def, p_off)
    t1 = t0 - secs

    # OT sudden death
    if ot_enabled:
        end_ot = inot & (scored | (p_def > 0) | (t1 <= 0))
    else:
        end_ot = np.zeros(m, bool)
    # halftime
    h = ~inot & (t0 > 1800) & (t1 <= 1800)
    t1 = np.where(h, 1800.0, t1)
    nph = np.where(h, ~frh[idx], nph)
    nl = np.where(h, 75, nl); nd = np.where(h, 1, nd); ny = np.where(h, 10, ny)
    end_reg = ~inot & ~h & (t1 <= 0)
    done = end_ot | end_reg
    tie = sh == sa
    start_ot = end_reg & tie & ot_enabled
    if start_ot.any():
        so = np.flatnonzero(start_ot)
        t1[so] = 600.0
        nph[so] = rng.random(so.size) < 0.5
        nl[so] = 75; nd[so] = 1; ny[so] = 10
        S["ot"][idx[so]] = True
    S["ph"][idx], S["yl"][idx], S["down"][idx], S["ytg"][idx] = nph, nl, nd, ny
    S["t"][idx], S["sh"][idx], S["sa"][idx] = t1, sh, sa
    active[idx[done & ~start_ot]] = False

# ----------------------------------------------------------------------------- season + playoffs
def seed_conference(score, conf):
    """score: (32,) wins+tiebreak. Returns list of 7 team indices ordered by seed + division-winner mask."""
    c = np.array([TIDX[t] for t in CONF[conf]])
    winners = []
    for dname, ts in DIVISIONS.items():
        if dname.startswith(conf):
            di = np.array([TIDX[t] for t in ts]); winners.append(di[np.argmax(score[di])])
    winners = np.array(winners); winners = winners[np.argsort(-score[winners])]
    rest = np.array([t for t in c if t not in set(winners)])
    wc = rest[np.argsort(-score[rest])][:3]
    return np.concatenate([winners, wc]), set(winners.tolist())

def playoff_deltas(draws, did, scn, team_int, mu, sd, neutral):
    """Playoff games: observed = full crowd at the home team's stadium intensity."""
    n = len(did)
    fill_o = np.ones(n); int_o = team_int
    out = make_deltas(draws, did, fill_o, int_o, scn, mu, sd)
    if neutral:
        for k in out: out[k] = np.zeros_like(out[k])
    return out

def sim_games_batch(base, draws, did, h, a, fill_o, int_o, scn, mu, sd, rng, neutral=False, team_int_for=None):
    if team_int_for is not None:
        dl = playoff_deltas(draws, did, scn, team_int_for, mu, sd, neutral)
    else:
        dl = make_deltas(draws, did, fill_o, int_o, scn, mu, sd)
    return play_games(base, h, a, dl, rng)

def run_scenario(base, draws, sched, scn, nseasons, chunk, draw_ids, seed, mu, sd, team_int):
    rng = np.random.default_rng(seed)
    G = len(sched)
    home = sched["home_idx"].values; away = sched["away_idx"].values
    fill_o = sched["crowd_fill"].values.astype(float); int_o = sched["intensity_mean"].values.astype(float)
    wins = np.zeros((nseasons, 32)); ties = np.zeros((nseasons, 32))
    pts = np.zeros(nseasons); hw = np.zeros(nseasons)
    cnt = np.zeros((nseasons, 12))
    po = {k: np.zeros((nseasons, 32), np.int8) for k in
          ["playoff", "div", "seed1", "conf_game", "sb", "champ"]}
    for c0 in range(0, nseasons, chunk):
        c1 = min(nseasons, c0 + chunk); S = c1 - c0
        did = np.repeat(draw_ids[c0:c1], G)
        h, a = np.tile(home, S), np.tile(away, S)
        sh, sa, C = sim_games_batch(base, draws, did, h, a, np.tile(fill_o, S), np.tile(int_o, S),
                                    scn, mu, sd, rng)
        sh, sa = sh.reshape(S, G), sa.reshape(S, G)
        pts[c0:c1] = (sh + sa).mean(axis=1)
        hw[c0:c1] = ((sh > sa) + 0.5 * (sh == sa)).mean(axis=1)
        cnt[c0:c1] = C.reshape(S, G, 12).sum(axis=1)
        for si in range(S):
            hwin = np.where(sh[si] > sa[si], 1.0, np.where(sh[si] == sa[si], 0.5, 0.0))
            np.add.at(wins[c0 + si], home, hwin); np.add.at(wins[c0 + si], away, 1 - hwin)
            np.add.at(ties[c0 + si], home, (sh[si] == sa[si]) * 1.0); np.add.at(ties[c0 + si], away, (sh[si] == sa[si]) * 1.0)
        # --- seeding
        sc = wins[c0:c1] + rng.random((S, 32)) * 1e-3
        seeds = {c: [seed_conference(sc[si], c) for si in range(S)] for c in ("AFC", "NFC")}
        for c in ("AFC", "NFC"):
            for si in range(S):
                order, dw = seeds[c][si]
                po["playoff"][c0 + si, order] = 1
                po["div"][c0 + si, list(dw)] = 1
                po["seed1"][c0 + si, order[0]] = 1
        # --- wild card (seed1 bye): 2v7, 3v6, 4v5 ; then divisional ; then conference title
        alive = {c: [list(seeds[c][si][0]) for si in range(S)] for c in ("AFC", "NFC")}  # team by seed order
        seed_of = {c: [{t: i + 1 for i, t in enumerate(alive[c][si])} for si in range(S)] for c in ("AFC", "NFC")}
        def play_round(pairs):    # pairs: list of (season_i, conf, high_team, low_team)
            if not pairs: return []
            hs = np.array([p[2] for p in pairs]); aw = np.array([p[3] for p in pairs])
            dd = np.array([draw_ids[c0 + p[0]] for p in pairs])
            sh_, sa_, _ = sim_games_batch(base, draws, dd, hs, aw, None, None, scn, mu, sd, rng,
                                          team_int_for=team_int[hs])
            coin = rng.random(len(pairs)) < 0.5
            return np.where(sh_ > sa_, hs, np.where(sa_ > sh_, aw, np.where(coin, hs, aw)))
        wc_pairs = []
        for c in ("AFC", "NFC"):
            for si in range(S):
                o = alive[c][si]
                wc_pairs += [(si, c, o[1], o[6]), (si, c, o[2], o[5]), (si, c, o[3], o[4])]
        wcw = play_round(wc_pairs)
        surv = {(c, si): [alive[c][si][0]] for c in ("AFC", "NFC") for si in range(S)}
        for p, w in zip(wc_pairs, wcw): surv[(p[1], p[0])].append(int(w))
        dv_pairs = []
        for (c, si), ts in surv.items():
            ts = sorted(ts, key=lambda t: seed_of[c][si][t])
            dv_pairs += [(si, c, ts[0], ts[3]), (si, c, ts[1], ts[2])]
        dvw = play_round(dv_pairs)
        surv2 = {}
        for p, w in zip(dv_pairs, dvw): surv2.setdefault((p[1], p[0]), []).append(int(w))
        cf_pairs = []
        for (c, si), ts in surv2.items():
            ts = sorted(ts, key=lambda t: seed_of[c][si][t])
            for t in ts: po["conf_game"][c0 + si, t] = 1
            cf_pairs.append((si, c, ts[0], ts[1]))
        cfw = play_round(cf_pairs)
        champs = {}
        for p, w in zip(cf_pairs, cfw):
            champs.setdefault(p[0], []).append(int(w)); po["sb"][c0 + p[0], int(w)] = 1
        # Super Bowl: neutral site, zero crowd shift
        sb = [(si, "SB", ch[0], ch[1]) for si, ch in champs.items()]
        hs = np.array([p[2] for p in sb]); aw = np.array([p[3] for p in sb])
        flip_ = rng.random(len(sb)) < 0.5
        hs, aw = np.where(flip_, aw, hs), np.where(flip_, hs, aw)
        dd = np.array([draw_ids[c0 + p[0]] for p in sb])
        sh_, sa_, _ = sim_games_batch(base, draws, dd, hs, aw, None, None, scn, mu, sd, rng,
                                      neutral=True, team_int_for=np.zeros(len(sb)))
        coin = rng.random(len(sb)) < 0.5
        win = np.where(sh_ > sa_, hs, np.where(sa_ > sh_, aw, np.where(coin, hs, aw)))
        for p, w in zip(sb, win): po["champ"][c0 + p[0], int(w)] = 1
    return dict(wins=wins, ties=ties, pts=pts, hw=hw, cnt=cnt, po=po)

# ----------------------------------------------------------------------------- reporting
def q(x, ax=0): return np.percentile(x, [2.5, 97.5], axis=ax)

def summarize(results, nseasons):
    team_rows, lg_rows = [], []
    for name, r in results.items():
        w, lo_hi = r["wins"], q(r["wins"])
        for i, t in enumerate(TEAMS):
            team_rows.append(dict(scenario=name, team=t, mean_wins=w[:, i].mean(), wins_p2_5=lo_hi[0, i],
                                  wins_p97_5=lo_hi[1, i], p_playoff=r["po"]["playoff"][:, i].mean(),
                                  p_division=r["po"]["div"][:, i].mean(), p_seed1=r["po"]["seed1"][:, i].mean(),
                                  p_conf_title_game=r["po"]["conf_game"][:, i].mean(),
                                  p_super_bowl=r["po"]["sb"][:, i].mean(), p_champion=r["po"]["champ"][:, i].mean()))
        c = r["cnt"]
        row = dict(scenario=name, points_per_game=r["pts"].mean(),
                   ppg_p2_5=q(r["pts"])[0], ppg_p97_5=q(r["pts"])[1],
                   home_win_pct=r["hw"].mean(), hw_p2_5=q(r["hw"])[0], hw_p97_5=q(r["hw"])[1])
        for vis, lab in ((0, "home_off"), (1, "visitor_off")):
            sn = c[:, 3 * 2 + vis]
            for k, nm in ((0, "false_start"), (1, "delay"), (2, "other_off_pen")):
                row[f"{nm}_rate_{lab}"] = (c[:, k * 2 + vis] / sn).mean()
            row[f"third_down_conv_{lab}"] = (c[:, 5 * 2 + vis] / np.maximum(c[:, 4 * 2 + vis], 1)).mean()
        lg_rows.append(row)
    return pd.DataFrame(team_rows), pd.DataFrame(lg_rows)

def effect_diagnostics(draws, sched, scns, mu, sd, out):
    """Print/save the shifts each scenario implies BEFORE simulating. Warn on implausible values."""
    n = len(sched); fo = sched["crowd_fill"].values.astype(float); io = sched["intensity_mean"].values.astype(float)
    rows = []
    for s in scns:
        if s.name == "Baseline": continue
        for did_label, did in (("mean", None), ("p2.5", 0.025), ("p97.5", 0.975)):
            ds = {}
            for k, name in MODEL_FOR.items():
                B = draws[name]
                b = B.mean(axis=0, keepdims=True) if did is None else np.percentile(B, did * 100, axis=0, keepdims=True)
                ds[k] = np.stack([effect(np.repeat(b, n, 0), v, (fo if s.fill is None else np.full(n, s.fill)),
                                         ((io if s.intensity is None else np.full(n, s.intensity)) - mu) / sd)
                                  - effect(np.repeat(b, n, 0), v, fo, (io - mu) / sd) for v in (0, 1)], axis=1)
            for k in ds:
                for vi, lab in enumerate(("home_off", "visitor_off")):
                    rows.append(dict(scenario=s.name, quantile=did_label, channel=k, offense=lab, mean_shift=float(ds[k][:, vi].mean())))
    df = pd.DataFrame(rows); df.to_csv(out / "scenario_shift_diagnostics.csv", index=False)
    m = df[df["quantile"] == "mean"].pivot_table(index=["scenario", "offense"], columns="channel", values="mean_shift")
    print("Scenario-implied shifts at posterior mean (logit for fs/dl/op, EPA/play for epa):")
    print(m.round(3).to_string())
    bad = df[(df["channel"].isin(["fs", "dl", "op"]) & (df["mean_shift"].abs() > 1.0)) |
             ((df["channel"] == "epa") & (df["mean_shift"].abs() > 0.15))]
    if len(bad):
        print(f"\nWARNING: {len(bad)} implausibly large shifts (|logit|>1 or |EPA|>0.15). Inspect scenario_shift_diagnostics.csv; "
              "these usually mean an unstable coefficient, not a real effect. Do not publish until resolved.\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="built/plays_2014_2026.parquet")
    ap.add_argument("--exposure", default="built/games_exposure.csv")
    ap.add_argument("--fitted", default="fitted")
    ap.add_argument("--schedule-season", type=int, required=True)
    ap.add_argument("--rating-seasons", type=int, nargs="*", default=None)
    ap.add_argument("--seasons", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--param-mode", choices=["draw", "mean"], default="draw")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default="sim_out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    plays = pd.read_parquet(a.plays) if a.plays.endswith(".parquet") else pd.read_csv(a.plays)
    base = Baseline(plays, a.rating_seasons)
    draws = load_draws(a.fitted)
    if a.param_mode == "mean":
        draws = {k: np.repeat(v.mean(axis=0, keepdims=True), len(v), axis=0) for k, v in draws.items()}
    ex = pd.read_csv(a.exposure)
    sched = ex[(ex["season"] == a.schedule_season) & (ex["game_type"] == "REG")].copy()
    bad = {t for t in set(sched["home_team"]) | set(sched["away_team"]) if t not in TIDX}
    if bad: raise SystemExit(f"team abbreviations not in DIVISIONS map: {bad}")
    sched["home_idx"] = sched["home_team"].map(TIDX); sched["away_idx"] = sched["away_team"].map(TIDX)
    mu, sd = ex["intensity_mean"].mean(), ex["intensity_mean"].std()
    # NB: matches exposure_index.py (z-score over the games table). Keep these identical.
    team_int = ex[ex["season"] == a.schedule_season].groupby("home_team")["intensity_mean"].mean() \
        .reindex(TEAMS).fillna(ex["intensity_mean"].mean()).values
    rng0 = np.random.default_rng(a.seed)
    draw_ids = rng0.integers(0, len(next(iter(draws.values()))), a.seasons)   # shared across scenarios

    scns = [s for s in SCENARIOS if a.scenarios is None or s.name in a.scenarios]
    effect_diagnostics(draws, sched, scns, mu, sd, out)
    results = {}
    for s in scns:
        t0 = time.time()
        results[s.name] = run_scenario(base, draws, sched, s, a.seasons, a.chunk, draw_ids, a.seed, mu, sd, team_int)
        print(f"{s.name:26s} {a.seasons} seasons in {time.time()-t0:6.0f}s")
    teams, league = summarize(results, a.seasons)
    teams.to_csv(out / "team_results.csv", index=False); league.to_csv(out / "league_summary.csv", index=False)
    np.savez_compressed(out / "wins_by_season.npz", **{k: v["wins"] for k, v in results.items()})
    # paired differences vs Baseline (same coefficient draws by season)
    if "Baseline" in results:
        rows = []
        for k, v in results.items():
            if k == "Baseline": continue
            d = v["wins"] - results["Baseline"]["wins"]; qq = q(d)
            for i, t in enumerate(TEAMS):
                rows.append(dict(scenario=k, team=t, d_wins=d[:, i].mean(), d_wins_p2_5=qq[0][i], d_wins_p97_5=qq[1][i]))
        pd.DataFrame(rows).to_csv(out / "paired_win_changes.csv", index=False)
    # validation of the baseline engine against the observed world
    obs = base.obs
    L = league.set_index("scenario")
    if "Baseline" in L.index:
        b = L.loc["Baseline"]
        print("\nBaseline validation (engine vs observed play data):")
        print(f"  false-start rate : sim home {b['false_start_rate_home_off']:.4f}/visitor {b['false_start_rate_visitor_off']:.4f}  obs {obs['false_start_rate']:.4f}")
        print(f"  3rd-down conv    : sim {b['third_down_conv_home_off']:.3f}/{b['third_down_conv_visitor_off']:.3f}  obs {obs['third_down_conv']:.3f}")
        hist = sched[["home_score", "away_score"]].dropna() if "home_score" in sched else None
        if hist is not None and len(hist):
            print(f"  points/game      : sim {b['points_per_game']:.1f}  obs {(hist.home_score + hist.away_score).mean():.1f}")
            print(f"  home win %       : sim {b['home_win_pct']:.3f}  obs {(hist.home_score > hist.away_score).mean():.3f}")
    json.dump(dict(seed=a.seed, seasons=a.seasons, param_mode=a.param_mode, schedule_season=a.schedule_season,
                   scenarios=[s.name for s in scns], epa_per_yard=base.k, intensity_mu=mu, intensity_sd=sd,
                   note="exposure is a modeled index, not measured dBA"), open(out / "run_meta.json", "w"), indent=2)
    print("Wrote", out)

if __name__ == "__main__":
    main()
