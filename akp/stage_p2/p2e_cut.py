"""P2e / B-gate: the one-method route's cut on x = delta_bic per window pixel.

The rule (`cut`): N unvetoed control anomalies with a valid window, k = round(TAIL_FRAC * N),
u the largest x below the top k; route closed if k < K_MIN. A generalized Pareto tail
(shape xi in [0, XI_MAX]) is fitted to the top-k exceedances over u by a w-weighted
likelihood, w = w_s of the anomaly's stratum (unvetoed searched land / unvetoed control land).
E(c) = (sum of the top-k weights) * S(c - u), an expected count over the searched area. Its
LEVEL upper bound is the LEVEL quantile over N_BOOT bootstrap replicates that resample control
tiles and rerun the same estimator (a failed fit or k < K_MIN is an infinite E). c is the
smallest value whose bound is within BUDGET. LEVEL and N_TAIL were set on synthetic data.
The real run fits at t_low, the highest tabulated P2b threshold at or below the operating t
whose anomaly count reaches N_TAIL: the greedy merge at t keeps a subset of the anomalies at
t_low, with the same windows, so E at t_low bounds E at t from above.
`split_half` runs the rule on each seeded half and applies the fixed two falsifiers to the other.
`simulate` runs the rule on synthetic control sets (the real tiles, strata and land; known
tails) and returns the true E at the rule's cut; main(--sim) writes p2e_cut_sim.csv.
Tile land: p2e_tile_land.csv (tile_id, stratum, unvetoed_km2), from the P2b control-run cells
and the statewide veto fraction (p2e_vetoes.py), written by main(--land).
Run: python -m akp.stage_p2.p2e_cut --land | --sim | --summary | --check (far-tail checks, p2e_tail_check.csv)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from akp.paths import data_dir

HERE = Path(__file__).parent
TAIL_FRAC = 0.25
K_MIN = 10
XI_MAX = 2.0
LEVEL = 0.995  # chosen on synthetic data (p2e_cut_sim_summary.csv)
N_TAIL = 160  # smallest tested anomaly count at which the rule passed its synthetic check
N_BOOT = 1000
BUDGET = 0.04  # The one-method route's share through P2b
LADDER = (10.0, 3.0, 1.0)
LADDER_ALPHA = 0.01
SEED = 20261114
N_SIM = 100
SIM_NS = (160, 320, 640)  # expected anomalies per synthetic control set
SIM_WORKERS = 4
SIM_TAILS = ("exponential", "normal", "gpd0.2", "gpd0.4")
SIM_LEVELS = (0.95, 0.99, 0.995)  # bound levels compared on synthetic data (rule check)
SIM_GATES = (0.2, 0.3, np.inf)  # route closed when the fitted shape exceeds the gate


def _logs(y: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    """log1p(xi * y / sigma) / xi, continuous at xi = 0 (where it is y / sigma)."""
    z = y / sigma
    return np.log1p(xi * z) / xi if xi > 1e-8 else z


def fit(x: np.ndarray, w: np.ndarray):
    """(u, sigma, xi, W_u) of the weighted tail fit, or None (k < K_MIN or a failed fit)."""
    k = int(round(TAIL_FRAC * len(x)))
    if k < K_MIN:
        return None
    o = np.argsort(-x, kind="stable")
    u = float(x[o[k]])
    y, wy = x[o[:k]] - u, w[o[:k]]
    if not (y > 0).any():
        return None

    def nll(p):
        s, xi = np.exp(p[0]), p[1]
        return -float(np.sum(wy * (-np.log(s) - (1 + xi) * _logs(y, s, xi))))

    s0 = np.log(np.mean(y))
    r = minimize(nll, x0=[s0, 0.1], bounds=[(s0 - 10, s0 + 10), (0.0, XI_MAX)], method="L-BFGS-B")
    if not r.success or not np.isfinite(r.fun):
        return None
    return u, float(np.exp(r.x[0])), float(r.x[1]), float(wy.sum())


def expected(f, c: float) -> float:
    """E(c) of a fit: weighted count above u times the tail probability beyond c (c >= u)."""
    if f is None:
        return np.inf
    u, s, xi, wu = f
    return wu * float(np.exp(-_logs(np.array([max(c - u, 0.0)]), s, xi)[0]))


def boot_fits(x: np.ndarray, w: np.ndarray, tile: np.ndarray, rng) -> list:
    """N_BOOT tail fits on control tiles resampled with replacement (None where a fit fails)."""
    tiles = np.unique(tile)
    idx = {t: np.flatnonzero(tile == t) for t in tiles}
    return [fit(x[p], w[p]) for p in
            (np.concatenate([idx[t] for t in rng.choice(tiles, size=len(tiles))]) for _ in range(N_BOOT))]


def cut(x: np.ndarray, w: np.ndarray, tile: np.ndarray, rng, budget: float = BUDGET,
        level: float = LEVEL, boots: list | None = None) -> dict:
    f = fit(x, w)
    if f is None:
        return dict(closed=True, c=np.inf, k=int(round(TAIL_FRAC * len(x))))
    boots = boot_fits(x, w, tile, rng) if boots is None else boots
    n_fail = sum(b is None for b in boots)

    def bound(c: float) -> float:
        return float(np.quantile([expected(b, c) for b in boots], level))

    out = dict(closed=False, k=int(round(TAIL_FRAC * len(x))), u=f[0], sigma=f[1], xi=f[2], w_u=f[3],
               boot_failed=n_fail)
    if n_fail > (1 - level) * N_BOOT:
        return out | dict(closed=True, c=np.inf)
    lo, hi = f[0], f[0] + f[1]
    while bound(hi) > budget:
        lo, hi = hi, hi + 2 * (hi - f[0])
        if hi - f[0] > 1e9 * f[1]:  # the bound never reaches the budget
            return out | dict(closed=True, c=np.inf)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if bound(mid) > budget else (lo, mid)
    return out | dict(c=hi, e_point=expected(f, hi), e_bound=bound(hi), boots=boots)


def weights(land: pd.DataFrame, searched: pd.Series) -> pd.Series:
    """w_s = unvetoed searched land / unvetoed control land, per stratum, for these tiles."""
    ctrl = land.groupby("stratum").unvetoed_km2.sum()
    return (searched / ctrl.where(ctrl > 0)).dropna()


def split_half(anom: pd.DataFrame, land: pd.DataFrame, searched: pd.Series, rng) -> dict:
    """The fixed split-half falsifiers; anom has tile_id, stratum, x."""
    tiles = np.array(sorted(land.tile_id.unique(), key=int))
    a_tiles = set(rng.permutation(tiles)[: len(tiles) // 2])
    half = {"A": land.tile_id.isin(a_tiles), "B": ~land.tile_id.isin(a_tiles)}
    ahalf = {"A": anom.tile_id.isin(a_tiles), "B": ~anom.tile_id.isin(a_tiles)}
    out = {}
    for fit_h, test_h in (("A", "B"), ("B", "A")):
        lf, lt = land[half[fit_h]], land[half[test_h]]
        af, at = anom[ahalf[fit_h]], anom[ahalf[test_h]]
        w = weights(lf, searched)
        x, wv = af.x.to_numpy(), af.stratum.map(w).fillna(0.0).to_numpy()
        c = cut(x, wv, af.tile_id.to_numpy(), rng)
        out[f"c_{fit_h}"] = c["c"]
        out[f"over_c_{fit_h}"] = int((at.x > c["c"]).sum())
        f = fit(x, wv)
        fired = False
        if f is not None and "boots" in c:
            ratio = (lt.groupby("stratum").unvetoed_km2.sum() / lf.groupby("stratum").unvetoed_km2.sum()).fillna(0.0)
            top = af.iloc[np.argsort(-x, kind="stable")[: int(round(TAIL_FRAC * len(x)))]]
            r = float(top.stratum.map(ratio).fillna(0.0).sum())  # B's expected count above u
            u, s, xi, wu = f
            rho = r / wu  # B's expected count per unit of A's searched-area count
            for m in LADDER:
                if m >= r:
                    continue  # the level would sit below u, outside the fitted tail
                q = np.log(r / m)  # S(y) = m / r
                level = u + (s * q if xi <= 1e-8 else s * np.expm1(xi * q) / xi)
                n = int((at.x > level).sum())
                # predictive: Poisson counts mixed over A's bootstrap fits (fit uncertainty included)
                means = np.array([rho * expected(b, level) for b in c["boots"]])
                p = float(np.mean([1.0 if not np.isfinite(mb) else poisson.sf(n - 1, mb) for mb in means]))
                fired |= p < LADDER_ALPHA
        out[f"ladder_fired_{fit_h}"] = fired
    out["fired"] = bool(out["over_c_A"] or out["over_c_B"] or out["ladder_fired_A"] or out["ladder_fired_B"])
    return out


def tile_land() -> pd.DataFrame:
    from akp.stage_p2.p2a_strata import strata_grid
    from akp.stage_p2.p2b_controls_run import compact_cells, stratum_namer

    _, s = strata_grid()
    _, _, name = stratum_namer(s)
    f = np.load(data_dir("p2e") / "veto_fraction.npy")
    rows = {}
    tiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str})
    for t in sorted(tiles.tile_id, key=int):
        res = json.loads((data_dir("p2b/control_run/tiles") / f"{t}.json").read_text())
        for r, c, a in compact_cells([res["cells"]]):
            r, c = int(r), int(c)
            k = name(r, c)
            if k == "offland":
                continue
            fr = float(f[r, c]) if 0 <= r < f.shape[0] and 0 <= c < f.shape[1] else 0.0
            rows[(t, k)] = rows.get((t, k), 0.0) + a * (1.0 - fr)
        del res
    out = pd.DataFrame([dict(tile_id=t, stratum=k, unvetoed_km2=round(v, 3)) for (t, k), v in rows.items()])
    out.to_csv(HERE / "p2e_tile_land.csv", index=False)
    return out


def _searched() -> pd.Series:
    v = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum")
    return v.searched_unvetoed_km2.drop("_total")


def _draw(kind: str, n: int, rng) -> np.ndarray:
    if kind == "exponential":
        return rng.exponential(1.0, n)
    if kind == "normal":
        return rng.normal(0.0, 1.0, n)
    xi = float(kind[3:])
    return rng.pareto(1 / xi, n) / xi  # GPD(shape xi, scale 1) is Lomax(1/xi) times 1/xi


def _survival(kind: str, c: float) -> float:
    from scipy.stats import genpareto, norm

    if kind == "exponential":
        return float(np.exp(-c)) if c > 0 else 1.0
    return float(norm.sf(c)) if kind == "normal" else float(genpareto.sf(c, float(kind[3:])))


def _ideal(kind: str, p: float) -> float:
    """The cut where the true tail probability is p (the cut a perfect model would set)."""
    from scipy.stats import genpareto, norm

    if kind == "exponential":
        return float(-np.log(p))
    return float(norm.isf(p)) if kind == "normal" else float(genpareto.isf(p, float(kind[3:])))


def simulate(args) -> dict:
    kind, n_exp, i = args
    rng = np.random.default_rng([SEED, SIM_TAILS.index(kind), n_exp, i])
    land = pd.read_csv(HERE / "p2e_tile_land.csv", dtype={"tile_id": str})
    searched = _searched()
    strata = sorted(land.stratum.unique())
    fs = dict(zip(strata, rng.uniform(0.5, 2.0, len(strata))))  # stratum-varying anomaly rates
    lam = land.stratum.map(fs) * land.unvetoed_km2
    rho0 = n_exp / lam.sum()
    n = rng.poisson(rho0 * lam.to_numpy())
    anom = pd.DataFrame(dict(tile_id=np.repeat(land.tile_id.to_numpy(), n), stratum=np.repeat(land.stratum.to_numpy(), n)))
    anom["x"] = _draw(kind, len(anom), rng)
    w = weights(land, searched)
    x, wv, tl = anom.x.to_numpy(), anom.stratum.map(w).fillna(0.0).to_numpy(), anom.tile_id.to_numpy()
    boots = boot_fits(x, wv, tl, rng)
    true_rate = sum(rho0 * fs.get(k, 0.0) * searched.get(k, 0.0) for k in searched.index)
    c_ideal = _ideal(kind, BUDGET / true_rate)
    sh = split_half(anom, land, searched, rng)
    rows = []
    for lv in SIM_LEVELS:
        c = cut(x, wv, tl, rng, level=lv, boots=boots)
        for gate in SIM_GATES:
            closed = c["closed"] or c["xi"] > gate
            e_true = 0.0 if closed else true_rate * _survival(kind, c["c"])
            rows.append(dict(tail=kind, n_expected=n_exp, sim=i, level=lv, gate=gate, n=len(anom), xi=c.get("xi"),
                             closed=closed, c=c["c"], c_ideal=c_ideal, e_true=e_true,
                             over_budget=bool(e_true > BUDGET), split_fired=sh["fired"]))
    return rows


def tail_check() -> pd.DataFrame:
    """Far-tail checks of the committed fits (p2e_cut{tag}.csv on p2e_anomalies{tag}.csv).

    Per operating point: the expected number of control anomalies above the largest observed x
    under the fit (k * S), and the chance of one or more. Per stratum, for an exponential fit
    (xi = 0): the stratum's exceedances over u summed, against Gamma(n, sigma), one-sided for a
    heavier tail than the pooled fit. Writes p2e_tail_check.csv.
    """
    from scipy.stats import gamma

    rows = []
    for tag in ("", "_t2"):
        c = pd.read_csv(HERE / f"p2e_cut{tag}.csv").iloc[0]
        an = pd.read_csv(HERE / f"p2e_anomalies{tag}.csv", dtype={"tile_id": str})
        u_an = an[~an.vetoed & an.valid & (an.score >= c.t_low)]
        f = (float(c.u), float(c.sigma), float(c.xi), float(c.k))
        x_max = float(u_an.x.max())
        lam = f[3] * float(np.exp(-_logs(np.array([x_max - f[0]]), f[1], f[2])[0]))
        rows.append(dict(t=c.t, scope="all", n=len(u_an), n_above_u=int((u_an.x > f[0]).sum()), x_max=round(x_max, 4),
                         expected_above_max=round(lam, 4), p_value=round(float(1 - np.exp(-lam)), 4)))
        for st, g in u_an.groupby("stratum"):
            e = g.x[g.x > f[0]] - f[0]
            p = float(gamma.sf(e.sum(), len(e), scale=f[1])) if len(e) and f[2] <= 1e-8 else np.nan
            rows.append(dict(t=c.t, scope=st, n=len(g), n_above_u=len(e), x_max=round(float(g.x.max()), 4),
                             expected_above_max=np.nan, p_value=round(p, 4) if p == p else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p2e_tail_check.csv", index=False)
    return out


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, 4000)
    f = fit(x, np.ones_like(x))
    assert f is not None and abs(f[1] - 2.0) < 0.3 and f[2] < 0.1, f  # exponential: sigma 2, xi 0
    from scipy.stats import genpareto

    g = _draw("gpd0.2", 200_000, rng)  # the synthetic heavy tail matches the survival used as truth
    assert abs(np.mean(g > 5.0) - genpareto.sf(5.0, 0.2)) < 0.01, np.mean(g > 5.0)
    assert abs(expected(f, f[0]) - f[3]) < 1e-9 and expected(f, f[0] + 10) < expected(f, f[0] + 5)
    print("p2e_cut self-check OK:", tuple(round(v, 3) for v in f))


def main() -> None:
    if "--land" in sys.argv:
        land = tile_land()
        print(f"{land.tile_id.nunique()} tiles, {land.unvetoed_km2.sum():.0f} unvetoed km2; wrote p2e_tile_land.csv")
        return
    if "--sim" in sys.argv:
        jobs = [(k, n, i) for k in SIM_TAILS for n in SIM_NS for i in range(N_SIM)]
        with ProcessPoolExecutor(SIM_WORKERS) as ex:
            out = pd.DataFrame([r for rows in ex.map(simulate, jobs) for r in rows])
        out.to_csv(HERE / "p2e_cut_sim.csv", index=False)
    if "--check" in sys.argv:
        print(tail_check().to_string(index=False))
        return
    if "--sim" in sys.argv or "--summary" in sys.argv:
        out = pd.read_csv(HERE / "p2e_cut_sim.csv")
        summ = out.groupby(["tail", "n_expected", "level", "gate"]).agg(
            over_budget=("over_budget", "mean"), closed=("closed", "mean"), split_fired=("split_fired", "mean"),
            c_over_ideal=("c", lambda c: float(np.median(c / out.loc[c.index, "c_ideal"])))).reset_index()
        summ.round(3).to_csv(HERE / "p2e_cut_sim_summary.csv", index=False)
        print(summ.round(3).to_string(index=False))
        return
    _selfcheck()


if __name__ == "__main__":
    main()
