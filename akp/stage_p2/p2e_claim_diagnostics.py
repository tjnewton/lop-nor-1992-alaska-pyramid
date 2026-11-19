"""Post-hoc diagnostics of the claim-zone run. Not pre-stated.

From the committed outputs (no claim tile is re-read, except by the replication):
- count excess, conditional on both counts: for a stratum, the claim count k1 on land L1 and the
  control count k0 on land L0. Under equal rates k1 | k1 + k0 is binomial with p = L1/(L1 + L0).
  One-sided p, and the rate ratio with its exact 95% interval.
- the claim anomalies above the control maximum of x (at t_low, and at t), and the number the
  frozen fit expects above it;
- the stratum's share of the weighted control exceedances over the fitted u;
- P_d at the claimed base and the largest tested relief, split into "found at t" and "passes c
  once found". Claim plants: eligible ones (owned, valid, unvetoed center); control plants:
  unvetoed, on land (third seed, unweighted). Fisher's exact test on the pass rates, and
  Mann-Whitney on the found plants' x;
- a replication of the claim-zone P_d at that cell with a fresh seed (REPLICATION_SEED), through
  the frozen runner's trial_at and the same hit rule as p2e_claim_run.inject.
Writes p2e_claim_diagnostics.csv (check, scope, value). Run: python -m akp.stage_p2.p2e_claim_diagnostics
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, fisher_exact, mannwhitneyu

from akp.stage_p2 import p2e_cut as pc
from akp.stage_p2.p2e_budget import CLAIMED
from akp.stage_p2.p2e_vintage import frozen_cut

HERE = Path(__file__).parent
REPLICATION_SEED = 20261118
STRATA = ("nopf|bedrock|forest", "nopf|bedrock|barren_ice")


def rate_ratio(k1: int, l1: float, k0: int, l0: float) -> tuple[float, float, float, float]:
    """(one-sided p, ratio, low, high) for claim rate over control rate, conditional on k1 + k0."""
    n, p0 = k1 + k0, l1 / (l1 + l0)
    p = binomtest(k1, n, p0, alternative="greater").pvalue
    lo = beta.ppf(0.025, k1, k0 + 1) if k1 > 0 else 0.0
    hi = beta.ppf(0.975, k1 + 1, k0) if k0 > 0 else np.inf
    odds = lambda q: q / (1 - q) * l0 / l1  # binomial share to rate ratio
    return p, (k1 / l1) / (k0 / l0) if k0 else np.inf, odds(lo), odds(hi) if np.isfinite(hi) and hi < 1 else np.inf


def _replicate(i: int) -> dict:
    from akp.stage_p2 import p2b_detector as det
    from akp.stage_p2.p2b_explore import VARIANTS
    from akp.stage_p2.p2e_claim_run import _inject_state, trial_at

    det.configure(**VARIANTS["V0"])
    return trial_at(i, CLAIMED["base"], CLAIMED["height"], *_inject_state(), claim_zone=True,
                    seed=REPLICATION_SEED)


def main() -> None:
    import shapely

    from akp.stage_p2.p2e_claim_run import _grid, _owners
    from akp.stage_p2.p2e_vetoes import veto_shapes

    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    t, fz = float(run.t), frozen_cut()
    c = float(fz.c)
    rows = []
    add = lambda check, scope, value: rows.append(dict(check=check, scope=scope, value=value))

    ca = pd.read_csv(HERE / "p2e_anomalies.csv", dtype={"tile_id": str})
    cu = ca[~ca.vetoed & ca.valid]
    an = pd.read_csv(HERE / "p2e_claim_anomalies.csv", dtype={"tile_id": str})
    au = an[~an.vetoed & an.valid]
    ctl_land = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum").control_unvetoed_km2
    from akp.stage_p2.p2e_claim_run import out_dir

    claim_land = pd.read_csv(out_dir() / "claim_land.csv").set_index("stratum").unvetoed_km2
    for st in STRATA:
        k1, k0 = int((au.stratum == st).sum()), int((cu.stratum == st).sum())
        p, rr, lo, hi = rate_ratio(k1, float(claim_land[st]), k0, float(ctl_land[st]))
        for k, v in (("claim_count", k1), ("control_count", k0), ("p_conditional", p), ("rate_ratio", rr),
                     ("rate_ratio_low", lo), ("rate_ratio_high", hi)):
            add(f"count_excess_{k}", st, v)

    for scope, cx, ax in (("t_low", cu.x, au.x), ("t", cu[cu.score >= t].x, au[au.above_t].x)):
        add("control_max_x", scope, float(cx.max()))
        add("claim_above_control_max", scope, int((ax > cx.max()).sum()))
    u, sg, xi, wu = float(fz.u), float(fz.sigma), float(fz.xi), float(fz.w_u)
    add("fit_expected_above_control_max", "t_low",
        wu * float(np.exp(-pc._logs(np.array([float(cu.x.max()) - u]), sg, xi)[0])))

    w = pc.weights(pd.read_csv(HERE / "p2e_tile_land.csv", dtype={"tile_id": str}), pc._searched())
    ex = cu[cu.x > float(fz.u)]
    ew = ex.stratum.map(w).fillna(0.0)
    add("exceedances_over_u", "all", len(ex))
    add("exceedances_over_u", STRATA[0], int((ex.stratum == STRATA[0]).sum()))
    add("weighted_share_of_exceedances", STRATA[0], float(ew[ex.stratum == STRATA[0]].sum() / ew.sum()))

    tr = pd.read_csv(HERE / "p2e_claim_trials.csv", dtype={"tile_id": str})
    cl = tr[(tr.base == CLAIMED["base"]) & (tr.height == CLAIMED["height"]) & (tr.miss.isna()) & ~tr.center_vetoed]
    cl_found = cl[(cl[f"score_{t:.2f}"] >= t) & cl.rep_owned & ~cl.rep_vetoed]
    ct = pd.read_csv(HERE / "p2e_pd_trials.csv", dtype={"tile_id": str})
    ct = ct[(ct.base == CLAIMED["base"]) & (ct.height == CLAIMED["height"]) & ~ct.vetoed & (ct.w > 0)]
    ct_found = ct[ct[f"score_{t:.2f}"] >= t]
    for scope, d, f in (("claim", cl, cl_found), ("control", ct, ct_found)):
        add("plants", scope, len(d))
        add("found_rate", scope, len(f) / len(d))
        add("pass_given_found", scope, float((f[f"x_{t:.2f}"] > c).mean()))
        add("median_x_found", scope, float(f[f"x_{t:.2f}"].median()))
    tab = [[int((cl_found[f"x_{t:.2f}"] > c).sum()), int((cl_found[f"x_{t:.2f}"] <= c).sum())],
           [int((ct_found[f"x_{t:.2f}"] > c).sum()), int((ct_found[f"x_{t:.2f}"] <= c).sum())]]
    add("fisher_p_pass_given_found", "claim vs control", float(fisher_exact(tab)[1]))
    add("mannwhitney_p_x_found", "claim vs control",
        float(mannwhitneyu(cl_found[f"x_{t:.2f}"].dropna(), ct_found[f"x_{t:.2f}"].dropna()).pvalue))

    from akp.stage_p2.p2b_injection import N_TRIALS
    from swe.detection import clopper_pearson

    with ProcessPoolExecutor(4) as exr:
        rep = pd.DataFrame(list(exr.map(_replicate, range(N_TRIALS))))
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    own, gt = _owners(), _grid()
    vet = lambda x, y: np.isfinite(x) and tree.query(shapely.Point(x, y), predicate="intersects").size > 0
    rx, ry = rep.get(f"rx_{t:.2f}", np.nan), rep.get(f"ry_{t:.2f}", np.nan)
    owned = [np.isfinite(x) and own.get(tuple(int(np.floor(v)) for v in (~gt * (x, y))[::-1])) == tid
             for x, y, tid in zip(rx, ry, rep.tile_id)]
    hit = (rep.get(f"score_{t:.2f}", np.nan) >= t) & (rep.get(f"x_{t:.2f}", np.nan) > c) & pd.Series(owned) \
        & ~pd.Series([vet(x, y) for x, y in zip(rx, ry)])
    k = int(hit.sum())
    lo, hi = clopper_pearson(k, len(rep))
    for kk, v in (("p_detect", k / len(rep)), ("ci_low", lo), ("ci_high", hi), ("n", len(rep))):
        add(f"replication_{kk}", "claimed base, largest relief", v)

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p2e_claim_diagnostics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
