"""Vintage check before the freeze: does the claim zone's DEM-project mix move the cut?

Metadata and control data only (no claim tile is opened).
- Claim land by project: each searched land cell of the P2a grid, counted at its unvetoed share
  ($AKP_DATA/p2e/veto_fraction.npy), goes to the most recently published claim tile whose
  bounding box holds the cell center (ties: file name). This approximates the owner rule,
  which needs the tiles' validity.
- Control land by project: p2e_tile_land.csv (unvetoed) joined to p2a_controls.csv.
- Groups: projects holding at least MIN_SHARE of the claim land and some control land; the rest
  pool into "other". r_g = claim share / control share.
- The anomalies' weights become w_s * r_g (product post-stratification: it assumes stratum and
  project independent), rescaled so the weighted control land equals the searched unvetoed land.
  The cut is recomputed with the same estimator, seed and level.
Writes p2e_vintage.csv (group, claim_km2, claim_share, control_km2, control_share, r) and
p2e_vintage_cut.csv (weighting, c, e_bound_at_frozen_c, the fit u, sigma, xi, w_u, and P_d at the claimed size at that c
for both seeds, with the trial weights). Run: python -m akp.stage_p2.p2e_vintage
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping

from akp.paths import data_dir
from akp.stage_p2 import p2e_cut as pc
from akp.stage_p2.p2a_strata import strata_grid, to_3338
from akp.stage_p2.p2e_budget import CLAIMED
from akp.stage_p2.p2e_claim_run import claim_tiles, searched_cells
from akp.stage_p2.p2e_pd import summary

HERE = Path(__file__).parent
MIN_SHARE = 0.05


def claim_land_by_project() -> pd.Series:
    transform, s = strata_grid()
    land = np.zeros(s.shape, bool)
    for r, c in searched_cells():
        land[r, c] = True
    keep = 1.0 - np.load(data_dir("p2e") / "veto_fraction.npy")
    owned = np.zeros(s.shape, bool)
    km2: dict[str, float] = {}
    for t in claim_tiles().sort_values(["publication_date", "file"], ascending=[False, True]).itertuples():
        g = to_3338(box(t.min_lon, t.min_lat, t.max_lon, t.max_lat).segmentize(0.01))
        m = ~geometry_mask([mapping(g)], s.shape, transform) & land & ~owned
        owned |= m
        km2[t.project] = km2.get(t.project, 0.0) + float(keep[m].sum())
    assert owned.sum() == land.sum(), "a searched land cell lies in no claim tile's bounding box"
    return pd.Series(km2)


def vintage(claim: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(group table, cuts) for claim unvetoed land by project: the cut and the stratum x
    vintage cut, each with its fit (u, sigma, xi, w_u), its bound at the c, and P_d."""
    ctiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str})[["tile_id", "project"]]
    land = pd.read_csv(HERE / "p2e_tile_land.csv", dtype={"tile_id": str})
    ctrl = land.merge(ctiles, on="tile_id").groupby("project").unvetoed_km2.sum()
    big = [p for p in claim.index if claim[p] / claim.sum() >= MIN_SHARE and ctrl.get(p, 0) > 0]
    grp = lambda p: p if p in big else "other"
    cg, kg = claim.groupby(grp).sum(), ctrl.groupby(grp).sum()
    tab = pd.DataFrame({"claim_km2": cg, "control_km2": kg}).fillna(0.0)
    assert (tab.control_km2 > 0).all(), "a claim group has no control land"
    tab["claim_share"] = tab.claim_km2 / tab.claim_km2.sum()
    tab["control_share"] = tab.control_km2 / tab.control_km2.sum()
    tab["r"] = tab.claim_share / tab.control_share
    tab.index.name = "group"

    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    an = pd.read_csv(HERE / "p2e_anomalies.csv", dtype={"tile_id": str})
    u = an[~an.vetoed & an.valid].merge(ctiles, on="tile_id")
    searched = pc._searched()
    ws = pc.weights(land, searched)
    r = ctiles.set_index("tile_id").project.map(grp).map(tab.r)
    lw = land.assign(w=land.stratum.map(ws).fillna(0.0) * land.tile_id.map(r))
    scale = searched.sum() / float((lw.unvetoed_km2 * lw.w).sum())
    x, tile = u.x.to_numpy(), u.tile_id.to_numpy()
    v = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum").loc["_total"]
    blind = float(1 - v.searched_unvetoed_km2 / v.searched_km2)
    trials = {sd: pd.read_csv(HERE / f"p2e_pd_trials{tag}.csv", dtype={"tile_id": str})
              for sd, tag in (("third", ""), ("fourth", "_t2"))}
    rows = []
    for name, w in (("stratum", u.stratum.map(ws).fillna(0.0).to_numpy()),
                    ("stratum x vintage", u.stratum.map(ws).fillna(0.0).to_numpy() * u.tile_id.map(r).to_numpy() * scale)):
        c = pc.cut(x, w, tile, np.random.default_rng(pc.SEED))
        b = float(np.quantile([pc.expected(f, float(run.c)) for f in c["boots"]], pc.LEVEL))
        row = dict(weighting=name, c=c["c"], e_bound_at_frozen_c=b, u=c["u"], sigma=c["sigma"], xi=c["xi"],
                   w_u=c["w_u"])
        for sd, df in trials.items():  # P_d at the claimed size at this weighting's cut (trial weights)
            p = summary(df, float(run.t), c["c"], blind)
            row[f"p_detect_{sd}"] = float(p[(p.base == CLAIMED["base"]) & (p.height == CLAIMED["height"])].p_detect.iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    assert np.isclose(out.c.iloc[0], run.c), "the weighting does not reproduce the committed cut"
    return tab, out


def frozen_cut() -> pd.Series:
    """The frozen cut: the largest c in p2e_vintage_cut.csv, with its fit."""
    v = pd.read_csv(HERE / "p2e_vintage_cut.csv")
    return v.loc[v.c.idxmax()]


def main() -> None:
    tab, out = vintage(claim_land_by_project())
    tab.round(4).to_csv(HERE / "p2e_vintage.csv")
    out.to_csv(HERE / "p2e_vintage_cut.csv", index=False)
    print(tab.round(3).to_string())
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
